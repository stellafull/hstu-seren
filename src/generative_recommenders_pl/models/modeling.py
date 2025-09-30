"""Multi-task HSTU modeling with serendipity head."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, Callable, Dict, Tuple

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata

from generative_recommenders_pl.models.generative_recommenders import (
    GenerativeRecommenders,
)
from generative_recommenders_pl.models.heads import RelHead, SerHead
from generative_recommenders_pl.models.losses.autoregressive_losses import (
    AutoregressiveLoss,
)
from generative_recommenders_pl.models.losses.multitask_losses import compute_losses
from generative_recommenders_pl.models.metrics.ranking_calc import (
    compute_ranking_metrics,
    compute_ser_metrics,
)
from generative_recommenders_pl.models.utils.features import seq_features_from_row


class HSTUSeren(GenerativeRecommenders):
    """HSTU backbone with multi-head objectives for serendipity modeling."""

    def __init__(
        self,
        *,
        datamodule,
        embeddings,
        preprocessor,
        sequence_encoder,
        postprocessor,
        similarity,
        negatives_sampler,
        candidate_index,
        loss: AutoregressiveLoss | DictConfig | None = None,
        optimizer,
        scheduler,
        configure_optimizer_params,
        gr_output_length: int,
        item_embedding_dim: int,
        compile_model: bool,
        tie_weights: bool = True,
        loss_weights: Dict[str, float] | DictConfig | None = None,
        pos_weight: float = 1.0,
        metrics: Dict[str, Any] | DictConfig | None = None,
        vocab_size: int | None = None,
        init_from_ckpt: str | None = None,
        freeze_base_epochs: int = 0,
    ) -> None:
        super().__init__(
            datamodule=datamodule,
            embeddings=embeddings,
            preprocessor=preprocessor,
            sequence_encoder=sequence_encoder,
            postprocessor=postprocessor,
            similarity=similarity,
            negatives_sampler=negatives_sampler,
            candidate_index=candidate_index,
            loss=loss,
            metrics=None,
            optimizer=optimizer,
            scheduler=scheduler,
            configure_optimizer_params=configure_optimizer_params,
            gr_output_length=gr_output_length,
            item_embedding_dim=item_embedding_dim,
            compile_model=compile_model,
        )

        embedding_module = getattr(self.embeddings, "_item_emb", None)
        if embedding_module is None:
            raise ValueError("HSTUSeren requires embeddings with an _item_emb attribute")

        weight = embedding_module.weight if tie_weights else None
        vocab = vocab_size or embedding_module.num_embeddings
        self.rel_head = RelHead(
            hidden_size=item_embedding_dim,
            vocab_size=vocab,
            tied_weight=weight,
        )
        self.ser_head = SerHead(item_embedding_dim)

        if not isinstance(self.loss, AutoregressiveLoss):
            raise ValueError(
                "HSTUSeren requires an AutoregressiveLoss for the relevance head."
            )
        if self.negatives_sampler is None or self.similarity is None:
            raise ValueError(
                "HSTUSeren requires both a negatives sampler and a similarity module."
            )

        weights = loss_weights or {"rel": 1.0, "ser": 1.0}
        if isinstance(weights, DictConfig):
            weights = dict(weights)
        self.loss_weights = {
            "rel": float(weights.get("rel", 1.0)),
            "ser": float(weights.get("ser", 1.0)),
        }
        self.pos_weight = float(max(pos_weight, 1e-6))

        self.freeze_base_epochs = max(int(freeze_base_epochs), 0)

        self._maybe_load_checkpoint(init_from_ckpt)
        if self.freeze_base_epochs > 0:
            self._set_backbone_requires_grad(enabled=False)

        metrics_cfg: Dict[str, Any]
        if metrics is None:
            metrics_cfg = {}
        elif isinstance(metrics, DictConfig):
            metrics_cfg = OmegaConf.to_container(metrics, resolve=True) or {}
        else:
            metrics_cfg = dict(metrics)
        ranking_cfg = metrics_cfg.get("ranking", {})
        ser_cfg = metrics_cfg.get("serendipity", {})
        self.metric_ks = sorted(set(ranking_cfg.get("ks", [5, 10])))
        self.ser_metric_ks = sorted(set(ser_cfg.get("ks", self.metric_ks)))

        self._metric_buffers: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
            lambda: {"ranks": [], "ser_mask": []}
        )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> tuple:
        seq_features, target_ids, _ = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )
        target_ids = target_ids.squeeze(1)

        target_positions = seq_features.past_lengths
        seq_features.past_ids.scatter_(
            dim=1,
            index=target_positions.view(-1, 1),
            src=target_ids.view(-1, 1),
        )

        # Include the supervised target token in the valid sequence length so the
        # encoder actually processes the appended position instead of masking it out.
        seq_features = seq_features._replace(past_lengths=target_positions + 1)
        prediction_positions = torch.clamp(target_positions - 1, min=0)

        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        encoded_embeddings, _ = super().forward(seq_features)
        batch_indices = torch.arange(
            encoded_embeddings.size(0), device=encoded_embeddings.device
        )
        hidden = encoded_embeddings[batch_indices, prediction_positions, :]

        logits_next = self.rel_head(hidden)
        logits_ser = self.ser_head(hidden)

        ser_labels = batch.get("target_ser_label")
        if ser_labels is None:
            ser_labels = torch.zeros_like(target_ids, dtype=logits_ser.dtype)
        ser_labels = ser_labels.to(self.device)

        outputs = {
            "logits_next": logits_next,
            "logits_ser": logits_ser,
            "hidden": hidden,
        }
        return outputs, target_ids, ser_labels

    def _compute_rel_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relevance loss via the configured autoregressive objective."""

        hidden = outputs["hidden"]
        valid_mask = target_ids != 0
        if not torch.any(valid_mask):
            # Return a zero tensor that still participates in autograd.
            return hidden.sum() * 0.0

        supervision_ids = target_ids[valid_mask]
        supervision_embeddings = self.embeddings.get_item_embeddings(supervision_ids)
        output_embeddings = hidden[valid_mask]
        supervision_weights = output_embeddings.new_ones(supervision_ids.shape)

        # Ensure negative sampler uses the latest item embeddings (tie-weights case).
        if hasattr(self.negatives_sampler, "_item_emb"):
            self.negatives_sampler._item_emb = self.embeddings._item_emb

        if hasattr(self.negatives_sampler, "process_batch"):
            presences = torch.ones_like(supervision_ids, dtype=torch.bool)
            self.negatives_sampler.process_batch(
                ids=supervision_ids,
                presences=presences,
                embeddings=supervision_embeddings,
            )

        try:
            return self.loss.jagged_forward(
                output_embeddings=output_embeddings,
                supervision_ids=supervision_ids,
                supervision_embeddings=supervision_embeddings,
                supervision_weights=supervision_weights,
                negatives_sampler=self.negatives_sampler,
                similarity=self.similarity,
            )
        except TypeError as exc:
            raise TypeError(
                "Configured autoregressive loss %s is incompatible with "
                "HSTUSeren relevance loss computation." % self.loss.__class__.__name__
            ) from exc

    def _aggregate_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        target_ids: torch.Tensor,
        ser_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute weighted relevance and ser losses for a batch."""

        rel_loss_fn: Callable[[Dict[str, torch.Tensor], torch.Tensor], torch.Tensor] | None = self._compute_rel_loss
        rel_loss, ser_loss = compute_losses(
            outputs,
            target_ids,
            ser_labels,
            pos_weight=self.pos_weight,
            rel_loss_fn=rel_loss_fn,
        )
        total_loss = self.loss_weights["rel"] * rel_loss + self.loss_weights["ser"] * ser_loss
        return total_loss, (rel_loss, ser_loss)

    def _compute_ranks(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        # break ties deterministically by adding tiny noise so uniform logits
        # don't give rank=1 for every item
        with torch.no_grad():
            noise = torch.rand_like(logits) * torch.finfo(logits.dtype).eps
        adjusted_logits = logits + noise
        target_scores = adjusted_logits.gather(1, target_ids.unsqueeze(1))
        higher_scores = (adjusted_logits > target_scores).sum(dim=1)
        ranks = higher_scores + 1
        return ranks.to(torch.int64)

    def _update_buffers(
        self,
        stage: str,
        ranks: torch.Tensor,
        ser_labels: torch.Tensor,
    ) -> None:
        self._metric_buffers[stage]["ranks"].append(ranks.detach().cpu())
        ser_mask = ser_labels.eq(1)
        self._metric_buffers[stage]["ser_mask"].append(
            ser_mask.detach().cpu()
        )

    def _finalize_metrics(self, stage: str) -> None:
        buffers = self._metric_buffers.get(stage)
        if not buffers or not buffers["ranks"]:
            return

        ranks = torch.cat(buffers["ranks"])
        ser_mask = torch.cat(buffers["ser_mask"])

        ranking_metrics = compute_ranking_metrics(ranks, self.metric_ks)
        ser_metrics = compute_ser_metrics(ranks, ser_mask, self.ser_metric_ks)

        for key, value in ranking_metrics.items():
            self.log(
                f"{stage}/{key}",
                value,
                prog_bar=(stage != "train"),
                on_step=False,
                on_epoch=True,
            )
        for key, value in ser_metrics.items():
            if torch.isnan(torch.tensor(value)):
                continue
            self.log(
                f"{stage}/{key}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        buffers["ranks"].clear()
        buffers["ser_mask"].clear()

    def _set_backbone_requires_grad(self, enabled: bool) -> None:
        modules = [
            getattr(self, "embeddings", None),
            getattr(self, "preprocessor", None),
            getattr(self, "sequence_encoder", None),
            getattr(self, "postprocessor", None),
        ]
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = enabled

    def _maybe_load_checkpoint(self, path: str | None) -> None:
        if path is None:
            return
        # Lightning checkpoints may store OmegaConf objects (e.g. ListConfig)
        # that require either allow-listing or falling back to the legacy
        # unpickling behaviour introduced prior to torch 2.6.
        torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        state = torch.load(path, **load_kwargs)
        state_dict = state.get("state_dict", state)
        self.load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs, target_ids, ser_labels = self._forward_batch(batch)
        loss, (rel_loss, ser_loss) = self._aggregate_losses(
            outputs, target_ids, ser_labels
        )
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_next", rel_loss, prog_bar=False)
        self.log("train/loss_ser", ser_loss, prog_bar=False)
        return loss

    def evaluation_step(
        self, batch: Dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        outputs, target_ids, ser_labels = self._forward_batch(batch)
        loss, (rel_loss, ser_loss) = self._aggregate_losses(
            outputs, target_ids, ser_labels
        )

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}/loss_next",
            rel_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}/loss_ser",
            ser_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        ranks = self._compute_ranks(outputs["logits_next"], target_ids)
        self._update_buffers(stage, ranks, ser_labels)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.evaluation_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._finalize_metrics("val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.evaluation_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._finalize_metrics("test")

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.freeze_base_epochs > 0 and self.current_epoch == self.freeze_base_epochs:
            self._set_backbone_requires_grad(enabled=True)


__all__ = ["HSTUSeren"]
