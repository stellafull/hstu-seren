"""Hybrid HSTU + Seren expert retrieval module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from generative_recommenders_pl.models.heads.seren_expert import SerendipityExpertHead
from generative_recommenders_pl.models.indexing.candidate_index import CandidateIndex
from generative_recommenders_pl.models.negatives_samples.negative_sampler import (
    InBatchNegativesSampler,
    NegativesSampler,
)
from generative_recommenders_pl.models.postprocessors.postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders_pl.models.preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders_pl.models.retrieval import Retrieval
from generative_recommenders_pl.models.similarity.ndp_module import NDPModule
from generative_recommenders_pl.models.utils import ops
from generative_recommenders_pl.models.utils.features import SequentialFeatures, \
    seq_features_from_row
from generative_recommenders_pl.utils.logger import RankedLogger
from generative_recommenders_pl.data.reco_dataset import RecoDataModule
from generative_recommenders_pl.models.embeddings import EmbeddingModule
from generative_recommenders_pl.models.embeddings.embeddings_sid import (
    LocalSIDEmbeddingModule,
)
from generative_recommenders_pl.models.losses.autoregressive_losses import (
    AutoregressiveLoss,
)
from generative_recommenders_pl.models.losses.ser_losses import (
    SerendipityBCELoss,
)
import torchmetrics

log = RankedLogger(__name__)


def _resolve_item_lookup_path(datamodule: Any) -> str | None:
    """Best-effort resolution of the item lookup CSV from the datamodule."""
    if datamodule is None:
        return None
    data_preprocessor = getattr(datamodule, "data_preprocessor", None)
    if data_preprocessor is None:
        return None
    lookup_fn = getattr(data_preprocessor, "item_lookup_csv", None)
    if lookup_fn is None or not callable(lookup_fn):
        return None
    try:
        return str(lookup_fn())
    except Exception as exc:  # pragma: no cover - defensive logging
        log.debug("Failed to resolve item lookup path from datamodule: %s", exc)
        return None


class HSTUSeren(Retrieval):
    """Retrieval model that augments HSTU backbone with a serendipity expert."""

    def __init__(
        self,
        datamodule: RecoDataModule | DictConfig,
        embeddings: EmbeddingModule | DictConfig,
        preprocessor: InputFeaturesPreprocessorModule | DictConfig,
        sequence_encoder: torch.nn.Module | DictConfig,
        postprocessor: OutputPostprocessorModule | DictConfig,
        similarity: NDPModule | DictConfig,
        negatives_sampler: NegativesSampler | DictConfig,
        candidate_index: CandidateIndex | DictConfig,
        loss: AutoregressiveLoss | DictConfig,
        metrics: torchmetrics.Metric | DictConfig,
        optimizer: torch.optim.Optimizer | DictConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler | DictConfig,
        configure_optimizer_params: DictConfig,
        gr_output_length: int,
        item_embedding_dim: int,
        compile_model: bool,
        seren_expert: SerendipityExpertHead | DictConfig,
        seren_loss_weight: float = 1.0,
        seren_score_alpha: float = 1.0,
        seren_loss: torch.nn.Module | DictConfig | None = None,
        ser_metrics: torchmetrics.Metric | DictConfig | None = None,
        transfer_learning: dict[str, Any] | DictConfig | None = None,
        pretrained_checkpoint_path: str | None = None,
        load_strict: bool = False,
        embedding_type: str = "local",
        embedding_configs: DictConfig | dict | None = None,
        semantic_id_prefix: str | None = None,
    ) -> None:
        resolved_lookup_path: str | None = None
        if embedding_type == "semantic_id":
            resolved_lookup_path = _resolve_item_lookup_path(datamodule)

        if isinstance(embeddings, DictConfig):
            with open_dict(embeddings):
                if (
                    embedding_type == "semantic_id"
                    and resolved_lookup_path
                    and "item_lookup_path" not in embeddings
                ):
                    embeddings.item_lookup_path = resolved_lookup_path
                if (
                    embedding_type == "semantic_id"
                    and "item_lookup_path" not in embeddings
                    and resolved_lookup_path is None
                ):
                    log.warning(
                        "Semantic ID embeddings configured without item_lookup_path; "
                        "attempted datamodule resolution failed."
                    )

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
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            configure_optimizer_params=configure_optimizer_params,
            gr_output_length=gr_output_length,
            item_embedding_dim=item_embedding_dim,
            compile_model=compile_model,
        )

        self.embedding_type: str = embedding_type
        self.embedding_configs: DictConfig | dict | None = embedding_configs
        self.semantic_id_prefix: str | None = semantic_id_prefix
        self.embedding_debug_str: str | None = None
        self._log_embedding_configuration()

        self.seren_expert: SerendipityExpertHead = self._init_seren_expert(
            seren_expert, item_embedding_dim
        )
        self.seren_loss_weight: float = float(seren_loss_weight)
        self.seren_score_alpha: float = float(seren_score_alpha)
        self._ser_loss_fn = self._init_seren_loss(seren_loss)
        self._seren_compiled: bool = False

        self.serendipity_metrics: torchmetrics.Metric | None = (
            self._instantiate_ser_metrics(ser_metrics)
        )

        self._transfer_cfg = self._resolve_transfer_cfg(transfer_learning)
        checkpoint_path = pretrained_checkpoint_path or self._transfer_cfg.get(
            "checkpoint_path"
        )
        load_strict_flag = bool(
            load_strict or self._transfer_cfg.get("strict", False)
        )
        if checkpoint_path:
            self._load_checkpoint_weights(checkpoint_path, strict=load_strict_flag)
        self._maybe_apply_transfer_learning(self._transfer_cfg)

    def _log_embedding_configuration(self) -> None:
        debug_str: str | None = None
        if hasattr(self.embeddings, "debug_str") and callable(
            getattr(self.embeddings, "debug_str")
        ):
            try:
                debug_str = self.embeddings.debug_str()
            except Exception as exc:  # pragma: no cover - defensive logging
                log.debug("Failed to obtain embedding debug string: %s", exc)
        module_name = self.embeddings.__class__.__name__
        if debug_str:
            log.info(
                "Using embedding module (%s): %s", module_name, debug_str
            )
        else:
            log.info("Using embedding module: %s", module_name)
        self.embedding_debug_str = debug_str
        if self.embedding_type == "semantic_id" and not isinstance(
            self.embeddings, LocalSIDEmbeddingModule
        ):
            log.warning(
                "Embedding type configured as semantic_id but module is %s",
                module_name,
            )
        if self.semantic_id_prefix:
            log.info("Semantic ID prefix resolved to %s", self.semantic_id_prefix)

    def _init_seren_expert(
        self,
        seren_expert: SerendipityExpertHead | DictConfig,
        item_embedding_dim: int,
    ) -> SerendipityExpertHead:
        if isinstance(seren_expert, DictConfig):
            kwargs: dict[str, Any] = {}
            if "user_dim" not in seren_expert:
                kwargs["user_dim"] = item_embedding_dim
            if "item_dim" not in seren_expert:
                kwargs["item_dim"] = item_embedding_dim
            return hydra.utils.instantiate(seren_expert, **kwargs)
        return seren_expert

    def _instantiate_ser_metrics(
        self, metric_cfg: torchmetrics.Metric | DictConfig | None
    ) -> torchmetrics.Metric | None:
        if metric_cfg is None:
            return None
        if isinstance(metric_cfg, DictConfig):
            return hydra.utils.instantiate(metric_cfg)
        return metric_cfg

    def _resolve_transfer_cfg(
        self, transfer_cfg: dict[str, Any] | DictConfig | None
    ) -> dict[str, Any]:
        if transfer_cfg is None:
            return {}
        if isinstance(transfer_cfg, DictConfig):
            return {
                k: v
                for k, v in OmegaConf.to_container(transfer_cfg, resolve=True).items()
                if v is not None
            }
        return {k: v for k, v in transfer_cfg.items() if v is not None}

    def _load_checkpoint_weights(self, checkpoint_path: str, strict: bool) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            log.warning("Checkpoint path %s not found; skipping load", path)
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        current_keys = set(self.state_dict().keys())
        incoming_keys = set(state_dict.keys())
        missing_keys = sorted(current_keys - incoming_keys)
        unexpected_keys = sorted(incoming_keys - current_keys)
        if missing_keys:
            log.info("Missing keys during checkpoint load: %s", missing_keys)
        if unexpected_keys:
            log.info("Unexpected keys during checkpoint load: %s", unexpected_keys)
        self.load_state_dict(state_dict, strict=strict)
        log.info("Loaded weights from %s", path)

    def _maybe_apply_transfer_learning(self, transfer_cfg: dict[str, Any]) -> None:
        if not transfer_cfg.get("enabled", False):
            return
        freeze_backbone = transfer_cfg.get("freeze_backbone", False)
        modules_to_freeze: list[str] = list(transfer_cfg.get("freeze_modules", []))
        if freeze_backbone:
            modules_to_freeze.extend(
                [
                    "embeddings",
                    "preprocessor",
                    "sequence_encoder",
                    "postprocessor",
                ]
            )
        for module_name in modules_to_freeze:
            module = getattr(self, module_name, None)
            if module is None:
                log.warning(
                    "Transfer learning requested freeze of missing module %s", module_name
                )
                continue
            for param in module.parameters():
                param.requires_grad = False
            log.info("Froze parameters in module %s", module_name)
        for module_name in transfer_cfg.get("unfreeze_modules", []):
            module = getattr(self, module_name, None)
            if module is None:
                log.warning(
                    "Transfer learning requested unfreeze of missing module %s",
                    module_name,
                )
                continue
            for param in module.parameters():
                param.requires_grad = True
            log.info("Unfroze parameters in module %s", module_name)

    def setup(self, stage: str) -> None:
        if (
            self.compile_model
            and stage in {"fit", None}
            and not hasattr(self, "net")
        ):
            # Provide a placeholder module so the base setup can compile without error.
            self.net = torch.nn.Identity()
        super().setup(stage)
        if not self.compile_model:
            return
        if stage not in {"fit", None}:
            return
        if self._seren_compiled:
            return
        if not hasattr(torch, "compile"):
            log.warning("torch.compile is unavailable; skipping seren module compilation")
            return
        try:
            self.sequence_encoder = torch.compile(self.sequence_encoder)
            self.seren_expert = torch.compile(self.seren_expert)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("Failed to compile serendipity modules: %s", exc)
            return
        self._seren_compiled = True
        log.info("Compiled sequence encoder and serendipity expert with torch.compile")

    def _compute_user_representation(
        self, seq_embeddings: torch.Tensor, past_lengths: torch.Tensor
    ) -> torch.Tensor:
        return ops.get_current_embeddings(past_lengths, seq_embeddings)

    def _compute_seren_logits(
        self, user_repr: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if user_repr.dim() == 2 and item_embeddings.dim() == 3:
            user_repr = user_repr.unsqueeze(1).expand(-1, item_embeddings.size(1), -1)
        elif user_repr.dim() == 3 and item_embeddings.dim() == 2:
            item_embeddings = item_embeddings.unsqueeze(1).expand_as(user_repr)
        return self.seren_expert(user_repr, item_embeddings)

    def _prepare_seq_features(
        self, batch: dict[str, Any]
    ) -> tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
        return seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1,
        )

    def _get_ser_labels(self, batch: dict[str, Any]) -> torch.Tensor:
        label = batch.get("target_ser_label")
        if label is None:
            return torch.zeros(len(batch["history_lengths"]), 1, device=self.device)
        label = label.to(self.device)
        if label.ndim == 0:
            label = label.unsqueeze(0)
        if label.ndim == 1:
            label = label.unsqueeze(1)
        return label.float()

    def _gather_target_embeddings(self, target_ids: torch.Tensor) -> torch.Tensor:
        target_embeddings = self.embeddings.get_item_embeddings(target_ids)
        if target_embeddings.ndim == 3 and target_embeddings.size(1) == 1:
            target_embeddings = target_embeddings.squeeze(1)
        return target_embeddings

    def forward(  # type: ignore[override]
        self, seq_features: SequentialFeatures
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(seq_features)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        seq_features, target_ids, _ = self._prepare_seq_features(batch)
        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lengths.view(-1, 1),
            src=target_ids.view(-1, 1),
        )
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        seq_embeddings, _ = super().forward(seq_features)
        user_repr = self._compute_user_representation(
            seq_embeddings, seq_features.past_lengths
        )

        supervision_ids = seq_features.past_ids
        if isinstance(self.negatives_sampler, InBatchNegativesSampler):
            in_batch_ids = supervision_ids.view(-1)
            self.negatives_sampler.process_batch(
                ids=in_batch_ids,
                presences=(in_batch_ids != 0),
                embeddings=self.embeddings.get_item_embeddings(in_batch_ids),
            )
        else:
            sampler_emb = getattr(self.embeddings, "_item_emb", None)
            if sampler_emb is None:
                if not hasattr(self, "_negatives_embedding_adapter"):
                    class _EmbeddingAdapter(torch.nn.Module):
                        def __init__(self, embedding_module: EmbeddingModule) -> None:
                            super().__init__()
                            self.embedding_module = embedding_module

                        def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
                            return self.embedding_module.get_item_embeddings(item_ids)

                    self._negatives_embedding_adapter = _EmbeddingAdapter(self.embeddings)
                sampler_emb = self._negatives_embedding_adapter
            self.negatives_sampler._item_emb = sampler_emb

        jagged_features = self.dense_to_jagged(
            lengths=seq_features.past_lengths,
            output_embeddings=seq_embeddings[:, :-1, :],
            supervision_ids=supervision_ids[:, 1:],
            supervision_embeddings=input_embeddings[:, 1:, :],
            supervision_weights=(supervision_ids[:, 1:] != 0).float(),
        )
        base_loss = self.loss.jagged_forward(
            negatives_sampler=self.negatives_sampler,
            similarity=self.similarity,
            **jagged_features,
        )

        ser_loss = base_loss.new_tensor(0.0)
        if self.seren_loss_weight > 0:
            target_embeddings = self._gather_target_embeddings(target_ids)
            ser_labels = self._get_ser_labels(batch)
            if ser_labels.numel() > 0:
                ser_logits = self._compute_seren_logits(user_repr, target_embeddings)
                loss_output = self._ser_loss_fn(
                    ser_logits.view_as(ser_labels), ser_labels
                )
                if isinstance(loss_output, tuple):
                    ser_loss, ser_metrics = loss_output
                else:
                    ser_loss = loss_output
                    ser_metrics = {}
                for metric_name, value in ser_metrics.items():
                    self.log(metric_name, value, on_step=True, on_epoch=True)

        total_loss = base_loss + self.seren_loss_weight * ser_loss

        self.log("train/base_loss", base_loss, on_step=True, on_epoch=True)
        self.log("train/seren_loss", ser_loss, on_step=True, on_epoch=True)
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    @torch.inference_mode()
    def retrieve(
        self,
        seq_features: SequentialFeatures,
        filter_past_ids: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_embeddings, _ = super().forward(seq_features)
        user_repr = self._compute_user_representation(
            seq_embeddings, seq_features.past_lengths
        )

        if self.candidate_index.embeddings is None:
            log.info(
                "Initializing candidate index embeddings with current item embeddings"
            )
            self.candidate_index.update_embeddings(
                self.negatives_sampler.normalize_embeddings(
                    self.embeddings.get_item_embeddings(self.candidate_index.ids)
                )
            )

        top_k_ids, base_scores = self.candidate_index.get_top_k_outputs(
            query_embeddings=user_repr,
            invalid_ids=(seq_features.past_ids if filter_past_ids else None),
        )
        apply_seren = (
            self.seren_score_alpha != 0.0 or self.serendipity_metrics is not None
        )
        if apply_seren:
            candidate_embeddings = self.embeddings.get_item_embeddings(top_k_ids)
            ser_logits = self._compute_seren_logits(user_repr, candidate_embeddings)
            ser_probs = torch.relu(ser_logits)
            combined_scores = base_scores + self.seren_score_alpha * ser_probs * ser_logits
            sort_indices = combined_scores.argsort(dim=1, descending=True)
            combined_scores = combined_scores.gather(1, sort_indices)
            top_k_ids = top_k_ids.gather(1, sort_indices)
            base_scores = base_scores.gather(1, sort_indices)
            ser_logits = ser_logits.gather(1, sort_indices)
        else:
            ser_logits = base_scores.new_zeros(base_scores.shape)
            combined_scores = base_scores
        return top_k_ids, combined_scores, base_scores, ser_logits

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if self.serendipity_metrics is not None:
            self.serendipity_metrics.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        seq_features, target_ids, _ = self._prepare_seq_features(batch)
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        seq_features = seq_features._replace(past_embeddings=input_embeddings)

        top_k_ids, combined_scores, base_scores, ser_logits = self.retrieve(
            seq_features
        )
        self.metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        if self.serendipity_metrics is not None:
            ser_labels = self._get_ser_labels(batch)
            self.serendipity_metrics.update(
                top_k_ids=top_k_ids,
                serendipity_scores=ser_logits,
                target_ids=target_ids,
                target_ser_labels=ser_labels,
            )
        self.log("val/base_scores", base_scores.mean(), on_epoch=True)
        self.log("val/combined_scores", combined_scores.mean(), on_epoch=True)

    def on_validation_epoch_end(self) -> Any:
        monitor = super().on_validation_epoch_end()
        if self.serendipity_metrics is not None:
            ser_results = self.serendipity_metrics.compute()
            for key, value in ser_results.items():
                self.log(f"val/{key}", value, on_epoch=True, prog_bar=False)
            self.serendipity_metrics.reset()
            if monitor is None and "monitor" in self.configure_optimizer_params:
                monitor_key = self.configure_optimizer_params["monitor"].split("/", 1)[
                    -1
                ]
                monitor = ser_results.get(monitor_key, monitor)
        return monitor

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        if self.serendipity_metrics is not None:
            self.serendipity_metrics.reset()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> Any:
        monitor = super().on_test_epoch_end()
        if self.serendipity_metrics is not None:
            ser_results = self.serendipity_metrics.compute()
            for key, value in ser_results.items():
                self.log(f"test/{key}", value, on_epoch=True, prog_bar=False)
            self.serendipity_metrics.reset()
            if monitor is None and "monitor" in self.configure_optimizer_params:
                monitor_key = self.configure_optimizer_params["monitor"].split("/", 1)[
                    -1
                ]
                monitor = ser_results.get(monitor_key, monitor)
        return monitor
    def _init_seren_loss(
        self, seren_loss: torch.nn.Module | DictConfig | None
    ) -> torch.nn.Module:
        if seren_loss is None:
            return SerendipityBCELoss()
        if isinstance(seren_loss, DictConfig):
            return hydra.utils.instantiate(seren_loss)
        return seren_loss
