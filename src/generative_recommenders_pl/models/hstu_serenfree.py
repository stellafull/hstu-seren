"""Stage-1 full-SID-in/full-SID-out HSTU-SerenFree model."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from generative_recommenders_pl.data.reco_dataset import RecoDataModule
from generative_recommenders_pl.models.serenfree import (
    DecoderMode,
    SIDComposer,
    SharedPrefixDecoder,
    relevance_loss,
    semantic_js_divergence,
    semantic_loss,
)
from generative_recommenders_pl.models.serenfree.losses import ai_collapse_diagnostics, lf_rank_loss, trie_marginal_nll
from generative_recommenders_pl.models.sequential_encoders.hstu import HSTU, TIMESTAMPS_KEY
from generative_recommenders_pl.models.utils.initialization import truncated_normal
from generative_recommenders_pl.scripts.evaluate_serenfree_retrieval import (
    EvalContexts,
    batched_constrained_beam_search,
    batched_constrained_beam_search_tensors,
    build_sid_transition_index,
    rank_items,
)
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


class HSTUSerenFreeStage1(L.LightningModule):
    """Relevance-only full-SID generation from item-level HSTU states."""

    def __init__(
        self,
        datamodule: RecoDataModule | DictConfig,
        sequence_encoder: HSTU | DictConfig,
        optimizer: torch.optim.Optimizer | DictConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler | DictConfig | None,
        sid_path: str,
        sid_lookup_path: str | None = None,
        semantic_id_prefix: str | None = None,
        configure_optimizer_params: DictConfig | dict[str, Any] | None = None,
        item_lookup_path: str | None = None,
        item_lookup_is_zero_based: bool = True,
        q1_size: int | None = None,
        q2_size: int | None = None,
        q3_size: int | None = None,
        dedup_size: int | None = None,
        embedding_dim: int = 256,
        input_dropout: float = 0.2,
        lambda_d: float = 1.0,
        lambda_i: float = 0.0,
        lambda_a: float = 0.0,
        lambda_g: float = 0.0,
        pseudo_ser_path: str | None = None,
        acceptable_window: int = 20,
        recent_window: int = 10,
        future_target_loss: str = "legacy_history_proxy",
        require_future_targets: bool = False,
        training_objective: str = "single_target",
        rating_positive_threshold: float = 4.0,
        detach_aux_context: bool = False,
        lambda_rank: float = 0.0,
        rank_temperature: float = 0.1,
        stopgrad_R_in_rank: bool = True,
        decoder_modes: list[str] | tuple[str, ...] | None = None,
        compile_model: bool = False,
        validation_ks: list[int] | tuple[int, ...] | None = None,
        validation_beam_size: int = 200,
        validation_eval_batch_size: int = 64,
        log_train_prefix_metrics: bool = False,
        log_train_ai_diagnostics: bool = False,
        next_transition_loss_positions: str = "all",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule", "sequence_encoder", "optimizer", "scheduler"])
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.configure_optimizer_params = (
            OmegaConf.to_container(configure_optimizer_params, resolve=True)
            if isinstance(configure_optimizer_params, DictConfig)
            else configure_optimizer_params
            or {"monitor": "val/full_sid_acc", "interval": "epoch", "frequency": 1}
        )
        self.lambda_d = float(lambda_d)
        self.lambda_i = float(lambda_i)
        self.lambda_a = float(lambda_a)
        self.lambda_g = float(lambda_g)
        self.acceptable_window = int(acceptable_window)
        self.recent_window = int(recent_window)
        self.future_target_loss = str(future_target_loss)
        self.require_future_targets = bool(require_future_targets)
        self.training_objective = str(training_objective)
        self.rating_positive_threshold = float(rating_positive_threshold)
        self.detach_aux_context = bool(detach_aux_context)
        self.lambda_rank = float(lambda_rank)
        self.rank_temperature = float(rank_temperature)
        self.stopgrad_R_in_rank = bool(stopgrad_R_in_rank)
        self.compile_model = bool(compile_model)
        self.validation_ks = tuple(int(k) for k in (validation_ks or (10, 20, 50, 100, 200)))
        self.validation_beam_size = int(validation_beam_size)
        self.validation_eval_batch_size = int(validation_eval_batch_size)
        self.log_train_prefix_metrics = bool(log_train_prefix_metrics)
        self.log_train_ai_diagnostics = bool(log_train_ai_diagnostics)
        self.next_transition_loss_positions = str(next_transition_loss_positions)
        if self.next_transition_loss_positions not in {"all", "last"}:
            raise ValueError("next_transition_loss_positions must be 'all' or 'last'")
        self._compiled = False
        self._validation_sid_index = None
        self._validation_sid_index_device = None
        pseudo_ser_items = self._load_pseudo_ser_items(pseudo_ser_path)
        self.register_buffer("pseudo_ser_items", pseudo_ser_items)

        sid_lookup = (
            self._load_prebuilt_sid_lookup(sid_lookup_path)
            if sid_lookup_path is not None
            else self._load_sid_lookup(
                sid_path=sid_path,
                item_lookup_path=item_lookup_path,
                num_items=getattr(datamodule, "max_item_id", None),
                lookup_is_zero_based=item_lookup_is_zero_based,
            )
        )
        self.register_buffer("sid_lookup", sid_lookup)
        vocab_sizes = self._resolve_vocab_sizes(
            sid_lookup=sid_lookup,
            q1_size=q1_size,
            q2_size=q2_size,
            q3_size=q3_size,
            dedup_size=dedup_size,
        )
        self.sid_composer = SIDComposer(
            vocab_sizes=vocab_sizes,
            embedding_dim=embedding_dim,
        )
        max_sequence_length = int(getattr(datamodule, "max_sequence_length"))
        self.position_embedding = torch.nn.Embedding(max_sequence_length, embedding_dim)
        self.embedding_dropout = torch.nn.Dropout(p=float(input_dropout))
        truncated_normal(
            self.position_embedding.weight.data,
            mean=0.0,
            std=(1.0 / embedding_dim) ** 0.5,
        )
        self.sequence_encoder = self._init_sequence_encoder(
            sequence_encoder=sequence_encoder,
            datamodule=datamodule,
            embedding_dim=embedding_dim,
        )
        self.decoder = SharedPrefixDecoder(
            hidden_dim=embedding_dim,
            vocab_sizes=vocab_sizes,
        )

    def setup(self, stage: str | None = None) -> None:
        if not self.compile_model or self._compiled or stage not in {"fit", None}:
            return
        self.sequence_encoder.compile()
        self.decoder.compile()
        self._compiled = True
        log.info("Compiled HSTU-SerenFree sequence encoder and decoder")

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, metrics = self._step(batch, require_aux_targets=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for key, value in metrics.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        loss, metrics = self._step(batch, require_aux_targets=False)
        batch_size = int(batch["target_ids"].numel())
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for key, value in metrics.items():
            self.log(
                f"val/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        for key, value in self._retrieval_metrics(batch).items():
            self.log(
                f"val/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        loss, metrics = self._step(batch, require_aux_targets=False)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        for key, value in metrics.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optimizer_factory = hydra.utils.instantiate(self.optimizer_cfg)
        optimizer = (
            optimizer_factory(params=self.parameters())
            if callable(optimizer_factory)
            else optimizer_factory
        )
        if self.scheduler_cfg is None:
            return optimizer
        scheduler_factory = hydra.utils.instantiate(self.scheduler_cfg)
        scheduler = (
            scheduler_factory(optimizer=optimizer)
            if callable(scheduler_factory)
            else scheduler_factory
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **self.configure_optimizer_params,
            },
        }

    def _step(
        self,
        batch: dict[str, Any],
        require_aux_targets: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        historical_ids = batch["historical_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        lengths = batch["history_lengths"].to(self.device)
        timestamps = batch["historical_timestamps"].to(self.device)

        history_sid = self._item_ids_to_sid(historical_ids)
        target_sid = self._item_ids_to_sid(target_ids)
        history_embeddings = self.sid_composer(history_sid)
        valid_mask = (history_sid != 0).any(dim=-1, keepdim=True).float()
        history_embeddings = self._add_position_embeddings(history_embeddings, valid_mask)
        encoded, _ = self.sequence_encoder(
            past_lengths=lengths,
            user_embeddings=history_embeddings,
            valid_mask=valid_mask,
            past_payloads={TIMESTAMPS_KEY: timestamps},
        )
        if self._uses_next_transition_policy():
            return self._step_next_transition_policy(
                batch=batch,
                historical_ids=historical_ids,
                target_ids=target_ids,
                history_sid=history_sid,
                encoded=encoded,
                target_sid=target_sid,
                require_aux_targets=require_aux_targets,
                compute_diagnostics=(not self.training) or self.log_train_prefix_metrics,
            )
        relevance_context = self._last_valid_state(encoded, lengths)
        recent_context = self._masked_recent_mean(encoded, lengths, self.recent_window)
        history_context = self._masked_history_mean(encoded, lengths)
        decoded = self.decoder(
            relevance_context,
            target_sid[:, :-1],
            mode=DecoderMode.RELEVANCE,
        )
        loss = relevance_loss(decoded, target_sid, lambda_d=self.lambda_d)
        metrics = self._prefix_metrics(decoded.as_list(), target_sid)
        if not require_aux_targets:
            return loss, metrics

        imminent = None
        acceptable = None
        if self.lambda_i > 0:
            imminent = self.decoder(
                recent_context,
                target_sid[:, :-1],
                mode=DecoderMode.IMMINENT,
            )
            i_targets = self._batch_future_sid_targets(batch, ("I_sids", "future_i_sids", "imminent_sids"))
            if self._uses_v2_future_targets():
                if i_targets is None:
                    raise RuntimeError(
                        "Missing future-window I_sids for V2 imminent training; "
                        "build them with tools/build_future_window_targets.py and use FutureWindowTargetDataset."
                    )
                imminent_loss = self._multi_positive_sid_loss(
                    contexts=recent_context,
                    target_sids=i_targets,
                    mode=DecoderMode.IMMINENT,
                )
            else:
                imminent_loss = semantic_loss(imminent, target_sid)
            loss = loss + self.lambda_i * imminent_loss
            metrics["imminent_loss"] = imminent_loss.detach()

        if self.lambda_a > 0:
            a_targets = self._batch_future_sid_targets(batch, ("A_sids", "future_a_sids", "acceptable_sids"))
            if self._uses_v2_future_targets():
                if a_targets is None:
                    raise RuntimeError(
                        "Missing future-window A_sids for V2 acceptable training; "
                        "the legacy history-window proxy is disabled for V2."
                    )
                acceptable_prefix = self._first_positive_prefix(a_targets)
                acceptable = self.decoder(
                    history_context,
                    acceptable_prefix,
                    mode=DecoderMode.ACCEPTABLE,
                )
                acceptable_loss = self._multi_positive_sid_loss(
                    contexts=history_context,
                    target_sids=a_targets,
                    mode=DecoderMode.ACCEPTABLE,
                )
            else:
                acceptable_targets = self._acceptable_semantic_targets(
                    historical_ids=historical_ids,
                    lengths=lengths,
                )
                acceptable = self.decoder(
                    history_context,
                    acceptable_targets,
                    mode=DecoderMode.ACCEPTABLE,
                )
                acceptable_loss = semantic_loss(acceptable, acceptable_targets)
            loss = loss + self.lambda_a * acceptable_loss
            metrics["acceptable_loss"] = acceptable_loss.detach()
            if imminent is not None and acceptable is not None:
                for key, value in ai_collapse_diagnostics(
                    acceptable.semantic_logits,
                    imminent.semantic_logits,
                    target_a=a_targets[:, 0, :-1] if a_targets is not None and a_targets.numel() else None,
                    target_i=i_targets[:, 0, :-1] if 'i_targets' in locals() and i_targets is not None and i_targets.numel() else None,
                ).items():
                    metrics[key] = value.detach()
                js = semantic_js_divergence(acceptable, imminent)
                metrics["js_pa_pi"] = js.detach()
                if not self._uses_v2_future_targets():
                    gap_loss = self._gap_loss(
                        acceptable=acceptable,
                        imminent=imminent,
                        target_sid=target_sid,
                        target_ids=target_ids,
                    )
                    if gap_loss is not None and self.lambda_g > 0:
                        loss = loss + self.lambda_g * gap_loss
                        metrics["gap_loss"] = gap_loss.detach()

        if self.lambda_rank > 0 and self._uses_v2_future_targets():
            rank_loss = self._batch_lf_rank_loss(
                batch,
                relevance_context=relevance_context,
                acceptable_context=history_context,
                imminent_context=recent_context,
            )
            if rank_loss is not None:
                loss = loss + self.lambda_rank * rank_loss
                metrics["lf_rank_loss"] = rank_loss.detach()
        return loss, metrics


    def _uses_next_transition_policy(self) -> bool:
        return self.training_objective in {
            "next_transition_policy",
            "rai_next_transition_policy",
        }

    def _uses_v2_future_targets(self) -> bool:
        return self.require_future_targets or self.future_target_loss in {"trie_marginal", "multi_positive"}

    def _step_next_transition_policy(
        self,
        batch: dict[str, Any],
        historical_ids: torch.Tensor,
        target_ids: torch.Tensor,
        history_sid: torch.Tensor,
        encoded: torch.Tensor,
        target_sid: torch.Tensor,
        require_aux_targets: bool,
        compute_diagnostics: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        targets = self._next_transition_targets(
            batch,
            historical_ids,
            target_ids,
            history_sid,
            target_sid,
        )
        prefix_tokens = targets["next_sids"][..., : self.decoder.num_semantic_levels]
        loss_context = encoded
        loss_prefix_tokens = prefix_tokens
        loss_target_sids = targets["next_sids"]
        loss_valid_mask = targets["valid_mask"]
        loss_accept_mask = targets["accept_mask"]
        if self.next_transition_loss_positions == "last":
            rows = torch.arange(encoded.size(0), device=encoded.device)
            cols = targets["terminal_positions"].clamp_min(0)
            loss_context = encoded[rows, cols]
            loss_prefix_tokens = prefix_tokens[rows, cols]
            loss_target_sids = targets["next_sids"][rows, cols]
            loss_valid_mask = targets["valid_mask"][rows, cols] & targets["terminal_mask"][rows, cols]
            loss_accept_mask = targets["accept_mask"][rows, cols] & targets["terminal_mask"][rows, cols]

        decoded_r = self.decoder(loss_context, loss_prefix_tokens, mode=DecoderMode.RELEVANCE)
        loss_r = self._masked_sid_ce(
            decoded_r.level_logits,
            loss_target_sids,
            loss_valid_mask,
            include_dedup=True,
        )
        loss = loss_r
        metrics: dict[str, torch.Tensor] = {"loss_r": loss_r.detach()}
        valid_transition_count = loss_valid_mask.sum()
        accepted_transition_count = loss_accept_mask.sum()
        if compute_diagnostics:
            metrics.update(
                self._sequence_prefix_metrics(
                    decoded_r.as_list(),
                    loss_target_sids,
                    loss_valid_mask,
                )
            )
            metrics["valid_transition_count"] = valid_transition_count.detach().to(torch.float32)
            metrics["accepted_transition_count"] = accepted_transition_count.detach().to(torch.float32)
            valid_count = valid_transition_count.clamp_min(1).to(torch.float32)
            metrics["accepted_transition_ratio"] = (
                accepted_transition_count.to(torch.float32) / valid_count
            ).detach()
        if not require_aux_targets:
            return loss, metrics

        aux_encoded = encoded.detach() if self.detach_aux_context else encoded
        decoded_i = None
        decoded_a = None
        if self.lambda_i > 0:
            recent_context = self._rolling_recent_mean(
                aux_encoded,
                targets["input_valid_mask"],
                self.recent_window,
            )
            if self.next_transition_loss_positions == "last":
                recent_context = recent_context[rows, cols]
            decoded_i = self.decoder(recent_context, loss_prefix_tokens, mode=DecoderMode.IMMINENT)
            loss_i = self._masked_sid_ce(
                decoded_i.semantic_logits,
                loss_target_sids[..., : self.decoder.num_semantic_levels],
                loss_valid_mask,
                include_dedup=False,
            )
            loss = loss + self.lambda_i * loss_i
            metrics["loss_i"] = loss_i.detach()

        if self.lambda_a > 0:
            acceptable_context = aux_encoded
            if self.next_transition_loss_positions == "last":
                acceptable_context = acceptable_context[rows, cols]
            decoded_a = self.decoder(acceptable_context, loss_prefix_tokens, mode=DecoderMode.ACCEPTABLE)
            loss_a = self._masked_sid_ce(
                decoded_a.semantic_logits,
                loss_target_sids[..., : self.decoder.num_semantic_levels],
                loss_accept_mask,
                include_dedup=False,
            )
            loss = loss + self.lambda_a * loss_a
            metrics["loss_a"] = loss_a.detach()

        if decoded_i is not None and decoded_a is not None and (
            compute_diagnostics or self.log_train_ai_diagnostics
        ):
            metrics.update(
                self._ai_policy_diagnostics(
                    decoded_a.semantic_logits,
                    decoded_i.semantic_logits,
                    loss_target_sids[..., : self.decoder.num_semantic_levels],
                    loss_valid_mask,
                    loss_accept_mask,
                )
            )
        return loss, metrics

    def _next_transition_targets(
        self,
        batch: dict[str, Any],
        historical_ids: torch.Tensor,
        target_ids: torch.Tensor,
        history_sid: torch.Tensor,
        target_sid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        historical_ratings = batch["historical_ratings"].to(self.device)
        target_ratings = batch["target_ratings"].to(self.device)
        input_valid = historical_ids != 0
        next_ids = torch.zeros_like(historical_ids)
        next_sids = torch.zeros_like(history_sid)
        next_ratings = torch.zeros_like(historical_ratings)
        adjacent_valid = input_valid[:, :-1] & input_valid[:, 1:]
        next_ids[:, :-1] = torch.where(
            adjacent_valid,
            historical_ids[:, 1:],
            next_ids[:, :-1],
        )
        next_sids[:, :-1] = torch.where(
            adjacent_valid.unsqueeze(-1),
            history_sid[:, 1:],
            next_sids[:, :-1],
        )
        next_ratings[:, :-1] = torch.where(
            adjacent_valid,
            historical_ratings[:, 1:],
            next_ratings[:, :-1],
        )
        positions = torch.arange(historical_ids.size(1), device=historical_ids.device)
        last_positions = torch.where(input_valid, positions.unsqueeze(0), -1).max(dim=1).values
        terminal_mask = torch.zeros_like(input_valid)
        has_history = last_positions >= 0
        if has_history.any():
            rows = has_history.nonzero(as_tuple=False).flatten()
            cols = last_positions[rows]
            next_ids[rows, cols] = target_ids[rows]
            next_sids[rows, cols] = target_sid[rows]
            next_ratings[rows, cols] = target_ratings[rows]
            terminal_mask[rows, cols] = True
        next_valid = next_ids != 0
        sid_valid = (next_sids != 0).all(dim=-1)
        valid_mask = input_valid & next_valid & sid_valid
        accept_mask = valid_mask & (next_ratings.to(torch.float32) >= self.rating_positive_threshold)
        return {
            "next_ids": next_ids,
            "next_sids": next_sids,
            "next_ratings": next_ratings,
            "input_valid_mask": input_valid,
            "valid_mask": valid_mask,
            "accept_mask": accept_mask,
            "terminal_positions": last_positions,
            "terminal_mask": terminal_mask,
        }

    def _masked_sid_ce(
        self,
        logits_by_level: list[torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor,
        *,
        include_dedup: bool,
    ) -> torch.Tensor:
        if not mask.any():
            return sum(logits.sum() for logits in logits_by_level) * 0.0
        levels = len(logits_by_level) if include_dedup else self.decoder.num_semantic_levels
        losses = []
        for level, logits in enumerate(logits_by_level[:levels]):
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_target = targets[..., level].reshape(-1).to(torch.long)
            flat_mask = mask.reshape(-1) & (flat_target != 0)
            if not flat_mask.any():
                losses.append(logits.sum() * 0.0)
                continue
            losses.append(
                torch.nn.functional.cross_entropy(
                    flat_logits[flat_mask],
                    flat_target[flat_mask],
                )
            )
        return torch.stack(losses).sum()

    def _sequence_prefix_metrics(
        self,
        logits_by_level: list[torch.Tensor],
        target_sid: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        correct_by_level = []
        valid_by_level = []
        for idx, logits in enumerate(logits_by_level):
            name = f"q{idx + 1}_acc" if idx < len(logits_by_level) - 1 else "d_acc"
            pred = logits.argmax(dim=-1)
            target = target_sid[..., idx]
            valid = mask & (target != 0)
            metrics[name] = (
                (pred[valid] == target[valid]).float().mean()
                if valid.any()
                else logits.new_tensor(0.0)
            )
            correct_by_level.append(pred == target)
            valid_by_level.append(valid)
        semantic_correct = torch.stack(correct_by_level[:-1], dim=-1).all(dim=-1)
        semantic_valid = torch.stack(valid_by_level[:-1], dim=-1).all(dim=-1)
        metrics["semantic_acc"] = (
            semantic_correct[semantic_valid].float().mean()
            if semantic_valid.any()
            else target_sid.new_tensor(0.0, dtype=torch.float32)
        )
        full_correct = torch.stack(correct_by_level, dim=-1).all(dim=-1)
        full_valid = torch.stack(valid_by_level, dim=-1).all(dim=-1)
        metrics["full_sid_acc"] = (
            full_correct[full_valid].float().mean()
            if full_valid.any()
            else target_sid.new_tensor(0.0, dtype=torch.float32)
        )
        return metrics

    def _rolling_recent_mean(
        self,
        encoded: torch.Tensor,
        valid_mask: torch.Tensor,
        recent_window: int,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = encoded.shape
        window = max(int(recent_window), 1)
        weights = valid_mask.to(encoded.dtype).unsqueeze(-1)
        weighted = encoded * weights
        zero_values = encoded.new_zeros((batch_size, 1, hidden_dim))
        zero_counts = encoded.new_zeros((batch_size, 1, 1))
        prefix_values = torch.cat([zero_values, weighted.cumsum(dim=1)], dim=1)
        prefix_counts = torch.cat([zero_counts, weights.cumsum(dim=1)], dim=1)
        end = torch.arange(1, seq_len + 1, device=encoded.device)
        start = (end - window).clamp_min(0)
        sums = prefix_values[:, end, :] - prefix_values[:, start, :]
        counts = prefix_counts[:, end, :] - prefix_counts[:, start, :]
        return sums / counts.clamp_min(1.0)

    def _ai_policy_diagnostics(
        self,
        acceptable_logits: list[torch.Tensor],
        imminent_logits: list[torch.Tensor],
        target_sids: torch.Tensor,
        valid_mask: torch.Tensor,
        accept_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        js_values = []
        entropy_a_values = []
        entropy_i_values = []
        target_gaps = []
        for level, (a_logits, i_logits) in enumerate(zip(acceptable_logits, imminent_logits)):
            a_log = torch.nn.functional.log_softmax(a_logits, dim=-1)
            i_log = torch.nn.functional.log_softmax(i_logits, dim=-1)
            a_prob = a_log.exp()
            i_prob = i_log.exp()
            mixture = 0.5 * (a_prob + i_prob)
            mixture_log = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
            js = 0.5 * (
                (a_prob * (a_log - mixture_log)).sum(dim=-1)
                + (i_prob * (i_log - mixture_log)).sum(dim=-1)
            )
            entropy_a = -(a_prob * a_log).sum(dim=-1)
            entropy_i = -(i_prob * i_log).sum(dim=-1)
            tok = target_sids[..., level].to(torch.long)
            safe_tok = tok.clamp(min=0, max=a_log.size(-1) - 1)
            gap = (a_log - i_log).gather(-1, safe_tok.unsqueeze(-1)).squeeze(-1)
            level_mask = valid_mask & (tok != 0)
            if level_mask.any():
                js_values.append(js[level_mask].mean())
                entropy_a_values.append(entropy_a[level_mask].mean())
                entropy_i_values.append(entropy_i[level_mask].mean())
                target_gaps.append(gap[level_mask])
        zero = acceptable_logits[0].sum() * 0.0
        metrics["js_a_i_sem_col_mean"] = torch.stack(js_values).mean().detach() if js_values else zero.detach()
        metrics["entropy_a_sem_col_mean"] = torch.stack(entropy_a_values).mean().detach() if entropy_a_values else zero.detach()
        metrics["entropy_i_sem_col_mean"] = torch.stack(entropy_i_values).mean().detach() if entropy_i_values else zero.detach()
        if target_gaps:
            all_gap = torch.cat(target_gaps)
            metrics["target_gap_a_minus_i_mean_all"] = all_gap.mean().detach()
            metrics["positive_boost_rate"] = (all_gap > 0).to(torch.float32).mean().detach()
        else:
            metrics["target_gap_a_minus_i_mean_all"] = zero.detach()
            metrics["positive_boost_rate"] = zero.detach()
        accepted_gaps = []
        for level, (a_logits, i_logits) in enumerate(zip(acceptable_logits, imminent_logits)):
            a_log = torch.nn.functional.log_softmax(a_logits, dim=-1)
            i_log = torch.nn.functional.log_softmax(i_logits, dim=-1)
            tok = target_sids[..., level].to(torch.long)
            safe_tok = tok.clamp(min=0, max=a_log.size(-1) - 1)
            gap = (a_log - i_log).gather(-1, safe_tok.unsqueeze(-1)).squeeze(-1)
            level_mask = accept_mask & (tok != 0)
            if level_mask.any():
                accepted_gaps.append(gap[level_mask])
        metrics["target_gap_a_minus_i_mean_accepted"] = (
            torch.cat(accepted_gaps).mean().detach() if accepted_gaps else zero.detach()
        )
        return metrics

    def _batch_future_sid_targets(
        self,
        batch: dict[str, Any],
        names: tuple[str, ...],
    ) -> torch.Tensor | None:
        for name in names:
            value = batch.get(name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                tensor = value.to(self.device, dtype=torch.long)
            else:
                tensor = torch.as_tensor(value, dtype=torch.long, device=self.device)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            if tensor.dim() != 3:
                raise ValueError(f"{name} must have shape [B, N, SID_COLUMNS]")
            if tensor.size(-1) == self.decoder.num_semantic_levels:
                pad = torch.zeros((*tensor.shape[:-1], 1), dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad], dim=-1)
            return tensor
        return None

    def _first_positive_prefix(self, target_sids: torch.Tensor) -> torch.Tensor:
        valid = (target_sids != 0).any(dim=-1)
        first_idx = valid.to(torch.long).argmax(dim=1)
        rows = torch.arange(target_sids.size(0), device=target_sids.device)
        return target_sids[rows, first_idx, : self.decoder.num_semantic_levels]

    def _multi_positive_sid_loss(
        self,
        contexts: torch.Tensor,
        target_sids: torch.Tensor,
        mode: DecoderMode,
    ) -> torch.Tensor:
        semantic_levels = tuple(range(self.decoder.num_semantic_levels))
        losses = []
        for row_idx in range(target_sids.size(0)):
            row_targets = target_sids[row_idx]
            row_targets = row_targets[(row_targets[:, : self.decoder.num_semantic_levels] != 0).all(dim=1)]
            if row_targets.numel() == 0:
                losses.append(contexts[row_idx].sum() * 0.0)
                continue

            def logits_fn(prefix: tuple[int, ...], level: int) -> torch.Tensor:
                prefix_tokens = torch.zeros(
                    (1, self.decoder.num_semantic_levels),
                    dtype=torch.long,
                    device=self.device,
                )
                if prefix:
                    prefix_tokens[0, : len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=self.device)
                decoded = self.decoder(
                    contexts[row_idx : row_idx + 1],
                    prefix_tokens,
                    mode=mode,
                )
                return decoded.level_logits[level][0]

            losses.append(trie_marginal_nll(logits_fn, row_targets, semantic_levels))
        return torch.stack(losses).mean()

    def _batch_lf_rank_loss(
        self,
        batch: dict[str, Any],
        relevance_context: torch.Tensor | None = None,
        acceptable_context: torch.Tensor | None = None,
        imminent_context: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if "rank_positive_mask" not in batch:
            return None
        positive_mask = batch["rank_positive_mask"].to(self.device).bool()
        geometry = batch.get("rank_geometry")
        geometry = (
            geometry.to(self.device, dtype=torch.float32)
            if isinstance(geometry, torch.Tensor)
            else torch.zeros_like(positive_mask, dtype=torch.float32, device=self.device)
        )
        if all(name in batch for name in ("rank_s_R", "rank_s_A", "rank_s_I")):
            return lf_rank_loss(
                batch["rank_s_R"].to(self.device),
                batch["rank_s_A"].to(self.device),
                batch["rank_s_I"].to(self.device),
                geometry,
                positive_mask,
                temperature=self.rank_temperature,
                stopgrad_r=self.stopgrad_R_in_rank,
            )
        rank_sids = self._batch_future_sid_targets(batch, ("rank_candidate_sids", "rank_sids"))
        if rank_sids is None:
            return None
        if relevance_context is None or acceptable_context is None or imminent_context is None:
            raise RuntimeError("Missing mode-specific contexts for V2 LF-rank candidate scoring")
        s_r = self._candidate_sid_scores(relevance_context, rank_sids, DecoderMode.RELEVANCE)
        s_a = self._candidate_sid_scores(acceptable_context, rank_sids, DecoderMode.ACCEPTABLE)
        s_i = self._candidate_sid_scores(imminent_context, rank_sids, DecoderMode.IMMINENT)
        return lf_rank_loss(
            s_r,
            s_a,
            s_i,
            geometry,
            positive_mask,
            temperature=self.rank_temperature,
            stopgrad_r=self.stopgrad_R_in_rank,
        )

    def _candidate_sid_scores(
        self,
        contexts: torch.Tensor,
        target_sids: torch.Tensor,
        mode: DecoderMode,
    ) -> torch.Tensor:
        batch_size, num_candidates, _ = target_sids.shape
        flat_context = contexts.unsqueeze(1).expand(-1, num_candidates, -1).reshape(
            batch_size * num_candidates,
            contexts.size(-1),
        )
        flat_sids = target_sids.reshape(batch_size * num_candidates, target_sids.size(-1))
        prefix = flat_sids[:, : self.decoder.num_semantic_levels]
        decoded = self.decoder(flat_context, prefix, mode=mode)
        scores = flat_context.new_zeros(flat_sids.size(0))
        valid = (prefix != 0).all(dim=-1)
        logits_by_level = (
            decoded.level_logits
            if mode == DecoderMode.RELEVANCE
            else decoded.semantic_logits
        )
        for level, logits in enumerate(logits_by_level):
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            tok = flat_sids[:, level].to(torch.long)
            safe_tok = tok.clamp(min=0, max=log_probs.size(-1) - 1)
            scores = scores + log_probs.gather(1, safe_tok.unsqueeze(1)).squeeze(1)
            valid = valid & (tok > 0)
        scores = scores.masked_fill(~valid, -1.0e9)
        return scores.reshape(batch_size, num_candidates)

    def _item_ids_to_sid(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.to(torch.long)
        valid = (item_ids >= 0) & (item_ids < self.sid_lookup.size(0))
        safe_ids = item_ids.clamp(min=0, max=self.sid_lookup.size(0) - 1)
        sid = self.sid_lookup[safe_ids]
        return sid.masked_fill(~valid.unsqueeze(-1), 0)

    @torch.inference_mode()
    def _retrieval_metrics(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        max_k = max(self.validation_ks)
        beam_size = max(max_k, self.validation_beam_size)
        target_ids = batch["target_ids"].to(self.device, dtype=torch.long)
        metrics = {
            f"hr@{k}": target_ids.new_tensor(0.0, dtype=torch.float32)
            for k in self.validation_ks
        }
        metrics.update(
            {
                f"ndcg@{k}": target_ids.new_tensor(0.0, dtype=torch.float32)
                for k in self.validation_ks
            }
        )
        if target_ids.numel() == 0:
            return metrics

        item_ids = []
        scores = []
        chunk_size = max(1, self.validation_eval_batch_size)
        for start in range(0, int(target_ids.numel()), chunk_size):
            chunk_item_ids, chunk_scores = self._rank_batch_item_tensors(
                batch=self._slice_batch(batch, start, start + chunk_size),
                beam_size=beam_size,
            )
            item_ids.append(chunk_item_ids)
            scores.append(chunk_scores)
        ranked_items = torch.cat(item_ids, dim=0)
        ranked_scores = torch.cat(scores, dim=0)
        history_ids = batch["historical_ids"].to(self.device, dtype=torch.long)
        seen = ranked_items.unsqueeze(-1).eq(history_ids.unsqueeze(1)).any(dim=-1)
        keep_target = ranked_items.eq(target_ids.unsqueeze(1))
        ranked_scores = ranked_scores.masked_fill(seen & ~keep_target, -torch.inf)
        top_count = min(max_k, ranked_items.size(1))
        top_pos = ranked_scores.topk(top_count, dim=1).indices
        ranked_items = ranked_items.gather(1, top_pos)
        target_matches = ranked_items.eq(target_ids.unsqueeze(1))
        rank_positions = torch.arange(
            ranked_items.size(1),
            device=ranked_items.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        first_rank = torch.where(target_matches, rank_positions, torch.inf).min(dim=1).values

        for k in self.validation_ks:
            hit = first_rank < float(k)
            metrics[f"hr@{k}"] = hit.to(torch.float32).mean()
            ndcg = torch.where(
                hit,
                1.0 / torch.log2(first_rank + 2.0),
                torch.zeros_like(first_rank),
            )
            metrics[f"ndcg@{k}"] = ndcg.mean()
        return metrics

    def _slice_batch(
        self,
        batch: dict[str, Any],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        batch_size = int(batch["target_ids"].shape[0])
        sliced: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == batch_size:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _rank_batch_items(
        self,
        batch: dict[str, Any],
        beam_size: int,
        max_k: int,
    ) -> list[list[int]]:
        historical_ids = batch["historical_ids"].to(self.device)
        lengths = batch["history_lengths"].to(self.device)
        timestamps = batch["historical_timestamps"].to(self.device)
        context = self._relevance_context(
            historical_ids=historical_ids,
            lengths=lengths,
            timestamps=timestamps,
        )
        contexts = EvalContexts(
            relevance=context,
            imminent=context,
            acceptable=context,
        )
        beams = batched_constrained_beam_search(
            decoder=self.decoder,
            contexts=contexts,
            sid_index=self._get_validation_sid_index(),
            beam_size=beam_size,
            aig=None,
        )
        ranked: list[list[int]] = []
        for row_idx, row_beams in enumerate(beams):
            target = int(batch["target_ids"][row_idx].detach().cpu().item())
            seen = {
                int(item)
                for item in historical_ids[row_idx].detach().cpu().tolist()
                if int(item) > 0
            }
            ranked.append(
                rank_items(
                    beams=row_beams,
                    seen=seen,
                    filter_history=True,
                    max_k=max_k,
                    target=target,
                )
            )
        return ranked

    def _rank_batch_item_tensors(
        self,
        batch: dict[str, Any],
        beam_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        historical_ids = batch["historical_ids"].to(self.device)
        lengths = batch["history_lengths"].to(self.device)
        timestamps = batch["historical_timestamps"].to(self.device)
        context = self._relevance_context(
            historical_ids=historical_ids,
            lengths=lengths,
            timestamps=timestamps,
        )
        contexts = EvalContexts(
            relevance=context,
            imminent=context,
            acceptable=context,
        )
        item_ids, scores, _ = batched_constrained_beam_search_tensors(
            decoder=self.decoder,
            contexts=contexts,
            sid_index=self._get_validation_sid_index(),
            beam_size=beam_size,
            aig=None,
        )
        return item_ids, scores

    def _relevance_context(
        self,
        historical_ids: torch.Tensor,
        lengths: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        history_sid = self._item_ids_to_sid(historical_ids)
        history_embeddings = self.sid_composer(history_sid)
        valid_mask = (history_sid != 0).any(dim=-1, keepdim=True).float()
        history_embeddings = self._add_position_embeddings(history_embeddings, valid_mask)
        encoded, _ = self.sequence_encoder(
            past_lengths=lengths,
            user_embeddings=history_embeddings,
            valid_mask=valid_mask,
            past_payloads={TIMESTAMPS_KEY: timestamps},
        )
        return self._last_valid_state(encoded, lengths)

    def _get_validation_sid_index(self):
        if (
            self._validation_sid_index is None
            or self._validation_sid_index_device != self.device
        ):
            sid_index, _ = build_sid_transition_index(
                self.sid_lookup.detach().cpu(),
                allow_duplicate_sids=False,
            )
            self._validation_sid_index = sid_index.to(self.device)
            self._validation_sid_index_device = self.device
        return self._validation_sid_index

    def _add_position_embeddings(
        self, history_embeddings: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = history_embeddings.shape
        positions = torch.arange(seq_len, device=history_embeddings.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        history_embeddings = (
            history_embeddings * (self.sid_composer.embedding_dim**0.5)
            + self.position_embedding(positions)
        )
        return self.embedding_dropout(history_embeddings) * valid_mask

    def _last_valid_state(self, encoded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = encoded.shape
        row_offsets = torch.arange(batch_size, device=encoded.device) * seq_len
        indices = (lengths.to(torch.long).clamp(min=1, max=seq_len) - 1) + row_offsets
        return encoded.reshape(batch_size * seq_len, hidden_dim)[indices]

    def _masked_history_mean(
        self,
        encoded: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = encoded.size(1)
        positions = torch.arange(seq_len, device=encoded.device).unsqueeze(0)
        mask = positions < lengths.to(torch.long).unsqueeze(1)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / denom

    def _masked_recent_mean(
        self,
        encoded: torch.Tensor,
        lengths: torch.Tensor,
        recent_window: int,
    ) -> torch.Tensor:
        seq_len = encoded.size(1)
        lengths = lengths.to(torch.long).clamp(min=1, max=seq_len)
        positions = torch.arange(seq_len, device=encoded.device).unsqueeze(0)
        starts = (lengths - max(int(recent_window), 1)).clamp_min(0).unsqueeze(1)
        mask = (positions >= starts) & (positions < lengths.unsqueeze(1))
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / denom

    def _acceptable_semantic_targets(
        self,
        historical_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Proxy acceptable target from the recent history window.

        The current dataloader exposes one next-item target and the preceding
        history, but not a separate future window. For Stage 2 on Amazon Movies
        we use the far edge of a recent history window as a broader semantic
        proxy and exclude the final dedup column from loss.
        """

        seq_len = historical_ids.size(1)
        clamped_lengths = lengths.to(torch.long).clamp(min=1, max=seq_len)
        window = max(self.acceptable_window, 1)
        offsets = (clamped_lengths - min(window, seq_len)).clamp_min(0)
        row_offsets = torch.arange(historical_ids.size(0), device=historical_ids.device) * seq_len
        item_ids = historical_ids.reshape(-1)[row_offsets + offsets]
        return self._item_ids_to_sid(item_ids)[:, :-1]

    def _prefix_metrics(
        self, logits_by_level: list[torch.Tensor], target_sid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        correct_by_level = []
        valid_by_level = []
        metrics = {}
        for idx, logits in enumerate(logits_by_level):
            name = f"q{idx + 1}_acc" if idx < len(logits_by_level) - 1 else "d_acc"
            pred = logits.argmax(dim=-1)
            target = target_sid[:, idx]
            valid = target != 0
            if valid.any():
                acc = (pred[valid] == target[valid]).float().mean()
            else:
                acc = logits.new_tensor(0.0)
            metrics[name] = acc
            correct_by_level.append(pred == target)
            valid_by_level.append(valid)

        semantic_correct = torch.stack(correct_by_level[:-1], dim=1).all(dim=1)
        semantic_valid = torch.stack(valid_by_level[:-1], dim=1).all(dim=1)
        if semantic_valid.any():
            metrics["semantic_acc"] = semantic_correct[semantic_valid].float().mean()
        else:
            metrics["semantic_acc"] = target_sid.new_tensor(0.0, dtype=torch.float32)

        full_correct = torch.stack(correct_by_level, dim=1).all(dim=1)
        full_valid = torch.stack(valid_by_level, dim=1).all(dim=1)
        if full_valid.any():
            metrics["full_sid_acc"] = full_correct[full_valid].float().mean()
        else:
            metrics["full_sid_acc"] = target_sid.new_tensor(0.0, dtype=torch.float32)
        return metrics

    def _gap_loss(
        self,
        acceptable,
        imminent,
        target_sid: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.pseudo_ser_items.numel() == 0:
            return None
        positive = torch.isin(target_ids.to(torch.long), self.pseudo_ser_items)
        if not positive.any():
            return None
        gaps = []
        for level, target in enumerate(target_sid[:, :-1].unbind(dim=-1)):
            log_p_a = torch.nn.functional.log_softmax(
                acceptable.semantic_logits[level],
                dim=-1,
            )
            log_p_i = torch.nn.functional.log_softmax(
                imminent.semantic_logits[level],
                dim=-1,
            )
            gaps.append((log_p_a - log_p_i).gather(1, target.unsqueeze(1)).squeeze(1))
        mean_gap = torch.stack(gaps, dim=1).mean(dim=1)
        return -mean_gap[positive].mean()

    def _init_sequence_encoder(
        self,
        sequence_encoder: HSTU | DictConfig,
        datamodule: RecoDataModule | DictConfig,
        embedding_dim: int,
    ) -> HSTU:
        if not isinstance(sequence_encoder, DictConfig):
            return sequence_encoder
        kwargs: dict[str, Any] = {}
        if "max_sequence_len" not in sequence_encoder:
            kwargs["max_sequence_len"] = getattr(datamodule, "max_sequence_length")
        if "max_output_len" not in sequence_encoder:
            kwargs["max_output_len"] = 0
        if "embedding_dim" not in sequence_encoder:
            kwargs["embedding_dim"] = embedding_dim
        if "item_embedding_dim" not in sequence_encoder:
            kwargs["item_embedding_dim"] = embedding_dim
        if "attention_dim" not in sequence_encoder:
            kwargs["attention_dim"] = embedding_dim
        if "linear_dim" not in sequence_encoder:
            kwargs["linear_dim"] = embedding_dim
        return hydra.utils.instantiate(sequence_encoder, **kwargs)

    def _resolve_vocab_sizes(
        self,
        sid_lookup: torch.Tensor,
        q1_size: int | None,
        q2_size: int | None,
        q3_size: int | None,
        dedup_size: int | None,
    ) -> tuple[int, ...]:
        max_values = sid_lookup.max(dim=0).values.tolist()
        legacy_configured = [q1_size, q2_size, q3_size, dedup_size]
        if sid_lookup.size(1) == 4:
            configured = legacy_configured
        else:
            if any(value is not None for value in legacy_configured):
                raise ValueError(
                    "q1/q2/q3/dedup size overrides only apply to 4-column SIDs; "
                    "leave them unset for variable-depth SID tables"
                )
            configured = [None] * sid_lookup.size(1)
        return tuple(
            int(value) if value is not None else int(max_token) + 1
            for value, max_token in zip(configured, max_values)
        )

    def _load_sid_lookup(
        self,
        sid_path: str,
        item_lookup_path: str | None,
        num_items: int | None,
        lookup_is_zero_based: bool,
    ) -> torch.Tensor:
        path = Path(sid_path)
        data = torch.load(path, map_location="cpu")
        item_ids_raw = data["item_ids"]
        sid_tokens = data["semantic_ids"].to(torch.long)
        if sid_tokens.dim() != 2 or sid_tokens.size(1) < 2:
            raise ValueError(
                "Expected full SID tensors shaped [items, semantic_levels + dedup]"
            )
        if sid_tokens.min().item() < 0:
            raise ValueError("Semantic ID tokens must be non-negative")
        # Reserve 0 for padding/missing items. Raw codebook token 0, including
        # dedup=0, is a valid SID token and is shifted to 1 for training.
        sid_tokens = sid_tokens + 1

        lookup_map = self._read_item_lookup(item_lookup_path, lookup_is_zero_based)
        mapped_ids = []
        kept_indices = []
        for idx, raw_id in enumerate(item_ids_raw):
            mapped = self._map_item_id(raw_id, lookup_map)
            if mapped is None or mapped <= 0:
                continue
            mapped_ids.append(mapped)
            kept_indices.append(idx)
        if not mapped_ids:
            raise ValueError("No semantic IDs mapped to training item ids")

        max_item_id = max(mapped_ids if num_items is None else [*mapped_ids, int(num_items)])
        lookup = torch.zeros((max_item_id + 1, sid_tokens.size(1)), dtype=torch.long)
        kept_sids = sid_tokens[torch.tensor(kept_indices, dtype=torch.long)]
        lookup[torch.tensor(mapped_ids, dtype=torch.long)] = kept_sids
        log.info(
            "Loaded %d full SID mappings from %s with %d SID columns (%d semantic + dedup)",
            len(mapped_ids),
            path,
            sid_tokens.size(1),
            sid_tokens.size(1) - 1,
        )
        return lookup

    def _load_prebuilt_sid_lookup(self, sid_lookup_path: str) -> torch.Tensor:
        data = torch.load(Path(sid_lookup_path), map_location="cpu")
        lookup = data["sid_lookup"] if isinstance(data, dict) else data
        if not isinstance(lookup, torch.Tensor):
            raise TypeError("Prebuilt SID lookup must be a tensor or contain sid_lookup")
        lookup = lookup.to(torch.long)
        if lookup.dim() != 2 or lookup.size(1) < 2:
            raise ValueError("Prebuilt SID lookup must have shape [items, SID columns]")
        log.info(
            "Loaded prebuilt SID lookup from %s with shape %s",
            sid_lookup_path,
            tuple(lookup.shape),
        )
        return lookup

    def _load_pseudo_ser_items(self, pseudo_ser_path: str | None) -> torch.Tensor:
        if pseudo_ser_path is None:
            return torch.empty(0, dtype=torch.long)
        import pandas as pd

        path = Path(pseudo_ser_path)
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError("pseudo_ser_path must be .parquet or .csv")
        if "item_id" not in frame.columns:
            raise ValueError("pseudo-ser file must contain an item_id column")
        if "label" in frame.columns:
            frame = frame[frame["label"].astype(int) == 1]
        return torch.tensor(frame["item_id"].astype(int).tolist(), dtype=torch.long)

    def _read_item_lookup(
        self, item_lookup_path: str | None, lookup_is_zero_based: bool
    ) -> dict[str, int] | None:
        if item_lookup_path is None:
            return None
        import pandas as pd

        lookup_df = pd.read_csv(item_lookup_path)
        offset = 1 if lookup_is_zero_based else 0
        return {
            str(orig).casefold(): int(norm) + offset
            for orig, norm in zip(
                lookup_df["original_item_id"].astype(str),
                lookup_df["normalized_item_id"].astype(int),
            )
        }

    def _map_item_id(self, raw_id: Any, lookup_map: dict[str, int] | None) -> int | None:
        key = str(raw_id).casefold()
        if lookup_map is not None and key in lookup_map:
            return lookup_map[key]
        try:
            value = int(raw_id)
        except (TypeError, ValueError):
            return None
        return value
