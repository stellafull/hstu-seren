from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
import torchmetrics
from omegaconf import DictConfig, ListConfig
try:  # omegaconf >= 2.3
    from omegaconf.base import ContainerMetadata
except ImportError:  # pragma: no cover - older versions
    ContainerMetadata = None

from generative_recommenders_pl.models.generative_recommenders import (
    GenerativeRecommenders,
)
from generative_recommenders_pl.models.negatives_samples.negative_sampler import (
    InBatchNegativesSampler,
)
from generative_recommenders_pl.models.postprocessors.ser_postprocessors import (
    CandidateSetBuilder,
)
from generative_recommenders_pl.models.ser_energy import SerEnergy
from generative_recommenders_pl.models.utils import ops
from generative_recommenders_pl.models.utils.features import (
    SequentialFeatures,
    seq_features_from_row,
)
from generative_recommenders_pl.utils.logger import RankedLogger
from generative_recommenders_pl.utils.surprise import (
    online_serendipity_scores,
)

log = RankedLogger(__name__)


class SerRetrieval(GenerativeRecommenders):
    """Single-head HSTU module with additive serendipity energy."""

    def __init__(
        self,
        *,
        candidate_builder: CandidateSetBuilder | DictConfig,
        ser_metrics: torchmetrics.Metric | DictConfig,
        ser_energy: Optional[SerEnergy | DictConfig] = None,
        use_ser_energy: bool = True,
        lambda_ser: float = 0.4,
        detach_base_steps: int = 10000,
        ser_label_source: str = "hybrid",
        eval_ser_label_source: Optional[str] = None,
        unexpectedness_quantile: float = 0.65,
        relevance_gate_quantile: Optional[float] = None,
        candidate_size: Optional[int] = None,
        init_from_ckpt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(candidate_builder, DictConfig):
            self.candidate_builder: CandidateSetBuilder = hydra.utils.instantiate(
                candidate_builder
            )
        else:
            self.candidate_builder = candidate_builder

        if isinstance(ser_metrics, DictConfig):
            self.ser_metrics: torchmetrics.Metric = hydra.utils.instantiate(ser_metrics)
        else:
            self.ser_metrics = ser_metrics

        if ser_energy is not None:
            if isinstance(ser_energy, DictConfig):
                self.ser_energy: SerEnergy | None = hydra.utils.instantiate(ser_energy)
            else:
                self.ser_energy = ser_energy
        else:
            self.ser_energy = None

        if candidate_size is not None and hasattr(self.candidate_builder, "candidate_size"):
            if self.candidate_builder.candidate_size != candidate_size:
                log.warning(
                    "candidate_size argument (%s) differs from builder setting (%s); using builder setting",
                    candidate_size,
                    self.candidate_builder.candidate_size,
                )

        self.use_ser_energy = use_ser_energy and self.ser_energy is not None
        self.lambda_ser = lambda_ser
        self.detach_base_steps = max(detach_base_steps, 0)
        self.ser_label_source = ser_label_source.lower()
        self.eval_ser_label_source = (
            eval_ser_label_source.lower()
            if eval_ser_label_source is not None
            else self.ser_label_source
        )
        self.unexpectedness_quantile = unexpectedness_quantile
        self.relevance_gate_quantile = relevance_gate_quantile
        self._objective = self.loss

        max_candidates = self.candidate_index.num_objects
        if hasattr(self.candidate_builder, "candidate_size"):
            candidate_limit = self.candidate_builder.candidate_size
            if getattr(self.candidate_builder, "ensure_target", False):
                candidate_limit += 1
            max_candidates = min(max_candidates, candidate_limit)

        if max_candidates <= 0:
            msg = "Candidate builder produced zero valid candidates"
            raise ValueError(msg)

        if hasattr(self.metrics, "k") and self.metrics.k > max_candidates:
            log.warning(
                "metrics.k (%s) exceeds available candidates (%s); clamping to %s",
                self.metrics.k,
                max_candidates,
                max_candidates,
            )
            self.metrics.k = max_candidates
        if hasattr(self.metrics, "at_k_list"):
            adjusted_at_k = sorted(
                {max(1, min(k, max_candidates)) for k in self.metrics.at_k_list}
            )
            if adjusted_at_k != list(self.metrics.at_k_list):
                log.warning(
                    "metrics.at_k_list adjusted to fit available candidates: %s",
                    adjusted_at_k,
                )
                self.metrics.at_k_list = adjusted_at_k

        if hasattr(self.ser_metrics, "at_k_list"):
            adjusted_ser_k = sorted(
                {max(1, min(k, max_candidates)) for k in self.ser_metrics.at_k_list}
            )
            if adjusted_ser_k != list(self.ser_metrics.at_k_list):
                log.warning(
                    "ser_metrics.at_k_list adjusted to fit available candidates: %s",
                    adjusted_ser_k,
                )
                self.ser_metrics.at_k_list = adjusted_ser_k
                if hasattr(self.ser_metrics, "_max_k"):
                    self.ser_metrics._max_k = adjusted_ser_k[-1]

        if init_from_ckpt:
            ckpt_path = Path(init_from_ckpt)
            if not ckpt_path.exists():
                msg = f"Checkpoint not found: {ckpt_path}"
                raise FileNotFoundError(msg)
            allowlisted = [ListConfig, DictConfig, ContainerMetadata, Any]
            checkpoint = None
            safe_loaded = False
            try:  # PyTorch >=2.6
                from torch.serialization import safe_globals as _safe_globals
            except (ImportError, AttributeError):  # pragma: no cover - older torch
                _safe_globals = None

            if _safe_globals is not None:
                try:
                    with _safe_globals([cls for cls in allowlisted if cls is not None]):
                        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
                        safe_loaded = True
                except Exception as exc:
                    log.warning(
                        "safe_globals load failed for %s (%s); retrying with explicit weights_only=False",
                        ckpt_path,
                        exc,
                    )
                    checkpoint = None
            else:
                try:
                    from torch.serialization import add_safe_globals as _add_safe_globals
                except (ImportError, AttributeError):  # pragma: no cover - older torch
                    _add_safe_globals = None
                if _add_safe_globals is not None:
                    _add_safe_globals([cls for cls in allowlisted if cls is not None])

            if checkpoint is None:
                load_kwargs = {"map_location": "cpu"}
                try:
                    checkpoint = torch.load(
                        str(ckpt_path), weights_only=False, **load_kwargs
                    )
                except TypeError:
                    checkpoint = torch.load(str(ckpt_path), **load_kwargs)
            state_dict = checkpoint.get("state_dict", checkpoint)
            load_result = self.load_state_dict(state_dict, strict=False)
            missing: Tuple[str, ...] | list[str]
            unexpected: Tuple[str, ...] | list[str]
            if load_result is None:
                missing, unexpected = (), ()
            elif isinstance(load_result, tuple):  # torch<2.1 compatibility
                missing, unexpected = load_result
            else:
                missing = getattr(load_result, "missing_keys", [])
                unexpected = getattr(load_result, "unexpected_keys", [])
            if missing:
                log.warning("Missing keys when loading %s: %s", ckpt_path, missing)
            if unexpected:
                log.warning(
                    "Unexpected keys when loading %s: %s", ckpt_path, unexpected
                )
            log.info("Loaded initialization from %s", ckpt_path)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _ensure_index_embeddings(self) -> None:
        if self.candidate_index.embeddings is None:
            log.info("Populating candidate index embeddings")
            normalized = self.negatives_sampler.normalize_embeddings(
                self.embeddings.get_item_embeddings(self.candidate_index.ids)
            )
            self.candidate_index.update_embeddings(normalized)

    def _update_negative_sampler(self, supervision_ids: torch.Tensor) -> None:
        if isinstance(self.negatives_sampler, InBatchNegativesSampler):
            in_batch_ids = supervision_ids.view(-1)
            presences = in_batch_ids != 0
            self.negatives_sampler.process_batch(
                ids=in_batch_ids,
                presences=presences,
                embeddings=self.embeddings.get_item_embeddings(in_batch_ids),
            )
        else:
            self.negatives_sampler._item_emb = self.embeddings._item_emb

    def _history_embeddings(
        self, seq_features: SequentialFeatures
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        past_ids = seq_features.past_ids
        device = past_ids.device
        history_mask = torch.arange(
            past_ids.size(1), device=device
        ).unsqueeze(0) < seq_features.past_lengths.unsqueeze(1)
        if seq_features.past_embeddings is None:
            embeddings = self.embeddings.get_item_embeddings(past_ids)
        else:
            embeddings = seq_features.past_embeddings
        history_embeddings = embeddings * history_mask.unsqueeze(-1)
        return history_embeddings, history_mask

    def _pos_in_recent(
        self,
        candidate_ids: torch.Tensor,
        seq_features: SequentialFeatures,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = candidate_ids.device
        history_ids = seq_features.past_ids
        idx = torch.arange(history_ids.size(1), device=device)
        idx = idx.view(1, 1, -1).float()
        matches = (
            candidate_ids.unsqueeze(-1) == history_ids.unsqueeze(1)
        ) & history_mask.unsqueeze(1)
        default = torch.full_like(idx, float("inf"))
        pos = torch.where(matches, idx, default).min(dim=-1)[0]
        seen = torch.isfinite(pos)
        pos = torch.where(seen, pos + 1.0, torch.zeros_like(pos))
        return pos

    def _build_energy_features(
        self,
        context_state: torch.Tensor,
        unexpectedness: torch.Tensor,
        pos_in_recent: torch.Tensor,
        history_lengths: torch.Tensor,
        base_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        scaled_pos = pos_in_recent / history_lengths.clamp(min=1.0).unsqueeze(1)
        max_cos = 1.0 - unexpectedness
        base_feature = base_logits
        if self.global_step < self.detach_base_steps:
            base_feature = base_feature.detach()
        features = torch.cat(
            [
                context_state,
                unexpectedness.unsqueeze(-1),
                max_cos.unsqueeze(-1),
                scaled_pos.unsqueeze(-1),
                base_feature.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = features.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        return features

    def _ser_labels(
        self,
        online_indicator: torch.Tensor,
        candidate_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_ser: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        label_source: str,
    ) -> torch.Tensor:
        label_source = label_source.lower()
        ser_labels = online_indicator.clone()
        target_ids = target_ids.view(-1, 1)
        target_mask = candidate_ids == target_ids
        if label_source == "online":
            pass
        elif label_source == "human":
            ser_labels = torch.zeros_like(ser_labels)
            human_pos = target_ser.view(-1, 1).float() * target_mask.float()
            ser_labels = torch.where(target_mask, human_pos, ser_labels)
        elif label_source == "hybrid":  # prefer human if available
            human_pos = target_ser.view(-1, 1).float() * target_mask.float()
            ser_labels = torch.where(target_mask, human_pos, ser_labels)
        else:
            msg = f"Unknown ser_label_source: {label_source}"
            raise ValueError(msg)
        ser_labels = ser_labels * candidate_mask.to(ser_labels.dtype)
        return ser_labels

    def _compute_logits(
        self,
        seq_features: SequentialFeatures,
        target_ids: torch.Tensor,
        target_ser: torch.Tensor,
        *,
        label_source: str,
    ) -> Dict[str, torch.Tensor]:
        history_embeddings, history_mask = self._history_embeddings(seq_features)
        history_lengths = seq_features.past_lengths.float()
        seq_features_with_emb = seq_features
        if seq_features.past_embeddings is None:
            embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
            seq_features_with_emb = seq_features._replace(past_embeddings=embeddings)

            history_embeddings = embeddings * history_mask.unsqueeze(-1)

        encoded_embeddings, _ = self.forward(seq_features_with_emb)
        user_state = ops.get_current_embeddings(
            seq_features.past_lengths, encoded_embeddings
        )
        self._ensure_index_embeddings()
        candidate_ids, candidate_mask = self.candidate_builder(
            query_embeddings=user_state,
            candidate_index=self.candidate_index,
            invalid_ids=seq_features.past_ids,
            target_ids=target_ids,
        )
        candidate_embeddings = self.embeddings.get_item_embeddings(candidate_ids)
        candidate_embeddings = self.negatives_sampler.normalize_embeddings(
            candidate_embeddings
        )
        user_state_norm = self.negatives_sampler.normalize_embeddings(user_state)
        base_logits = self.similarity(
            input_embeddings=user_state_norm,
            item_embeddings=candidate_embeddings,
            item_sideinfo=None,
            item_ids=candidate_ids,
        )[0]
        base_logits = torch.where(
            candidate_mask, base_logits, torch.full_like(base_logits, -1e9)
        )
        base_prob = torch.softmax(base_logits, dim=-1)
        base_prob = base_prob * candidate_mask.to(base_prob.dtype)
        base_prob = base_prob / base_prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        unexpectedness, online_indicator = online_serendipity_scores(
            candidate_embeddings=candidate_embeddings,
            recent_embeddings=history_embeddings,
            recent_mask=history_mask,
            base_prob=base_prob,
            candidate_mask=candidate_mask,
            quantile=self.unexpectedness_quantile,
            relevance_quantile=self.relevance_gate_quantile,
        )
        pos_in_recent = self._pos_in_recent(
            candidate_ids=candidate_ids,
            seq_features=seq_features,
            history_mask=history_mask,
        )
        context_state = user_state.unsqueeze(1).expand_as(candidate_embeddings)

        adjusted_logits = base_logits
        energy = None
        if self.use_ser_energy:
            features = self._build_energy_features(
                context_state=context_state,
                unexpectedness=unexpectedness,
                pos_in_recent=pos_in_recent,
                history_lengths=history_lengths,
                base_logits=base_logits,
                candidate_mask=candidate_mask,
            )
            energy = self.ser_energy(features)
            energy = energy.masked_fill(~candidate_mask, 0.0)
            adjusted_logits = base_logits + self.lambda_ser * energy

        adjusted_logits = torch.where(
            candidate_mask,
            adjusted_logits,
            torch.full_like(adjusted_logits, -1e9),
        )

        ser_labels = self._ser_labels(
            online_indicator=online_indicator,
            candidate_ids=candidate_ids,
            target_ids=target_ids,
            target_ser=target_ser,
            candidate_mask=candidate_mask,
            label_source=label_source,
        )

        return {
            "base_logits": base_logits,
            "adjusted_logits": adjusted_logits,
            "candidate_ids": candidate_ids,
            "candidate_mask": candidate_mask,
            "ser_labels": ser_labels,
            "unexpectedness": unexpectedness,
            "pos_in_recent": pos_in_recent,
            "context_state": context_state,
            "energy": energy,
        }

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        seq_features, target_ids, _, target_ser = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=0,
        )
        supervision_ids = seq_features.past_ids
        self._update_negative_sampler(supervision_ids)
        outputs = self._compute_logits(
            seq_features=seq_features,
            target_ids=target_ids,
            target_ser=target_ser,
            label_source=self.ser_label_source,
        )
        loss, metrics = self._objective(
            logits=outputs["adjusted_logits"],
            baseline_logits=outputs["base_logits"].detach(),
            ser_labels=outputs["ser_labels"],
            mask=outputs["candidate_mask"],
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        for key, value in metrics.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=False)
        if outputs["energy"] is not None:
            energy_values = outputs["energy"].masked_select(outputs["candidate_mask"])
            avg_energy = (
                energy_values.mean()
                if energy_values.numel() > 0
                else torch.tensor(0.0, device=self.device)
            )
        else:
            avg_energy = torch.tensor(0.0, device=self.device)
        self.log("train/avg_energy", avg_energy, on_step=True, on_epoch=True)

        unexpected_vals = outputs["unexpectedness"].masked_select(
            outputs["candidate_mask"]
        )
        avg_unexpected = (
            unexpected_vals.mean()
            if unexpected_vals.numel() > 0
            else torch.tensor(0.0, device=self.device)
        )
        self.log(
            "train/avg_unexpectedness",
            avg_unexpected,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset()
        self.ser_metrics.reset()
        self._ensure_index_embeddings()

    def validation_step(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        seq_features, target_ids, _, target_ser = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=0,
        )
        outputs = self._compute_logits(
            seq_features=seq_features,
            target_ids=target_ids,
            target_ser=target_ser,
            label_source=self.eval_ser_label_source,
        )
        adjusted_logits = outputs["adjusted_logits"]
        candidate_ids = outputs["candidate_ids"]
        candidate_mask = outputs["candidate_mask"]
        ser_labels = outputs["ser_labels"]

        k = self.metrics.k
        top_scores, top_indices = adjusted_logits.topk(k=k, dim=-1)
        top_ids = candidate_ids.gather(1, top_indices)
        self.metrics.update(top_k_ids=top_ids, target_ids=target_ids)

        ser_ranked = ser_labels.gather(1, top_indices)
        self.ser_metrics.update(ser_ranked)

        return top_scores.mean()

    def on_validation_epoch_end(self) -> None:
        rel_metrics = self.metrics.compute()
        ser_metrics = self.ser_metrics.compute()
        for key, value in rel_metrics.items():
            self.log(f"val/{key}", value, prog_bar=True)
        for key, value in ser_metrics.items():
            self.log(f"val/{key}", value, prog_bar=False)
        self.metrics.reset()
        self.ser_metrics.reset()

    def on_test_epoch_start(self) -> None:
        self.metrics.reset()
        self.ser_metrics.reset()
        self._ensure_index_embeddings()

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        rel_metrics = self.metrics.compute()
        ser_metrics = self.ser_metrics.compute()
        for key, value in rel_metrics.items():
            self.log(f"test/{key}", value)
        for key, value in ser_metrics.items():
            self.log(f"test/{key}", value)
        self.metrics.reset()
        self.ser_metrics.reset()
