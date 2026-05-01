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
from generative_recommenders_pl.models.sequential_encoders.hstu import HSTU, TIMESTAMPS_KEY
from generative_recommenders_pl.models.utils.initialization import truncated_normal
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
        acceptable_window: int = 20,
        recent_window: int = 10,
        compile_model: bool = False,
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
        self.acceptable_window = int(acceptable_window)
        self.recent_window = int(recent_window)
        self.compile_model = bool(compile_model)
        self._compiled = False

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
        loss, metrics = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for key, value in metrics.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        loss, metrics = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for key, value in metrics.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        loss, metrics = self._step(batch)
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

    def _step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        recent_context = self._last_valid_state(encoded, lengths)
        history_context = self._masked_history_mean(encoded, lengths)
        decoded = self.decoder(recent_context, target_sid[:, :-1], mode=DecoderMode.RELEVANCE)
        loss = relevance_loss(decoded, target_sid, lambda_d=self.lambda_d)
        metrics = self._prefix_metrics(decoded.as_list(), target_sid)

        if self.lambda_i > 0:
            imminent = self.decoder(
                recent_context,
                target_sid[:, :-1],
                mode=DecoderMode.IMMINENT,
            )
            imminent_loss = semantic_loss(imminent, target_sid)
            loss = loss + self.lambda_i * imminent_loss
            metrics["imminent_loss"] = imminent_loss.detach()
        else:
            imminent = None

        if self.lambda_a > 0:
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
            if imminent is not None:
                metrics["js_pa_pi"] = semantic_js_divergence(
                    acceptable,
                    imminent,
                ).detach()
        return loss, metrics

    def _item_ids_to_sid(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.to(torch.long)
        valid = (item_ids >= 0) & (item_ids < self.sid_lookup.size(0))
        safe_ids = item_ids.clamp(min=0, max=self.sid_lookup.size(0) - 1)
        sid = self.sid_lookup[safe_ids]
        return sid.masked_fill(~valid.unsqueeze(-1), 0)

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
