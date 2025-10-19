"""SASRec-based retrieval Lightning module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
import torchmetrics
from omegaconf import DictConfig, OmegaConf, open_dict

from generative_recommenders_pl.data.reco_dataset import RecoDataModule
from generative_recommenders_pl.models.embeddings import EmbeddingModule
from generative_recommenders_pl.models.embeddings.embeddings_sid import (
    LocalSIDEmbeddingModule,
)
from generative_recommenders_pl.models.indexing.candidate_index import CandidateIndex
from generative_recommenders_pl.models.losses.autoregressive_losses import (
    AutoregressiveLoss,
)
from generative_recommenders_pl.models.negatives_samples.negative_sampler import (
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
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

__all__ = ["SASRec"]


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


class SASRec(Retrieval):
    """Retrieval model that swaps in the SASRec sequential encoder backbone."""

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
        embedding_type: str = "local",
        embedding_configs: DictConfig | dict | None = None,
        semantic_id_prefix: str | None = None,
        transfer_learning: dict[str, Any] | DictConfig | None = None,
        pretrained_checkpoint_path: str | None = None,
        load_strict: bool = False,
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

        self._sequence_encoder_compiled: bool = False

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
            log.info("Using embedding module (%s): %s", module_name, debug_str)
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
                    "Transfer learning requested freeze of missing module %s",
                    module_name,
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
            self.net = torch.nn.Identity()
        super().setup(stage)
        if not self.compile_model:
            return
        if stage not in {"fit", None}:
            return
        if self._sequence_encoder_compiled:
            return
        if not hasattr(torch, "compile"):
            log.warning("torch.compile is unavailable; skipping SASRec compilation")
            return
        try:
            self.sequence_encoder = torch.compile(self.sequence_encoder)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("Failed to compile sequence encoder: %s", exc)
            return
        self._sequence_encoder_compiled = True
        log.info("Compiled sequence encoder with torch.compile")
