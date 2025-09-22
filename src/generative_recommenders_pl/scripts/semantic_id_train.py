from __future__ import annotations

import glob
from pathlib import Path
from typing import Sequence

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from generative_recommenders_pl.models.semantic_id.embedding_generator import (
    load_parquet_embeddings,
)
from generative_recommenders_pl.models.semantic_id.residual_quantizer import (
    ResidualVectorQuantizer,
)
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"

DTYPE_LOOKUP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def _resolve_torch_dtype(
    value: torch.dtype | str | None,
    *,
    default: torch.dtype,
) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    key = str(value).lower().strip()
    key = key.replace("torch.", "")
    try:
        return DTYPE_LOOKUP[key]
    except KeyError as exc:  # pragma: no cover - defensive
        allowed = ", ".join(sorted(DTYPE_LOOKUP))
        raise ValueError(f"Unsupported dtype '{value}'. Allowed: {allowed}.") from exc


def _normalize_ref(reference: str, category: str | None = None) -> str:
    ref = reference.strip()
    if not ref:
        raise ValueError("Empty config reference provided.")
    if ref.startswith("semantic_id/"):
        return ref
    if "/" in ref:
        return f"semantic_id/{ref}"
    if category:
        return f"semantic_id/{category}/{ref}"
    return f"semantic_id/{ref}"


def _load_config(reference: str, *, category: str | None = None) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        return compose(config_name=_normalize_ref(reference, category))


def _merge_with_overlay(base: DictConfig, overlay: DictConfig | None) -> DictConfig:
    if overlay is None:
        return base
    return OmegaConf.merge(base, overlay)


def _resolve_embedding_sources(entry_cfg: DictConfig) -> Path | list[Path]:
    path_value = entry_cfg.get("path")
    paths_value = entry_cfg.get("paths")
    glob_value = entry_cfg.get("path_glob")

    has_path = path_value is not None

    paths_list: list[Path] | None = None
    if paths_value is not None:
        paths_list = [Path(candidate) for candidate in paths_value]
    has_paths = bool(paths_list)

    glob_pattern = None
    if glob_value is not None:
        candidate = str(glob_value).strip()
        if candidate:
            glob_pattern = candidate
    has_glob = glob_pattern is not None

    specified = [has_path, has_paths, has_glob]
    if sum(specified) == 0:
        raise ValueError(
            "Each training embedding entry must define one of 'path', 'paths', or 'path_glob'."
        )
    if sum(specified) > 1:
        raise ValueError(
            "Each training embedding entry must specify only one of 'path', 'paths', or 'path_glob'."
        )

    if has_path:
        return Path(path_value)

    if has_paths and paths_list is not None:
        return paths_list

    matches = sorted(Path(match) for match in glob.glob(glob_pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"No embedding parquet files matched glob pattern '{glob_pattern}'."
        )
    return matches


def _resolved_configs(
    cfg: DictConfig,
    selector_key: str,
    overlay_keys: Sequence[str],
    *,
    category: str,
) -> list[DictConfig]:
    overlay_cfg = None
    if overlay_keys:
        overlay_cfg = OmegaConf.masked_copy(cfg, list(overlay_keys))
        if overlay_cfg is not None and len(overlay_cfg) == 0:
            overlay_cfg = None

    explicit = cfg.get(selector_key)
    if explicit:
        return [_merge_with_overlay(_load_config(str(explicit), category=category), overlay_cfg)]

    references = cfg.get("refs")
    if references:
        return [
            _merge_with_overlay(_load_config(str(reference), category=category), overlay_cfg)
            for reference in references
        ]

    return [_merge_with_overlay(cfg, overlay_cfg)]


def run(cfg: DictConfig) -> None:
    resolved_cfgs = _resolved_configs(
        cfg,
        selector_key="train_cfg",
        overlay_keys=("training",),
        category="training",
    )

    if cfg.get("train_cfg") or cfg.get("refs"):
        for resolved in resolved_cfgs:
            run(resolved)
        return

    cfg = resolved_cfgs[0]

    log.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))

    entries = cfg.training.embeddings
    quantizer_cfg = cfg.training.quantizer
    storage_dtype = _resolve_torch_dtype(
        quantizer_cfg.get("storage_dtype"), default=torch.float16
    )
    compute_dtype = _resolve_torch_dtype(
        quantizer_cfg.get("compute_dtype"), default=torch.float32
    )

    quantizer_path = Path(cfg.training.output.quantizer_path)
    semantic_ids_path = Path(cfg.training.output.semantic_ids_path)
    overwrite = bool(cfg.training.output.get("overwrite", False))

    if not overwrite and quantizer_path.exists():
        log.info("Quantizer %s already exists and overwrite disabled. Skipping.", quantizer_path)
        return

    if cfg.training.get("seed"):
        L.seed_everything(int(cfg.training.seed), workers=True)

    all_ids: list[str] = []
    tensors: list[torch.Tensor] = []
    for entry_cfg in entries:
        sources = _resolve_embedding_sources(entry_cfg)
        dtype_value = entry_cfg.get("dtype")
        target_dtype = _resolve_torch_dtype(dtype_value, default=torch.float16)

        item_ids, embeddings = load_parquet_embeddings(sources, target_dtype=target_dtype)
        prefix = entry_cfg.get("id_prefix")
        if prefix:
            item_ids = [f"{prefix}{item_id}" for item_id in item_ids]

        all_ids.extend(item_ids)
        tensors.append(embeddings.to(storage_dtype).contiguous())

    embeddings_tensor = torch.cat(tensors, dim=0).contiguous()
    tensors.clear()
    log.info("Training on %s embeddings with dimension %s", len(all_ids), embeddings_tensor.shape[1])

    codebook_width = quantizer_cfg.codebook_width
    if isinstance(codebook_width, (list, tuple)):
        widths = [int(width) for width in codebook_width]
    else:
        widths = [int(codebook_width)] * int(quantizer_cfg.num_hierarchies)

    quantizer = ResidualVectorQuantizer(
        num_layers=int(quantizer_cfg.num_hierarchies),
        codebook_size=widths,
        chunk_size=int(quantizer_cfg.chunk_size),
        max_iterations=int(quantizer_cfg.max_iterations),
        tolerance=float(quantizer_cfg.tolerance),
        normalize_residuals=bool(quantizer_cfg.normalize_residuals),
        seed=quantizer_cfg.get("seed"),
        device=quantizer_cfg.get("device"),
        use_mini_batch=bool(quantizer_cfg.get("use_mini_batch", False)),
        mini_batch_size=quantizer_cfg.get("mini_batch_size"),
        mini_batch_epochs=quantizer_cfg.get("mini_batch_epochs"),
        shuffle_mini_batches=quantizer_cfg.get("shuffle_mini_batches", True),
        storage_dtype=storage_dtype,
        compute_dtype=compute_dtype,
    )

    result = quantizer.fit(embeddings_tensor, all_ids)

    quantizer_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_ids_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        quantizer_path.unlink(missing_ok=True)
        semantic_ids_path.unlink(missing_ok=True)

    result.save(quantizer_path)
    torch.save(
        {
            "item_ids": all_ids,
            "semantic_ids": result.semantic_id_tensor(),
        },
        semantic_ids_path,
    )

    log.info(
        "Saved quantizer to %s and semantic IDs to %s",
        quantizer_path,
        semantic_ids_path,
    )


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="semantic_id/training/sid_train_all",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
