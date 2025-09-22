from __future__ import annotations

import glob
from pathlib import Path
from typing import Sequence

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from generative_recommenders_pl.models.semantic_id.embedding_generator import load_parquet_embeddings
from generative_recommenders_pl.models.semantic_id.residual_quantizer import ResidualQuantizerResult
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
    key = str(value).lower().strip().replace("torch.", "")
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
            "Each inference embedding entry must define one of 'path', 'paths', or 'path_glob'."
        )
    if sum(specified) > 1:
        raise ValueError(
            "Each inference embedding entry must specify only one of 'path', 'paths', or 'path_glob'."
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
        selector_key="infer_cfg",
        overlay_keys=("inference",),
        category="inference",
    )

    if cfg.get("infer_cfg") or cfg.get("refs"):
        for resolved in resolved_cfgs:
            run(resolved)
        return

    cfg = resolved_cfgs[0]

    log.info("Inference configuration:\n%s", OmegaConf.to_yaml(cfg))

    entries = cfg.inference.embeddings
    quantizer_path = Path(cfg.inference.quantizer_path)
    output_path = Path(cfg.inference.output_path)

    if not quantizer_path.exists():
        raise FileNotFoundError(f"Quantizer file {quantizer_path} does not exist.")
    if output_path.exists() and not bool(cfg.inference.get("overwrite", False)):
        log.info("Inference output %s already exists and overwrite disabled.", output_path)
        return

    seen_ids: set[str] = set()
    unique_ids: list[str] = []
    embedding_chunks: list[torch.Tensor] = []
    duplicates_dropped = 0

    for entry_cfg in entries:
        sources = _resolve_embedding_sources(entry_cfg)
        dtype_value = entry_cfg.get("dtype")
        target_dtype = _resolve_torch_dtype(dtype_value, default=torch.float16)

        item_ids, embeddings = load_parquet_embeddings(sources, target_dtype=target_dtype)
        prefix = entry_cfg.get("id_prefix")
        if prefix:
            item_ids = [f"{prefix}{item_id}" for item_id in item_ids]

        keep_indices: list[int] = []
        for idx, item_id in enumerate(item_ids):
            if item_id in seen_ids:
                duplicates_dropped += 1
                continue
            seen_ids.add(item_id)
            keep_indices.append(idx)

        if not keep_indices:
            continue

        unique_ids.extend(item_ids[index] for index in keep_indices)

        if len(keep_indices) == len(item_ids):
            filtered_embeddings = embeddings
        else:
            index_tensor = torch.tensor(
                keep_indices,
                dtype=torch.long,
                device=embeddings.device,
            )
            filtered_embeddings = embeddings.index_select(0, index_tensor)

        embedding_chunks.append(filtered_embeddings.contiguous())

    if not embedding_chunks:
        raise ValueError("No embeddings available for inference after deduplication.")

    embeddings_tensor = torch.cat(embedding_chunks, dim=0).contiguous()
    embedding_chunks.clear()

    if duplicates_dropped:
        log.info("Dropped %s duplicate embedding(s) during inference.", duplicates_dropped)

    if cfg.inference.get("device"):
        embeddings_tensor = embeddings_tensor.to(cfg.inference.device)

    result = ResidualQuantizerResult.load(quantizer_path)
    normalize_residuals = cfg.inference.normalize_residuals
    if normalize_residuals is None:
        normalize_residuals = result.normalize_residuals

    storage_dtype_override = cfg.inference.get("storage_dtype")
    compute_dtype_override = cfg.inference.get("compute_dtype")

    effective_result = ResidualQuantizerResult(
        codebooks=result.codebooks,
        assignments=result.assignments,
        item_ids=result.item_ids,
        normalize_residuals=bool(normalize_residuals),
        storage_dtype=_resolve_torch_dtype(
            storage_dtype_override, default=result.storage_dtype
        ),
        compute_dtype=_resolve_torch_dtype(
            compute_dtype_override, default=result.compute_dtype
        ),
    )

    assignments = effective_result.quantize(
        embeddings_tensor,
        chunk_size=int(cfg.inference.chunk_size),
    ).cpu()

    if assignments.shape[1] != len(result.codebooks):
        raise RuntimeError(
            "Quantizer assignments dimension mismatch: expected %s layers, got %s."
            % (len(result.codebooks), assignments.shape[1])
        )

    dedup_cardinality = cfg.inference.get("dedup_slot_cardinality", 256)
    dedup_cardinality = 256 if dedup_cardinality is None else int(dedup_cardinality)
    if dedup_cardinality <= 0:
        raise ValueError("dedup_slot_cardinality must be a positive integer.")

    dedup_slot = torch.empty(assignments.size(0), dtype=torch.long, device=assignments.device)
    collision_counts: dict[tuple[int, ...], int] = {}
    total_collisions = 0
    max_offset = 0

    for index in range(assignments.size(0)):
        key = tuple(int(value) for value in assignments[index].tolist())
        offset = collision_counts.get(key, 0)
        dedup_slot[index] = offset
        collision_counts[key] = offset + 1
        if offset:
            total_collisions += 1
        if offset > max_offset:
            max_offset = offset

    if max_offset >= dedup_cardinality:
        raise RuntimeError(
            "Deduplication slot overflowed %s values for semantic IDs (max offset %s)."
            % (dedup_cardinality, max_offset)
        )

    if total_collisions:
        log.info(
            "Appended dedup slot for %s colliding embedding(s); max cardinality per code was %s.",
            total_collisions,
            max_offset + 1,
        )

    assignments = torch.cat([assignments, dedup_slot.unsqueeze(1)], dim=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"item_ids": unique_ids, "semantic_ids": assignments}, output_path)
    log.info("Saved %s semantic IDs to %s", assignments.shape[0], output_path)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="semantic_id/inference/sid_infer_all",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
