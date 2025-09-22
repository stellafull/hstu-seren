from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import hydra
import lightning as L
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from generative_recommenders_pl.models.semantic_id.embedding_generator import (
    ItemText,
    generate_item_embeddings,
    iter_amazon_metadata,
    iter_serendipity_movies,
)
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"

DATASET_READERS: dict[str, Callable[[Path], Iterable[ItemText]]] = {
    "amazon_meta": iter_amazon_metadata,
    "serendipity_movies": iter_serendipity_movies,
}


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


def _dataset_records(dataset_cfg: DictConfig) -> Iterable[ItemText]:
    dataset_type = str(dataset_cfg.get("type"))
    reader = DATASET_READERS.get(dataset_type)
    if reader is None:
        raise ValueError(f"Unsupported dataset type '{dataset_type}'.")

    input_path = Path(dataset_cfg.get("input_path", ""))
    return reader(input_path)


def run(cfg: DictConfig) -> None:
    resolved_cfgs = _resolved_configs(
        cfg,
        selector_key="embed_cfg",
        overlay_keys=("embedding", "seed"),
        category="embedding",
    )

    if cfg.get("embed_cfg") or cfg.get("refs"):
        for resolved in resolved_cfgs:
            run(resolved)
        return

    cfg = resolved_cfgs[0]

    log.info("Embedding configuration:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        L.seed_everything(int(cfg.seed), workers=True)

    dataset_cfg = cfg.embedding.dataset
    model_cfg = cfg.embedding.model
    output_cfg = cfg.embedding.output

    output_path = Path(output_cfg.path)
    if output_path.exists() and not bool(output_cfg.get("overwrite", False)):
        log.info("Embeddings already exist at %s and overwrite disabled.", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _dataset_records(dataset_cfg)

    generate_item_embeddings(
        records,
        model_name=model_cfg.name,
        batch_size=int(model_cfg.batch_size),
        device=model_cfg.get("device"),
        normalize_embeddings=bool(model_cfg.get("normalize", True)),
        output_path=output_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        use_flash_attention=bool(model_cfg.get("use_flash_attention", True)),
        use_fp16=bool(model_cfg.get("use_fp16", True)),
        padding_side=model_cfg.get("padding_side", "left"),
        truncate_dim=model_cfg.get("truncate_dim"),
        storage_dtype=model_cfg.get("storage_dtype"),
        compute_dtype=model_cfg.get("compute_dtype"),
        max_items_per_file=output_cfg.get("max_items_per_file"),
        output_format=output_cfg.get("format"),
        resume=bool(output_cfg.get("resume", False)),
    )

    log.info("Finished embedding generation. Output stored at %s", output_path)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="semantic_id/embedding/emb_all",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
