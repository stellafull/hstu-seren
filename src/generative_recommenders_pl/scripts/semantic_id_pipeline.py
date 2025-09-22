from __future__ import annotations

from pathlib import Path
from typing import Sequence

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from generative_recommenders_pl.scripts.semantic_id_embed import run as run_embedding
from generative_recommenders_pl.scripts.semantic_id_infer import run as run_inference
from generative_recommenders_pl.scripts.semantic_id_train import run as run_training
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


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
    category: str | None = None,
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


def _as_sequence(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value) if value else []


def run(cfg: DictConfig) -> None:
    resolved_cfgs = _resolved_configs(
        cfg,
        selector_key="pipeline_cfg",
        overlay_keys=("pipeline",),
    )

    if cfg.get("pipeline_cfg") or cfg.get("refs"):
        for resolved in resolved_cfgs:
            run(resolved)
        return

    cfg = resolved_cfgs[0]

    log.info("Pipeline plan:\n%s", OmegaConf.to_yaml(cfg))

    for embedding_ref in _as_sequence(cfg.pipeline.get("embeddings")):
        run_embedding(_load_config(embedding_ref, category="embedding"))

    for training_ref in _as_sequence(cfg.pipeline.get("trainings")):
        run_training(_load_config(training_ref, category="training"))

    for inference_ref in _as_sequence(cfg.pipeline.get("inference")):
        run_inference(_load_config(inference_ref, category="inference"))


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="semantic_id/sid_all",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
