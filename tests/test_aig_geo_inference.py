from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from generative_recommenders_pl.models.serenfree.v2_levels import semantic_non_dedup_levels
from generative_recommenders_pl.scripts.evaluate_serenfree_retrieval import (
    AIGConfig,
    EvalContexts,
    LateFusionConfig,
    audit_pipeline,
    build_sid_transition_index,
    load_manifest_meta,
    score_candidate_sids,
)
from generative_recommenders_pl.serenfree.geometry import relevance_safe_geometry_boost


FORBIDDEN_AIG_LEVEL_PATTERNS = (
    "aig_levels: [1, 2]",
    'configured("aig_levels", [1, 2])',
)


def _dummy_args(**overrides) -> Namespace:
    data = {
        "checkpoint": Path("tmp/checkpoint.ckpt"),
        "experiment": "tiny-exp",
        "split": "test",
        "filter_history": True,
        "beam_size": 20,
        "ks": [10],
        "aig": True,
        "aig_alpha": 0.5,
        "aig_levels": "adaptive_semantic_non_dedup",
        "aig_gate_min_acceptable_prob": 0.05,
        "pseudo_ser_path": None,
        "late_fusion": True,
        "candidate_M": 20,
        "relevance_floor_rank": 10,
        "beta": 0.25,
        "geometry_levels": "adaptive_semantic_non_dedup",
        "ring_low_quantile": 0.60,
        "ring_high_quantile": 0.95,
        "geometry_recency_decay": 0.85,
        "geometry_epsilon": 1e-6,
        "geometry_history_len": None,
    }
    data.update(overrides)
    return Namespace(**data)


def _dummy_cfg():
    return OmegaConf.create(
        {"model": {"sid_lookup_path": "tmp/sid_lookup.pt", "sid_path": "tmp/sid.pt"}}
    )


def _dummy_model(num_sid_columns: int = 4):
    return SimpleNamespace(
        sid_composer=object(),
        sequence_encoder=object(),
        decoder=SimpleNamespace(heads=[object()] * num_sid_columns),
    )


def test_adaptive_aig_excludes_dedup():
    assert semantic_non_dedup_levels(4, "adaptive_semantic_non_dedup") == (0, 1, 2)


def test_explicit_dedup_aig_level_is_rejected_by_helper():
    with pytest.raises(ValueError):
        semantic_non_dedup_levels(4, [4])


def test_duplicate_full_sid_fails_by_default():
    sid_lookup = torch.tensor(
        [
            [1, 10, 100],
            [1, 10, 100],
            [2, 20, 200],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="duplicate full SIDs"):
        build_sid_transition_index(sid_lookup, allow_duplicate_sids=False)


def test_duplicate_full_sid_can_be_allowed_for_audit_only():
    sid_lookup = torch.tensor(
        [
            [1, 10, 100],
            [1, 10, 100],
            [2, 20, 200],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )

    sid_index, audit = build_sid_transition_index(sid_lookup, allow_duplicate_sids=True)

    assert audit["duplicate_full_sid_count"] == 1
    assert audit["candidate_items"] == 3
    assert sid_index.num_sid_columns == 3
    assert int(sid_index.child_valid[-1].sum().item()) == 2


def test_audit_pipeline_marks_adaptive_aig_as_non_dedup():
    audit = audit_pipeline(
        model=_dummy_model(),
        trie_audit={"sid_columns": 4, "dedup_column": 3},
        cfg=_dummy_cfg(),
        args=_dummy_args(aig_levels="adaptive_semantic_non_dedup"),
    )

    assert audit["aig_levels"] == "adaptive_semantic_non_dedup"
    assert audit["aig_touches_dedup"] is False


def test_audit_pipeline_flags_explicit_dedup_level():
    audit = audit_pipeline(
        model=_dummy_model(),
        trie_audit={"sid_columns": 4, "dedup_column": 3},
        cfg=_dummy_cfg(),
        args=_dummy_args(aig_levels=[4]),
    )

    assert audit["aig_levels"] == [4]
    assert audit["aig_touches_dedup"] is True


def test_audit_marks_inline_aig_disabled_when_late_fusion_is_enabled():
    audit = audit_pipeline(
        model=_dummy_model(),
        trie_audit={"sid_columns": 4, "dedup_column": 3},
        cfg=_dummy_cfg(),
        args=_dummy_args(aig=True, late_fusion=True, aig_alpha=0.5),
    )

    assert audit["inline_aig_enabled"] is False
    assert audit["inline_aig_disabled_by_late_fusion"] is True
    assert audit["late_fusion_ai_gap_alpha"] == 0.5


def test_audit_pipeline_records_manifest_hashes(tmp_path):
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "manifest_meta.json").write_text(
        """{"protocol":"SER_EVENT_FULL_CATALOG","denominator":"num_ser_queries","manifest_hash":"mh","item_universe_hash":"ih","sid_lookup_hash":"sh","trie_hash":"th"}"""
    )
    cfg = OmegaConf.create(
        {
            "model": {"sid_lookup_path": "tmp/sid_lookup.pt", "sid_path": "tmp/sid.pt"},
            "serenfree_eval": {"manifest_dir": str(manifest_dir)},
        }
    )

    audit = audit_pipeline(
        model=_dummy_model(),
        trie_audit={"sid_columns": 4, "dedup_column": 3},
        cfg=cfg,
        args=_dummy_args(),
    )

    assert audit["loo_manifest_hash"] == "mh"
    assert audit["eval_protocol"] == "SER_EVENT_FULL_CATALOG"
    assert audit["eval_denominator"] == "num_ser_queries"
    assert audit["item_universe_hash"] == "ih"
    assert audit["manifest_sid_lookup_hash"] == "sh"
    assert audit["manifest_trie_hash"] == "th"
    assert audit["loo_manifest_meta_path"] == str(manifest_dir / "manifest_meta.json")


@pytest.mark.parametrize("root", [Path("configs"), Path("src"), Path("evaluator")])
def test_no_hard_coded_nonadaptive_aig_levels_regression(root: Path):
    if not root.exists():
        return

    hits: list[str] = []
    for path in root.rglob("*"):
        if path.is_dir() or path.suffix not in {".py", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_AIG_LEVEL_PATTERNS:
            if pattern in text:
                hits.append(f"{path}: {pattern}")

    assert not hits, "Forbidden AIG level regression(s):\n" + "\n".join(hits)


def test_low_relevance_high_geometry_not_boosted():
    rel = torch.tensor([5.0, 1.0])
    geo = torch.tensor([0.5, 0.9])
    ranks = torch.tensor([1, 999])

    scored, mask, _ = relevance_safe_geometry_boost(
        rel,
        geo,
        ranks,
        beta=1.0,
        relevance_floor_rank=10,
        low_q=0.0,
        high_q=1.0,
    )

    assert not bool(mask[1])
    assert scored[1].item() == rel[1].item()


def test_late_fusion_config_parses_v2_knobs():
    args = _dummy_args(candidate_M=1000, relevance_floor_rank=500, beta=0.25)
    cfg = LateFusionConfig.from_args(args)
    assert cfg is not None
    assert cfg.candidate_m == 1000
    assert cfg.relevance_floor_rank == 500
    assert cfg.beta == 0.25
    assert cfg.alpha == 0.5
    assert cfg.geometry_epsilon == 1e-6


def test_load_manifest_meta_falls_back_to_split_ratings_path(tmp_path):
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "manifest_meta.json").write_text('{"manifest_hash":"from-split"}')
    cfg = OmegaConf.create(
        {
            "data": {
                "test_dataset": {
                    "ratings_file": str(manifest_dir / "loo_eval.parquet")
                }
            }
        }
    )
    args = _dummy_args(split="test")

    meta = load_manifest_meta(cfg, args)

    assert meta["manifest_hash"] == "from-split"


def test_late_fusion_scores_modes_with_separate_contexts():
    class TinyDecoder(torch.nn.Module):
        num_semantic_levels = 1
        prefix_dim = 2
        mode_embedding = torch.nn.Embedding(3, 2)
        heads = torch.nn.ModuleList([torch.nn.Linear(2, 3, bias=False), torch.nn.Linear(2, 3, bias=False)])
        def _prefix_parts(self, prefix_tokens, level, empty_prefix):
            return []
        def _state(self, context, mode_emb, level, prefix_parts):
            return context
    decoder = TinyDecoder()
    decoder.heads[0].weight.data = torch.eye(3, 2)
    decoder.heads[1].weight.data.zero_()
    contexts = SimpleNamespace(
        relevance=torch.tensor([[3.0, 0.0], [3.0, 0.0]]),
        acceptable=torch.tensor([[0.0, 3.0], [0.0, 3.0]]),
        imminent=torch.tensor([[3.0, 0.0], [3.0, 0.0]]),
    )
    candidate_sids = torch.tensor([[1, 1], [2, 1]])
    scores = score_candidate_sids(decoder, contexts, candidate_sids, alpha=1.0)
    assert scores[0] > scores[1]


def test_late_fusion_candidate_score_uses_relevance_dedup_only():
    class TinyDecoder(torch.nn.Module):
        num_semantic_levels = 1
        prefix_dim = 2
        mode_embedding = torch.nn.Embedding(3, 2)
        heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(2, 3, bias=False),
                torch.nn.Linear(2, 3, bias=False),
            ]
        )

        def _prefix_parts(self, prefix_tokens, level, empty_prefix):
            return []

        def _state(self, context, mode_emb, level, prefix_parts):
            return context

    decoder = TinyDecoder()
    decoder.heads[0].weight.data.zero_()
    decoder.heads[1].weight.data = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]]
    )
    contexts = EvalContexts(
        relevance=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        acceptable=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        imminent=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )
    candidate_sids = torch.tensor([[1, 1], [1, 2]])

    scores = score_candidate_sids(decoder, contexts, candidate_sids, alpha=10.0)

    assert scores[1] > scores[0]
