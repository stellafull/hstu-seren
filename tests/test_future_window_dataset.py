import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from generative_recommenders_pl.data.reco_dataset import (
    DynamicFutureWindowTargetDataset,
    FutureWindowTargetDataset,
    LOOManifestEvalDataset,
    RecoDataset,
)
from generative_recommenders_pl.serenfree.geometry import prefix_surprise


def test_future_window_dataset_emits_ai_and_rank_batch_fields(tmp_path):
    train = tmp_path / "loo_train.parquet"
    pd.DataFrame({
        "dataset": ["tiny"],
        "split_version": ["loo_v1"],
        "user_id": [1],
        "train_items": ["[1,2,3,4]"],
        "train_ratings": ["[5,4,5,5]"],
        "train_timestamps": ["[1,2,3,4]"],
        "train_sids": ["[[1,1,1],[1,2,1],[2,1,1],[2,2,1]]"],
        "num_train_items": [4],
    }).to_parquet(train, index=False)
    future = tmp_path / "future.parquet"
    subprocess.check_call([
        sys.executable,
        "tools/build_future_window_targets.py",
        "--loo-train", str(train),
        "--output", str(future),
        "--acceptable-min-gap", "2",
        "--acceptable-window", "3",
        "--rating-threshold", "4",
    ])
    ds = FutureWindowTargetDataset(str(future), padding_length=5, emit_rank_from_future_targets=True)
    sample = ds[0]
    assert sample["historical_ids"].shape == (4,)
    assert sample["I_sids"].ndim == 2
    assert sample["A_sids"].ndim == 2
    assert "rank_candidate_sids" in sample and "rank_positive_mask" in sample
    batch = torch.utils.data.default_collate([sample])
    assert batch["I_sids"].ndim == 3
    assert batch["A_sids"].ndim == 3
    assert torch.isfinite(batch["rank_geometry"]).all()


def test_future_window_dataset_applies_shift_id_by_to_items(tmp_path):
    future = tmp_path / "future.parquet"
    pd.DataFrame({
        "dataset": ["tiny"],
        "user_id": [1],
        "position_t": [0],
        "history_items": ["[10,20]"],
        "history_ratings": ["[5,5]"],
        "history_timestamps": ["[1,2]"],
        "history_sids": ["[[1,1,1],[1,2,1]]"],
        "timestamp_t": [2],
        "R_item": [30],
        "R_rating": [5.0],
        "I_items": ["[]"],
        "I_sids": ["[]"],
        "A_items": ["[]"],
        "A_sids": ["[]"],
    }).to_parquet(future, index=False)

    sample = FutureWindowTargetDataset(str(future), padding_length=5, shift_id_by=1)[0]

    assert sample["historical_ids"].tolist()[:2] == [11, 21]
    assert int(sample["target_ids"]) == 31


def test_future_window_rank_candidates_dedup_overlap_and_geometry(tmp_path):
    future = tmp_path / "future.parquet"
    pd.DataFrame({
        "dataset": ["tiny"],
        "split_version": ["loo_v1"],
        "user_id": [1],
        "position_t": [0],
        "history_items": ["[10,20]"],
        "history_ratings": ["[5,5]"],
        "history_timestamps": ["[1,2]"],
        "history_sids": ["[[1,1,1],[1,2,1]]"],
        "history_hash": ["h"],
        "timestamp_t": [2],
        "R_item": [30],
        "R_sid": ["[2,1,1]"],
        "R_rating": [5.0],
        "I_items": ["[3,4,3]"],
        "I_sids": ["[[3,1,1],[4,1,1],[3,1,2]]"],
        "A_items": ["[4,5,5,6]"],
        "A_sids": ["[[4,1,1],[5,1,1],[5,1,2],[6,1,1]]"],
    }).to_parquet(future, index=False)

    ds = FutureWindowTargetDataset(
        str(future),
        padding_length=5,
        max_i_targets=3,
        max_a_targets=3,
        emit_rank_from_future_targets=True,
    )
    sample = ds[0]

    assert sample["rank_candidate_sids"].tolist()[:4] == [
        [5, 1, 1],
        [6, 1, 1],
        [3, 1, 1],
        [4, 1, 1],
    ]
    assert sample["rank_positive_mask"].tolist()[:4] == [True, True, False, False]
    expected = prefix_surprise(
        torch.tensor([[1, 1, 1], [1, 2, 1]]),
        sample["rank_candidate_sids"],
    )
    assert torch.allclose(sample["rank_geometry"], expected)


def test_future_window_rank_geometry_respects_configured_history_len(tmp_path):
    frame = pd.DataFrame(
        {
            "dataset": ["tiny"],
            "user_id": [1],
            "position_t": [2],
            "history_items": ["[1,2]"],
            "history_ratings": ["[5,5]"],
            "history_timestamps": ["[10,20]"],
            "history_sids": ["[[1,1,1],[2,2,1]]"],
            "timestamp_t": [20],
            "R_item": [3],
            "R_rating": [5],
            "I_items": ["[]"],
            "I_sids": ["[]"],
            "A_items": ["[1,2]"],
            "A_sids": ["[[1,1,1],[2,2,1]]"],
        }
    )
    future = tmp_path / "future.parquet"
    frame.to_parquet(future, index=False)

    ds = FutureWindowTargetDataset(
        str(future),
        padding_length=5,
        emit_rank_from_future_targets=True,
        rank_geometry_history_len=1,
        rank_geometry_epsilon=1e-4,
    )
    sample = ds[0]

    assert sample["rank_geometry"][0].item() == pytest.approx(
        -torch.log(torch.tensor(1e-4)).item()
    )
    assert sample["rank_geometry"][1].item() == pytest.approx(0.0)


def test_dynamic_future_window_dataset_constructs_sparse_targets_from_loo_rows(tmp_path):
    train = tmp_path / "loo_train.parquet"
    pd.DataFrame({
        "dataset": ["tiny"],
        "split_version": ["loo_v1"],
        "user_id": [1],
        "train_items": ["[10,20,10,30,40,50,60]"],
        "train_ratings": ["[5,5,5,4,5,4,5]"],
        "train_timestamps": ["[1,2,3,4,5,6,7]"],
        "train_sids": ["[[1,1,1],[1,1,2],[1,1,1],[2,1,1],[2,1,2],[3,1,1],[3,1,2]]"],
        "num_train_items": [7],
    }).to_parquet(train, index=False)

    ds = DynamicFutureWindowTargetDataset(
        str(train),
        padding_length=5,
        shift_id_by=1,
        imminent_window=1,
        acceptable_min_gap=2,
        acceptable_window=5,
        rating_positive_threshold=4,
        max_a_targets=2,
        emit_rank_from_future_targets=True,
    )
    sample = ds[1]

    assert len(ds) == 6
    assert sample["historical_ids"].tolist()[:2] == [11, 21]
    assert int(sample["target_ids"]) == 11
    assert sample["I_sids"].shape == (3, 3)
    assert sample["A_sids"].tolist()[:2] == [[2, 1, 2], [3, 1, 2]]
    assert sample["rank_candidate_sids"].tolist()[:3] == [
        [2, 1, 2],
        [3, 1, 2],
        [0, 0, 0],
    ]
    assert sample["rank_positive_mask"].tolist()[:3] == [True, True, False]


def test_loo_manifest_eval_dataset_reads_frozen_eval_schema(tmp_path):
    manifest = tmp_path / "loo_eval.parquet"
    pd.DataFrame(
        {
            "user_id": [7],
            "history_items": ["[1,2,3]"],
            "history_ratings": ["[5,4,3]"],
            "history_timestamps": ["[10,20,30]"],
            "target_item": [4],
            "target_rating": [5.0],
            "target_timestamp": [40],
            "target_ser_label": [1],
        }
    ).to_parquet(manifest, index=False)

    ds = LOOManifestEvalDataset(
        str(manifest),
        padding_length=4,
        shift_id_by=1,
        chronological=True,
    )
    sample = ds[0]

    assert sample["historical_ids"].tolist() == [2, 3, 4]
    assert int(sample["history_lengths"]) == 3
    assert int(sample["target_ids"]) == 5
    assert int(sample["target_ser_label"]) == 1


def test_reco_dataset_safe_parser_supports_common_formats_and_rejects_code(tmp_path):
    frame = pd.DataFrame(
        {
            "user_id": [1, 2],
            "sequence_item_ids": ["1,2,3", "[4, 5, 6]"],
            "sequence_ratings": ["5,4,3", "[3, 4, 5]"],
            "sequence_timestamps": ["10,20,30", "[40, 50, 60]"],
        }
    )
    ds = RecoDataset(frame, padding_length=4, ignore_last_n=0, chronological=True)
    assert int(ds[0]["target_ids"]) == 3
    assert int(ds[1]["target_ids"]) == 6

    marker = tmp_path / "executed"
    bad = pd.DataFrame(
        {
            "user_id": [3],
            "sequence_item_ids": [f"__import__('pathlib').Path('{marker}').touch()"],
            "sequence_ratings": ["1,2,3"],
            "sequence_timestamps": ["1,2,3"],
        }
    )
    bad_ds = RecoDataset(bad, padding_length=4, ignore_last_n=0)
    try:
        bad_ds[0]
    except ValueError:
        pass
    assert not marker.exists()


def test_v2_model_has_fail_fast_future_target_wiring():
    # Keep this static: importing the full model loads fbgemm/CUDA on the remote
    # image even for CPU-only tests. Config-compose tests cover the class path;
    # this assertion pins the fail-fast V2 hooks in source.
    src = Path("src/generative_recommenders_pl/models/hstu_serenfree.py").read_text()
    assert "require_future_targets" in src
    assert "future_target_loss" in src
    assert "_batch_future_sid_targets" in src
    assert "_candidate_sid_scores" in src
    assert "rank_candidate_sids" in src
    assert "acceptable_context" in src and "imminent_context" in src
    assert "Missing future-window" in src
