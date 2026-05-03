import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from generative_recommenders_pl.data.reco_dataset import FutureWindowTargetDataset, RecoDataset
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
