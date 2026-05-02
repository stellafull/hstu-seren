import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from generative_recommenders_pl.data.reco_dataset import FutureWindowTargetDataset


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
