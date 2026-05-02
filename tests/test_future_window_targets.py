import hashlib
import json
import subprocess
import sys

import pandas as pd


def test_future_targets_track_seen_exclusions_and_caps(tmp_path):
    train = tmp_path / "loo_train.parquet"
    pd.DataFrame(
        {
            "dataset": ["tiny"],
            "split_version": ["loo_v1"],
            "user_id": [1],
            "train_items": ['[10,20,10,30,40,50,60]'],
            "train_ratings": ['[5,5,5,4,5,4,5]'],
            "train_timestamps": ['[1,2,3,4,5,6,7]'],
            "train_sids": ['[[1,1,1],[1,1,2],[1,1,1],[2,1,1],[2,1,2],[3,1,1],[3,1,2]]'],
            "num_train_items": [7],
        }
    ).to_parquet(train, index=False)
    out = tmp_path / "future.parquet"

    subprocess.check_call(
        [
            sys.executable,
            "tools/build_future_window_targets.py",
            "--loo-train",
            str(train),
            "--output",
            str(out),
            "--imminent-window",
            "1",
            "--acceptable-min-gap",
            "2",
            "--acceptable-window",
            "5",
            "--rating-threshold",
            "4",
            "--max-a-targets",
            "2",
        ]
    )

    df = pd.read_parquet(out)
    row = df[df.position_t == 1].iloc[0]
    assert row.R_item == 10
    assert row.seen_excluded_count == 1
    assert row.empty_I
    assert json.loads(row.I_items) == []
    assert row.num_A_targets == 2
    assert json.loads(row.A_items) == [40, 60]
    assert json.loads(row.A_offsets) == [3, 5]
    assert json.loads(row.A_ratings) == [5.0, 5.0]
    expected_hash = hashlib.sha256(json.dumps([10, 20], separators=(",", ":")).encode("utf-8")).hexdigest()
    assert row.history_hash == expected_hash
