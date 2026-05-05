import subprocess
import sys

import pandas as pd


FORBIDDEN_SER_FIELDS = {
    "sequence_ser_label",
    "target_ser_label",
    "s_ser_find",
    "s_ser_imp",
    "s_ser_rec",
    "m_ser_find",
    "m_ser_imp",
    "m_ser_rec",
}


def test_training_and_miner_outputs_exclude_ser_labels(tmp_path):
    src = tmp_path / "seq.csv"
    pd.DataFrame(
        {
            "user_id": [1],
            "sequence_item_ids": ["1,2,1,3,4,5"],
            "sequence_ratings": ["5,4,5,5,4,5"],
            "sequence_timestamps": ["10,20,30,40,50,60"],
            "sequence_ser_label": ["0,0,0,0,0,1"],
        }
    ).to_csv(src, index=False)

    manifest_dir = tmp_path / "manifest"
    subprocess.check_call(
        [
            sys.executable,
            "tools/build_loo_manifest.py",
            "--dataset",
            "tiny",
            "--input",
            str(src),
            "--output-dir",
            str(manifest_dir),
        ]
    )

    future_targets = tmp_path / "future.parquet"
    subprocess.check_call(
        [
            sys.executable,
            "tools/build_future_window_targets.py",
            "--loo-train",
            str(manifest_dir / "loo_train.parquet"),
            "--output",
            str(future_targets),
            "--imminent-window",
            "1",
            "--acceptable-min-gap",
            "2",
            "--acceptable-window",
            "4",
            "--rating-threshold",
            "4",
        ]
    )

    mined = tmp_path / "lf_candidates.parquet"
    subprocess.check_call(
        [
            sys.executable,
            "tools/mine_label_free_ser_candidates.py",
            "--future-targets",
            str(future_targets),
            "--output",
            str(mined),
        ]
    )

    ser_event_manifest = tmp_path / "ser_event_manifest"
    subprocess.check_call(
        [
            sys.executable,
            "tools/build_ser_event_manifest.py",
            "--dataset",
            "tiny",
            "--input",
            str(src),
            "--output-dir",
            str(ser_event_manifest),
        ]
    )

    eval_df = pd.read_parquet(manifest_dir / "loo_eval.parquet")
    train_df = pd.read_parquet(manifest_dir / "loo_train.parquet")
    mined_df = pd.read_parquet(mined)
    ser_event_train_df = pd.read_parquet(
        ser_event_manifest / "strict_train.parquet"
    )

    assert "target_ser_label" in eval_df.columns
    assert FORBIDDEN_SER_FIELDS.isdisjoint(train_df.columns)
    assert FORBIDDEN_SER_FIELDS.isdisjoint(mined_df.columns)
    assert FORBIDDEN_SER_FIELDS.isdisjoint(ser_event_train_df.columns)
