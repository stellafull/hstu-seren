import json
import subprocess
import sys

import pandas as pd

from generative_recommenders_pl.data.reco_dataset import LOOManifestEvalDataset


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


def test_ser_event_manifest_builds_prefix_queries_and_strict_train(tmp_path):
    src = tmp_path / "seq.csv"
    pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "sequence_item_ids": ["10,11,12,13,14", "20,21,22", "30,31,32"],
            "sequence_ratings": ["5,4,5,4,5", "5,5,5", "5,5,5"],
            "sequence_timestamps": ["10,20,30,40,50", "10,20,30", "10,20,30"],
            "sequence_ser_label": ["0,1,0,1,1", "0,0,1", "0,0,0"],
        }
    ).to_csv(src, index=False)

    out = tmp_path / "ser_event"
    completed = subprocess.check_output(
        [
            sys.executable,
            "tools/build_ser_event_manifest.py",
            "--dataset",
            "tiny",
            "--input",
            str(src),
            "--output-dir",
            str(out),
            "--min-prefix",
            "1",
            "--rating-threshold",
            "4",
            "--single-positive-dest",
            "val",
        ],
        text=True,
    )
    cli = json.loads(completed)
    meta = json.loads((out / "manifest_meta.json").read_text())
    val_df = pd.read_parquet(out / "val_eval.parquet")
    test_df = pd.read_parquet(out / "test_eval.parquet")
    train_df = pd.read_parquet(out / "strict_train.parquet")

    assert cli["protocol"] == "SER_EVENT_FULL_CATALOG"
    assert meta["denominator"] == "num_ser_queries"
    assert meta["num_eligible_ser_positive_events"] == 4
    assert meta["num_val_rows"] == 2
    assert meta["num_test_rows"] == 1

    user1_val = val_df[val_df.user_id == 1].iloc[0]
    assert int(user1_val.position_t) == 3
    assert json.loads(user1_val.history_items) == [10, 11, 12]
    assert int(user1_val.target_item) == 13
    assert int(user1_val.target_ser_label) == 1

    user1_test = test_df[test_df.user_id == 1].iloc[0]
    assert int(user1_test.position_t) == 4
    assert json.loads(user1_test.history_items) == [10, 11, 12, 13]
    assert int(user1_test.target_item) == 14

    user2_val = val_df[val_df.user_id == 2].iloc[0]
    assert int(user2_val.position_t) == 2
    assert int(user2_val.target_item) == 22

    user1_train = train_df[train_df.user_id == 1].iloc[0]
    assert json.loads(user1_train.train_items) == [10, 11, 12]
    assert int(user1_train.strict_truncated_before_position) == 3
    assert FORBIDDEN_SER_FIELDS.isdisjoint(train_df.columns)

    ds = LOOManifestEvalDataset(
        str(out / "val_eval.parquet"),
        padding_length=5,
        shift_id_by=1,
        chronological=True,
    )
    sample = ds[0]
    assert int(sample["target_ser_label"]) == 1
    assert int(sample["history_lengths"]) >= 2


def test_ser_event_manifest_applies_prefix_and_rating_filters(tmp_path):
    src = tmp_path / "seq.csv"
    pd.DataFrame(
        {
            "user_id": [1],
            "sequence_item_ids": ["1,2,3,4"],
            "sequence_ratings": ["5,5,3,5"],
            "sequence_timestamps": ["1,2,3,4"],
            "sequence_ser_label": ["1,0,1,1"],
        }
    ).to_csv(src, index=False)

    out = tmp_path / "ser_event"
    subprocess.check_call(
        [
            sys.executable,
            "tools/build_ser_event_manifest.py",
            "--dataset",
            "tiny",
            "--input",
            str(src),
            "--output-dir",
            str(out),
            "--min-prefix",
            "2",
            "--rating-threshold",
            "4",
            "--single-positive-dest",
            "test",
        ]
    )

    meta = json.loads((out / "manifest_meta.json").read_text())
    test_df = pd.read_parquet(out / "test_eval.parquet")

    assert meta["num_ser_positive_events"] == 3
    assert meta["num_eligible_ser_positive_events"] == 1
    assert meta["num_val_rows"] == 0
    assert meta["num_test_rows"] == 1
    assert int(test_df.iloc[0].position_t) == 3
    assert json.loads(test_df.iloc[0].history_items) == [1, 2, 3]
