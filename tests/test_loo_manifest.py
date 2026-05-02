import json
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


def build_manifest(src, out_dir):
    subprocess.check_call(
        [
            sys.executable,
            "tools/build_loo_manifest.py",
            "--dataset",
            "tiny",
            "--input",
            str(src),
            "--output-dir",
            str(out_dir),
        ]
    )
    return json.loads((out_dir / "manifest_meta.json").read_text())


def test_build_loo_manifest_hash_is_stable_and_input_sensitive(tmp_path):
    src = tmp_path / "seq.csv"
    pd.DataFrame(
        {
            "user_id": [1, 2],
            "sequence_item_ids": ["1,2,3", "4,5,6"],
            "sequence_ratings": ["5,4,5", "3,4,5"],
            "sequence_timestamps": ["10,20,30", "10,20,30"],
            "sequence_ser_label": ["0,0,1", "0,1,0"],
        }
    ).to_csv(src, index=False)

    meta_a = build_manifest(src, tmp_path / "out_a")
    meta_b = build_manifest(src, tmp_path / "out_b")
    assert meta_a["manifest_hash"] == meta_b["manifest_hash"]

    changed = tmp_path / "seq_changed.csv"
    pd.DataFrame(
        {
            "user_id": [1, 2],
            "sequence_item_ids": ["1,2,8", "4,5,6"],
            "sequence_ratings": ["5,4,5", "3,4,5"],
            "sequence_timestamps": ["10,20,30", "10,20,30"],
            "sequence_ser_label": ["0,0,1", "0,1,0"],
        }
    ).to_csv(changed, index=False)
    meta_changed = build_manifest(changed, tmp_path / "out_changed")
    assert meta_changed["manifest_hash"] != meta_a["manifest_hash"]


def test_build_loo_manifest_and_audit_outputs(tmp_path):
    src = tmp_path / "seq.csv"
    pd.DataFrame(
        {
            "user_id": [1],
            "sequence_item_ids": ["1,2,3"],
            "sequence_ratings": ["5,4,5"],
            "sequence_timestamps": ["10,20,30"],
            "sequence_ser_label": ["0,0,1"],
        }
    ).to_csv(src, index=False)

    out = tmp_path / "manifest"
    meta = build_manifest(src, out)
    eval_df = pd.read_parquet(out / "loo_eval.parquet")
    train_df = pd.read_parquet(out / "loo_train.parquet")

    assert eval_df.loc[0, "target_item"] == 3
    assert json.loads(eval_df.loc[0, "history_items"]) == [1, 2]
    assert eval_df.loc[0, "target_ser_label"] == 1
    assert FORBIDDEN_SER_FIELDS.isdisjoint(train_df.columns)
    assert meta["protocol"] == "LOO_FULL_CATALOG"
    assert "manifest_hash" in meta

    audit = json.loads(
        subprocess.check_output(
            [sys.executable, "tools/audit_loo_manifest.py", str(out)],
            text=True,
        )
    )
    assert audit["protocol"] == "LOO_FULL_CATALOG"
    assert audit["train_forbidden_ser_columns"] == []
    assert audit["num_eval_rows"] == 1


def test_reachability_audit_smoke_outputs_required_fields(tmp_path):
    src = tmp_path / "seq.csv"
    pd.DataFrame(
        {
            "user_id": [1],
            "sequence_item_ids": ["7,8,9"],
            "sequence_ratings": ["4,5,5"],
            "sequence_timestamps": ["10,20,30"],
            "sequence_ser_label": ["0,0,1"],
        }
    ).to_csv(src, index=False)

    out = tmp_path / "manifest"
    build_manifest(src, out)
    report = json.loads(
        subprocess.check_output(
            [sys.executable, "tools/audit_ser_target_reachability.py", str(out / "loo_eval.parquet")],
            text=True,
        )
    )

    required_fields = {
        "num_eval_users",
        "num_ser_targets",
        "ser_target_in_item_map_rate",
        "ser_target_in_sid_lookup_rate",
        "ser_target_in_trie_rate",
        "ser_target_seen_in_history_rate",
        "ser_target_removed_by_history_filter_rate",
        "oracle_prefix_reachable_rate",
    }
    assert required_fields.issubset(report)
    assert report["num_eval_users"] == 1
    assert report["num_ser_targets"] == 1
