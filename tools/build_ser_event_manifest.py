#!/usr/bin/env python
"""Build SER_EVENT_FULL_CATALOG manifests from sequence-level ser labels.

Each eval row is a prefix-qualified ser-positive event:
``(user, items before t, target item at t)`` where ``target_ser_label == 1``.
The output schema intentionally matches ``LOOManifestEvalDataset`` so the same
full-catalog evaluator can compare LOO and ser-event protocols.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def parse_seq(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        return list(ast.literal_eval(text))
    return [x for x in text.split(",") if x != ""]


def dumps(value: Any) -> str:
    return json.dumps(
        value,
        separators=(",", ":"),
        default=lambda x: x.item() if hasattr(x, "item") else str(x),
    )


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def file_sha(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sid_lookup(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    import torch

    obj = torch.load(p, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.to(torch.long)
    if isinstance(obj, dict):
        if "sid_lookup" in obj:
            return obj["sid_lookup"].to(torch.long)
        if "semantic_ids" in obj:
            return obj["semantic_ids"].to(torch.long) + 1
    raise ValueError(f"Unsupported SID lookup: {path}")


def sid_for(item: int, sid_lookup, item_shift: int = 0) -> list[int]:
    lookup_item = int(item) + int(item_shift)
    if sid_lookup is None or lookup_item < 0 or lookup_item >= sid_lookup.size(0):
        return []
    return [int(x) for x in sid_lookup[lookup_item].tolist()]


def has_sid(item: int, sid_lookup, item_shift: int = 0) -> bool:
    sid = sid_for(item, sid_lookup, item_shift=item_shift)
    return bool(sid) and all(x != 0 for x in sid)


def stable_fraction(value: Any) -> float:
    digest = hashlib.sha256(str(value).encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def choose_single_positive_split(user_id: Any, args: argparse.Namespace) -> str | None:
    if args.single_positive_dest == "skip":
        return None
    if args.single_positive_dest in {"val", "test"}:
        return args.single_positive_dest
    return "val" if stable_fraction(user_id) < args.single_positive_val_rate else "test"


def make_eval_row(
    *,
    args: argparse.Namespace,
    user: Any,
    items: list[int],
    ratings: list[float],
    times: list[int],
    sids: list[list[int]],
    pos: int,
    split: str,
    sid_lookup,
) -> dict[str, Any]:
    history = items[:pos]
    target = int(items[pos])
    return {
        "dataset": args.dataset,
        "protocol": "SER_EVENT_FULL_CATALOG",
        "split_version": args.split_version,
        "query_split": split,
        "query_id": f"{user}:{pos}",
        "user_id": user,
        "raw_user_id": user,
        "position_t": int(pos),
        "prefix_len": int(len(history)),
        "history_items": dumps(history),
        "history_ratings": dumps(ratings[:pos]),
        "history_timestamps": dumps(times[:pos]),
        "target_item": target,
        "target_rating": ratings[pos],
        "target_timestamp": times[pos],
        "target_ser_label": 1,
        "target_sid": dumps(sid_for(target, sid_lookup, args.sid_item_shift)),
        "history_sids": dumps(sids[:pos]),
        "num_history": int(len(history)),
        "target_in_item_universe": True,
        "target_in_sid_lookup": has_sid(target, sid_lookup, args.sid_item_shift),
        "target_in_trie": has_sid(target, sid_lookup, args.sid_item_shift),
    }


EVAL_COLUMNS = [
    "dataset",
    "protocol",
    "split_version",
    "query_split",
    "query_id",
    "user_id",
    "raw_user_id",
    "position_t",
    "prefix_len",
    "history_items",
    "history_ratings",
    "history_timestamps",
    "target_item",
    "target_rating",
    "target_timestamp",
    "target_ser_label",
    "target_sid",
    "history_sids",
    "num_history",
    "target_in_item_universe",
    "target_in_sid_lookup",
    "target_in_trie",
]


TRAIN_COLUMNS = [
    "dataset",
    "protocol",
    "split_version",
    "user_id",
    "train_items",
    "train_ratings",
    "train_timestamps",
    "train_sids",
    "num_train_items",
    "strict_truncated_before_position",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--split-version", default="ser_event_v1")
    ap.add_argument("--sid-lookup", default=None)
    ap.add_argument("--sid-item-shift", type=int, default=0)
    ap.add_argument("--trie", default=None)
    ap.add_argument("--min-prefix", type=int, default=1)
    ap.add_argument("--rating-threshold", type=float, default=4.0)
    ap.add_argument(
        "--single-positive-dest",
        choices=("hash", "val", "test", "skip"),
        default="hash",
    )
    ap.add_argument("--single-positive-val-rate", type=float, default=0.5)
    args = ap.parse_args()

    if args.min_prefix < 1:
        raise ValueError("--min-prefix must be at least 1")
    if not 0.0 <= args.single_positive_val_rate <= 1.0:
        raise ValueError("--single-positive-val-rate must be in [0, 1]")

    out = args.output_dir or Path("tmp/ser_event_manifest") / args.dataset / args.split_version
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input) if args.input.suffix == ".csv" else pd.read_parquet(args.input)
    sid_lookup = load_sid_lookup(args.sid_lookup)

    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    item_universe: set[int] = set()
    num_ser_positive_events = 0
    num_eligible_events = 0
    num_single_positive_users = 0
    num_multi_positive_users = 0
    num_no_eligible_users = 0

    for row in df.itertuples(index=False):
        d = row._asdict()
        items = [int(float(x)) for x in parse_seq(d.get("sequence_item_ids"))]
        ratings = [float(x) for x in parse_seq(d.get("sequence_ratings"))]
        times = [int(float(x)) for x in parse_seq(d.get("sequence_timestamps"))]
        ser = [int(float(x)) for x in parse_seq(d.get("sequence_ser_label"))]
        if len(items) < 2:
            continue
        if not ratings:
            ratings = [1.0] * len(items)
        if not times:
            times = list(range(len(items)))
        n = min(len(items), len(ratings), len(times))
        items = items[:n]
        ratings = ratings[:n]
        times = times[:n]
        ser = ser[:n] if len(ser) >= n else [0] * n
        order = sorted(range(n), key=lambda i: times[i])
        items = [items[i] for i in order]
        ratings = [ratings[i] for i in order]
        times = [times[i] for i in order]
        ser = [ser[i] for i in order]
        sids = [sid_for(int(item), sid_lookup, args.sid_item_shift) for item in items]
        item_universe.update(items)
        user = d.get("user_id")

        ser_positions = [idx for idx, label in enumerate(ser) if int(label) == 1]
        eligible = [
            idx
            for idx in ser_positions
            if idx >= args.min_prefix and ratings[idx] >= args.rating_threshold
        ]
        num_ser_positive_events += len(ser_positions)
        num_eligible_events += len(eligible)

        selected: dict[str, int] = {}
        if len(eligible) >= 2:
            selected["val"] = eligible[-2]
            selected["test"] = eligible[-1]
            num_multi_positive_users += 1
        elif len(eligible) == 1:
            split = choose_single_positive_split(user, args)
            if split is not None:
                selected[split] = eligible[0]
            num_single_positive_users += 1
        else:
            num_no_eligible_users += 1

        for split, pos in selected.items():
            target_rows = val_rows if split == "val" else test_rows
            target_rows.append(
                make_eval_row(
                    args=args,
                    user=user,
                    items=items,
                    ratings=ratings,
                    times=times,
                    sids=sids,
                    pos=pos,
                    split=split,
                    sid_lookup=sid_lookup,
                )
            )

        first_holdout = min(selected.values()) if selected else len(items)
        train_items = items[:first_holdout]
        train_rows.append(
            {
                "dataset": args.dataset,
                "protocol": "SER_EVENT_FULL_CATALOG",
                "split_version": args.split_version,
                "user_id": user,
                "train_items": dumps(train_items),
                "train_ratings": dumps(ratings[:first_holdout]),
                "train_timestamps": dumps(times[:first_holdout]),
                "train_sids": dumps(sids[:first_holdout]),
                "num_train_items": int(len(train_items)),
                "strict_truncated_before_position": int(first_holdout),
            }
        )

    val_df = pd.DataFrame(val_rows, columns=EVAL_COLUMNS)
    test_df = pd.DataFrame(test_rows, columns=EVAL_COLUMNS)
    train_df = pd.DataFrame(train_rows, columns=TRAIN_COLUMNS)
    all_eval_df = pd.concat([val_df, test_df], ignore_index=True)

    val_path = out / "val_eval.parquet"
    test_path = out / "test_eval.parquet"
    all_eval_path = out / "all_eval.parquet"
    train_path = out / "strict_train.parquet"
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    all_eval_df.to_parquet(all_eval_path, index=False)
    train_df.to_parquet(train_path, index=False)

    meta = {
        "protocol": "SER_EVENT_FULL_CATALOG",
        "base_relevance_protocol": "LOO_FULL_CATALOG",
        "split_version": args.split_version,
        "k_values": [10, 20, 50, 100, 200],
        "min_prefix": int(args.min_prefix),
        "rating_threshold": float(args.rating_threshold),
        "single_positive_dest": args.single_positive_dest,
        "single_positive_val_rate": float(args.single_positive_val_rate),
        "denominator": "num_ser_queries",
        "num_input_users": int(len(df)),
        "num_train_rows": int(len(train_df)),
        "num_val_rows": int(len(val_df)),
        "num_test_rows": int(len(test_df)),
        "num_eval_rows": int(len(all_eval_df)),
        "num_eval_users": int(all_eval_df["user_id"].nunique()) if len(all_eval_df) else 0,
        "num_val_users": int(val_df["user_id"].nunique()) if len(val_df) else 0,
        "num_test_users": int(test_df["user_id"].nunique()) if len(test_df) else 0,
        "num_ser_positive_events": int(num_ser_positive_events),
        "num_eligible_ser_positive_events": int(num_eligible_events),
        "num_single_positive_users": int(num_single_positive_users),
        "num_multi_positive_users": int(num_multi_positive_users),
        "num_no_eligible_users": int(num_no_eligible_users),
        "train_policy": "strict_truncate_before_first_heldout_ser_event",
        "item_universe_size": int(len(item_universe)),
        "sid_item_shift": int(args.sid_item_shift),
        "sid_lookup_hash": file_sha(args.sid_lookup),
        "trie_hash": file_sha(args.trie),
        "item_universe_hash": sha_obj(sorted(item_universe)),
        "created_from_git_commit": subprocess.getoutput("git rev-parse HEAD 2>/dev/null"),
    }
    meta["manifest_hash"] = sha_obj(
        {
            "val": val_rows,
            "test": test_rows,
            "strict_train": train_rows,
            "meta": {k: v for k, v in meta.items() if k != "manifest_hash"},
        }
    )
    (out / "manifest_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "val": str(val_path),
                "test": str(test_path),
                "all_eval": str(all_eval_path),
                "strict_train": str(train_path),
                **meta,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
