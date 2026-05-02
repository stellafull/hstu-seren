#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("manifest_dir", type=Path); args=ap.parse_args()
    eval_df=pd.read_parquet(args.manifest_dir/"loo_eval.parquet"); train_df=pd.read_parquet(args.manifest_dir/"loo_train.parquet")
    blocked={"sequence_ser_label","target_ser_label","s_ser_find","s_ser_imp","s_ser_rec","m_ser_find","m_ser_imp","m_ser_rec"}
    forbidden=[c for c in train_df.columns if c in blocked]
    report={
      "protocol":"LOO_FULL_CATALOG", "num_eval_rows":int(len(eval_df)), "num_train_rows":int(len(train_df)),
      "train_forbidden_ser_columns": forbidden,
      "target_in_sid_lookup_rate": float(eval_df["target_in_sid_lookup"].mean()) if len(eval_df) else 0.0,
      "target_in_trie_rate": float(eval_df["target_in_trie"].mean()) if len(eval_df) else 0.0,
      "history_filter_drop_target_rate": float(eval_df.apply(lambda r: r["target_item"] in set(json.loads(r["history_items"])), axis=1).mean()) if len(eval_df) else 0.0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if forbidden: raise SystemExit(2)
if __name__=="__main__": main()
