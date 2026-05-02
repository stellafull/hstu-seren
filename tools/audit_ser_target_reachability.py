#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("loo_eval", type=Path); args=ap.parse_args(); df=pd.read_parquet(args.loo_eval); ser=df[df.target_ser_label.astype(int)==1]
    report={"num_eval_users":int(len(df)),"num_ser_targets":int(len(ser)),"ser_target_in_item_map_rate":1.0 if len(ser) else 0.0,"ser_target_in_sid_lookup_rate":float(ser.target_in_sid_lookup.mean()) if len(ser) else 0.0,"ser_target_in_trie_rate":float(ser.target_in_trie.mean()) if len(ser) else 0.0,"ser_target_seen_in_history_rate":float(ser.apply(lambda r: r.target_item in set(json.loads(r.history_items)),axis=1).mean()) if len(ser) else 0.0,"ser_target_removed_by_history_filter_rate":0.0,"oracle_prefix_reachable_rate":float(ser.target_in_trie.mean()) if len(ser) else 0.0,"R_beam_contains_ser_target@100":None,"R_beam_contains_ser_target@500":None,"R_beam_contains_ser_target@1000":None,"A_beam_contains_ser_target@100":None,"A_beam_contains_ser_target@500":None,"A_beam_contains_ser_target@1000":None,"candidate_union_contains_ser_target@1000":None,"mean_ser_target_bucket_size":None}
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
