#!/usr/bin/env python
"""Mine V2 label-free context-level serendipity candidates from future targets."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def loads(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return []
    return x if isinstance(x, list) else []

def dumps(x): return json.dumps(x, separators=(",", ":"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--future-targets', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--max-pos-per-context', type=int, default=4)
    args=ap.parse_args()
    df=pd.read_parquet(args.future_targets)
    rows=[]
    for r in df.itertuples(index=False):
        A_items=loads(r.A_items); A_sids=loads(r.A_sids); A_offsets=loads(r.A_offsets); A_ratings=loads(r.A_ratings)
        I_items=loads(r.I_items); I_sids=loads(r.I_sids)
        for idx,item in enumerate(A_items[:args.max_pos_per_context]):
            rows.append({
                'dataset':r.dataset,'split_version':getattr(r,'split_version','loo_v1'),'user_id':r.user_id,'position_t':int(r.position_t),'timestamp_t':int(r.timestamp_t),'history_hash':str(r.history_hash),'history_items':getattr(r,'history_items','[]'),'history_ratings':getattr(r,'history_ratings','[]'),'history_timestamps':getattr(r,'history_timestamps','[]'),
                'pos_item_id':int(item),'pos_sid':dumps(A_sids[idx] if idx < len(A_sids) else []),'pos_type':'future_accepted_ring_unchecked','pos_rating':float(A_ratings[idx] if idx < len(A_ratings) else 1.0),'pos_future_offset':int(A_offsets[idx] if idx < len(A_offsets) else -1),'pos_logp_R':None,'pos_logp_A':None,'pos_logp_I':None,'pos_A_minus_I':None,'pos_G_prefix':None,'pos_G_centroid':None,'pos_G_qwen':None,'pos_popularity':None,'pos_weight':1.0,
                'neg_item_ids':dumps(I_items),'neg_sids':dumps(I_sids),'neg_types':dumps(['imminent']*len(I_items)),'neg_logp_R':dumps([]),'neg_logp_A':dumps([]),'neg_logp_I':dumps([]),'neg_A_minus_I':dumps([]),'neg_G_prefix':dumps([]),'neg_G_centroid':dumps([]),'neg_G_qwen':dumps([]),'neg_popularity':dumps([]),'candidate_source':'future_window_targets','teacher_checkpoint':'','reason':'strong_positive_future_accepted_no_ser_labels'
            })
    out=pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output,index=False)
    print({'output':str(args.output),'rows':len(out)})
if __name__=='__main__': main()
