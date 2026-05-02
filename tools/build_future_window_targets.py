#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import pandas as pd

def loads(x: Any):
    if isinstance(x, str): return json.loads(x)
    if hasattr(x, 'tolist'): return x.tolist()
    return x if isinstance(x, list) else []
def dumps(x: Any) -> str: return json.dumps(x, separators=(',', ':'), default=lambda v: v.item() if hasattr(v, 'item') else str(v))
def stable_hash(x: Any) -> str: return hashlib.sha256(dumps(x).encode()).hexdigest()
def weight(r): return 1.0 if r >= 4 else (0.5 if r == 3 else 0.2)
def cap_a(items,maxn): return sorted(items, key=lambda x:(-x[2], x[1]))[:maxn]
def cap_i(items,maxn): return sorted(items, key=lambda x:x[1])[:maxn]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--loo-train', required=True, type=Path); ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--imminent-window', type=int, default=3); ap.add_argument('--acceptable-min-gap', type=int, default=2); ap.add_argument('--acceptable-window', type=int, default=50)
    ap.add_argument('--rating-threshold', type=float, default=4.0); ap.add_argument('--max-i-targets', type=int, default=3); ap.add_argument('--max-a-targets', type=int, default=32); ap.add_argument('--include-seen', action='store_true')
    args=ap.parse_args(); df=pd.read_parquet(args.loo_train); rows=[]
    for r in df.itertuples(index=False):
        items=list(map(int, loads(r.train_items))); ratings=list(map(float, loads(r.train_ratings))); times=list(map(int, loads(r.train_timestamps))); sids=loads(r.train_sids) if hasattr(r,'train_sids') else [[] for _ in items]
        for t in range(len(items)-1):
            seen=set(items[:t+1]); R=items[t+1]; I=[]; A=[]; seen_excluded=0
            for d in range(1,args.imminent_window+1):
                pos=t+d
                if pos>=len(items): break
                if not args.include_seen and items[pos] in seen:
                    seen_excluded += 1; continue
                I.append((items[pos],d,ratings[pos],sids[pos]))
            for d in range(args.acceptable_min_gap,args.acceptable_window+1):
                pos=t+d
                if pos>=len(items): break
                if ratings[pos] < args.rating_threshold: continue
                if not args.include_seen and items[pos] in seen:
                    seen_excluded += 1; continue
                A.append((items[pos],d,ratings[pos],sids[pos]))
            I=cap_i(I,args.max_i_targets); A=cap_a(A,args.max_a_targets)
            rows.append({'dataset':r.dataset,'split_version':getattr(r,'split_version','loo_v1'),'user_id':r.user_id,'position_t':t,'history_items':dumps(items[:t+1]),'history_ratings':dumps(ratings[:t+1]),'history_timestamps':dumps(times[:t+1]),'history_hash':stable_hash(items[:t+1]),'timestamp_t':times[t],'R_item':R,'R_sid':dumps(sids[t+1]),'R_rating':ratings[t+1],'R_weight':weight(ratings[t+1]),'I_items':dumps([x[0] for x in I]),'I_sids':dumps([x[3] for x in I]),'I_offsets':dumps([x[1] for x in I]),'I_ratings':dumps([x[2] for x in I]),'num_I_targets':len(I),'empty_I':len(I)==0,'A_items':dumps([x[0] for x in A]),'A_sids':dumps([x[3] for x in A]),'A_offsets':dumps([x[1] for x in A]),'A_ratings':dumps([x[2] for x in A]),'num_A_targets':len(A),'empty_A':len(A)==0,'seen_excluded_count':seen_excluded})
    args.output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_parquet(args.output,index=False); print({'output':str(args.output),'rows':len(rows)})
if __name__=='__main__': main()
