#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_ROWS = 25000
COLUMNS = [
    'dataset', 'split_version', 'user_id', 'position_t',
    'history_items', 'history_ratings', 'history_timestamps', 'history_sids',
    'history_hash', 'timestamp_t',
    'R_item', 'R_sid', 'R_rating', 'R_weight',
    'I_items', 'I_sids', 'I_offsets', 'I_ratings', 'num_I_targets', 'empty_I',
    'A_items', 'A_sids', 'A_offsets', 'A_ratings', 'num_A_targets', 'empty_A',
    'seen_excluded_count',
]

def loads(x: Any):
    if isinstance(x, str): return json.loads(x)
    if hasattr(x, 'tolist'): return x.tolist()
    return x if isinstance(x, list) else []
def dumps(x: Any) -> str: return json.dumps(x, separators=(',', ':'), default=lambda v: v.item() if hasattr(v, 'item') else str(v))
def stable_hash(x: Any) -> str: return hashlib.sha256(dumps(x).encode()).hexdigest()
def weight(r): return 1.0 if r >= 4 else (0.5 if r == 3 else 0.2)
def dedup_by_item(items, key):
    out=[]; seen=set()
    for row in sorted(items, key=key):
        item=int(row[0])
        if item in seen: continue
        seen.add(item); out.append(row)
    return out
def cap_a(items,maxn,blocked_items=frozenset()):
    ranked=dedup_by_item(items, key=lambda x:(-x[2], x[1]))
    return [x for x in ranked if int(x[0]) not in blocked_items][:maxn]
def cap_i(items,maxn): return dedup_by_item(items, key=lambda x:x[1])[:maxn]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--loo-train', required=True, type=Path); ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--imminent-window','--imminent_window','--near-session-window','--near_session_window', dest='imminent_window', type=int, default=3)
    ap.add_argument('--acceptable-min-gap','--acceptable_min_gap','--future-session-min-gap','--future_session_min_gap', dest='acceptable_min_gap', type=int, default=4)
    ap.add_argument('--acceptable-window','--acceptable_window','--future-session-window','--future_session_window', dest='acceptable_window', type=int, default=50)
    ap.add_argument('--rating-threshold','--rating-positive-threshold','--rating_positive_threshold', dest='rating_threshold', type=float, default=4.0)
    ap.add_argument('--max-i-targets','--max_i_targets', dest='max_i_targets', type=int, default=3); ap.add_argument('--max-a-targets','--max_a_targets', dest='max_a_targets', type=int, default=32)
    seen=ap.add_mutually_exclusive_group(); seen.add_argument('--include-seen','--include_seen', dest='include_seen', action='store_true', default=False); seen.add_argument('--exclude-seen','--exclude_seen', dest='include_seen', action='store_false')
    args=ap.parse_args(); df=pd.read_parquet(args.loo_train); rows=[]; writer=None; schema=None; written=0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def flush_rows():
        nonlocal rows, writer, schema, written
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=COLUMNS)
        if schema is None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            schema = table.schema
            writer = pq.ParquetWriter(args.output, schema)
        else:
            table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        writer.write_table(table)
        written += len(rows)
        print({'written_rows': written}, flush=True)
        rows = []

    for r in df.itertuples(index=False):
        items=list(map(int, loads(r.train_items))); ratings=list(map(float, loads(r.train_ratings))); times=list(map(int, loads(r.train_timestamps))); sids=loads(r.train_sids) if hasattr(r,'train_sids') else [[] for _ in items]
        seen=set()
        for t in range(len(items)-1):
            seen.add(items[t]); R=items[t+1]; I=[]; A=[]; seen_excluded=0
            for d in range(1,args.imminent_window+1):
                pos=t+d
                if pos>=len(items): break
                if not args.include_seen and items[pos] in seen:
                    seen_excluded += 1; continue
                I.append((items[pos],d,ratings[pos],sids[pos]))
            a_start=max(int(args.acceptable_min_gap), int(args.imminent_window)+1)
            for d in range(a_start,args.acceptable_window+1):
                pos=t+d
                if pos>=len(items): break
                if ratings[pos] < args.rating_threshold: continue
                if not args.include_seen and items[pos] in seen:
                    seen_excluded += 1; continue
                A.append((items[pos],d,ratings[pos],sids[pos]))
            i_blocked={int(x[0]) for x in I}; I=cap_i(I,args.max_i_targets); A=cap_a(A,args.max_a_targets, i_blocked)
            rows.append({'dataset':r.dataset,'split_version':getattr(r,'split_version','loo_v1'),'user_id':r.user_id,'position_t':t,'history_items':dumps(items[:t+1]),'history_ratings':dumps(ratings[:t+1]),'history_timestamps':dumps(times[:t+1]),'history_sids':dumps(sids[:t+1]),'history_hash':stable_hash(items[:t+1]),'timestamp_t':times[t],'R_item':R,'R_sid':dumps(sids[t+1]),'R_rating':ratings[t+1],'R_weight':weight(ratings[t+1]),'I_items':dumps([x[0] for x in I]),'I_sids':dumps([x[3] for x in I]),'I_offsets':dumps([x[1] for x in I]),'I_ratings':dumps([x[2] for x in I]),'num_I_targets':len(I),'empty_I':len(I)==0,'A_items':dumps([x[0] for x in A]),'A_sids':dumps([x[3] for x in A]),'A_offsets':dumps([x[1] for x in A]),'A_ratings':dumps([x[2] for x in A]),'num_A_targets':len(A),'empty_A':len(A)==0,'seen_excluded_count':seen_excluded})
            if len(rows) >= CHUNK_ROWS:
                flush_rows()
    flush_rows()
    if writer is not None:
        writer.close()
    else:
        pd.DataFrame([], columns=COLUMNS).to_parquet(args.output,index=False)
    print({'output':str(args.output),'rows':written})
if __name__=='__main__': main()
