#!/usr/bin/env python
"""Build frozen LOO_FULL_CATALOG manifests.

List-valued columns are stored as compact JSON strings rather than arbitrary
Python objects so parquet round-trips consistently across pandas/pyarrow.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, subprocess
from pathlib import Path
from typing import Any
import pandas as pd


def parse_seq(value: Any) -> list[Any]:
    if isinstance(value, list): return value
    if hasattr(value, 'tolist'): return value.tolist()
    if pd.isna(value): return []
    text=str(value).strip()
    if not text: return []
    if text.startswith('['): return list(ast.literal_eval(text))
    return [x for x in text.split(',') if x!='']


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(',', ':'), default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def loads(value: Any) -> Any:
    if isinstance(value, str): return json.loads(value)
    if hasattr(value, 'tolist'): return value.tolist()
    return value


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_sha(path: str | Path | None) -> str:
    if not path: return ''
    p=Path(path)
    if not p.exists(): return ''
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()


def load_sid_lookup(path: str | None):
    if not path: return None
    p=Path(path)
    if not p.exists(): return None
    import torch
    obj=torch.load(p, map_location='cpu')
    if isinstance(obj, torch.Tensor): return obj.to(torch.long)
    if isinstance(obj, dict):
        if 'sid_lookup' in obj: return obj['sid_lookup'].to(torch.long)
        if 'semantic_ids' in obj: return obj['semantic_ids'].to(torch.long)+1
    raise ValueError(f'Unsupported SID lookup: {path}')


def sid_for(item: int, sid_lookup, item_shift: int = 0) -> list[int]:
    lookup_item = int(item) + int(item_shift)
    if sid_lookup is None or lookup_item < 0 or lookup_item >= sid_lookup.size(0): return []
    return [int(x) for x in sid_lookup[lookup_item].tolist()]


def has_sid(item: int, sid_lookup, item_shift: int = 0) -> bool:
    sid=sid_for(item, sid_lookup, item_shift=item_shift)
    return bool(sid) and all(x != 0 for x in sid)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--output-dir', type=Path, default=None); ap.add_argument('--split-version', default='loo_v1')
    ap.add_argument('--sid-lookup', default=None); ap.add_argument('--sid-item-shift', type=int, default=0); ap.add_argument('--trie', default=None)
    args=ap.parse_args(); out=args.output_dir or Path('tmp/loo_manifest')/args.dataset; out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.input) if args.input.suffix=='.csv' else pd.read_parquet(args.input)
    sid_lookup=load_sid_lookup(args.sid_lookup)
    eval_rows=[]; train_rows=[]; item_universe=set(); ser_count=0
    for row in df.itertuples(index=False):
        d=row._asdict()
        items=[int(float(x)) for x in parse_seq(d.get('sequence_item_ids'))]
        ratings=[float(x) for x in parse_seq(d.get('sequence_ratings'))]
        times=[int(float(x)) for x in parse_seq(d.get('sequence_timestamps'))]
        ser=[int(float(x)) for x in parse_seq(d.get('sequence_ser_label'))] if 'sequence_ser_label' in d else []
        if len(items) < 2: continue
        if not ratings: ratings=[1.0]*len(items)
        if not times: times=list(range(len(items)))
        n=min(len(items),len(ratings),len(times)); items=items[:n]; ratings=ratings[:n]; times=times[:n]
        order=sorted(range(n), key=lambda i: times[i]); items=[items[i] for i in order]; ratings=[ratings[i] for i in order]; times=[times[i] for i in order]
        ser=[ser[i] for i in order] if ser and len(ser)>=n else [0]*n
        user=d.get('user_id'); target=int(items[-1]); history=[int(x) for x in items[:-1]]; item_universe.update(items)
        target_ser=int(ser[-1]) if ser else 0; ser_count += int(target_ser==1)
        target_sid=sid_for(target, sid_lookup, args.sid_item_shift); history_sids=[sid_for(int(x), sid_lookup, args.sid_item_shift) for x in history]
        eval_rows.append({'dataset':args.dataset,'split_version':args.split_version,'user_id':user,'raw_user_id':user,'history_items':dumps(history),'history_ratings':dumps(ratings[:-1]),'history_timestamps':dumps(times[:-1]),'target_item':target,'target_rating':ratings[-1],'target_timestamp':times[-1],'target_ser_label':target_ser,'target_sid':dumps(target_sid),'history_sids':dumps(history_sids),'num_history':len(history),'target_in_item_universe':True,'target_in_sid_lookup':has_sid(target,sid_lookup,args.sid_item_shift),'target_in_trie':has_sid(target,sid_lookup,args.sid_item_shift)})
        train_rows.append({'dataset':args.dataset,'split_version':args.split_version,'user_id':user,'train_items':dumps(history),'train_ratings':dumps(ratings[:-1]),'train_timestamps':dumps(times[:-1]),'train_sids':dumps(history_sids),'num_train_items':len(history)})
    eval_df=pd.DataFrame(eval_rows); train_df=pd.DataFrame(train_rows)
    eval_path=out/'loo_eval.parquet'; train_path=out/'loo_train.parquet'; eval_df.to_parquet(eval_path,index=False); train_df.to_parquet(train_path,index=False)
    meta={'protocol':'LOO_FULL_CATALOG','k_values':[10,20,50,100,200],'num_eval_rows':int(len(eval_df)),'num_ser_target_rows':int(ser_count),'item_universe_size':int(len(item_universe)),'sid_item_shift':int(args.sid_item_shift),'sid_lookup_coverage':float(eval_df['target_in_sid_lookup'].mean()) if len(eval_df) else 0.0,'trie_coverage':float(eval_df['target_in_trie'].mean()) if len(eval_df) else 0.0,'item_universe_hash':sha_obj(sorted(item_universe)),'sid_lookup_hash':file_sha(args.sid_lookup),'trie_hash':file_sha(args.trie),'created_from_git_commit':subprocess.getoutput('git rev-parse HEAD 2>/dev/null')}
    meta['manifest_hash']=sha_obj({'eval':eval_rows,'train':train_rows,'meta':{k:v for k,v in meta.items() if k!='manifest_hash'}})
    (out/'manifest_meta.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'eval':str(eval_path),'train':str(train_path),**meta}, sort_keys=True))
if __name__=='__main__': main()
