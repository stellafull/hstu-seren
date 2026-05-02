import json, subprocess, sys
import pandas as pd

def test_lf_rank_miner_context_schema(tmp_path):
    future=tmp_path/'future.parquet'
    pd.DataFrame({'dataset':['tiny'],'split_version':['loo_v1'],'user_id':[1],'position_t':[0],'timestamp_t':[10],'history_hash':['h'],'A_items':['[3]'],'A_sids':['[[1,2,3]]'],'A_offsets':['[2]'],'A_ratings':['[5]'],'I_items':['[2]'],'I_sids':['[[1,1,1]]']}).to_parquet(future,index=False)
    out=tmp_path/'ctx.parquet'
    subprocess.check_call([sys.executable,'tools/mine_label_free_ser_candidates.py','--future-targets',str(future),'--output',str(out)])
    df=pd.read_parquet(out)
    assert df.loc[0,'pos_item_id']==3
    assert df.loc[0,'candidate_source']=='future_window_targets'
    forbidden={'target_ser_label','sequence_ser_label','s_ser_find','s_ser_imp','s_ser_rec','m_ser_find','m_ser_imp','m_ser_rec'}
    assert forbidden.isdisjoint(df.columns)
    assert json.loads(df.loc[0,'neg_types']) == ['imminent']
