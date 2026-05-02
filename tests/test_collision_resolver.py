import torch
from generative_recommenders_pl.serenfree.collision_resolver import sid_buckets, resolve_bucket, collision_stats

def test_collision_resolver_tiebreak_and_stats():
    sid_lookup=torch.tensor([[0,0,0],[1,2,1],[1,2,1],[2,1,1]])
    buckets=sid_buckets(torch.tensor([1,2,3]), sid_lookup)
    assert buckets[(1,2,1)] == [1,2]
    item, scores=resolve_bucket([1,2], 1.0, popularity={1:10,2:1}, eta_pop=1.0)
    assert item == 2
    stats=collision_stats([2,1], outputs_from_collision=1, total_outputs=2)
    assert stats.max_collision_bucket_size == 2
    assert stats.fraction_outputs_from_collision_bucket == 0.5
