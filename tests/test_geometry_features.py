import torch
import pytest

from generative_recommenders_pl.models.serenfree.v2_levels import semantic_non_dedup_levels
from generative_recommenders_pl.serenfree.geometry import prefix_surprise, relevance_safe_geometry_boost


def test_adaptive_levels_exclude_dedup_and_reject_dedup():
    assert semantic_non_dedup_levels(4) == (0, 1, 2)
    assert semantic_non_dedup_levels(3, [1, 2]) == (0, 1)
    with pytest.raises(ValueError):
        semantic_non_dedup_levels(3, [3])


def test_prefix_surprise_and_relevance_guard():
    hist = torch.tensor([[1, 2, 9], [1, 2, 8], [3, 4, 7]])
    cand = torch.tensor([[1, 2, 1], [5, 6, 1]])
    g = prefix_surprise(hist, cand, levels=[1, 2])
    assert g[0] < g[1]
    rel = torch.tensor([10.0, 1.0, 9.0])
    geo = torch.tensor([0.7, 0.9, 0.8])
    ranks = torch.tensor([1, 1000, 2])
    scored, mask, _ = relevance_safe_geometry_boost(rel, geo, ranks, beta=1.0, relevance_floor_rank=10, low_q=0.0, high_q=1.0)
    assert not mask[1]
    assert scored[1] == rel[1]


def test_prefix_surprise_respects_max_history_and_epsilon():
    hist = torch.tensor([[1, 1, 1], [2, 2, 1]])
    cand = torch.tensor([[1, 1, 1], [2, 2, 1]])

    all_history = prefix_surprise(hist, cand, levels=[1, 2], epsilon=1e-4)
    recent_only = prefix_surprise(hist, cand, levels=[1, 2], epsilon=1e-4, max_history=1)

    assert all_history[0] < recent_only[0]
    assert recent_only[0].item() == pytest.approx(-torch.log(torch.tensor(1e-4)).item())
    assert recent_only[1] < all_history[1]
