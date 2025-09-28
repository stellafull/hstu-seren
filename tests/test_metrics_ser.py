import math

import torch

from generative_recommenders_pl.models.metrics.ranking_calc import (
    compute_ranking_metrics,
    compute_ser_metrics,
)


def test_ranking_metrics_basic():
    ranks = torch.tensor([1, 3, 6])
    metrics = compute_ranking_metrics(ranks, ks=[5, 10])
    assert math.isclose(metrics["hr@5"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(metrics["hr@10"], 1.0, rel_tol=1e-6)
    expected_ndcg5 = (1.0 + 1.0 / math.log2(4)) / 3
    assert math.isclose(metrics["ndcg@5"], expected_ndcg5, rel_tol=1e-5)
    expected_ndcg10 = (
        1.0 + 1.0 / math.log2(4) + 1.0 / math.log2(7)
    ) / 3
    assert math.isclose(metrics["ndcg@10"], expected_ndcg10, rel_tol=1e-5)


def test_ser_metrics_filters_positive_targets():
    ranks = torch.tensor([1, 7, 2, 4])
    ser_mask = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    metrics = compute_ser_metrics(ranks, ser_mask, ks=[5, 10])

    assert math.isclose(metrics["hr_ser@5"], 1.0, rel_tol=1e-6)
    assert math.isclose(metrics["hr_ser@10"], 1.0, rel_tol=1e-6)

    ndcg5 = (
        1.0
        + 1.0 / math.log2(3)
        + 1.0 / math.log2(5)
    ) / 3
    assert math.isclose(metrics["ndcg_ser@5"], ndcg5, rel_tol=1e-5)

    ndcg10 = ndcg5
    assert math.isclose(metrics["ndcg_ser@10"], ndcg10, rel_tol=1e-5)
