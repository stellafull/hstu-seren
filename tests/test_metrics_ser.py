import math

import pytest
import torch

from generative_recommenders_pl.models.metrics.ser_metrics import (
    SerMetrics,
    compute_ser_metrics,
)


@pytest.mark.parametrize("at_k_list", ([1, 3], [2, 4]))
def test_ser_metrics_hr_and_ndcg(at_k_list):
    scores = torch.tensor(
        [
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )

    metric = SerMetrics(at_k_list=at_k_list)
    metric.update(scores)
    results = metric.compute()

    for k in at_k_list:
        hr_expected = sum(scores[:, :k].max(dim=1).values.tolist()) / scores.size(0)
        discounts = 1.0 / torch.log2(
            torch.arange(2, k + 2, dtype=scores.dtype, device=scores.device)
        )
        sliced = scores[:, :k]
        dcg = (sliced * discounts).sum(dim=1)
        ideal_sorted, _ = torch.sort(sliced, dim=1, descending=True)
        ideal_dcg = (ideal_sorted * discounts).sum(dim=1)
        ndcg_expected = torch.where(ideal_dcg > 0, dcg / ideal_dcg, torch.zeros_like(dcg)).mean()

        assert math.isclose(
            results[f"hr_ser@{k}"].item(),
            hr_expected,
            rel_tol=1e-6,
        )
        assert math.isclose(
            results[f"ndcg_ser@{k}"].item(),
            ndcg_expected.item(),
            rel_tol=1e-6,
        )


def test_compute_ser_metrics_wrapper_matches_metric():
    scores = torch.tensor(
        [
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    at_k_list = [1, 3]

    metric = SerMetrics(at_k_list=at_k_list)
    metric.update(scores)
    expected = metric.compute()
    computed = compute_ser_metrics(scores, at_k_list)

    for key, value in computed.items():
        assert math.isclose(value, expected[key].item(), rel_tol=1e-6)
