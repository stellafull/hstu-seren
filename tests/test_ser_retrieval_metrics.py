import pytest
import torch

from generative_recommenders_pl.models.metrics.ser_retrival import (
    SerendipityRetrievalMetrics,
)


def test_serendipity_metrics_compute():
    top_k_ids = torch.tensor(
        [
            [11, 12, 13],
            [21, 22, 23],
            [31, 32, 33],
        ]
    )
    ser_scores = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    target_ids = torch.tensor([[11], [22], [99]])
    target_ser_labels = torch.tensor([[1], [1], [1]])

    metric = SerendipityRetrievalMetrics(k=3, at_k_list=[1, 2, 3])
    metric.update(top_k_ids, ser_scores, target_ids, target_ser_labels)
    result = metric.compute()

    assert result["ndcg_ser@1"].item() == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert result["ndcg_ser@2"].item() == pytest.approx(0.5436433, abs=1e-5)
    assert result["ndcg_ser@3"].item() == pytest.approx(0.5436433, abs=1e-5)
    assert result["hr_ser@1"].item() == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert result["hr_ser@2"].item() == pytest.approx(2.0 / 3.0, abs=1e-5)
    assert result["hr_ser@3"].item() == pytest.approx(2.0 / 3.0, abs=1e-5)


def test_serendipity_metrics_ignore_non_ser_labels():
    top_k_ids = torch.tensor([[1, 2, 3]])
    ser_scores = torch.tensor([[1, 0, 0]])
    target_ids = torch.tensor([[1]])
    target_ser_labels = torch.tensor([[0]])

    metric = SerendipityRetrievalMetrics(k=3, at_k_list=[1, 2])
    metric.update(top_k_ids, ser_scores, target_ids, target_ser_labels)
    result = metric.compute()

    assert result["ndcg_ser@1"].item() == pytest.approx(0.0, abs=1e-5)
    assert result["ndcg_ser@2"].item() == pytest.approx(0.0, abs=1e-5)
    assert result["hr_ser@1"].item() == pytest.approx(0.0, abs=1e-5)
    assert result["hr_ser@2"].item() == pytest.approx(0.0, abs=1e-5)
