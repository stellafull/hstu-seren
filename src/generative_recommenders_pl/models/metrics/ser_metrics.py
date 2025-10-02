"""Serendipity-aware retrieval metrics."""

from __future__ import annotations

from typing import Iterable

import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat


class SerMetrics(torchmetrics.Metric):
    """Compute serendipity-oriented retrieval metrics over top-k results.

    The metric expects per-example serendipity indicators for the ranked list of
    recommendations. Each indicator should be ``0`` or ``1`` and correspond to a
    position in the ranked results (higher score means more serendipitous).

    For every ``k`` in ``at_k_list`` the following metrics are reported:

    ``hr_ser@k``
        Indicator that at least one serendipitous item (``serend_score == 1``)
        appears in the top-``k`` recommendations.

    ``ndcg_ser@k``
        Discounted cumulative gain of the serendipity indicators within the
        top-``k`` positions using the standard ``1 / log2(i + 1)`` discount.
    """

    is_differentiable = False

    def __init__(
        self,
        at_k_list: Iterable[int],
        *,
        condition_on_positive: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ks = list({int(k) for k in at_k_list})
        if not ks:
            msg = "at_k_list must contain at least one positive integer"
            raise ValueError(msg)
        if any(k <= 0 for k in ks):
            msg = "All values in at_k_list must be positive integers"
            raise ValueError(msg)

        self.at_k_list = sorted(ks)
        self._max_k = self.at_k_list[-1]
        self.condition_on_positive = condition_on_positive

        self.add_state("ser_scores", default=[], dist_reduce_fx="cat")

    def update(self, serendipity_scores: torch.Tensor) -> None:  # type: ignore[override]
        """Accumulate serendipity indicators for a batch of ranked lists.

        Args:
            serendipity_scores: A tensor of shape ``(batch_size, top_k)`` with
                binary serendipity indicators for each ranked position.
        """

        if serendipity_scores.ndim != 2:
            msg = "serendipity_scores must be a 2D tensor of shape (batch, top_k)"
            raise ValueError(msg)
        if serendipity_scores.size(1) < self._max_k:
            msg = (
                "Received fewer positions than required by at_k_list: "
                f"got {serendipity_scores.size(1)}, expected at least {self._max_k}"
            )
            raise ValueError(msg)

        scores = serendipity_scores.to(torch.float32)
        self.ser_scores.append(scores)

    def compute(self) -> dict[str, torch.Tensor]:  # type: ignore[override]
        if not self.ser_scores:
            return {
                f"hr_ser@{k}": torch.tensor(0.0, device=self.device)
                for k in self.at_k_list
            } | {
                f"ndcg_ser@{k}": torch.tensor(0.0, device=self.device)
                for k in self.at_k_list
            }

        scores = dim_zero_cat(self.ser_scores)
        if self.condition_on_positive:
            mask = scores.sum(dim=1) > 0
            if mask.any():
                scores = scores[mask]
            else:
                return {
                    f"hr_ser@{k}": torch.tensor(0.0, device=self.device)
                    for k in self.at_k_list
                } | {
                    f"ndcg_ser@{k}": torch.tensor(0.0, device=self.device)
                    for k in self.at_k_list
                }
        discounts = 1.0 / torch.log2(
            torch.arange(2, self._max_k + 2, device=scores.device, dtype=scores.dtype)
        )

        metrics: dict[str, torch.Tensor] = {}

        for k in self.at_k_list:
            sliced = scores[:, :k]
            metrics[f"hr_ser@{k}"] = (
                (sliced > 0).any(dim=1).to(scores.dtype).mean()
            )
            dcg = (sliced * discounts[:k]).sum(dim=1)
            ideal_sorted, _ = torch.sort(sliced, dim=1, descending=True)
            ideal_dcg = (ideal_sorted * discounts[:k]).sum(dim=1)
            normalized = torch.where(
                ideal_dcg > 0,
                dcg / ideal_dcg,
                torch.zeros_like(dcg),
            )
            metrics[f"ndcg_ser@{k}"] = normalized.mean()

        return metrics

    def reset(self) -> None:  # type: ignore[override]
        super().reset()


def compute_ser_metrics(
    serendipity_scores: torch.Tensor,
    at_k_list: Iterable[int],
    *,
    condition_on_positive: bool = False,
) -> dict[str, float]:
    """Convenience wrapper to compute serendipity metrics on a tensor input."""

    metric = SerMetrics(
        at_k_list=at_k_list,
        condition_on_positive=condition_on_positive,
    )
    metric.update(serendipity_scores)
    output = metric.compute()
    return {name: value.item() for name, value in output.items()}


__all__ = ["SerMetrics", "compute_ser_metrics"]
