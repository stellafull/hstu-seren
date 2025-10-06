"""Serendipity-focused retrieval metrics."""

from __future__ import annotations

import torch
import torchmetrics
import torchmetrics.utilities.data


class SerendipityRetrievalMetrics(torchmetrics.Metric):
    """Compute serendipity-aware retrieval metrics over top-k recommendations.

    The metric expects pre-ranked top-k recommendations together with per-item
    serendipity scores and the ground-truth target ids. Only examples whose
    target serendipity label equals one contribute to the aggregation. Hit rate
    measures how often the serendipitous target appears within the cutoffs,
    while NDCG rewards ranking the serendipitous target closer to the first
    position.
    """

    def __init__(self, k: int, at_k_list: list[int], **kwargs) -> None:
        super().__init__(**kwargs)
        if k <= 0:
            msg = "k must be positive"
            raise ValueError(msg)
        if not at_k_list:
            msg = "at_k_list must contain at least one cutoff"
            raise ValueError(msg)

        unique_cutoffs = sorted(set(at_k_list))
        if unique_cutoffs[-1] > k:
            msg = "All cutoffs in at_k_list must be <= k"
            raise ValueError(msg)

        self.k = k
        self.at_k_list = unique_cutoffs

        self.add_state("top_k_ids", default=[], dist_reduce_fx="cat")
        self.add_state("ser_scores", default=[], dist_reduce_fx="cat")
        self.add_state("target_ids", default=[], dist_reduce_fx="cat")
        self.add_state("target_ser_labels", default=[], dist_reduce_fx="cat")

    def update(
        self,
        top_k_ids: torch.Tensor,
        serendipity_scores: torch.Tensor,
        target_ids: torch.Tensor,
        target_ser_labels: torch.Tensor,
        **kwargs,
    ) -> None:
        if top_k_ids.ndim != 2:
            msg = "top_k_ids must be a 2D tensor [batch, k]"
            raise ValueError(msg)
        if top_k_ids.size(1) != self.k:
            msg = "top_k_ids second dimension must equal k"
            raise ValueError(msg)
        if serendipity_scores.size() != top_k_ids.size():
            msg = "serendipity_scores must match top_k_ids shape"
            raise ValueError(msg)
        if target_ids.ndim == 1:
            target_ids = target_ids.unsqueeze(-1)
        if target_ids.ndim != 2 or target_ids.size(1) != 1:
            msg = "target_ids must be shape [batch, 1]"
            raise ValueError(msg)
        if target_ser_labels.ndim == 1:
            target_ser_labels = target_ser_labels.unsqueeze(-1)
        if target_ser_labels.ndim != 2 or target_ser_labels.size(1) != 1:
            msg = "target_ser_labels must be shape [batch, 1]"
            raise ValueError(msg)

        self.top_k_ids.append(top_k_ids)
        self.ser_scores.append(serendipity_scores.to(dtype=torch.float32))
        self.target_ids.append(target_ids)
        self.target_ser_labels.append(target_ser_labels.to(dtype=torch.int64))

    def compute(self) -> dict[str, torch.Tensor]:
        if not self.top_k_ids:
            return {
                **{f"ndcg_ser@{at_k}": torch.tensor(0.0) for at_k in self.at_k_list},
                **{f"hr_ser@{at_k}": torch.tensor(0.0) for at_k in self.at_k_list},
            }

        top_k_ids = torchmetrics.utilities.data.dim_zero_cat(self.top_k_ids)
        ser_scores = torchmetrics.utilities.data.dim_zero_cat(self.ser_scores)
        target_ids = torchmetrics.utilities.data.dim_zero_cat(self.target_ids)
        target_ser_labels = torchmetrics.utilities.data.dim_zero_cat(
            self.target_ser_labels
        ).view(-1)

        mask = target_ser_labels == 1
        if not torch.any(mask):
            device = top_k_ids.device
            return {
                **{
                    f"ndcg_ser@{at_k}": torch.tensor(0.0, device=device)
                    for at_k in self.at_k_list
                },
                **{
                    f"hr_ser@{at_k}": torch.tensor(0.0, device=device)
                    for at_k in self.at_k_list
                },
            }

        filtered_top_k_ids = top_k_ids[mask]
        filtered_target_ids = target_ids[mask]

        device = filtered_top_k_ids.device
        dtype = ser_scores.dtype if ser_scores.numel() > 0 else torch.float32
        discounts = 1.0 / torch.log2(
            torch.arange(2, self.k + 2, device=device, dtype=dtype)
        )

        results: dict[str, torch.Tensor] = {}
        for at_k in self.at_k_list:
            cutoff = min(at_k, self.k)
            if cutoff == 0:
                zero = torch.zeros(1, device=device, dtype=dtype).squeeze()
                results[f"ndcg_ser@{at_k}"] = zero
                results[f"hr_ser@{at_k}"] = zero
                continue

            cutoff_discounts = discounts[:cutoff]
            ids_at_k = filtered_top_k_ids[:, :cutoff]
            target_ids_at_k = filtered_target_ids.expand_as(ids_at_k)
            target_matches = (ids_at_k == target_ids_at_k).to(dtype=dtype)

            dcg = (target_matches * cutoff_discounts.unsqueeze(0)).sum(dim=1)
            ndcg = dcg
            if ndcg.numel() > 0:
                results[f"ndcg_ser@{at_k}"] = ndcg.mean()
            else:
                results[f"ndcg_ser@{at_k}"] = torch.zeros(
                    1, device=device, dtype=dtype
                ).squeeze()

            hits = target_matches.any(dim=1).to(dtype=dtype)
            if hits.numel() > 0:
                results[f"hr_ser@{at_k}"] = hits.mean()
            else:
                results[f"hr_ser@{at_k}"] = torch.zeros(
                    1, device=device, dtype=dtype
                ).squeeze()

        return results
