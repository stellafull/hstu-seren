"""Ranking metrics utilities for top-K evaluation."""

from __future__ import annotations

from typing import Iterable

import torch


def _ensure_tensor(values: torch.Tensor | Iterable[int]) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(list(values), dtype=torch.long)


def compute_ranking_metrics(
    ranks: torch.Tensor | Iterable[int],
    ks: Iterable[int],
) -> dict[str, float]:
    """Compute HR@K and NDCG@K from 1-based ranks."""

    ranks_t = _ensure_tensor(ranks).to(torch.float32)
    if ranks_t.numel() == 0:
        return {f"hr@{k}": 0.0 for k in ks} | {f"ndcg@{k}": 0.0 for k in ks}

    logs = torch.log2(ranks_t + 1.0)
    metrics: dict[str, float] = {}
    for k in ks:
        mask = ranks_t <= k
        hits = mask.to(torch.float32)
        hr = hits.mean() if hits.numel() else torch.tensor(0.0)
        ndcg = torch.where(mask, 1.0 / logs, torch.zeros_like(logs)).mean()
        metrics[f"hr@{k}"] = hr.item()
        metrics[f"ndcg@{k}"] = ndcg.item()
    return metrics


def compute_ser_metrics(
    ranks: torch.Tensor | Iterable[int],
    ser_mask: torch.Tensor | Iterable[bool],
    ks: Iterable[int],
) -> dict[str, float]:
    """Compute HR_ser@K and NDCG_ser@K for ser-positive targets."""

    ranks_t = _ensure_tensor(ranks)
    mask_t = _ensure_tensor(ser_mask).to(torch.bool)
    filtered = ranks_t[mask_t]
    if filtered.numel() == 0:
        return {
            f"hr_ser@{k}": float("nan") for k in ks
        } | {
            f"ndcg_ser@{k}": float("nan") for k in ks
        }

    filtered = filtered.to(torch.float32)
    logs = torch.log2(filtered + 1.0)
    metrics: dict[str, float] = {}
    for k in ks:
        mask = filtered <= k
        hits = mask.to(torch.float32)
        hr = hits.mean() if hits.numel() else torch.tensor(0.0)
        ndcg = torch.where(mask, 1.0 / logs, torch.zeros_like(logs)).mean()
        metrics[f"hr_ser@{k}"] = hr.item()
        metrics[f"ndcg_ser@{k}"] = ndcg.item()
    return metrics


__all__ = ["compute_ranking_metrics", "compute_ser_metrics"]
