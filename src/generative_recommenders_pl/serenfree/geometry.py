"""Label-free geometry features for V2 SerenFree."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from generative_recommenders_pl.models.serenfree.v2_levels import semantic_non_dedup_levels


@dataclass(frozen=True)
class RingBounds:
    low: float
    high: float


def prefix_surprise(history_sids: torch.Tensor, candidate_sids: torch.Tensor, *, levels="adaptive_semantic_non_dedup", recency_decay: float = 0.85, epsilon: float = 1e-6) -> torch.Tensor:
    selected = semantic_non_dedup_levels(candidate_sids.size(1), levels)
    hist = history_sids[:, list(selected)].to(torch.long)
    cand = candidate_sids[:, list(selected)].to(torch.long)
    valid = (hist != 0).all(dim=1)
    if not valid.any():
        return torch.full((candidate_sids.size(0),), -math.log(float(epsilon)), device=candidate_sids.device)
    hist = hist[valid]
    n = hist.size(0)
    weights = torch.tensor([float(recency_decay) ** (n - 1 - i) for i in range(n)], dtype=torch.float32, device=candidate_sids.device)
    denom = weights.sum().clamp_min(epsilon)
    return torch.stack([-torch.log((weights[(hist == row).all(dim=1)].sum() / denom).clamp_min(epsilon)) for row in cand])


def content_distance(history_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor) -> torch.Tensor:
    if history_embeddings.numel() == 0:
        return torch.ones(candidate_embeddings.size(0), dtype=candidate_embeddings.dtype, device=candidate_embeddings.device)
    return 1.0 - F.normalize(candidate_embeddings, dim=-1).matmul(F.normalize(history_embeddings, dim=-1).T).max(dim=1).values


def quantile_ring(values: torch.Tensor, low_q: float = 0.60, high_q: float = 0.95) -> RingBounds:
    if values.numel() == 0:
        return RingBounds(0.0, 0.0)
    return RingBounds(float(torch.quantile(values.float(), float(low_q)).item()), float(torch.quantile(values.float(), float(high_q)).item()))


def normalize_in_ring(values: torch.Tensor, bounds: RingBounds) -> torch.Tensor:
    return ((values.float() - bounds.low) / max(bounds.high - bounds.low, 1e-12)).clamp(0.0, 1.0)


def relevance_safe_geometry_boost(relevance_scores: torch.Tensor, geometry_scores: torch.Tensor, ranks: torch.Tensor, *, beta: float, relevance_floor_rank: int, low_q: float = 0.60, high_q: float = 0.95) -> tuple[torch.Tensor, torch.Tensor, RingBounds]:
    bounds = quantile_ring(geometry_scores, low_q, high_q)
    guarded = (geometry_scores >= bounds.low) & (geometry_scores <= bounds.high) & (ranks.to(torch.long) <= int(relevance_floor_rank))
    boost = float(beta) * normalize_in_ring(geometry_scores, bounds)
    return relevance_scores + torch.where(guarded, boost, torch.zeros_like(boost)), guarded, bounds


def centroid_distance(history_sids: torch.Tensor, candidate_sids: torch.Tensor, centroid_tables: list[torch.Tensor], *, levels="adaptive_semantic_non_dedup", level_weights=None) -> torch.Tensor:
    selected = tuple(level for level in semantic_non_dedup_levels(candidate_sids.size(1), levels) if level < len(centroid_tables))
    if not selected or history_sids.numel() == 0:
        return torch.zeros(candidate_sids.size(0), dtype=torch.float32, device=candidate_sids.device)
    weights = list(level_weights or [1.0] * len(selected))
    out = []
    for cand in candidate_sids.to(torch.long):
        best = None
        for hist in history_sids.to(torch.long):
            dist = torch.zeros((), dtype=torch.float32, device=candidate_sids.device)
            for w, level in zip(weights, selected):
                ci = int(cand[level]); hi = int(hist[level])
                if ci <= 0 or hi <= 0:
                    continue
                table = centroid_tables[level].to(candidate_sids.device)
                dist = dist + float(w) * (table[ci] - table[hi]).pow(2).sum()
            best = dist if best is None else torch.minimum(best, dist)
        out.append(best if best is not None else torch.zeros((), device=candidate_sids.device))
    return torch.stack(out)
