from __future__ import annotations

from typing import Optional

import torch

_EPS = 1e-8


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(x.norm(dim=-1, keepdim=True), min=_EPS)


def max_cos_to_recent(
    candidate_embeddings: torch.Tensor,
    recent_embeddings: torch.Tensor,
    recent_mask: torch.Tensor,
) -> torch.Tensor:
    """Return maximum cosine similarity between candidates and recent history."""

    if candidate_embeddings.ndim != 3:
        msg = "candidate_embeddings must be (batch, num_candidates, dim)"
        raise ValueError(msg)
    if recent_embeddings.ndim != 3:
        msg = "recent_embeddings must be (batch, history_len, dim)"
        raise ValueError(msg)
    if recent_mask.ndim != 2:
        msg = "recent_mask must be (batch, history_len)"
        raise ValueError(msg)

    if candidate_embeddings.size(0) != recent_embeddings.size(0):
        msg = "Batch size mismatch between candidates and history"
        raise ValueError(msg)
    if recent_embeddings.size(0) != recent_mask.size(0):
        msg = "recent_mask must align with history embeddings"
        raise ValueError(msg)

    norm_candidates = _normalize(candidate_embeddings)
    norm_recent = _normalize(recent_embeddings)

    similarities = torch.matmul(norm_candidates, norm_recent.transpose(-1, -2))
    similarities = similarities.masked_fill(~recent_mask.unsqueeze(1), float("-inf"))
    max_sim, _ = similarities.max(dim=-1)
    max_sim = torch.nan_to_num(max_sim, nan=0.0, neginf=0.0)
    return max_sim.clamp(min=-1.0, max=1.0)


def compute_unexpectedness(
    candidate_embeddings: torch.Tensor,
    recent_embeddings: torch.Tensor,
    recent_mask: torch.Tensor,
) -> torch.Tensor:
    max_cos = max_cos_to_recent(candidate_embeddings, recent_embeddings, recent_mask)
    return 1.0 - max_cos


def _per_row_quantile(
    values: torch.Tensor,
    quantile: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not 0.0 < quantile < 1.0:
        msg = "quantile must lie in (0, 1)"
        raise ValueError(msg)

    if values.size() != mask.size():
        msg = "values and mask must share shape"
        raise ValueError(msg)

    padded = torch.where(mask, values, torch.full_like(values, float("inf")))
    sorted_vals, _ = torch.sort(padded, dim=-1, descending=False)

    counts = mask.sum(dim=-1)
    safe_counts = torch.clamp(counts, min=1)
    positions = torch.round((safe_counts.to(values.dtype) - 1.0) * quantile).long()
    positions = torch.clamp(positions, min=0, max=sorted_vals.size(-1) - 1)
    gathered = sorted_vals.gather(-1, positions.unsqueeze(-1)).squeeze(-1)

    default_threshold = torch.zeros_like(gathered)
    return torch.where(counts > 0, gathered, default_threshold)


def compute_serendipity_indicator(
    unexpectedness: torch.Tensor,
    base_prob: torch.Tensor,
    mask: torch.Tensor,
    quantile: float,
    relevance_quantile: Optional[float] = None,
) -> torch.Tensor:
    tau = _per_row_quantile(unexpectedness, quantile, mask)
    indicator = unexpectedness >= tau.unsqueeze(-1)
    indicator &= mask
    if relevance_quantile is not None:
        rho = _per_row_quantile(base_prob, relevance_quantile, mask)
        indicator &= base_prob >= rho.unsqueeze(-1)
    return indicator.to(unexpectedness.dtype)


def online_serendipity_scores(
    candidate_embeddings: torch.Tensor,
    recent_embeddings: torch.Tensor,
    recent_mask: torch.Tensor,
    base_prob: torch.Tensor,
    candidate_mask: torch.Tensor,
    quantile: float,
    relevance_quantile: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unexpectedness and binary serendipity indicator."""

    unexpectedness = compute_unexpectedness(
        candidate_embeddings, recent_embeddings, recent_mask
    )
    indicator = compute_serendipity_indicator(
        unexpectedness=unexpectedness,
        base_prob=base_prob,
        mask=candidate_mask,
        quantile=quantile,
        relevance_quantile=relevance_quantile,
    )
    return unexpectedness, indicator
