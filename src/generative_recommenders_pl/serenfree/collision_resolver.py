"""Collision-aware SID resolver for V2 SerenFree."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CollisionStats:
    mean_collision_bucket_size: float
    p95_collision_bucket_size: float
    max_collision_bucket_size: int
    collision_resolver_used_rate: float
    fraction_outputs_from_collision_bucket: float


def sid_buckets(item_ids: torch.Tensor, sid_lookup: torch.Tensor) -> dict[tuple[int, ...], list[int]]:
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for item in item_ids.to(torch.long).tolist():
        sid = tuple(int(x) for x in sid_lookup[int(item)].tolist())
        if all(token != 0 for token in sid):
            buckets[sid].append(int(item))
    return dict(buckets)


def resolve_bucket(bucket_items: list[int], sid_score: float, popularity: dict[int, float] | None = None, eta_pop: float = 0.0) -> tuple[int, dict[int, float]]:
    import math
    scores = {int(item): float(sid_score) - float(eta_pop) * math.log1p(float((popularity or {}).get(int(item), 0.0))) for item in bucket_items}
    return max(scores, key=scores.get), scores


def collision_stats(bucket_sizes: list[int], outputs_from_collision: int, total_outputs: int) -> CollisionStats:
    if not bucket_sizes:
        return CollisionStats(0.0, 0.0, 0, 0.0, 0.0)
    vals = torch.tensor(bucket_sizes, dtype=torch.float32)
    used = sum(1 for x in bucket_sizes if x > 1)
    return CollisionStats(float(vals.mean()), float(torch.quantile(vals, 0.95)), int(vals.max()), used / max(len(bucket_sizes), 1), outputs_from_collision / max(total_outputs, 1))
