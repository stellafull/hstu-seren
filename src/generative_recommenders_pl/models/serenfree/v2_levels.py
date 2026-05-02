"""V2 SID level selection helpers.

Configs may use ``adaptive_semantic_non_dedup`` instead of hard-coding level 1/2.
Public config levels are 1-based; returned levels are 0-based.
"""

from __future__ import annotations

from collections.abc import Iterable


def semantic_non_dedup_levels(
    num_sid_columns: int,
    configured: str | Iterable[int] | None = "adaptive_semantic_non_dedup",
) -> tuple[int, ...]:
    if num_sid_columns < 2:
        raise ValueError("SID must include semantic levels and dedup")
    semantic_count = int(num_sid_columns) - 1
    if configured is None or configured == "adaptive_semantic_non_dedup":
        return tuple(range(semantic_count))
    if isinstance(configured, str):
        raise ValueError(f"Unsupported level selector: {configured!r}")
    levels: list[int] = []
    for raw in configured:
        level = int(raw)
        if level <= 0:
            raise ValueError("Configured levels are 1-based")
        zero = level - 1
        if zero >= semantic_count:
            raise ValueError(f"Configured level {level} touches dedup/out-of-range")
        if zero not in levels:
            levels.append(zero)
    return tuple(levels)


def touches_dedup(
    num_sid_columns: int,
    configured: str | Iterable[int] | None = "adaptive_semantic_non_dedup",
) -> bool:
    try:
        semantic_non_dedup_levels(num_sid_columns, configured)
    except ValueError:
        return True
    return False
