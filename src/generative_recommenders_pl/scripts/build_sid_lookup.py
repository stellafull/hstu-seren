"""Build item-id indexed full-SID lookup tensors for SerenFree training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build [normalized item id] -> shifted full-SID lookup. "
            "This does not create semantic IDs; it only reindexes an existing "
            "full-SID artifact for dataloader integer ids."
        )
    )
    parser.add_argument("--sid-path", required=True, type=Path)
    parser.add_argument("--item-lookup-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--normalized-id-offset", type=int, default=1)
    parser.add_argument("--sid-token-shift", type=int, default=1)
    parser.add_argument("--max-item-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sid_lookup, metadata = build_sid_lookup(
        sid_path=args.sid_path,
        item_lookup_path=args.item_lookup_path,
        normalized_id_offset=args.normalized_id_offset,
        sid_token_shift=args.sid_token_shift,
        max_item_id=args.max_item_id,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sid_lookup": sid_lookup, **metadata}, args.output_path)
    print(
        "saved",
        args.output_path,
        "shape",
        tuple(sid_lookup.shape),
        "mapped",
        metadata["mapped_count"],
        "sid_columns",
        metadata["sid_columns"],
        "max_by_col",
        sid_lookup.max(dim=0).values.tolist(),
        flush=True,
    )


def build_sid_lookup(
    sid_path: Path,
    item_lookup_path: Path,
    normalized_id_offset: int = 1,
    sid_token_shift: int = 1,
    max_item_id: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    sid_data = torch.load(sid_path, map_location="cpu")
    item_ids_raw = sid_data["item_ids"]
    sid_tokens = sid_data["semantic_ids"].to(torch.long)
    if sid_tokens.dim() != 2 or sid_tokens.size(1) < 2:
        raise ValueError("Expected full SID tensor shaped [items, semantic levels + dedup]")
    if sid_tokens.min().item() < 0:
        raise ValueError("SID tokens must be non-negative before shifting")

    sid_tokens = sid_tokens + int(sid_token_shift)
    lookup_map, lookup_max = read_item_lookup(
        item_lookup_path=item_lookup_path,
        normalized_id_offset=normalized_id_offset,
    )

    mapped_ids: list[int] = []
    kept_indices: list[int] = []
    for idx, raw_id in enumerate(item_ids_raw):
        mapped = lookup_map.get(str(raw_id).casefold())
        if mapped is None or mapped <= 0:
            continue
        mapped_ids.append(mapped)
        kept_indices.append(idx)
    if not mapped_ids:
        raise ValueError("No full SID rows mapped to normalized item ids")

    output_max_item_id = max(mapped_ids)
    output_max_item_id = max(output_max_item_id, lookup_max)
    if max_item_id is not None:
        output_max_item_id = max(output_max_item_id, int(max_item_id))

    sid_lookup = torch.zeros(
        (output_max_item_id + 1, sid_tokens.size(1)),
        dtype=torch.long,
    )
    index = torch.tensor(kept_indices, dtype=torch.long)
    mapped = torch.tensor(mapped_ids, dtype=torch.long)
    sid_lookup[mapped] = sid_tokens[index]

    metadata: dict[str, Any] = {
        "mapped_count": len(mapped_ids),
        "unmapped_count": len(item_ids_raw) - len(mapped_ids),
        "sid_columns": sid_tokens.size(1),
        "semantic_levels": sid_tokens.size(1) - 1,
        "dedup_column": sid_tokens.size(1) - 1,
        "normalized_id_offset": int(normalized_id_offset),
        "sid_token_shift": int(sid_token_shift),
        "source_sid_path": str(sid_path),
        "item_lookup_path": str(item_lookup_path),
    }
    return sid_lookup, metadata


def read_item_lookup(
    item_lookup_path: Path,
    normalized_id_offset: int,
) -> tuple[dict[str, int], int]:
    lookup_map: dict[str, int] = {}
    max_item_id = 0
    with item_lookup_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            normalized = int(row["normalized_item_id"]) + int(normalized_id_offset)
            lookup_map[str(row["original_item_id"]).casefold()] = normalized
            max_item_id = max(max_item_id, normalized)
    return lookup_map, max_item_id


if __name__ == "__main__":
    main()
