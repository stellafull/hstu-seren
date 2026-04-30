"""Valid-SID trie and constrained beam search."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from generative_recommenders_pl.models.serenfree.prefix_decoder import (
    DecoderMode,
    SharedPrefixDecoder,
)


@dataclass(frozen=True)
class SIDBeam:
    """One constrained decoding result."""

    item_id: int
    sid: tuple[int, ...]
    score: float


class SIDTrie:
    """Trie over valid full SID tuples."""

    def __init__(self) -> None:
        self._root: dict[int, dict] = {}
        self.num_sid_columns: int | None = None

    @classmethod
    def from_items(cls, item_ids: torch.Tensor, sid_tokens: torch.Tensor) -> "SIDTrie":
        if item_ids.dim() != 1:
            raise ValueError("item_ids must be 1D")
        if sid_tokens.dim() != 2 or sid_tokens.size(0) != item_ids.numel():
            raise ValueError("sid_tokens must have shape [num_items, sid_columns]")
        trie = cls()
        for item_id, sid in zip(item_ids.tolist(), sid_tokens.to(torch.long).tolist()):
            trie.insert(int(item_id), tuple(int(token) for token in sid))
        return trie

    def insert(self, item_id: int, sid: tuple[int, ...]) -> None:
        if len(sid) < 2:
            raise ValueError("sid must include semantic levels and dedup")
        if self.num_sid_columns is None:
            self.num_sid_columns = len(sid)
        elif len(sid) != self.num_sid_columns:
            raise ValueError("all SIDs in one trie must have the same depth")
        node = self._root
        for token in sid:
            node = node.setdefault(int(token), {})
        node["_item_id"] = int(item_id)

    def allowed_tokens(self, prefix: tuple[int, ...]) -> list[int]:
        node = self._node(prefix)
        return sorted(token for token in node.keys() if isinstance(token, int))

    def item_id(self, sid: tuple[int, ...]) -> int | None:
        node = self._node(sid)
        value = node.get("_item_id")
        return int(value) if value is not None else None

    def _node(self, prefix: tuple[int, ...]) -> dict:
        node = self._root
        for token in prefix:
            next_node = node.get(int(token))
            if next_node is None:
                return {}
            node = next_node
        return node


def constrained_beam_search(
    decoder: SharedPrefixDecoder,
    context: torch.Tensor,
    trie: SIDTrie,
    beam_size: int,
    mode: DecoderMode | int = DecoderMode.RELEVANCE,
) -> list[list[SIDBeam]]:
    """Decode valid full SIDs for each batch row using trie constraints."""

    if context.dim() != 2:
        raise ValueError("context must have shape [B, H]")
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    if trie.num_sid_columns is None:
        raise ValueError("trie is empty")

    results: list[list[SIDBeam]] = []
    for row_idx in range(context.size(0)):
        row_context = context[row_idx : row_idx + 1]
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for level in range(trie.num_sid_columns):
            expanded: list[tuple[tuple[int, ...], float]] = []
            for prefix, score in beams:
                allowed = trie.allowed_tokens(prefix)
                if not allowed:
                    continue
                prefix_tensor = _prefix_tensor(
                    prefix, decoder.num_semantic_levels, row_context.device
                )
                logits = decoder(row_context, prefix_tensor, mode=mode).as_list()[level][0]
                log_probs = F.log_softmax(logits, dim=-1)
                for token in allowed:
                    expanded.append(((*prefix, token), score + float(log_probs[token].item())))
            beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:beam_size]

        row_results: list[SIDBeam] = []
        for sid_prefix, score in beams:
            item_id = trie.item_id(sid_prefix)
            if item_id is not None:
                row_results.append(SIDBeam(item_id=item_id, sid=sid_prefix, score=score))
        results.append(row_results)
    return results


def _prefix_tensor(
    prefix: tuple[int, ...], num_semantic_levels: int, device: torch.device
) -> torch.Tensor:
    padded = list(prefix[:num_semantic_levels])
    padded.extend([0] * (num_semantic_levels - len(padded)))
    return torch.tensor([padded], dtype=torch.long, device=device)
