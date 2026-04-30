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
    sid: tuple[int, int, int, int]
    score: float


class SIDTrie:
    """Trie over valid `(q1, q2, q3, d)` tuples."""

    def __init__(self) -> None:
        self._root: dict[int, dict] = {}

    @classmethod
    def from_items(cls, item_ids: torch.Tensor, sid_tokens: torch.Tensor) -> "SIDTrie":
        if item_ids.dim() != 1:
            raise ValueError("item_ids must be 1D")
        if sid_tokens.shape != (item_ids.numel(), 4):
            raise ValueError("sid_tokens must have shape [num_items, 4]")
        trie = cls()
        for item_id, sid in zip(item_ids.tolist(), sid_tokens.to(torch.long).tolist()):
            trie.insert(int(item_id), tuple(int(token) for token in sid))
        return trie

    def insert(self, item_id: int, sid: tuple[int, int, int, int]) -> None:
        if len(sid) != 4:
            raise ValueError("sid must contain exactly four tokens")
        node = self._root
        for token in sid:
            node = node.setdefault(int(token), {})
        node["_item_id"] = int(item_id)

    def allowed_tokens(self, prefix: tuple[int, ...]) -> list[int]:
        node = self._node(prefix)
        return sorted(token for token in node.keys() if isinstance(token, int))

    def item_id(self, sid: tuple[int, int, int, int]) -> int | None:
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
    """Decode valid SIDs for each batch row using trie-constrained beams."""

    if context.dim() != 2:
        raise ValueError("context must have shape [B, H]")
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")

    results: list[list[SIDBeam]] = []
    for row_idx in range(context.size(0)):
        row_context = context[row_idx : row_idx + 1]
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for level in range(4):
            expanded: list[tuple[tuple[int, ...], float]] = []
            for prefix, score in beams:
                allowed = trie.allowed_tokens(prefix)
                if not allowed:
                    continue
                prefix_tensor = _prefix_tensor(prefix, row_context.device)
                logits = decoder(row_context, prefix_tensor, mode=mode).as_list()[level][0]
                log_probs = F.log_softmax(logits, dim=-1)
                for token in allowed:
                    expanded.append(((*prefix, token), score + float(log_probs[token].item())))
            beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:beam_size]

        row_results: list[SIDBeam] = []
        for sid_prefix, score in beams:
            if len(sid_prefix) != 4:
                continue
            sid = sid_prefix  # type: ignore[assignment]
            item_id = trie.item_id(sid)
            if item_id is not None:
                row_results.append(SIDBeam(item_id=item_id, sid=sid, score=score))
        results.append(row_results)
    return results


def _prefix_tensor(prefix: tuple[int, ...], device: torch.device) -> torch.Tensor:
    padded = list(prefix[:3])
    padded.extend([0] * (3 - len(padded)))
    return torch.tensor([padded], dtype=torch.long, device=device)
