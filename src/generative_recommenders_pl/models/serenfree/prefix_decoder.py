"""Shared prefix decoder and relevance loss for SID generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn.functional as F


class DecoderMode(IntEnum):
    """Decoder horizon modes."""

    RELEVANCE = 0
    IMMINENT = 1
    ACCEPTABLE = 2


@dataclass(frozen=True)
class PrefixDecoderOutput:
    """Logits for `q1 -> q2 -> q3 -> d` SID generation."""

    q1_logits: torch.Tensor
    q2_logits: torch.Tensor
    q3_logits: torch.Tensor
    dedup_logits: torch.Tensor

    def as_list(self) -> list[torch.Tensor]:
        return [self.q1_logits, self.q2_logits, self.q3_logits, self.dedup_logits]


class SharedPrefixDecoder(torch.nn.Module):
    """Decode a full SID with one mode-conditioned prefix network."""

    def __init__(
        self,
        hidden_dim: int,
        q1_size: int,
        q2_size: int,
        q3_size: int,
        dedup_size: int,
        prefix_dim: int | None = None,
        mode_count: int = 3,
        level_count: int = 4,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if min(q1_size, q2_size, q3_size, dedup_size) <= padding_idx:
            raise ValueError("vocabulary sizes must be greater than padding_idx")

        self.hidden_dim = int(hidden_dim)
        self.prefix_dim = int(prefix_dim or hidden_dim)
        self.padding_idx = int(padding_idx)

        self.mode_embedding = torch.nn.Embedding(mode_count, hidden_dim)
        self.level_embedding = torch.nn.Embedding(level_count, hidden_dim)
        self.q1_embedding = torch.nn.Embedding(q1_size, self.prefix_dim, padding_idx=padding_idx)
        self.q2_embedding = torch.nn.Embedding(q2_size, self.prefix_dim, padding_idx=padding_idx)
        self.q3_embedding = torch.nn.Embedding(q3_size, self.prefix_dim, padding_idx=padding_idx)

        input_dim = hidden_dim + hidden_dim + hidden_dim + self.prefix_dim * 3
        self.prefix_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.q1_head = torch.nn.Linear(hidden_dim, q1_size)
        self.q2_head = torch.nn.Linear(hidden_dim, q2_size)
        self.q3_head = torch.nn.Linear(hidden_dim, q3_size)
        self.dedup_head = torch.nn.Linear(hidden_dim, dedup_size)

    def forward(
        self,
        context: torch.Tensor,
        prefix_tokens: torch.Tensor | None = None,
        mode: DecoderMode | int = DecoderMode.RELEVANCE,
    ) -> PrefixDecoderOutput:
        if context.dim() < 2:
            raise ValueError("context must have shape [..., hidden_dim]")
        if context.size(-1) != self.hidden_dim:
            raise ValueError("context hidden dimension does not match decoder")

        shape = context.shape[:-1]
        if prefix_tokens is None:
            prefix_tokens = torch.zeros((*shape, 3), dtype=torch.long, device=context.device)
        if prefix_tokens.shape != (*shape, 3):
            raise ValueError("prefix_tokens must have shape [..., 3] for (q1, q2, q3)")
        prefix_tokens = prefix_tokens.to(torch.long)

        mode_ids = torch.full(shape, int(mode), dtype=torch.long, device=context.device)
        mode_emb = self.mode_embedding(mode_ids)
        q1, q2, q3 = prefix_tokens.unbind(dim=-1)
        empty_prefix = torch.zeros((*shape, self.prefix_dim), dtype=context.dtype, device=context.device)

        q1_state = self._state(context, mode_emb, level=0, prefix_parts=[empty_prefix, empty_prefix, empty_prefix])
        q2_state = self._state(
            context,
            mode_emb,
            level=1,
            prefix_parts=[self.q1_embedding(q1), empty_prefix, empty_prefix],
        )
        q3_state = self._state(
            context,
            mode_emb,
            level=2,
            prefix_parts=[self.q1_embedding(q1), self.q2_embedding(q2), empty_prefix],
        )
        dedup_state = self._state(
            context,
            mode_emb,
            level=3,
            prefix_parts=[self.q1_embedding(q1), self.q2_embedding(q2), self.q3_embedding(q3)],
        )
        return PrefixDecoderOutput(
            q1_logits=self.q1_head(q1_state),
            q2_logits=self.q2_head(q2_state),
            q3_logits=self.q3_head(q3_state),
            dedup_logits=self.dedup_head(dedup_state),
        )

    def _state(
        self,
        context: torch.Tensor,
        mode_emb: torch.Tensor,
        level: int,
        prefix_parts: list[torch.Tensor],
    ) -> torch.Tensor:
        level_ids = torch.full(
            context.shape[:-1], level, dtype=torch.long, device=context.device
        )
        level_emb = self.level_embedding(level_ids)
        return self.prefix_mlp(torch.cat([context, mode_emb, level_emb, *prefix_parts], dim=-1))


def relevance_loss(
    output: PrefixDecoderOutput,
    targets: torch.Tensor,
    lambda_d: float = 1.0,
    ignore_index: int = 0,
) -> torch.Tensor:
    """Cross-entropy loss for `q1, q2, q3, d` targets."""

    if targets.size(-1) != 4:
        raise ValueError("targets must end with four values: (q1, q2, q3, d)")
    weights = [1.0, 1.0, 1.0, float(lambda_d)]
    losses = []
    for logits, target, weight in zip(output.as_list(), targets.unbind(dim=-1), weights):
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1).to(torch.long),
                ignore_index=ignore_index,
            )
            * weight
        )
    return torch.stack(losses).sum()
