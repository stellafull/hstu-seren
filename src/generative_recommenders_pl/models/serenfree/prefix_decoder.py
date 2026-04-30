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
    """Logits for variable-depth full-SID generation."""

    level_logits: list[torch.Tensor]

    @property
    def semantic_logits(self) -> list[torch.Tensor]:
        return self.level_logits[:-1]

    @property
    def dedup_logits(self) -> torch.Tensor:
        return self.level_logits[-1]

    @property
    def q1_logits(self) -> torch.Tensor:
        return self.level_logits[0]

    @property
    def q2_logits(self) -> torch.Tensor:
        return self.level_logits[1]

    @property
    def q3_logits(self) -> torch.Tensor:
        return self.level_logits[2]

    def as_list(self) -> list[torch.Tensor]:
        return self.level_logits


class SharedPrefixDecoder(torch.nn.Module):
    """Decode a full SID with one mode-conditioned prefix network."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_sizes: list[int] | tuple[int, ...] | None = None,
        prefix_dim: int | None = None,
        mode_count: int = 3,
        padding_idx: int = 0,
        q1_size: int | None = None,
        q2_size: int | None = None,
        q3_size: int | None = None,
        dedup_size: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_sizes is None:
            if None in {q1_size, q2_size, q3_size, dedup_size}:
                raise ValueError("vocab_sizes or q1/q2/q3/dedup sizes are required")
            vocab_sizes = [int(q1_size), int(q2_size), int(q3_size), int(dedup_size)]
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if len(vocab_sizes) < 2:
            raise ValueError("full SID must include at least one semantic level and dedup")
        if min(vocab_sizes) <= padding_idx:
            raise ValueError("all vocabulary sizes must be greater than padding_idx")

        self.hidden_dim = int(hidden_dim)
        self.prefix_dim = int(prefix_dim or hidden_dim)
        self.padding_idx = int(padding_idx)
        self.vocab_sizes = [int(size) for size in vocab_sizes]
        self.num_sid_columns = len(self.vocab_sizes)
        self.num_semantic_levels = self.num_sid_columns - 1

        self.mode_embedding = torch.nn.Embedding(mode_count, hidden_dim)
        self.level_embedding = torch.nn.Embedding(self.num_sid_columns, hidden_dim)
        self.prefix_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(size, self.prefix_dim, padding_idx=padding_idx)
                for size in self.vocab_sizes[:-1]
            ]
        )
        input_dim = hidden_dim + hidden_dim + hidden_dim + self.prefix_dim * self.num_semantic_levels
        self.prefix_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, size) for size in self.vocab_sizes]
        )

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
            prefix_tokens = torch.zeros(
                (*shape, self.num_semantic_levels),
                dtype=torch.long,
                device=context.device,
            )
        if prefix_tokens.shape != (*shape, self.num_semantic_levels):
            raise ValueError(
                f"prefix_tokens must have shape [..., {self.num_semantic_levels}]"
            )
        prefix_tokens = prefix_tokens.to(torch.long)

        mode_ids = torch.full(shape, int(mode), dtype=torch.long, device=context.device)
        mode_emb = self.mode_embedding(mode_ids)
        empty_prefix = torch.zeros(
            (*shape, self.prefix_dim), dtype=context.dtype, device=context.device
        )

        logits = []
        for level, head in enumerate(self.heads):
            prefix_parts = self._prefix_parts(prefix_tokens, level, empty_prefix)
            state = self._state(context, mode_emb, level, prefix_parts)
            logits.append(head(state))
        return PrefixDecoderOutput(level_logits=logits)

    def _prefix_parts(
        self,
        prefix_tokens: torch.Tensor,
        level: int,
        empty_prefix: torch.Tensor,
    ) -> list[torch.Tensor]:
        parts = []
        known_semantic = min(level, self.num_semantic_levels)
        for prefix_level in range(self.num_semantic_levels):
            if prefix_level < known_semantic:
                parts.append(self.prefix_embeddings[prefix_level](prefix_tokens[..., prefix_level]))
            else:
                parts.append(empty_prefix)
        return parts

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
    """Cross-entropy loss for full-SID targets; final column is dedup."""

    if targets.size(-1) != len(output.level_logits):
        raise ValueError("targets must have the same number of columns as logits")
    weights = [1.0] * (len(output.level_logits) - 1) + [float(lambda_d)]
    losses = []
    for logits, target, weight in zip(output.level_logits, targets.unbind(dim=-1), weights):
        flat_target = target.reshape(-1).to(torch.long)
        valid = flat_target != ignore_index
        if not valid.any():
            losses.append(logits.sum() * 0.0)
            continue
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                flat_target,
                ignore_index=ignore_index,
            )
            * weight
        )
    return torch.stack(losses).sum()
