"""Thin HSTU state wrapper for HSTU-SerenFree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HSTUStateOutput:
    """Hidden states plus horizon pooling views."""

    hidden_states: torch.Tensor
    recent_state: torch.Tensor
    history_state: torch.Tensor
    cache_states: Any = None


class HSTUStateWrapper(torch.nn.Module):
    """Expose HSTU hidden states and V1 horizon pools without changing HSTU."""

    def __init__(self, sequence_encoder: torch.nn.Module, recent_window: int = 10) -> None:
        super().__init__()
        if recent_window <= 0:
            raise ValueError("recent_window must be positive")
        self.sequence_encoder = sequence_encoder
        self.recent_window = int(recent_window)

    def forward(
        self,
        past_lengths: torch.Tensor,
        user_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        past_payloads: dict[str, torch.Tensor],
        **encoder_kwargs: Any,
    ) -> HSTUStateOutput:
        encoded = self.sequence_encoder(
            past_lengths=past_lengths,
            user_embeddings=user_embeddings,
            valid_mask=valid_mask,
            past_payloads=past_payloads,
            **encoder_kwargs,
        )
        if isinstance(encoded, tuple):
            hidden_states, cache_states = encoded
        else:
            hidden_states, cache_states = encoded, None

        history_mask = _length_mask(past_lengths, hidden_states.size(1))
        recent_mask = _recent_length_mask(
            past_lengths, hidden_states.size(1), self.recent_window
        )
        return HSTUStateOutput(
            hidden_states=hidden_states,
            recent_state=_masked_mean(hidden_states, recent_mask),
            history_state=_masked_mean(hidden_states, history_mask),
            cache_states=cache_states,
        )


def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions < lengths.to(torch.long).unsqueeze(1)


def _recent_length_mask(
    lengths: torch.Tensor, max_len: int, recent_window: int
) -> torch.Tensor:
    lengths = lengths.to(torch.long)
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    starts = (lengths - int(recent_window)).clamp_min(0).unsqueeze(1)
    ends = lengths.unsqueeze(1)
    return (positions >= starts) & (positions < ends)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom
