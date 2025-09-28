"""Prediction heads for multi-task HSTU models."""

from __future__ import annotations

import torch
from torch import nn


class RelHead(nn.Module):
    """Linear head for next-item prediction with optional weight tying."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        tied_weight: nn.Parameter | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, vocab_size, bias=bias)
        if tied_weight is not None:
            if tied_weight.size(0) != vocab_size:
                raise ValueError(
                    "Tied weight size does not match vocab size for RelHead"
                )
            self.fc.weight = tied_weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to next-item logits."""

        return self.fc(hidden)


class SerHead(nn.Module):
    """Binary classification head for serendipity detection."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.fc(hidden).squeeze(-1)
        return logits


__all__ = ["RelHead", "SerHead"]
