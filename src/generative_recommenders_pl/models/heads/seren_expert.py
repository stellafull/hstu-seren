"""Serendipity scoring head built on top of HSTU user representations."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


class SerendipityExpertHead(nn.Module):
    """Small MLP that predicts serendipity likelihood for a recommendation pair."""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        context_dim: int | None = None,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            user_dim: Dimension of the target-aware user representation.
            item_dim: Dimension of the target item embedding.
            context_dim: Dimension of optional context embedding concatenated at input.
            hidden_dims: Hidden sizes for the MLP trunk.
            dropout: Dropout rate applied after each activation.
        """
        super().__init__()
        if user_dim <= 0 or item_dim <= 0:
            msg = "user_dim and item_dim must be positive"
            raise ValueError(msg)
        context_dim_value = context_dim or 0
        if context_dim_value < 0:
            msg = "context_dim must be non-negative when provided"
            raise ValueError(msg)
        if dropout < 0 or dropout >= 1:
            msg = "dropout must be in [0, 1)"
            raise ValueError(msg)

        self._input_dim = user_dim + item_dim + context_dim_value
        layers: list[nn.Module] = []
        in_dim = self._input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                msg = "hidden dimensions must be positive"
                raise ValueError(msg)
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        self.classifier = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(
        self,
        user_target_repr: torch.Tensor,
        item_embedding: torch.Tensor,
        context_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for serendipity given user-target and item embeddings."""
        if user_target_repr.dim() != item_embedding.dim():
            msg = "user and item tensors must share rank"
            raise ValueError(msg)
        features = [user_target_repr, item_embedding]
        if context_embedding is not None:
            if context_embedding.dim() != user_target_repr.dim():
                msg = "context tensor rank must match user tensor rank"
                raise ValueError(msg)
            features.append(context_embedding)
        concatenated = torch.cat(features, dim=-1)
        hidden = self.mlp(concatenated)
        logits = self.classifier(hidden)
        return logits.squeeze(-1)

    @staticmethod
    def predict_probability(logits: torch.Tensor) -> torch.Tensor:
        """Sigmoid transform to map logits to serendipity probabilities."""
        return torch.sigmoid(logits)
