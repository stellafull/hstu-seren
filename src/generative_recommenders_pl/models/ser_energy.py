from __future__ import annotations

from typing import Optional

import torch

from generative_recommenders_pl.models.utils.initialization import (
    init_mlp_xavier_weights_zero_bias,
)


class SerEnergy(torch.nn.Module):
    """Tiny MLP that produces additive serendipity energy adjustments."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: Optional[str] = "gelu",
    ) -> None:
        super().__init__()

        if num_layers < 1:
            msg = "SerEnergy requires at least one layer"
            raise ValueError(msg)

        layers: list[torch.nn.Module] = []
        in_dim = d_in
        act: torch.nn.Module
        if activation is None:
            act = torch.nn.Identity()
        elif activation.lower() == "gelu":
            act = torch.nn.GELU()
        elif activation.lower() == "relu":
            act = torch.nn.ReLU()
        else:
            msg = f"Unsupported activation '{activation}'"
            raise ValueError(msg)

        for layer_idx in range(num_layers - 1):
            layers.extend(
                [
                    torch.nn.Linear(in_dim, d_hidden),
                    act,
                ]
            )
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            in_dim = d_hidden

        layers.append(torch.nn.Linear(in_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self.apply(init_mlp_xavier_weights_zero_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return additive serendipity energy for a candidate set.

        Args:
            features: Tensor with shape ``(batch, num_candidates, d_in)``.

        Returns:
            Tensor with shape ``(batch, num_candidates)`` containing energy values.
        """

        if features.ndim != 3:
            msg = "features must be a 3D tensor (batch, candidates, dim)"
            raise ValueError(msg)
        return self.mlp(features).squeeze(-1)
