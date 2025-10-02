from __future__ import annotations

from typing import Dict, Tuple

import torch


class SCSTSerLoss(torch.nn.Module):
    """Placeholder SCST objective for future experimentation."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        ser_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError(
            "SCSTSerLoss is not implemented yet. Use SoftNDCGSerLoss instead."
        )
