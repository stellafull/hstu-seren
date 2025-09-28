"""Loss helpers for multi-task next-item and serendipity heads."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    target_ids: torch.Tensor,
    ser_labels: torch.Tensor,
    *,
    pos_weight: float,
    rel_loss_fn: Optional[
        Callable[[Dict[str, torch.Tensor], torch.Tensor], torch.Tensor]
    ] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute relevance and serendipity losses for multi-task output.

    Args:
        outputs: Dict with keys ``logits_next`` and ``logits_ser``.
        target_ids: Tensor of shape [B] with next-item targets.
        ser_labels: Tensor of shape [B] with binary serendipity labels.
        pos_weight: Positive class weight for ser loss (neg/pos ratio).
        rel_loss_fn: Optional callable that computes the relevance loss given
            the model outputs and supervised target ids. When ``None``, a standard
            cross-entropy loss over ``logits_next`` is used.

    Returns:
        Tuple of (next-item CE loss, serendipity BCE loss).
    """

    logits_next = outputs["logits_next"]
    logits_ser = outputs["logits_ser"]

    if rel_loss_fn is None:
        loss_next = F.cross_entropy(logits_next, target_ids)
    else:
        loss_next = rel_loss_fn(outputs, target_ids)
    if not torch.is_floating_point(ser_labels):
        ser_labels = ser_labels.to(dtype=logits_ser.dtype)
    pos_weight_tensor = torch.tensor(
        [pos_weight],
        device=logits_ser.device,
        dtype=logits_ser.dtype,
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    loss_ser = bce(logits_ser, ser_labels)
    return loss_next, loss_ser


__all__ = ["compute_losses"]
