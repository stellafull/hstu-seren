"""Binary cross entropy objectives for serendipity supervision."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class _SerendipityLossBase(torch.nn.Module):
    """Shared utilities for serendipity loss implementations."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            msg = "reduction must be either 'mean' or 'sum'"
            raise ValueError(msg)
        self.reduction = reduction

    @staticmethod
    def _prepare_mask(
        mask: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(reference, dtype=reference.dtype)
        if mask.size() != reference.size():
            msg = "mask must share shape with predictions"
            raise ValueError(msg)
        return mask.to(dtype=reference.dtype, device=reference.device)

    def _reduce(
        self,
        losses: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_losses = losses * mask
        normalizer = mask.sum()
        safe_normalizer = torch.clamp(normalizer, min=1.0)
        if self.reduction == "mean":
            loss = masked_losses.sum() / safe_normalizer
        else:
            loss = masked_losses.sum()
        return loss, safe_normalizer

    @staticmethod
    def _metrics(
        loss: torch.Tensor,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        normalizer: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        masked_probabilities = probabilities * mask
        masked_targets = targets * mask
        positive_rate = masked_targets.sum() / normalizer
        mean_probability = masked_probabilities.sum() / normalizer
        accuracy = (
            torch.eq(probabilities >= 0.5, targets >= 0.5)
            .to(dtype=probabilities.dtype)
            .mul(mask)
            .sum()
            / normalizer
        )
        return {
            "train/ser_bce": loss.detach(),
            "train/ser_positive_rate": positive_rate.detach(),
            "train/ser_prob_mean": mean_probability.detach(),
            "train/ser_accuracy": accuracy.detach(),
        }


class SerendipityBCELoss(_SerendipityLossBase):
    """Binary cross entropy loss on logits for serendipity labels."""

    def __init__(
        self,
        reduction: str = "mean",
        pos_weight: float | None = None,
    ) -> None:
        super().__init__(reduction=reduction)
        self._pos_weight = float(pos_weight) if pos_weight is not None else None

    def forward(
        self,
        logits: torch.Tensor,
        ser_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.size() != ser_labels.size():
            msg = "logits and ser_labels must share shape"
            raise ValueError(msg)
        targets = ser_labels.to(dtype=logits.dtype, device=logits.device)
        mask_tensor = self._prepare_mask(mask, logits)
        pos_weight_tensor = (
            torch.tensor(self._pos_weight, dtype=logits.dtype, device=logits.device)
            if self._pos_weight is not None
            else None
        )
        losses = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight_tensor,
        )
        loss, normalizer = self._reduce(losses, mask_tensor)
        with torch.no_grad():
            probabilities = torch.sigmoid(logits)
            metrics = self._metrics(
                loss=loss,
                probabilities=probabilities,
                targets=targets,
                mask=mask_tensor,
                normalizer=normalizer,
            )
        return loss, metrics


class SerendipityFocalLoss(_SerendipityLossBase):
    """Focal loss on logits for serendipity labels."""

    def __init__(
        self,
        reduction: str = "mean",
        gamma: float = 2.0,
        alpha: float | None = 0.25,
    ) -> None:
        super().__init__(reduction=reduction)
        if gamma < 0.0:
            msg = "gamma must be non-negative"
            raise ValueError(msg)
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            msg = "alpha must be in [0, 1]"
            raise ValueError(msg)
        self.gamma = float(gamma)
        self.alpha = float(alpha) if alpha is not None else None

    def forward(
        self,
        logits: torch.Tensor,
        ser_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.size() != ser_labels.size():
            msg = "logits and ser_labels must share shape"
            raise ValueError(msg)
        targets = ser_labels.to(dtype=logits.dtype, device=logits.device)
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        else:
            alpha_factor = torch.ones_like(probabilities)
        losses = (
            F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            * focal_weight
            * alpha_factor
        )
        mask_tensor = self._prepare_mask(mask, logits)
        loss, normalizer = self._reduce(losses, mask_tensor)
        with torch.no_grad():
            metrics = self._metrics(
                loss=loss,
                probabilities=probabilities,
                targets=targets,
                mask=mask_tensor,
                normalizer=normalizer,
            )
            metrics["train/ser_focal"] = metrics.pop("train/ser_bce")
        return loss, metrics


class SerendipityProbabilityBCELoss(_SerendipityLossBase):
    """Binary cross entropy loss on probabilities for serendipity labels."""

    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-7,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__(reduction=reduction)
        if eps <= 0 or eps >= 0.5:
            msg = "eps must be in (0, 0.5)"
            raise ValueError(msg)
        self.eps = float(eps)
        self._pos_weight = float(pos_weight) if pos_weight is not None else None

    def forward(
        self,
        probabilities: torch.Tensor,
        ser_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if probabilities.size() != ser_labels.size():
            msg = "probabilities and ser_labels must share shape"
            raise ValueError(msg)
        clamped_probabilities = probabilities.clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )
        targets = ser_labels.to(dtype=clamped_probabilities.dtype, device=clamped_probabilities.device)
        mask_tensor = self._prepare_mask(mask, clamped_probabilities)
        if self._pos_weight is not None:
            pos_weight_tensor = torch.tensor(
                self._pos_weight,
                dtype=clamped_probabilities.dtype,
                device=clamped_probabilities.device,
            )
            positive_term = targets * torch.log(clamped_probabilities)
            negative_term = (1 - targets) * torch.log1p(-clamped_probabilities)
            weighted_positive = pos_weight_tensor * positive_term
            losses = -(weighted_positive + negative_term)
        else:
            losses = F.binary_cross_entropy(
                clamped_probabilities,
                targets,
                reduction="none",
            )
        loss, normalizer = self._reduce(losses, mask_tensor)
        with torch.no_grad():
            metrics = self._metrics(
                loss=loss,
                probabilities=clamped_probabilities,
                targets=targets,
                mask=mask_tensor,
                normalizer=normalizer,
            )
        return loss, metrics
