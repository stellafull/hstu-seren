from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

_EPS = 1e-8


class SoftNDCGSerLoss(torch.nn.Module):
    """Soft NDCG-style objective for serendipity-aware training."""

    def __init__(
        self,
        k: int = 10,
        beta: float = 0.05,
        soft_rank_strength: float = 1.0,
        cutoff_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if k <= 0:
            msg = "k must be positive"
            raise ValueError(msg)
        self.k = k
        self.beta = beta
        self.soft_rank_strength = soft_rank_strength
        self.cutoff_temperature = cutoff_temperature

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask.to(dtype=torch.bool)
        neg_inf = torch.tensor(-math.inf, dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(mask_bool, logits, neg_inf)
        has_valid = mask_bool.any(dim=-1, keepdim=True)
        max_logits = masked_logits.max(dim=-1, keepdim=True).values
        max_logits = torch.where(has_valid, max_logits, torch.zeros_like(max_logits))
        exps = torch.exp(masked_logits - max_logits)
        exps = torch.where(mask_bool, exps, torch.zeros_like(exps))
        sums = exps.sum(dim=-1, keepdim=True)
        safe_sums = torch.clamp(sums, min=_EPS)
        return exps / safe_sums

    @staticmethod
    def _soft_rank(
        logits: torch.Tensor, mask: torch.Tensor, tau: float
    ) -> torch.Tensor:
        mask_bool = mask.to(dtype=torch.bool)
        mask_float = mask_bool.to(dtype=logits.dtype)
        valid_logits = torch.where(mask_bool, logits, torch.zeros_like(logits))

        pair_mask = mask_float.unsqueeze(-1) * mask_float.unsqueeze(-2)
        diff = (valid_logits.unsqueeze(-1) - valid_logits.unsqueeze(-2)) * pair_mask

        num_pairs = pair_mask.sum(dim=(-1, -2)) - mask_float.sum(dim=-1)
        avg_abs_diff = diff.abs().sum(dim=(-1, -2)) / torch.clamp(num_pairs, min=1.0)
        base_tau = max(tau, 1e-3)
        adaptive_tau = torch.clamp(avg_abs_diff * base_tau, min=1e-3)
        adaptive_tau = adaptive_tau.unsqueeze(-1).unsqueeze(-1)

        pairwise = torch.sigmoid(-diff / adaptive_tau) * pair_mask
        pairwise = pairwise - torch.diag_embed(torch.diagonal(pairwise, dim1=-2, dim2=-1))
        return pairwise.sum(dim=-1) + 1.0

    def forward(
        self,
        logits: torch.Tensor,
        baseline_logits: torch.Tensor,
        ser_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.size() != baseline_logits.size():
            msg = "logits and baseline_logits must share shape"
            raise ValueError(msg)
        if logits.size() != ser_labels.size():
            msg = "logits and ser_labels must share shape"
            raise ValueError(msg)
        if logits.size() != mask.size():
            msg = "logits and mask must share shape"
            raise ValueError(msg)

        mask_bool = mask.to(dtype=torch.bool)

        soft_ranks = self._soft_rank(
            logits,
            mask_bool,
            tau=self.soft_rank_strength,
        )
        discounts = 1.0 / torch.log2(soft_ranks + 1.0)
        cutoff = torch.sigmoid((self.k + 0.5 - soft_ranks) / self.cutoff_temperature)
        gains = ser_labels * mask_bool.to(ser_labels.dtype)
        dcg = (gains * discounts * cutoff).sum(dim=-1)

        hits = gains.sum(dim=-1)
        device = logits.device
        dtype = logits.dtype
        discount_vec = 1.0 / torch.log2(
            torch.arange(2, self.k + 2, device=device, dtype=dtype)
        )
        cumsum_discounts = torch.cumsum(discount_vec, dim=0)
        hits_int = hits.clamp(min=0.0, max=float(self.k)).long()
        safe_indices = torch.clamp(hits_int - 1, min=-1)
        idcg = torch.where(
            hits_int > 0,
            cumsum_discounts.index_select(0, torch.clamp(safe_indices, min=0)),
            torch.zeros_like(hits, dtype=dtype),
        )
        safe_idcg = torch.clamp(idcg, min=_EPS)
        ndcg = torch.where(idcg > 0, dcg / safe_idcg, torch.zeros_like(dcg))

        p = self._masked_softmax(logits, mask_bool)
        q = self._masked_softmax(baseline_logits, mask_bool)
        kl = (p * (torch.log(p + _EPS) - torch.log(q + _EPS))).sum(dim=-1)

        loss = -ndcg.mean() + self.beta * kl.mean()
        metrics = {
            "train/ndcg_ser_soft": ndcg.mean(),
            "train/kl_to_base": kl.mean(),
        }
        return loss, metrics
