from __future__ import annotations

import torch


def gumbel_top_k(logits: torch.Tensor, k: int, tau: float = 1.0) -> torch.Tensor:
    """Sample top-k indices using the Gumbel-Top-k trick without replacement."""

    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)
    if logits.size(-1) < k:
        msg = "k cannot exceed logits dimension"
        raise ValueError(msg)

    noise = torch.rand_like(logits)
    gumbels = -torch.log(-torch.log(noise + 1e-12) + 1e-12)
    scores = logits + tau * gumbels
    return scores.topk(k, dim=-1).indices


class CandidateSetBuilder(torch.nn.Module):
    """Builds candidate pools for serendipity-aware reranking."""

    def __init__(
        self,
        candidate_size: int,
        ensure_target: bool = True,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if candidate_size <= 0:
            msg = "candidate_size must be positive"
            raise ValueError(msg)
        self.candidate_size = candidate_size
        self.ensure_target = ensure_target
        self.pad_id = pad_id

    def forward(
        self,
        query_embeddings: torch.Tensor,
        candidate_index,
        invalid_ids: torch.Tensor | None,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = min(self.candidate_size, candidate_index.num_objects)
        batch_size = query_embeddings.size(0)
        top_ids, _ = candidate_index.get_top_k_outputs(
            query_embeddings=query_embeddings,
            k=top_k,
            invalid_ids=invalid_ids,
        )

        if top_ids.size(1) < self.candidate_size:
            pad_width = self.candidate_size - top_ids.size(1)
            pad_tensor = torch.full(
                (batch_size, pad_width),
                fill_value=self.pad_id,
                dtype=top_ids.dtype,
                device=top_ids.device,
            )
            top_ids = torch.cat([top_ids, pad_tensor], dim=1)

        candidate_ids = top_ids[:, : self.candidate_size]
        target_ids = target_ids.view(batch_size, 1)
        candidate_mask = candidate_ids != self.pad_id
        candidate_mask |= candidate_ids == target_ids

        if not self.ensure_target:
            return candidate_ids, candidate_mask

        has_target = (candidate_ids == target_ids).any(dim=1, keepdim=True)
        if has_target.all():
            return candidate_ids, candidate_mask

        # Allocate an extra slot to append the missing targets without displacing retrieved negatives.
        pad_column_ids = torch.full(
            (batch_size, 1),
            fill_value=self.pad_id,
            dtype=candidate_ids.dtype,
            device=candidate_ids.device,
        )
        pad_column_mask = torch.zeros(
            (batch_size, 1),
            dtype=torch.bool,
            device=candidate_mask.device,
        )
        candidate_ids = torch.cat([candidate_ids, pad_column_ids], dim=1)
        candidate_mask = torch.cat([candidate_mask, pad_column_mask], dim=1)

        missing_rows = ~has_target.squeeze(1)
        candidate_ids[missing_rows, -1] = target_ids[missing_rows, 0]
        candidate_mask[missing_rows, -1] = True

        missing_after_append = ~candidate_mask.any(dim=1)
        if missing_after_append.any():
            candidate_mask[missing_after_append, -1] = True

        return candidate_ids, candidate_mask
