"""Item-level full-SID composer for HSTU-SerenFree."""

from __future__ import annotations

import torch


class SIDComposer(torch.nn.Module):
    """Compose `(q1, q2, q3, d)` into one item-level embedding.

    The first three semantic levels are concatenated and projected. The dedup
    token conditions the semantic embedding with a gated additive block, keeping
    the history item-level as required by V1.
    """

    def __init__(
        self,
        q1_size: int,
        q2_size: int,
        q3_size: int,
        dedup_size: int,
        embedding_dim: int,
        hidden_dim: int | None = None,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if min(q1_size, q2_size, q3_size, dedup_size) <= padding_idx:
            raise ValueError("vocabulary sizes must be greater than padding_idx")

        self.embedding_dim = int(embedding_dim)
        self.padding_idx = int(padding_idx)
        hidden = int(hidden_dim or embedding_dim * 2)

        self.q1_embedding = torch.nn.Embedding(
            q1_size, embedding_dim, padding_idx=padding_idx
        )
        self.q2_embedding = torch.nn.Embedding(
            q2_size, embedding_dim, padding_idx=padding_idx
        )
        self.q3_embedding = torch.nn.Embedding(
            q3_size, embedding_dim, padding_idx=padding_idx
        )
        self.dedup_embedding = torch.nn.Embedding(
            dedup_size, embedding_dim, padding_idx=padding_idx
        )
        self.semantic_mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 3, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, embedding_dim),
        )
        self.dedup_gate = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.dedup_delta = torch.nn.Linear(embedding_dim, embedding_dim)
        self.output_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, sid_tokens: torch.Tensor) -> torch.Tensor:
        if sid_tokens.size(-1) != 4:
            raise ValueError(
                "sid_tokens must end with four values: (q1, q2, q3, d)"
            )
        sid_tokens = sid_tokens.to(torch.long)
        q1, q2, q3, dedup = sid_tokens.unbind(dim=-1)

        q_embeddings = torch.cat(
            [
                self.q1_embedding(q1),
                self.q2_embedding(q2),
                self.q3_embedding(q3),
            ],
            dim=-1,
        )
        semantic = self.semantic_mlp(q_embeddings)
        dedup_embedding = self.dedup_embedding(dedup)
        gate = torch.sigmoid(self.dedup_gate(torch.cat([semantic, dedup_embedding], dim=-1)))
        composed = semantic + gate * self.dedup_delta(dedup_embedding)
        composed = self.output_norm(composed)

        padding_mask = (sid_tokens == self.padding_idx).all(dim=-1, keepdim=True)
        return composed.masked_fill(padding_mask, 0.0)
