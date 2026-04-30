"""Item-level full-SID composer for HSTU-SerenFree."""

from __future__ import annotations

import torch


class SIDComposer(torch.nn.Module):
    """Compose variable-depth full SIDs into one item-level embedding.

    `vocab_sizes` contains one entry per SID column. The final column is always
    the dedup slot; all preceding columns are semantic codebook levels.
    """

    def __init__(
        self,
        vocab_sizes: list[int] | tuple[int, ...] | None = None,
        embedding_dim: int = 256,
        hidden_dim: int | None = None,
        padding_idx: int = 0,
        q1_size: int | None = None,
        q2_size: int | None = None,
        q3_size: int | None = None,
        dedup_size: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_sizes is None:
            if None in {q1_size, q2_size, q3_size, dedup_size}:
                raise ValueError("vocab_sizes or q1/q2/q3/dedup sizes are required")
            vocab_sizes = [int(q1_size), int(q2_size), int(q3_size), int(dedup_size)]
        if len(vocab_sizes) < 2:
            raise ValueError("full SID must include at least one semantic level and dedup")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if min(vocab_sizes) <= padding_idx:
            raise ValueError("all vocabulary sizes must be greater than padding_idx")

        self.vocab_sizes = [int(size) for size in vocab_sizes]
        self.num_sid_columns = len(self.vocab_sizes)
        self.num_semantic_levels = self.num_sid_columns - 1
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = int(padding_idx)
        hidden = int(hidden_dim or embedding_dim * 2)

        self.semantic_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(size, embedding_dim, padding_idx=padding_idx)
                for size in self.vocab_sizes[:-1]
            ]
        )
        self.dedup_embedding = torch.nn.Embedding(
            self.vocab_sizes[-1], embedding_dim, padding_idx=padding_idx
        )
        self.semantic_mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * self.num_semantic_levels, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, embedding_dim),
        )
        self.dedup_gate = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.dedup_delta = torch.nn.Linear(embedding_dim, embedding_dim)
        self.output_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, sid_tokens: torch.Tensor) -> torch.Tensor:
        if sid_tokens.size(-1) != self.num_sid_columns:
            raise ValueError(
                f"sid_tokens must end with {self.num_sid_columns} SID columns"
            )
        sid_tokens = sid_tokens.to(torch.long)
        semantic_tokens = sid_tokens[..., :-1]
        dedup = sid_tokens[..., -1]

        semantic_parts = [
            embedding(semantic_tokens[..., level])
            for level, embedding in enumerate(self.semantic_embeddings)
        ]
        semantic = self.semantic_mlp(torch.cat(semantic_parts, dim=-1))
        dedup_embedding = self.dedup_embedding(dedup)
        gate = torch.sigmoid(self.dedup_gate(torch.cat([semantic, dedup_embedding], dim=-1)))
        composed = semantic + gate * self.dedup_delta(dedup_embedding)
        composed = self.output_norm(composed)

        padding_mask = (sid_tokens == self.padding_idx).all(dim=-1, keepdim=True)
        return composed.masked_fill(padding_mask, 0.0)
