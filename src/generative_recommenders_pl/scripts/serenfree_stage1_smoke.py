"""CPU smoke test for Stage 1 HSTU-SerenFree building blocks."""

from __future__ import annotations

import torch

from generative_recommenders_pl.models.serenfree import (
    HSTUStateWrapper,
    SIDComposer,
    SIDTrie,
    SharedPrefixDecoder,
    constrained_beam_search,
    relevance_loss,
)
from generative_recommenders_pl.models.sequential_encoders.hstu import HSTU


class _IdentityEncoder(torch.nn.Module):
    def forward(
        self,
        past_lengths: torch.Tensor,
        user_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        past_payloads: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, None]:
        del past_lengths, valid_mask, past_payloads
        return user_embeddings, None


def _build_tiny_hstu(embedding_dim: int) -> HSTU:
    return HSTU(
        max_sequence_len=3,
        max_output_len=0,
        embedding_dim=embedding_dim,
        item_embedding_dim=embedding_dim,
        num_blocks=1,
        num_heads=1,
        linear_dim=embedding_dim,
        attention_dim=embedding_dim,
        normalization="rel_bias",
        linear_config="uvqk",
        linear_activation="silu",
        linear_dropout_rate=0.0,
        attn_dropout_rate=0.0,
        enable_relative_attention_bias=False,
    )


def _run_smoke(sequence_encoder: torch.nn.Module, label: str) -> str:
    sid_tokens = torch.tensor(
        [
            [[1, 1, 1, 1], [1, 2, 3, 4], [2, 1, 3, 5]],
            [[1, 2, 3, 4], [2, 1, 3, 5], [0, 0, 0, 0]],
        ]
    )
    past_lengths = torch.tensor([3, 2])
    composer = SIDComposer(
        q1_size=6,
        q2_size=6,
        q3_size=6,
        dedup_size=6,
        embedding_dim=8,
    )
    user_embeddings = composer(sid_tokens)
    valid_mask = (sid_tokens[..., 0] != 0).unsqueeze(-1).float()
    wrapper = HSTUStateWrapper(sequence_encoder, recent_window=2)
    state = wrapper(
        past_lengths=past_lengths,
        user_embeddings=user_embeddings,
        valid_mask=valid_mask,
        past_payloads={},
    )

    decoder = SharedPrefixDecoder(
        hidden_dim=8,
        q1_size=6,
        q2_size=6,
        q3_size=6,
        dedup_size=6,
    )
    targets = torch.tensor([[1, 2, 3, 4], [2, 1, 3, 5]])
    decoded = decoder(state.recent_state, targets[:, :3])
    loss = relevance_loss(decoded, targets)
    loss.backward()

    trie = SIDTrie.from_items(
        item_ids=torch.tensor([100, 200, 300]),
        sid_tokens=torch.tensor([[1, 1, 1, 1], [1, 2, 3, 4], [2, 1, 3, 5]]),
    )
    beams = constrained_beam_search(decoder, state.recent_state, trie, beam_size=2)
    if not all(row for row in beams):
        raise RuntimeError("beam search returned an empty row")
    return (
        f"{label}:hidden={tuple(state.hidden_states.shape)} "
        f"loss={loss.item():.4f} beam0={beams[0][0].sid}:{beams[0][0].item_id}"
    )


def main() -> None:
    torch.manual_seed(7)
    identity_result = _run_smoke(_IdentityEncoder(), "identity")
    hstu_result = _run_smoke(_build_tiny_hstu(embedding_dim=8), "hstu")
    print("SERENFREE_STAGE1_CPU_SMOKE_OK", identity_result, hstu_result)


if __name__ == "__main__":
    main()
