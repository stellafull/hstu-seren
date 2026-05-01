import torch

from generative_recommenders_pl.models.hstu_serenfree import HSTUSerenFreeStage1
from generative_recommenders_pl.models.serenfree import (
    DecoderMode,
    HSTUStateWrapper,
    SIDComposer,
    SIDTrie,
    SharedPrefixDecoder,
    constrained_beam_search,
    relevance_loss,
    semantic_js_divergence,
    semantic_loss,
)
from generative_recommenders_pl.scripts.evaluate_serenfree_retrieval import AIGConfig


class _OffsetEncoder(torch.nn.Module):
    def forward(
        self,
        past_lengths,
        user_embeddings,
        valid_mask,
        past_payloads,
        **kwargs,
    ):
        del past_lengths, valid_mask, past_payloads, kwargs
        return user_embeddings + 1.0, ["cache"]


def test_sid_composer_outputs_item_level_embeddings_and_masks_padding():
    composer = SIDComposer(
        q1_size=8,
        q2_size=9,
        q3_size=10,
        dedup_size=11,
        embedding_dim=6,
    )
    sid_tokens = torch.tensor(
        [
            [[1, 2, 3, 4], [0, 0, 0, 0]],
            [[1, 2, 3, 5], [2, 3, 4, 5]],
        ]
    )

    output = composer(sid_tokens)

    assert output.shape == (2, 2, 6)
    assert torch.all(output[0, 1] == 0)
    assert not torch.allclose(output[0, 0], output[1, 0])
    output.sum().backward()
    assert composer.semantic_embeddings[0].weight.grad is not None


def test_variable_depth_sid_modules_treat_last_column_as_dedup():
    vocab_sizes = [5, 6, 7, 8, 9]
    composer = SIDComposer(vocab_sizes=vocab_sizes, embedding_dim=6)
    decoder = SharedPrefixDecoder(hidden_dim=6, vocab_sizes=vocab_sizes)
    sid_tokens = torch.tensor([[1, 2, 3, 4, 1], [2, 3, 4, 5, 1]])

    embeddings = composer(sid_tokens)
    output = decoder(embeddings, sid_tokens[:, :-1])
    loss = relevance_loss(output, sid_tokens)

    assert embeddings.shape == (2, 6)
    assert len(output.semantic_logits) == 4
    assert output.dedup_logits.shape == (2, 9)
    assert loss.item() > 0


def test_shared_prefix_decoder_relevance_loss_backpropagates():
    decoder = SharedPrefixDecoder(
        hidden_dim=8,
        q1_size=5,
        q2_size=6,
        q3_size=7,
        dedup_size=8,
    )
    context = torch.randn(3, 8)
    targets = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ]
    )
    output = decoder(context, targets[:, :3], mode=DecoderMode.RELEVANCE)

    assert output.q1_logits.shape == (3, 5)
    assert output.q2_logits.shape == (3, 6)
    assert output.q3_logits.shape == (3, 7)
    assert output.dedup_logits.shape == (3, 8)

    loss = relevance_loss(output, targets, lambda_d=0.5)
    assert loss.item() > 0
    loss.backward()
    assert decoder.heads[0].weight.grad is not None


def test_relevance_loss_handles_all_padding_targets_without_nan():
    decoder = SharedPrefixDecoder(hidden_dim=4, vocab_sizes=[3, 3, 3])
    context = torch.randn(2, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = relevance_loss(decoder(context, targets[:, :-1]), targets)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_semantic_loss_excludes_final_dedup_head():
    decoder = SharedPrefixDecoder(hidden_dim=4, vocab_sizes=[5, 6, 7])
    context = torch.randn(2, 4)
    targets = torch.tensor([[1, 2, 3], [2, 3, 4]])

    output = decoder(context, targets[:, :-1], mode=DecoderMode.IMMINENT)
    loss = semantic_loss(output, targets)
    loss.backward()

    assert loss.item() > 0
    assert decoder.heads[0].weight.grad is not None
    assert decoder.heads[1].weight.grad is not None
    assert decoder.heads[2].weight.grad is None


def test_semantic_js_divergence_is_finite():
    decoder = SharedPrefixDecoder(hidden_dim=4, vocab_sizes=[5, 6, 7])
    context = torch.randn(2, 4)
    targets = torch.tensor([[1, 2, 3], [2, 3, 4]])

    left = decoder(context, targets[:, :-1], mode=DecoderMode.IMMINENT)
    right = decoder(context, targets[:, :-1], mode=DecoderMode.ACCEPTABLE)

    divergence = semantic_js_divergence(left, right)

    assert torch.isfinite(divergence)
    assert divergence.item() >= 0.0


def test_aig_config_biases_semantic_levels_not_dedup():
    config = AIGConfig(alpha=0.5, levels_zero_based=(0, 1), gate_min_acceptable_prob=0.0)

    assert config.applies_to(0)
    assert config.applies_to(1)
    assert not config.applies_to(2)
    assert not config.applies_to(3)


def test_gap_loss_uses_pseudo_ser_items_and_semantic_heads_only():
    decoder = SharedPrefixDecoder(hidden_dim=4, vocab_sizes=[5, 6, 7, 8])
    context = torch.randn(2, 4)
    target_sid = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    acceptable = decoder(context, target_sid[:, :-1], mode=DecoderMode.ACCEPTABLE)
    imminent = decoder(context, target_sid[:, :-1], mode=DecoderMode.IMMINENT)

    holder = object.__new__(HSTUSerenFreeStage1)
    holder.pseudo_ser_items = torch.tensor([20])

    loss = HSTUSerenFreeStage1._gap_loss(
        holder,
        acceptable,
        imminent,
        target_sid,
        torch.tensor([10, 20]),
    )

    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert decoder.heads[0].weight.grad is not None
    assert decoder.heads[1].weight.grad is not None
    assert decoder.heads[2].weight.grad is not None
    assert decoder.heads[3].weight.grad is None


def test_hstu_state_wrapper_exposes_recent_and_history_pools():
    wrapper = HSTUStateWrapper(_OffsetEncoder(), recent_window=2)
    user_embeddings = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    output = wrapper(
        past_lengths=torch.tensor([3, 4]),
        user_embeddings=user_embeddings,
        valid_mask=torch.ones(2, 4, 1),
        past_payloads={},
    )

    hidden = user_embeddings + 1.0
    assert torch.equal(output.hidden_states, hidden)
    assert output.cache_states == ["cache"]
    assert torch.allclose(output.history_state[0], hidden[0, :3].mean(dim=0))
    assert torch.allclose(output.history_state[1], hidden[1, :4].mean(dim=0))
    assert torch.allclose(output.recent_state[0], hidden[0, 1:3].mean(dim=0))
    assert torch.allclose(output.recent_state[1], hidden[1, 2:4].mean(dim=0))


def test_sid_trie_constrained_beam_search_returns_only_valid_items():
    decoder = SharedPrefixDecoder(
        hidden_dim=8,
        q1_size=6,
        q2_size=6,
        q3_size=6,
        dedup_size=6,
    )
    item_ids = torch.tensor([10, 20, 30])
    sid_tokens = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 2, 3, 4],
            [2, 1, 3, 5],
        ]
    )
    trie = SIDTrie.from_items(item_ids, sid_tokens)
    context = torch.randn(2, 8)

    results = constrained_beam_search(decoder, context, trie, beam_size=2)

    assert len(results) == 2
    valid_items = set(item_ids.tolist())
    valid_sids = {tuple(row.tolist()) for row in sid_tokens}
    for row in results:
        assert 1 <= len(row) <= 2
        for beam in row:
            assert beam.item_id in valid_items
            assert beam.sid in valid_sids


if __name__ == "__main__":
    test_sid_composer_outputs_item_level_embeddings_and_masks_padding()
    test_variable_depth_sid_modules_treat_last_column_as_dedup()
    test_shared_prefix_decoder_relevance_loss_backpropagates()
    test_relevance_loss_handles_all_padding_targets_without_nan()
    test_semantic_loss_excludes_final_dedup_head()
    test_semantic_js_divergence_is_finite()
    test_aig_config_biases_semantic_levels_not_dedup()
    test_gap_loss_uses_pseudo_ser_items_and_semantic_heads_only()
    test_hstu_state_wrapper_exposes_recent_and_history_pools()
    test_sid_trie_constrained_beam_search_returns_only_valid_items()
    print("SERENFREE_STAGE1_TESTS_OK")
