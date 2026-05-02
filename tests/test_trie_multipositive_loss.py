import itertools
import torch
import torch.nn.functional as F

from generative_recommenders_pl.models.serenfree.losses import lf_rank_loss, trie_marginal_nll


def test_trie_marginal_nll_matches_bruteforce():
    torch.manual_seed(0)
    vocab = 4
    logits = {
        ((), 0): torch.randn(vocab),
        ((1,), 1): torch.randn(vocab),
        ((2,), 1): torch.randn(vocab),
        ((1, 2), 2): torch.randn(vocab),
        ((1, 3), 2): torch.randn(vocab),
        ((2, 1), 2): torch.randn(vocab),
    }
    targets = torch.tensor([[1, 2, 3], [1, 3, 2], [2, 1, 1]])

    def fn(prefix, level):
        return logits[(tuple(prefix), level)]

    loss = trie_marginal_nll(fn, targets, (0, 1, 2))
    masses = []
    for sid in targets.tolist():
        prefix = ()
        score = 0.0
        for level, tok in enumerate(sid):
            score = score + F.log_softmax(logits[(prefix, level)], dim=-1)[tok]
            prefix = (*prefix, tok)
        masses.append(score)
    expected = -torch.logsumexp(torch.stack(masses), dim=0)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_lf_rank_stopgrad_r():
    s_r = torch.tensor([[1.0, 0.5]], requires_grad=True)
    s_a = torch.tensor([[0.1, 1.0]], requires_grad=True)
    s_i = torch.tensor([[0.2, 0.1]], requires_grad=True)
    g = torch.tensor([[0.0, 0.5]], requires_grad=True)
    loss = lf_rank_loss(s_r, s_a, s_i, g, torch.tensor([[False, True]]), stopgrad_r=True)
    loss.backward()
    assert s_r.grad is None
    assert s_a.grad is not None


def test_lf_rank_skips_rows_without_positive_candidates():
    s_r = torch.tensor([[1.0, 0.5], [0.2, 0.1]], requires_grad=True)
    s_a = torch.tensor([[0.1, 1.0], [0.0, 0.0]], requires_grad=True)
    s_i = torch.tensor([[0.2, 0.1], [0.0, 0.0]], requires_grad=True)
    g = torch.zeros_like(s_r, requires_grad=True)
    mask = torch.tensor([[False, True], [False, False]])
    loss = lf_rank_loss(s_r, s_a, s_i, g, mask, stopgrad_r=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert s_a.grad is not None
