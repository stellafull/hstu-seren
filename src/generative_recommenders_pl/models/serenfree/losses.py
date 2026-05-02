"""V2 label-free SerenFree losses and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TargetTrieNode:
    children: dict[int, "TargetTrieNode"]


def _insert(root: TargetTrieNode, sid: tuple[int, ...]) -> None:
    node = root
    for token in sid:
        node = node.children.setdefault(int(token), TargetTrieNode({}))


def build_target_trie(target_sids: torch.Tensor, levels: tuple[int, ...] | None = None) -> TargetTrieNode:
    if target_sids.dim() != 2:
        raise ValueError("target_sids must be [targets, sid_columns]")
    if levels is None:
        levels = tuple(range(target_sids.size(1)))
    root = TargetTrieNode({})
    seen: set[tuple[int, ...]] = set()
    for row in target_sids.to(torch.long).tolist():
        sid = tuple(int(row[level]) for level in levels)
        if any(token == 0 for token in sid) or sid in seen:
            continue
        seen.add(sid)
        _insert(root, sid)
    return root


def trie_marginal_nll(
    level_logits_fn: Callable[[tuple[int, ...], int], torch.Tensor],
    target_sids: torch.Tensor,
    levels: tuple[int, ...],
) -> torch.Tensor:
    """-log sum over target SID probabilities with prefix-conditioned logits."""
    root = build_target_trie(target_sids, levels=levels)
    if not root.children:
        return torch.zeros((), device=target_sids.device, dtype=torch.float32)
    frontier: list[tuple[TargetTrieNode, tuple[int, ...], torch.Tensor]] = [
        (root, (), torch.zeros((), device=target_sids.device, dtype=torch.float32))
    ]
    for level in levels:
        next_frontier: list[tuple[TargetTrieNode, tuple[int, ...], torch.Tensor]] = []
        for node, prefix, node_logp in frontier:
            logits = level_logits_fn(prefix, int(level))
            log_probs = F.log_softmax(logits, dim=-1)
            for token, child in sorted(node.children.items()):
                next_frontier.append((child, (*prefix, int(token)), node_logp + log_probs[int(token)]))
        frontier = next_frontier
    return -torch.logsumexp(torch.stack([score for _, _, score in frontier]), dim=0)


def trie_marginal_nll_from_log_probs(
    log_probs_by_level: list[torch.Tensor],
    target_sids: torch.Tensor,
    levels: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Same loss for tests when each prefix-node log-prob table is precomputed."""
    if levels is None:
        levels = tuple(range(target_sids.size(1)))
    if len(log_probs_by_level) != len(levels):
        raise ValueError("log_probs_by_level must align with levels")
    root = build_target_trie(target_sids, levels=levels)
    if not root.children:
        return log_probs_by_level[0].new_tensor(0.0)
    frontier: list[tuple[TargetTrieNode, torch.Tensor]] = [(root, log_probs_by_level[0].new_tensor(0.0))]
    for log_probs in log_probs_by_level:
        if log_probs.size(0) < len(frontier):
            raise ValueError("not enough prefix-node rows")
        nxt: list[tuple[TargetTrieNode, torch.Tensor]] = []
        for row_idx, (node, node_logp) in enumerate(frontier):
            for token, child in sorted(node.children.items()):
                nxt.append((child, node_logp + log_probs[row_idx, int(token)]))
        frontier = nxt
    return -torch.logsumexp(torch.stack([score for _, score in frontier]), dim=0)


def lf_rank_loss(
    s_r: torch.Tensor,
    s_a: torch.Tensor,
    s_i: torch.Tensor,
    geometry: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    theta_ai: float = 1.0,
    theta_g: float = 1.0,
    temperature: float = 0.1,
    stopgrad_r: bool = True,
) -> torch.Tensor:
    if positive_mask.dim() == 1:
        positive_mask = positive_mask.unsqueeze(0)
        s_r = s_r.unsqueeze(0)
        s_a = s_a.unsqueeze(0)
        s_i = s_i.unsqueeze(0)
        geometry = geometry.unsqueeze(0)
    positive_mask = positive_mask.bool()
    rows_with_positive = positive_mask.any(dim=-1)
    if not rows_with_positive.any():
        return s_r.sum() * 0.0
    base_r = s_r.detach() if stopgrad_r else s_r
    z = base_r + float(theta_ai) * (s_a - s_i) + float(theta_g) * geometry
    scaled = z / float(temperature)
    scaled = scaled[rows_with_positive]
    row_positive_mask = positive_mask[rows_with_positive]
    pos_mass = torch.logsumexp(scaled.masked_fill(~row_positive_mask, -torch.inf), dim=-1)
    all_mass = torch.logsumexp(scaled, dim=-1)
    return -(pos_mass - all_mass).mean()


def ai_collapse_diagnostics(
    acceptable_logits: list[torch.Tensor],
    imminent_logits: list[torch.Tensor],
    target_a: torch.Tensor | None = None,
    target_i: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for idx, (a_logits, i_logits) in enumerate(zip(acceptable_logits, imminent_logits), start=1):
        a_log = F.log_softmax(a_logits, dim=-1)
        i_log = F.log_softmax(i_logits, dim=-1)
        a = a_log.exp(); i = i_log.exp(); m = 0.5 * (a + i)
        m_log = m.clamp_min(torch.finfo(m.dtype).tiny).log()
        metrics[f"js_A_I_level{idx}"] = 0.5 * ((a * (a_log - m_log)).sum(-1) + (i * (i_log - m_log)).sum(-1)).mean()
        metrics[f"entropy_A_level{idx}"] = -(a * a_log).sum(-1).mean()
        metrics[f"entropy_I_level{idx}"] = -(i * i_log).sum(-1).mean()
        if target_a is not None and target_a.size(-1) >= idx:
            tok = target_a[..., idx - 1].to(torch.long)
            gap = (a_log - i_log).gather(-1, tok.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            metrics[f"mean_A_minus_I_on_A_targets_level{idx}"] = gap[tok != 0].mean() if (tok != 0).any() else gap.sum() * 0.0
        if target_i is not None and target_i.size(-1) >= idx:
            tok = target_i[..., idx - 1].to(torch.long)
            gap = (a_log - i_log).gather(-1, tok.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            metrics[f"mean_A_minus_I_on_I_targets_level{idx}"] = gap[tok != 0].mean() if (tok != 0).any() else gap.sum() * 0.0
    return metrics
