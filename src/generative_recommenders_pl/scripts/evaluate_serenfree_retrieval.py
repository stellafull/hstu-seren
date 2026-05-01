"""Evaluate SerenFree full-SID generation with retrieval metrics.

This script does not build semantic IDs. It evaluates an existing
HSTU-SerenFree checkpoint by constrained full-SID decoding, then computes
HR/NDCG@K against dataloader target item ids.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from generative_recommenders_pl.models.serenfree import DecoderMode, SIDBeam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--experiment", default="serenfree_stage1_amazon_movies")
    parser.add_argument("--config-name", default="train.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--no-filter-history", action="store_true")
    parser.add_argument("--allow-duplicate-sids", action="store_true")
    parser.add_argument("--aig", action="store_true")
    parser.add_argument("--aig-alpha", type=float, default=None)
    parser.add_argument("--aig-levels", type=int, nargs="+", default=None)
    parser.add_argument("--aig-gate-min-acceptable-prob", type=float, default=None)
    parser.add_argument("--pseudo-ser-path", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override, e.g. --override data.batch_size=128",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    apply_eval_config(args, cfg)
    ks = sorted(set(args.ks))
    if ks[-1] > args.beam_size:
        raise ValueError("beam-size must be at least the largest requested K")

    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)
    # Lightning checkpoints include OmegaConf metadata; this is a locally
    # produced checkpoint, so load the full object instead of weights-only.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        print(
            json.dumps(
                {"state_dict_missing": missing, "state_dict_unexpected": unexpected},
                indent=2,
            ),
            flush=True,
        )

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    datamodule.setup("fit" if args.split in {"train", "val"} else "test")
    dataloader = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }[args.split]()

    sid_index, trie_audit = build_sid_transition_index(
        model.sid_lookup.detach().cpu(),
        allow_duplicate_sids=args.allow_duplicate_sids,
    )
    audit = audit_pipeline(model=model, trie_audit=trie_audit, cfg=cfg, args=args)
    aig_config = AIGConfig.from_args(args)
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        sid_index=sid_index,
        ks=ks,
        beam_size=args.beam_size,
        device=device,
        filter_history=args.filter_history,
        max_batches=args.max_batches,
        progress_every=args.progress_every,
        aig=aig_config,
        pseudo_ser_labels=load_pseudo_ser_labels(args.pseudo_ser_path),
    )
    output = {"audit": audit, "metrics": metrics}
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def load_config(args: argparse.Namespace) -> Any:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    repo_root = Path(__file__).resolve().parents[3]
    overrides = [f"experiment={args.experiment}", *args.override]
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name=args.config_name, overrides=overrides)


def apply_eval_config(args: argparse.Namespace, cfg: Any) -> None:
    eval_cfg = cfg.get("serenfree_eval", None)
    eval_cfg = (
        OmegaConf.to_container(eval_cfg, resolve=True)
        if eval_cfg is not None
        else {}
    )

    def configured(name: str, default: Any) -> Any:
        return eval_cfg.get(name, default)

    if args.checkpoint is None:
        checkpoint = configured("checkpoint", None)
        if checkpoint is None:
            raise ValueError(
                "Evaluation checkpoint is required; pass --checkpoint or set "
                "serenfree_eval.checkpoint in the experiment YAML."
            )
        args.checkpoint = Path(str(checkpoint))
    if args.split is None:
        args.split = str(configured("split", "test"))
    if args.device is None:
        args.device = str(
            configured("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
    if args.beam_size is None:
        args.beam_size = int(configured("beam_size", 400))
    if args.ks is None:
        args.ks = [int(k) for k in configured("ks", [10, 50, 100, 200])]
    if args.max_batches is None:
        max_batches = configured("max_batches", None)
        args.max_batches = None if max_batches is None else int(max_batches)
    if args.output_json is None:
        output_json = configured("output_json", None)
        args.output_json = None if output_json is None else Path(str(output_json))
    args.filter_history = (
        False if args.no_filter_history else bool(configured("filter_history", True))
    )
    args.allow_duplicate_sids = bool(
        args.allow_duplicate_sids or configured("allow_duplicate_sids", False)
    )
    args.aig = bool(args.aig or configured("aig", False))
    if args.aig_alpha is None:
        args.aig_alpha = float(configured("aig_alpha", 0.0))
    if args.aig_levels is None:
        args.aig_levels = [
            int(level) for level in configured("aig_levels", [1, 2])
        ]
    if args.aig_gate_min_acceptable_prob is None:
        args.aig_gate_min_acceptable_prob = float(
            configured("aig_gate_min_acceptable_prob", 0.0)
        )
    if args.pseudo_ser_path is None:
        pseudo_ser_path = configured("pseudo_ser_path", None)
        args.pseudo_ser_path = (
            None if pseudo_ser_path is None else Path(str(pseudo_ser_path))
        )
    if args.progress_every is None:
        args.progress_every = int(configured("progress_every", 10))


@dataclass(frozen=True)
class SIDTransitionIndex:
    """Dense transition tables for valid full-SID constrained decoding."""

    num_sid_columns: int
    child_tokens: list[torch.Tensor]
    child_next_nodes: list[torch.Tensor]
    child_item_ids: list[torch.Tensor]
    child_valid: list[torch.Tensor]

    def to(self, device: torch.device) -> "SIDTransitionIndex":
        return SIDTransitionIndex(
            num_sid_columns=self.num_sid_columns,
            child_tokens=[tensor.to(device) for tensor in self.child_tokens],
            child_next_nodes=[tensor.to(device) for tensor in self.child_next_nodes],
            child_item_ids=[tensor.to(device) for tensor in self.child_item_ids],
            child_valid=[tensor.to(device) for tensor in self.child_valid],
        )


@dataclass(frozen=True)
class EvalContexts:
    """Decoder contexts for relevance, imminent, and acceptable modes."""

    relevance: torch.Tensor
    imminent: torch.Tensor
    acceptable: torch.Tensor


def build_sid_transition_index(
    sid_lookup: torch.Tensor,
    allow_duplicate_sids: bool,
) -> tuple[SIDTransitionIndex, dict[str, Any]]:
    if sid_lookup.dim() != 2 or sid_lookup.size(1) < 2:
        raise ValueError("sid_lookup must have shape [items, semantic levels + dedup]")
    valid = (sid_lookup != 0).all(dim=1)
    item_ids = torch.arange(sid_lookup.size(0), dtype=torch.long)[valid]
    sid_tokens = sid_lookup[valid].to(torch.long)

    sid_tuples = [tuple(int(token) for token in row.tolist()) for row in sid_tokens]
    duplicate_count = len(sid_tuples) - len(set(sid_tuples))
    if duplicate_count and not allow_duplicate_sids:
        raise ValueError(
            f"Found {duplicate_count} duplicate full SIDs. "
            "Dedup should make full SIDs unique; pass --allow-duplicate-sids only "
            "if duplicate handling is intentional."
        )

    sid_index = build_transition_tables(item_ids=item_ids, sid_tokens=sid_tokens)
    audit = {
        "candidate_items": int(item_ids.numel()),
        "duplicate_full_sid_count": int(duplicate_count),
        "sid_columns": int(sid_lookup.size(1)),
        "semantic_levels": int(sid_lookup.size(1) - 1),
        "dedup_column": int(sid_lookup.size(1) - 1),
        "zero_padding_rows": int((~valid).sum().item()),
        "max_token_by_column": sid_lookup.max(dim=0).values.tolist(),
        "min_nonzero_token_by_column": [
            int(sid_lookup[:, idx][sid_lookup[:, idx] != 0].min().item())
            for idx in range(sid_lookup.size(1))
        ],
    }
    return sid_index, audit


def build_transition_tables(
    item_ids: torch.Tensor,
    sid_tokens: torch.Tensor,
) -> SIDTransitionIndex:
    num_sid_columns = int(sid_tokens.size(1))
    prefix_to_node: list[dict[tuple[int, ...], int]] = [
        {} for _ in range(num_sid_columns)
    ]
    prefix_to_node[0][()] = 0
    children_by_level: list[dict[int, dict[int, tuple[int, int]]]] = [
        defaultdict(dict) for _ in range(num_sid_columns)
    ]

    seen_full_sids: set[tuple[int, ...]] = set()
    for item_id, sid_row in zip(item_ids.tolist(), sid_tokens.tolist()):
        sid = tuple(int(token) for token in sid_row)
        if sid in seen_full_sids:
            continue
        seen_full_sids.add(sid)

        prefix: tuple[int, ...] = ()
        node = 0
        for level, token in enumerate(sid):
            if level < num_sid_columns - 1:
                next_prefix = (*prefix, token)
                next_node = prefix_to_node[level + 1].setdefault(
                    next_prefix,
                    len(prefix_to_node[level + 1]),
                )
                children_by_level[level][node][token] = (next_node, -1)
                prefix = next_prefix
                node = next_node
            else:
                children_by_level[level][node][token] = (-1, int(item_id))

    child_tokens: list[torch.Tensor] = []
    child_next_nodes: list[torch.Tensor] = []
    child_item_ids: list[torch.Tensor] = []
    child_valid: list[torch.Tensor] = []
    for level in range(num_sid_columns):
        num_nodes = len(prefix_to_node[level])
        max_children = max(
            (len(children_by_level[level].get(node, {})) for node in range(num_nodes)),
            default=0,
        )
        if max_children <= 0:
            raise ValueError(f"SID transition level {level} has no children")

        tokens = torch.zeros((num_nodes, max_children), dtype=torch.long)
        next_nodes = torch.zeros((num_nodes, max_children), dtype=torch.long)
        item_tensor = torch.full((num_nodes, max_children), -1, dtype=torch.long)
        valid = torch.zeros((num_nodes, max_children), dtype=torch.bool)
        for node in range(num_nodes):
            children = sorted(children_by_level[level].get(node, {}).items())
            for col, (token, (next_node, item_id)) in enumerate(children):
                tokens[node, col] = int(token)
                next_nodes[node, col] = max(int(next_node), 0)
                item_tensor[node, col] = int(item_id)
                valid[node, col] = True

        child_tokens.append(tokens)
        child_next_nodes.append(next_nodes)
        child_item_ids.append(item_tensor)
        child_valid.append(valid)

    return SIDTransitionIndex(
        num_sid_columns=num_sid_columns,
        child_tokens=child_tokens,
        child_next_nodes=child_next_nodes,
        child_item_ids=child_item_ids,
        child_valid=child_valid,
    )


def audit_pipeline(
    model: torch.nn.Module,
    trie_audit: dict[str, Any],
    cfg: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "checkpoint": str(args.checkpoint),
        "experiment": args.experiment,
        "split": args.split,
        "filter_history": bool(args.filter_history),
        "beam_size": int(args.beam_size),
        "ks": args.ks,
        "model_class": type(model).__name__,
        "full_sid_in": hasattr(model, "sid_composer") and not hasattr(model, "embeddings"),
        "full_sid_out": hasattr(model, "decoder")
        and len(model.decoder.heads) == trie_audit["sid_columns"],
        "item_level_hstu": hasattr(model, "sequence_encoder"),
        "aig_bias_supported": True,
        "aig_enabled": bool(args.aig),
        "aig_alpha": float(args.aig_alpha),
        "aig_levels": args.aig_levels,
        "aig_touches_dedup": (trie_audit["dedup_column"] + 1) in args.aig_levels,
        "aig_gate_min_acceptable_prob": float(args.aig_gate_min_acceptable_prob),
        "pseudo_ser_path": None
        if args.pseudo_ser_path is None
        else str(args.pseudo_ser_path),
        "dedup_relevance_only_for_stage1": True,
        "sid_lookup_path": str(cfg.model.get("sid_lookup_path")),
        "sid_path": str(cfg.model.sid_path),
        **trie_audit,
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    sid_index: SIDTransitionIndex,
    ks: list[int],
    beam_size: int,
    device: torch.device,
    filter_history: bool,
    max_batches: int | None,
    progress_every: int,
    aig: "AIGConfig | None",
    pseudo_ser_labels: set[int] | None,
) -> dict[str, Any]:
    sid_index = sid_index.to(device)
    totals = {
        "all": new_totals(ks),
        "eligible": new_totals(ks),
        "ser": new_totals(ks, prefix="_ser"),
    }
    counts = defaultdict(int)
    max_k = max(ks)
    started_at = time.monotonic()
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        contexts = encode_contexts(model=model, batch=batch, device=device)
        beams_by_row = batched_constrained_beam_search(
            decoder=model.decoder,
            contexts=contexts,
            sid_index=sid_index,
            beam_size=beam_size,
            aig=aig,
        )

        target_ids = batch["target_ids"].to(torch.long)
        history_ids = batch["historical_ids"].to(torch.long)
        target_ser_labels = batch.get("target_ser_label")
        target_sids = model._item_ids_to_sid(target_ids.to(device)).cpu()
        target_has_sid = (target_sids != 0).all(dim=1)

        for row_idx, beams in enumerate(beams_by_row):
            target = int(target_ids[row_idx].item())
            history = history_ids[row_idx]
            seen = {
                int(item)
                for item in history.tolist()
                if int(item) > 0
            }
            if target in seen:
                counts["target_in_history"] += 1
            ranked = rank_items(
                beams=beams,
                seen=seen,
                filter_history=filter_history,
                max_k=max_k,
            )
            update_totals(totals["all"], ranked, target, ks)
            counts["all"] += 1
            if bool(target_has_sid[row_idx].item()):
                update_totals(totals["eligible"], ranked, target, ks)
                counts["eligible"] += 1
            else:
                counts["unmapped_target"] += 1
            has_true_ser_label = (
                target_ser_labels is not None
                and int(target_ser_labels[row_idx].item()) == 1
            )
            has_pseudo_ser_label = pseudo_ser_labels is not None and target in pseudo_ser_labels
            if has_true_ser_label or has_pseudo_ser_label:
                update_totals(totals["ser"], ranked, target, ks, prefix="_ser")
                counts["ser_label"] += 1
                if has_true_ser_label:
                    counts["true_ser_label"] += 1
                if has_pseudo_ser_label:
                    counts["pseudo_ser_label"] += 1

        if progress_every > 0 and (batch_idx + 1) % progress_every == 0:
            elapsed = time.monotonic() - started_at
            seen_batches = batch_idx + 1
            progress = {
                "progress": {
                    "batch": seen_batches,
                    "batches": total_batches,
                    "examples": counts["all"],
                    "elapsed_sec": round(elapsed, 2),
                    "examples_per_sec": round(counts["all"] / max(elapsed, 1e-9), 2),
                }
            }
            print(json.dumps(progress, sort_keys=True), flush=True)

    result = {
        "counts": dict(counts),
        "all": finalize_totals(totals["all"], counts["all"], ks),
        "eligible_targets_only": finalize_totals(
            totals["eligible"],
            counts["eligible"],
            ks,
        ),
        "ser_targets_only": finalize_totals(
            totals["ser"],
            counts["ser_label"],
            ks,
            prefix="_ser",
        ),
    }
    return result


@torch.inference_mode()
def encode_contexts(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> EvalContexts:
    historical_ids = batch["historical_ids"].to(device)
    lengths = batch["history_lengths"].to(device)
    timestamps = batch["historical_timestamps"].to(device)

    history_sid = model._item_ids_to_sid(historical_ids)
    history_embeddings = model.sid_composer(history_sid)
    valid_mask = (history_sid != 0).any(dim=-1, keepdim=True).float()
    history_embeddings = model._add_position_embeddings(history_embeddings, valid_mask)
    encoded, _ = model.sequence_encoder(
        past_lengths=lengths,
        user_embeddings=history_embeddings,
        valid_mask=valid_mask,
        past_payloads={"timestamps": timestamps},
    )
    relevance = model._last_valid_state(encoded, lengths)
    if hasattr(model, "_masked_recent_mean"):
        imminent = model._masked_recent_mean(
            encoded,
            lengths,
            getattr(model, "recent_window", 10),
        )
    else:
        imminent = relevance
    acceptable = (
        model._masked_history_mean(encoded, lengths)
        if hasattr(model, "_masked_history_mean")
        else relevance
    )
    return EvalContexts(
        relevance=relevance,
        imminent=imminent,
        acceptable=acceptable,
    )


@torch.inference_mode()
def batched_constrained_beam_search(
    decoder: torch.nn.Module,
    contexts: EvalContexts,
    sid_index: SIDTransitionIndex,
    beam_size: int,
    aig: "AIGConfig | None" = None,
) -> list[list[SIDBeam]]:
    context = contexts.relevance
    batch_size = int(context.size(0))
    semantic_levels = int(decoder.num_semantic_levels)
    scores = context.new_zeros((batch_size, 1))
    node_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=context.device)
    prefix_tokens = torch.zeros(
        (batch_size, 1, semantic_levels),
        dtype=torch.long,
        device=context.device,
    )
    sid_tokens = torch.zeros(
        (batch_size, 1, sid_index.num_sid_columns),
        dtype=torch.long,
        device=context.device,
    )
    item_ids = torch.full((batch_size, 1), -1, dtype=torch.long, device=context.device)

    for level in range(sid_index.num_sid_columns):
        beam_count = int(scores.size(1))
        logits = decoder_level_logits(
            decoder=decoder,
            context=context.unsqueeze(1).expand(-1, beam_count, -1),
            prefix_tokens=prefix_tokens,
            level=level,
            mode=DecoderMode.RELEVANCE,
        )
        if aig is not None and aig.applies_to(level):
            acceptable_logits = decoder_level_logits(
                decoder=decoder,
                context=contexts.acceptable.unsqueeze(1).expand(-1, beam_count, -1),
                prefix_tokens=prefix_tokens,
                level=level,
                mode=DecoderMode.ACCEPTABLE,
            )
            imminent_logits = decoder_level_logits(
                decoder=decoder,
                context=contexts.imminent.unsqueeze(1).expand(-1, beam_count, -1),
                prefix_tokens=prefix_tokens,
                level=level,
                mode=DecoderMode.IMMINENT,
            )
            logits = apply_aig_bias(
                relevance_logits=logits,
                acceptable_logits=acceptable_logits,
                imminent_logits=imminent_logits,
                alpha=aig.alpha,
                gate_min_acceptable_prob=aig.gate_min_acceptable_prob,
            )
        log_probs = F.log_softmax(logits, dim=-1)

        flat_nodes = node_ids.reshape(-1)
        child_tokens = sid_index.child_tokens[level].index_select(0, flat_nodes)
        child_next_nodes = sid_index.child_next_nodes[level].index_select(0, flat_nodes)
        child_item_ids = sid_index.child_item_ids[level].index_select(0, flat_nodes)
        child_valid = sid_index.child_valid[level].index_select(0, flat_nodes)

        child_count = int(child_tokens.size(1))
        child_tokens = child_tokens.view(batch_size, beam_count, child_count)
        child_next_nodes = child_next_nodes.view(batch_size, beam_count, child_count)
        child_item_ids = child_item_ids.view(batch_size, beam_count, child_count)
        child_valid = child_valid.view(batch_size, beam_count, child_count)

        child_scores = log_probs.gather(dim=-1, index=child_tokens)
        candidate_scores = (scores.unsqueeze(-1) + child_scores).masked_fill(
            ~child_valid,
            -torch.inf,
        )
        flat_scores = candidate_scores.reshape(batch_size, -1)
        top_count = min(int(beam_size), int(flat_scores.size(1)))
        top_scores, top_pos = flat_scores.topk(top_count, dim=-1)
        finite = torch.isfinite(top_scores)
        parent_idx = torch.div(top_pos, child_count, rounding_mode="floor")
        selected_tokens = child_tokens.reshape(batch_size, -1).gather(1, top_pos)
        selected_next_nodes = child_next_nodes.reshape(batch_size, -1).gather(1, top_pos)
        selected_item_ids = child_item_ids.reshape(batch_size, -1).gather(1, top_pos)
        selected_tokens = selected_tokens.masked_fill(~finite, 0)
        selected_next_nodes = selected_next_nodes.masked_fill(~finite, 0)
        selected_item_ids = selected_item_ids.masked_fill(~finite, -1)

        sid_tokens = sid_tokens.gather(
            dim=1,
            index=parent_idx.unsqueeze(-1).expand(-1, -1, sid_index.num_sid_columns),
        )
        sid_tokens[:, :, level] = selected_tokens
        if level < semantic_levels:
            prefix_tokens = prefix_tokens.gather(
                dim=1,
                index=parent_idx.unsqueeze(-1).expand(-1, -1, semantic_levels),
            )
            prefix_tokens[:, :, level] = selected_tokens
        if level < sid_index.num_sid_columns - 1:
            node_ids = selected_next_nodes
        else:
            item_ids = selected_item_ids
        scores = top_scores

    scores_cpu = scores.detach().cpu()
    item_ids_cpu = item_ids.detach().cpu()
    sid_tokens_cpu = sid_tokens.detach().cpu()
    results: list[list[SIDBeam]] = []
    for row_idx in range(batch_size):
        row_results: list[SIDBeam] = []
        for beam_idx in range(item_ids_cpu.size(1)):
            item_id = int(item_ids_cpu[row_idx, beam_idx].item())
            score = float(scores_cpu[row_idx, beam_idx].item())
            if item_id < 0 or not math.isfinite(score):
                continue
            sid = tuple(int(token) for token in sid_tokens_cpu[row_idx, beam_idx].tolist())
            row_results.append(SIDBeam(item_id=item_id, sid=sid, score=score))
        results.append(row_results)
    return results


def decoder_level_logits(
    decoder: torch.nn.Module,
    context: torch.Tensor,
    prefix_tokens: torch.Tensor,
    level: int,
    mode: DecoderMode,
) -> torch.Tensor:
    shape = context.shape[:-1]
    mode_ids = torch.full(
        shape,
        int(mode),
        dtype=torch.long,
        device=context.device,
    )
    mode_emb = decoder.mode_embedding(mode_ids)
    empty_prefix = torch.zeros(
        (*shape, decoder.prefix_dim),
        dtype=context.dtype,
        device=context.device,
    )
    prefix_parts = decoder._prefix_parts(prefix_tokens, level, empty_prefix)
    state = decoder._state(context, mode_emb, level, prefix_parts)
    return decoder.heads[level](state)


@dataclass(frozen=True)
class AIGConfig:
    """Inference-only Acceptability-Imminence Gap settings."""

    alpha: float
    levels_zero_based: tuple[int, ...]
    gate_min_acceptable_prob: float

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AIGConfig | None":
        if not args.aig or args.aig_alpha <= 0:
            return None
        return cls(
            alpha=float(args.aig_alpha),
            levels_zero_based=tuple(int(level) - 1 for level in args.aig_levels),
            gate_min_acceptable_prob=float(args.aig_gate_min_acceptable_prob),
        )

    def applies_to(self, level: int) -> bool:
        return int(level) in self.levels_zero_based


def apply_aig_bias(
    relevance_logits: torch.Tensor,
    acceptable_logits: torch.Tensor,
    imminent_logits: torch.Tensor,
    alpha: float,
    gate_min_acceptable_prob: float,
) -> torch.Tensor:
    log_p_a = F.log_softmax(acceptable_logits, dim=-1)
    log_p_i = F.log_softmax(imminent_logits, dim=-1)
    gap = log_p_a - log_p_i
    if gate_min_acceptable_prob > 0:
        gate = (log_p_a.exp() >= gate_min_acceptable_prob).to(relevance_logits.dtype)
        gap = gap * gate
    return relevance_logits + float(alpha) * gap


def load_pseudo_ser_labels(path: Path | None) -> set[int] | None:
    """Load mined pseudo-ser positive item ids for ser-target metrics."""

    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Pseudo-ser label file not found: {path}")

    import pandas as pd

    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Pseudo-ser labels must be stored as .parquet or .csv")
    if "item_id" not in frame.columns:
        raise ValueError("Pseudo-ser label file must contain an item_id column")
    if "label" in frame.columns:
        frame = frame[frame["label"].astype(int) == 1]
    return {int(item_id) for item_id in frame["item_id"].tolist()}


def rank_items(
    beams: list[SIDBeam],
    seen: set[int],
    filter_history: bool,
    max_k: int,
) -> list[int]:
    ranked = []
    used = set()
    for beam in beams:
        if filter_history and beam.item_id in seen:
            continue
        if beam.item_id in used:
            continue
        ranked.append(beam.item_id)
        used.add(beam.item_id)
        if len(ranked) >= max_k:
            break
    return ranked


def new_totals(ks: list[int], prefix: str = "") -> dict[str, float]:
    totals: dict[str, float] = {}
    for k in ks:
        totals[f"hr{prefix}@{k}"] = 0.0
        totals[f"ndcg{prefix}@{k}"] = 0.0
    return totals


def update_totals(
    totals: dict[str, float],
    ranked: list[int],
    target: int,
    ks: list[int],
    prefix: str = "",
) -> None:
    for k in ks:
        topk = ranked[:k]
        try:
            rank = topk.index(target)
        except ValueError:
            continue
        totals[f"hr{prefix}@{k}"] += 1.0
        totals[f"ndcg{prefix}@{k}"] += 1.0 / math.log2(rank + 2)


def finalize_totals(
    totals: dict[str, float],
    count: int,
    ks: list[int],
    prefix: str = "",
) -> dict[str, float]:
    if count <= 0:
        return {key: 0.0 for key in totals}
    finalized = {}
    for k in ks:
        finalized[f"hr{prefix}@{k}"] = totals[f"hr{prefix}@{k}"] / count
        finalized[f"ndcg{prefix}@{k}"] = totals[f"ndcg{prefix}@{k}"] / count
    return finalized


if __name__ == "__main__":
    main()
