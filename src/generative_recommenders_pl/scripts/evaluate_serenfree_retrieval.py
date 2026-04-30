"""Evaluate SerenFree full-SID generation with retrieval metrics.

This script does not build semantic IDs. It evaluates an existing
HSTU-SerenFree checkpoint by constrained full-SID decoding, then computes
HR/NDCG@K against dataloader target item ids.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from generative_recommenders_pl.models.serenfree import DecoderMode, SIDBeam, SIDTrie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--experiment", default="serenfree_stage1_amazon_movies")
    parser.add_argument("--config-name", default="train.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beam-size", type=int, default=400)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100, 200])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--no-filter-history", action="store_true")
    parser.add_argument("--allow-duplicate-sids", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override, e.g. --override data.batch_size=128",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ks = sorted(set(args.ks))
    if ks[-1] > args.beam_size:
        raise ValueError("beam-size must be at least the largest requested K")

    cfg = load_config(args)
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

    trie, trie_audit = build_trie_from_sid_lookup(
        model.sid_lookup.detach().cpu(),
        allow_duplicate_sids=args.allow_duplicate_sids,
    )
    audit = audit_pipeline(model=model, trie_audit=trie_audit, cfg=cfg, args=args)
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        trie=trie,
        ks=ks,
        beam_size=args.beam_size,
        device=device,
        filter_history=not args.no_filter_history,
        max_batches=args.max_batches,
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


def build_trie_from_sid_lookup(
    sid_lookup: torch.Tensor,
    allow_duplicate_sids: bool,
) -> tuple[SIDTrie, dict[str, Any]]:
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

    trie = SIDTrie.from_items(item_ids=item_ids, sid_tokens=sid_tokens)
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
    return trie, audit


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
        "filter_history": not args.no_filter_history,
        "beam_size": int(args.beam_size),
        "ks": args.ks,
        "model_class": type(model).__name__,
        "full_sid_in": hasattr(model, "sid_composer") and not hasattr(model, "embeddings"),
        "full_sid_out": hasattr(model, "decoder")
        and len(model.decoder.heads) == trie_audit["sid_columns"],
        "item_level_hstu": hasattr(model, "sequence_encoder"),
        "aig_present": any("aig" in name.lower() for name, _ in model.named_modules()),
        "dedup_relevance_only_for_stage1": True,
        "sid_lookup_path": str(cfg.model.get("sid_lookup_path")),
        "sid_path": str(cfg.model.sid_path),
        **trie_audit,
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    trie: SIDTrie,
    ks: list[int],
    beam_size: int,
    device: torch.device,
    filter_history: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    totals = {"all": new_totals(ks), "eligible": new_totals(ks)}
    counts = defaultdict(int)
    max_k = max(ks)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        context = encode_context(model=model, batch=batch, device=device)
        beams_by_row = batched_constrained_beam_search(
            decoder=model.decoder,
            context=context,
            trie=trie,
            beam_size=beam_size,
        )

        target_ids = batch["target_ids"].to(torch.long)
        history_ids = batch["historical_ids"].to(torch.long)
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

    return {
        "counts": dict(counts),
        "all": finalize_totals(totals["all"], counts["all"], ks),
        "eligible_targets_only": finalize_totals(
            totals["eligible"],
            counts["eligible"],
            ks,
        ),
    }


@torch.inference_mode()
def encode_context(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
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
    return model._last_valid_state(encoded, lengths)


@torch.inference_mode()
def batched_constrained_beam_search(
    decoder: torch.nn.Module,
    context: torch.Tensor,
    trie: SIDTrie,
    beam_size: int,
) -> list[list[SIDBeam]]:
    if trie.num_sid_columns is None:
        raise ValueError("Cannot decode with an empty SID trie")
    rows: list[list[tuple[tuple[int, ...], float]]] = [[((), 0.0)] for _ in range(context.size(0))]

    for level in range(trie.num_sid_columns):
        flat_rows: list[int] = []
        flat_prefixes: list[tuple[int, ...]] = []
        flat_scores: list[float] = []
        for row_idx, row_beams in enumerate(rows):
            for prefix, score in row_beams:
                flat_rows.append(row_idx)
                flat_prefixes.append(prefix)
                flat_scores.append(score)
        if not flat_rows:
            break

        row_index = torch.tensor(flat_rows, dtype=torch.long, device=context.device)
        prefix_tensor = prefix_tensor_from_prefixes(
            prefixes=flat_prefixes,
            num_semantic_levels=decoder.num_semantic_levels,
            device=context.device,
        )
        logits = decoder(
            context.index_select(dim=0, index=row_index),
            prefix_tensor,
            mode=DecoderMode.RELEVANCE,
        ).as_list()[level]
        log_probs = F.log_softmax(logits, dim=-1)

        expanded_by_row: list[list[tuple[tuple[int, ...], float]]] = [
            [] for _ in range(context.size(0))
        ]
        for beam_idx, prefix in enumerate(flat_prefixes):
            allowed = trie.allowed_tokens(prefix)
            if not allowed:
                continue
            allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=context.device)
            allowed_scores = log_probs[beam_idx].index_select(0, allowed_tensor).cpu()
            row_idx = flat_rows[beam_idx]
            base_score = flat_scores[beam_idx]
            for token, score in zip(allowed, allowed_scores.tolist()):
                expanded_by_row[row_idx].append(((*prefix, int(token)), base_score + float(score)))

        rows = [
            sorted(row_expanded, key=lambda candidate: candidate[1], reverse=True)[:beam_size]
            for row_expanded in expanded_by_row
        ]

    results: list[list[SIDBeam]] = []
    for row_beams in rows:
        row_results = []
        for sid, score in row_beams:
            item_id = trie.item_id(sid)
            if item_id is not None:
                row_results.append(SIDBeam(item_id=item_id, sid=sid, score=score))
        results.append(row_results)
    return results


def prefix_tensor_from_prefixes(
    prefixes: list[tuple[int, ...]],
    num_semantic_levels: int,
    device: torch.device,
) -> torch.Tensor:
    tokens = torch.zeros(
        (len(prefixes), num_semantic_levels),
        dtype=torch.long,
        device=device,
    )
    for row_idx, prefix in enumerate(prefixes):
        usable = prefix[:num_semantic_levels]
        if usable:
            tokens[row_idx, : len(usable)] = torch.tensor(
                usable,
                dtype=torch.long,
                device=device,
            )
    return tokens


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


def new_totals(ks: list[int]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for k in ks:
        totals[f"hr@{k}"] = 0.0
        totals[f"ndcg@{k}"] = 0.0
    return totals


def update_totals(
    totals: dict[str, float],
    ranked: list[int],
    target: int,
    ks: list[int],
) -> None:
    for k in ks:
        topk = ranked[:k]
        try:
            rank = topk.index(target)
        except ValueError:
            continue
        totals[f"hr@{k}"] += 1.0
        totals[f"ndcg@{k}"] += 1.0 / math.log2(rank + 2)


def finalize_totals(
    totals: dict[str, float],
    count: int,
    ks: list[int],
) -> dict[str, float]:
    if count <= 0:
        return {key: 0.0 for key in totals}
    finalized = {}
    for k in ks:
        finalized[f"hr@{k}"] = totals[f"hr@{k}"] / count
        finalized[f"ndcg@{k}"] = totals[f"ndcg@{k}"] / count
    return finalized


if __name__ == "__main__":
    main()
