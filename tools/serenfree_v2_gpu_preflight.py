#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra import compose, initialize_config_dir

from generative_recommenders_pl.data.reco_dataset import (
    DynamicFutureWindowTargetDataset,
    FutureWindowTargetDataset,
    LOOManifestTrainDataset,
)
from generative_recommenders_pl.models.serenfree import SIDBeam
from generative_recommenders_pl.scripts.evaluate_serenfree_retrieval import rank_items
from generative_recommenders_pl.utils.omegaconf_resolvers import register_safe_resolvers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="serenfree_v2_r_pretrain_movielens")
    parser.add_argument("--config-name", default="train.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requested CUDA, but torch.cuda.is_available() is false")

    cfg = load_config(args)
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    datamodule.setup("fit")
    train_dataset = datamodule.train_dataset
    if isinstance(train_dataset, (DynamicFutureWindowTargetDataset, FutureWindowTargetDataset)):
        raise RuntimeError("V2 R/A/I policy preflight found a future-window train dataset")
    if not isinstance(train_dataset, LOOManifestTrainDataset):
        raise RuntimeError(f"Expected LOOManifestTrainDataset, got {type(train_dataset).__name__}")

    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    forbidden_batch_fields = {"I_sids", "A_sids", "future_i_sids", "future_a_sids"}
    present_forbidden = sorted(forbidden_batch_fields.intersection(batch))
    if present_forbidden:
        raise RuntimeError(f"Future target fields leaked into policy batch: {present_forbidden}")

    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    check_pad_collision(model, datamodule, batch, device)
    check_synthetic_target_construction(model, device)
    check_eval_gather(model, device)
    check_history_filtering()

    with torch.no_grad():
        loss, metrics = model._step(batch, require_aux_targets=True)
    if not torch.isfinite(loss.detach().cpu()):
        raise RuntimeError("Preflight model step produced a non-finite loss")

    summary: dict[str, Any] = {
        "experiment": args.experiment,
        "device": str(device),
        "train_dataset": type(train_dataset).__name__,
        "batch_size": int(batch["target_ids"].numel()),
        "loss": float(loss.detach().cpu().item()),
        "metrics": {
            key: float(value.detach().cpu().item())
            for key, value in metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_config(args: argparse.Namespace) -> Any:
    register_safe_resolvers()
    repo_root = Path(__file__).resolve().parents[1]
    overrides = [f"experiment={args.experiment}", *args.override]
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name=args.config_name, overrides=overrides)


def check_pad_collision(model: Any, datamodule: Any, batch: dict[str, torch.Tensor], device: torch.device) -> None:
    ids = batch["historical_ids"]
    target_ids = batch["target_ids"]
    real_ids = torch.cat([ids[ids != 0], target_ids[target_ids != 0]])
    if real_ids.numel() == 0 or int(real_ids.min().item()) <= 0:
        raise RuntimeError("Real item ids must be strictly positive after shift_id_by")
    if int(real_ids.max().item()) > int(datamodule.max_item_id):
        raise RuntimeError("Batch item id exceeds datamodule.max_item_id")

    lookup = model.sid_lookup.detach().to(device)
    valid_lookup = lookup[(lookup != 0).all(dim=1)]
    if valid_lookup.numel() == 0:
        raise RuntimeError("SID lookup has no fully valid rows")
    if int(valid_lookup.min().item()) <= 0:
        raise RuntimeError("Real SID codes must be strictly positive; 0 is reserved for padding")
    if not (lookup[0] == 0).all().item():
        raise RuntimeError("SID lookup row 0 must be padding zeros")


def check_synthetic_target_construction(model: Any, device: torch.device) -> None:
    historical = torch.tensor([[0, 10, 20, 30], [10, 20, 30, 0]], device=device)
    target = torch.tensor([40, 40], device=device)
    ratings = torch.tensor([[0, 5, 3, 5], [5, 3, 5, 0]], device=device)
    target_ratings = torch.tensor([5, 2], device=device)
    sid_lookup = torch.zeros_like(model.sid_lookup[:41].to(device))
    for item in (10, 20, 30, 40):
        sid_lookup[item] = torch.arange(1, model.sid_lookup.size(1) + 1, device=device) + item
    old_lookup = model.sid_lookup.detach().clone()
    model.sid_lookup[:41].copy_(sid_lookup.to(model.sid_lookup.device))
    try:
        targets = model._next_transition_targets(
            {
                "historical_ratings": ratings,
                "target_ratings": target_ratings,
            },
            historical,
            target,
            model._item_ids_to_sid(historical),
            model._item_ids_to_sid(target),
        )
    finally:
        model.sid_lookup.copy_(old_lookup.to(model.sid_lookup.device))
    if targets["next_ids"].tolist()[0] != [0, 20, 30, 40]:
        raise RuntimeError("Left-padding synthetic next_ids check failed")
    if targets["valid_mask"].tolist()[0] != [False, True, True, True]:
        raise RuntimeError("Left-padding synthetic valid_mask check failed")
    if targets["next_ids"].tolist()[1] != [20, 30, 40, 0]:
        raise RuntimeError("Right-padding synthetic next_ids check failed")
    if targets["valid_mask"].tolist()[1] != [True, True, True, False]:
        raise RuntimeError("Right-padding synthetic valid_mask check failed")
    if targets["accept_mask"].sum().item() == 0:
        zero = torch.randn(2, 4, 3, device=device).sum() * 0.0
        if zero.device != device:
            raise RuntimeError("Synthetic zero-loss device check failed")


def check_eval_gather(model: Any, device: torch.device) -> None:
    encoded = torch.arange(2 * 4 * 3, dtype=torch.float32, device=device).reshape(2, 4, 3)
    lengths = torch.tensor([2, 4], device=device)
    gathered = model._last_valid_state(encoded, lengths)
    expected = torch.stack([encoded[0, 1], encoded[1, 3]])
    if not torch.equal(gathered, expected):
        raise RuntimeError("Eval last-state gather is not length-aware")
    if torch.equal(gathered[0], encoded[0, -1]):
        raise RuntimeError("Eval gather appears to use hidden[:, -1] for padded rows")


def check_history_filtering() -> None:
    beams = [
        SIDBeam(item_id=5, sid=(1, 1, 1, 1), score=3.0),
        SIDBeam(item_id=7, sid=(1, 1, 1, 2), score=2.0),
    ]
    ranked = rank_items(beams, seen={5}, filter_history=True, max_k=2, target=5)
    if ranked[:1] != [5]:
        raise RuntimeError("History filtering removed repeated ground-truth target")


if __name__ == "__main__":
    main()
