"""Mine label-free pseudo-ser positives for HSTU-SerenFree.

V1 mining is intentionally offline and teacher-scored. It does not update the
teacher and writes a small table usable by retrieval evaluation and Stage 4
fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from generative_recommenders_pl.models.serenfree import DecoderMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--experiment", default="serenfree_stage2_amazon_movies")
    parser.add_argument("--config-name", default="train.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--min-gap", type=float, default=0.5)
    parser.add_argument("--top-per-batch", type=int, default=256)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    datamodule.setup("fit" if args.split in {"train", "val"} else "test")
    dataloader = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }[args.split]()

    rows = mine(model, dataloader, device, args.max_batches, args.min_gap, args.top_per_batch)
    output = pd.DataFrame(rows).drop_duplicates("item_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        output.to_parquet(args.output, index=False)
    elif args.output.suffix == ".csv":
        output.to_csv(args.output, index=False)
    else:
        raise ValueError("output must be .parquet or .csv")
    print(
        {
            "output": str(args.output),
            "rows": int(len(output)),
            "min_gap": float(args.min_gap),
        },
        flush=True,
    )


def load_config(args: argparse.Namespace):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    repo_root = Path(__file__).resolve().parents[3]
    overrides = [f"experiment={args.experiment}", *args.override]
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name=args.config_name, overrides=overrides)


@torch.inference_mode()
def mine(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None,
    min_gap: float,
    top_per_batch: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        target_ids = batch["target_ids"].to(device)
        historical_ids = batch["historical_ids"].to(device)
        lengths = batch["history_lengths"].to(device)
        timestamps = batch["historical_timestamps"].to(device)
        target_sid = model._item_ids_to_sid(target_ids)

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
        recent = model._masked_recent_mean(encoded, lengths, getattr(model, "recent_window", 10))
        history = model._masked_history_mean(encoded, lengths)

        prefix = target_sid[:, :-1]
        acceptable = model.decoder(history, prefix, mode=DecoderMode.ACCEPTABLE)
        imminent = model.decoder(recent, prefix, mode=DecoderMode.IMMINENT)
        gaps = []
        for level, target in enumerate(target_sid[:, :-1].unbind(dim=-1)):
            log_p_a = F.log_softmax(acceptable.semantic_logits[level], dim=-1)
            log_p_i = F.log_softmax(imminent.semantic_logits[level], dim=-1)
            gaps.append((log_p_a - log_p_i).gather(1, target.unsqueeze(1)).squeeze(1))
        score = torch.stack(gaps, dim=1).mean(dim=1)
        selected = torch.nonzero(score >= float(min_gap), as_tuple=False).flatten()
        if selected.numel() > top_per_batch:
            _, top_idx = score[selected].topk(top_per_batch)
            selected = selected[top_idx]
        for idx in selected.tolist():
            rows.append(
                {
                    "item_id": int(target_ids[idx].item()),
                    "label": 1,
                    "score": float(score[idx].item()),
                    "source_batch": int(batch_idx),
                    "reason": "teacher_aig_gap",
                }
            )
    return rows


if __name__ == "__main__":
    main()
