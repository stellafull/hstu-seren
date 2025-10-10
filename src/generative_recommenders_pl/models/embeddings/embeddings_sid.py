"""Semantic ID enhanced embedding modules."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint

from generative_recommenders_pl.models.embeddings.embeddings import EmbeddingModule
from generative_recommenders_pl.models.utils.initialization import truncated_normal
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


class LocalSIDEmbeddingModule(EmbeddingModule):
    """Embedding module that augments local embeddings with semantic IDs."""

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        semantic_id_path: str,
        quantizer_path: str | None = None,
        emb_tbl_size: int | None = None,
        emb_method: str = "RQsum-indEmb",
        emb_comb_method: str = "projgate",
        scale_sid_emb: bool = True,
        used_uniq_layer: bool = False,
        use_num_codebook_layers: int = 0,
        missing_item_strategy: str = "fallback_to_individual",
        apply_checkpoint: bool = False,
        item_lookup_path: str | None = None,
        lookup_is_zero_based: bool = True,
        lookup_casefold: bool = True,
    ) -> None:
        super().__init__()

        self._item_embedding_dim = int(item_embedding_dim)
        self._num_items = int(num_items)
        self._scale_sid_emb = bool(scale_sid_emb)
        self._emb_method = emb_method
        self._emb_comb_method_name_str = emb_comb_method
        self._missing_item_strategy = missing_item_strategy
        self._use_checkpoint = bool(apply_checkpoint)

        codebook, lookup, has_sid_mask = self._load_semantic_id_artifacts(
            semantic_id_path=semantic_id_path,
            quantizer_path=quantizer_path,
            num_items=self._num_items,
            use_num_codebook_layers=use_num_codebook_layers,
            item_lookup_path=item_lookup_path,
            lookup_is_zero_based=lookup_is_zero_based,
            lookup_casefold=lookup_casefold,
        )

        if codebook is not None:
            self.register_buffer("_SID_codebook", codebook)
            self._num_layers = codebook.shape[0]
            self._num_codes = codebook.shape[1]
        else:
            self._SID_codebook = None
            self._num_layers = lookup.shape[-1]
            self._num_codes = int(lookup.max().item()) + 1 if lookup.numel() > 0 else 1

        if used_uniq_layer and self._emb_method not in {"prefixN-indEmb", "RQsum-indEmb"}:
            self._num_layers += 1

        if use_num_codebook_layers > 0:
            self._num_layers = min(self._num_layers, int(use_num_codebook_layers))

        self.register_buffer("_lookup", lookup)
        self.register_buffer("_has_sid_mask", has_sid_mask)

        if self._emb_method == "RQsum-indEmb":
            self._emb_tbl_size = self._num_layers * self._num_codes
            log.info(
                "Using RQsum-indEmb method; setting embedding table size to %s",
                self._emb_tbl_size,
            )
        else:
            if emb_tbl_size is None:
                raise ValueError(
                    "Parameter emb_tbl_size must be provided for method %s"
                    % self._emb_method
                )
            self._emb_tbl_size = int(emb_tbl_size)

        self._item_emb_sid = torch.nn.Embedding(
            self._emb_tbl_size + 1, self._item_embedding_dim, padding_idx=0
        )

        if self._emb_method in {"prefixN-indEmb", "RQsum-indEmb"}:
            self._item_emb_individual = torch.nn.Embedding(
                self._num_items + 1, self._item_embedding_dim, padding_idx=0
            )
            self._emb_comb_method = self._init_emb_comb_method(emb_comb_method)
        else:
            self._item_emb_individual = None
            self._emb_comb_method = None

        self.reset_params()

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if name.startswith("_item_emb"):
                log.info(
                    "Initialize %s as truncated normal: %s params",
                    name,
                    params.data.size(),
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                log.info("Skipping initializing params %s - not configured", name)

    def debug_str(self) -> str:
        base = (
            f"local_SID_emb_d{self._item_embedding_dim}_m{self._emb_method}_"
            f"tbl{self._emb_tbl_size}_l{self._num_layers}_c{self._num_codes}"
        )
        if self._item_emb_individual is not None:
            base += f"-indEmb_{self._emb_comb_method_name_str}"
        else:
            base += "-indEmb_NONE"
        return base

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim

    def _load_semantic_id_artifacts(
        self,
        semantic_id_path: str,
        quantizer_path: str | None,
        num_items: int,
        use_num_codebook_layers: int,
        item_lookup_path: str | None,
        lookup_is_zero_based: bool,
        lookup_casefold: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        sid_path = Path(semantic_id_path)
        if not sid_path.is_file():
            raise FileNotFoundError(f"Semantic ID file not found: {sid_path}")

        data = torch.load(sid_path, map_location="cpu")
        semantic_ids = data.get("semantic_ids")
        item_ids_raw = data.get("item_ids")
        if semantic_ids is None or item_ids_raw is None:
            raise ValueError(
                "Semantic ID file must contain 'semantic_ids' and 'item_ids' entries"
            )

        if not torch.is_tensor(semantic_ids):
            semantic_ids = torch.tensor(semantic_ids, dtype=torch.long)
        semantic_ids = semantic_ids.to(torch.long)

        if semantic_ids.dim() != 2:
            raise ValueError(
                "semantic_ids must be a 2D tensor, got shape %s" % (semantic_ids.shape,)
            )

        def _normalize_key(key: str) -> str:
            return key.casefold() if lookup_casefold else key

        lookup_map: dict[str, int] | None = None
        if item_lookup_path is not None:
            lookup_file = Path(item_lookup_path)
            if not lookup_file.is_file():
                raise FileNotFoundError(
                    f"Item lookup file not found: {lookup_file}"
                )
            import pandas as pd

            lookup_df = pd.read_csv(lookup_file)
            if "original_item_id" not in lookup_df.columns or "normalized_item_id" not in lookup_df.columns:
                raise ValueError(
                    "Item lookup file %s must contain 'original_item_id' and 'normalized_item_id' columns"
                    % lookup_file
                )
            add_one = 1 if lookup_is_zero_based else 0
            lookup_map = {
                _normalize_key(str(orig)): int(norm) + add_one
                for orig, norm in zip(
                    lookup_df["original_item_id"].astype(str),
                    lookup_df["normalized_item_id"].astype(int),
                )
            }

        mapped_item_ids: list[int] = []
        keep_indices: list[int] = []
        skipped_out_of_range = 0
        skipped_missing_lookup = 0

        for idx, raw_id in enumerate(item_ids_raw):
            value: int | None
            try:
                value = int(raw_id)
            except (TypeError, ValueError):
                if lookup_map is None:
                    raise ValueError(
                        "Semantic ID item '%s' is non-numeric; provide item_lookup_path to map original IDs"
                        % raw_id
                    )
                lookup_key = _normalize_key(str(raw_id))
                value = lookup_map.get(lookup_key)
                if value is None:
                    skipped_missing_lookup += 1
                    continue

            if value <= 0 or value > num_items:
                skipped_out_of_range += 1
                continue

            mapped_item_ids.append(int(value))
            keep_indices.append(idx)

        if not mapped_item_ids:
            raise ValueError(
                "No valid semantic ID entries after mapping; check lookup alignment"
            )

        if skipped_missing_lookup:
            log.warning(
                "Skipped %d semantic IDs with missing item lookup entries (path=%s)",
                skipped_missing_lookup,
                item_lookup_path,
            )
        if skipped_out_of_range:
            log.warning(
                "Skipped %d semantic IDs out of valid range [1, %d]",
                skipped_out_of_range,
                num_items,
            )

        semantic_ids = semantic_ids[keep_indices]
        item_ids = torch.tensor(mapped_item_ids, dtype=torch.long)

        num_layers = semantic_ids.size(1)
        lookup = torch.zeros((num_items + 1, num_layers), dtype=torch.long)
        has_sid_mask = torch.zeros((num_items + 1,), dtype=torch.bool)

        lookup[item_ids.long()] = semantic_ids[: item_ids.numel()]
        has_sid_mask[item_ids.long()] = True

        codebook = None
        if quantizer_path is not None:
            quant_path = Path(quantizer_path)
            if not quant_path.is_file():
                log.warning("Quantizer file not found: %s", quant_path)
            else:
                quantizer = torch.load(quant_path, map_location="cpu")
                codebooks_raw = quantizer.get("codebooks")
                if codebooks_raw is not None:
                    if isinstance(codebooks_raw, torch.Tensor):
                        codebook = codebooks_raw.to(torch.float32)
                    elif isinstance(codebooks_raw, list):
                        codebook = torch.stack(
                            [cb.to(torch.float32) for cb in codebooks_raw], dim=0
                        )
                    else:
                        raise TypeError(
                            "Unsupported codebook type: %s" % type(codebooks_raw)
                        )
        if codebook is not None and use_num_codebook_layers > 0:
            codebook = codebook[:use_num_codebook_layers]
        return codebook, lookup, has_sid_mask

    def _get_emb_from_idx(
        self, item_ids: torch.Tensor, emb_tbl: torch.nn.Embedding
    ) -> torch.Tensor:
        idx_min = torch.min(item_ids)
        idx_max = torch.max(item_ids)
        if idx_min < 0 or idx_max >= emb_tbl.num_embeddings:
            raise IndexError(
                f"item_ids (min: {idx_min}, max: {idx_max}) are out of bounds for emb table size {emb_tbl.num_embeddings}"
            )
        return emb_tbl(item_ids)

    def _init_emb_comb_method(self, emb_comb_method: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        method = emb_comb_method.lower()
        if method == "sum":
            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                return embs1 + embs2

        elif method == "fc":
            self._emb_comb_fc = torch.nn.Linear(
                self._item_embedding_dim * 2, self._item_embedding_dim
            )

            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                embs = torch.cat((embs1, embs2), dim=-1)
                return self._emb_comb_fc(embs)

        elif method == "mlp":
            self._emb_comb_mlp = torch.nn.Sequential(
                torch.nn.Linear(
                    self._item_embedding_dim * 2, self._item_embedding_dim * 2
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    self._item_embedding_dim * 2, self._item_embedding_dim
                ),
            )

            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                embs = torch.cat((embs1, embs2), dim=-1)
                return self._emb_comb_mlp(embs)

        elif method == "scalargate":
            self._emb_comb_sg = torch.nn.Linear(
                self._item_embedding_dim * 2, 1
            )

            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                embs = torch.cat((embs1, embs2), dim=-1)
                gate = torch.sigmoid(self._emb_comb_sg(embs))
                return gate * embs1 + (1 - gate) * embs2

        elif method == "vectorgate":
            self._emb_comb_vg = torch.nn.Linear(
                self._item_embedding_dim * 2, self._item_embedding_dim
            )

            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                embs = torch.cat((embs1, embs2), dim=-1)
                gate = torch.sigmoid(self._emb_comb_vg(embs))
                return gate * embs1 + (1 - gate) * embs2

        elif method == "scalarweight":
            self.weight_alpha = torch.nn.Parameter(
                torch.tensor(0.5, dtype=self._item_emb_sid.weight.dtype)
            )

            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                return self.weight_alpha * embs1 + (1 - self.weight_alpha) * embs2

        elif method == "projgate":
            self._emb_comb_proj_gate = torch.nn.Linear(
                self._item_embedding_dim * 2, self._item_embedding_dim
            )

            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                concat = torch.cat((embs1, embs2), dim=-1)
                fused = self._emb_comb_proj_gate(concat)
                gate = torch.sigmoid(self._emb_comb_proj_gate(concat))
                return gate * fused + (1 - gate) * embs2

        else:
            raise ValueError(
                "Unknown embedding combination method: %s" % emb_comb_method
            )

        log.info("Initialized embedding combination method: %s", emb_comb_method)
        return combiner

    def SID2embID(self, sid_tup: torch.Tensor) -> torch.Tensor:
        coeffs = [self._num_codes**i for i in range(sid_tup.shape[-1])]
        coeffs = list(reversed(coeffs))
        coeff_tensor = sid_tup.new_tensor(coeffs)
        view_shape = (1,) * (sid_tup.dim() - 1) + (sid_tup.shape[-1],)
        coeff_tensor = coeff_tensor.view(*view_shape)
        emb_id = sid_tup * coeff_tensor
        return emb_id.sum(dim=-1)

    def SID2multidigits_RQsum(self, sid_tup: torch.Tensor) -> torch.Tensor:
        offsets = [self._num_codes * i for i in range(sid_tup.shape[-1])]
        offsets = list(reversed(offsets))
        offset_tensor = sid_tup.new_tensor(offsets)
        view_shape = (1,) * (sid_tup.dim() - 1) + (sid_tup.shape[-1],)
        offset_tensor = offset_tensor.view(*view_shape)
        return sid_tup + offset_tensor

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        index_ids = item_ids.clamp(min=0, max=self._num_items).to(torch.long)
        sid_tup = self._lookup[index_ids]
        sid_tup = sid_tup[..., : self._num_layers]

        if self._emb_method == "Ngram":
            emb_id = self.SID2embID(sid_tup)
            emb_id_mod = torch.remainder(emb_id, self._emb_tbl_size).to(torch.long)
            sid_embs = self._get_emb_from_idx(emb_id_mod, self._item_emb_sid)
        else:
            sid_embs = torch.zeros(
                (*item_ids.shape, self._item_embedding_dim),
                dtype=self._item_emb_sid.weight.dtype,
                device=self._item_emb_sid.weight.device,
            )

            if self._emb_method == "RQsum-indEmb":
                emb_id_tups = self.SID2multidigits_RQsum(sid_tup)
                for layer_i in range(emb_id_tups.shape[-1]):
                    emb = self._get_emb_from_idx(
                        emb_id_tups[..., layer_i], self._item_emb_sid
                    )
                    sid_embs += emb
            else:
                for prefix_i in range(1, self._num_layers + 1):
                    emb_id = self.SID2embID(sid_tup[..., :prefix_i])
                    emb_id_mod = torch.remainder(emb_id, self._emb_tbl_size).to(
                        torch.long
                    )
                    emb = self._get_emb_from_idx(emb_id_mod, self._item_emb_sid)
                    sid_embs += emb

        if self._scale_sid_emb:
            layers_count = self._num_layers if self._emb_method != "Ngram" else 1
            scale = sid_embs.new_tensor(float(layers_count)).sqrt()
            sid_embs = sid_embs / scale

        if self._item_emb_individual is None:
            return sid_embs

        ind_emb = self._get_emb_from_idx(index_ids, self._item_emb_individual)
        ind_emb = ind_emb.view(*item_ids.shape, self._item_embedding_dim)

        if sid_embs.shape != ind_emb.shape:
            sid_embs = sid_embs.view_as(ind_emb)

        if self._use_checkpoint:
            combined = checkpoint(
                self._emb_comb_method, sid_embs, ind_emb, use_reentrant=False
            )
        else:
            combined = self._emb_comb_method(sid_embs, ind_emb)

        has_sid_mask = self._has_sid_mask[index_ids]
        has_sid_mask = has_sid_mask.view(*item_ids.shape)
        if self._missing_item_strategy == "fallback_to_individual":
            combined = torch.where(
                has_sid_mask.unsqueeze(-1), combined, ind_emb
            )
        elif self._missing_item_strategy == "zeros_if_missing":
            combined = torch.where(
                has_sid_mask.unsqueeze(-1), combined, torch.zeros_like(combined)
            )

        return combined
