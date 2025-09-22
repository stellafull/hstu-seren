from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

_EPS = 1e-12

_DTYPE_NAME_TO_TORCH = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}

_DTYPE_TORCH_TO_NAME = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}


def _resolve_dtype_config(
    value: torch.dtype | str | None,
    *,
    default: torch.dtype,
) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    key = str(value).lower().strip()
    key = key.replace("torch.", "")
    try:
        return _DTYPE_NAME_TO_TORCH[key]
    except KeyError as exc:  # pragma: no cover - defensive
        allowed = ", ".join(sorted(_DTYPE_NAME_TO_TORCH))
        raise ValueError(f"Unsupported dtype '{value}'. Allowed: {allowed}.") from exc


def _dtype_to_name(dtype: torch.dtype) -> str:
    return _DTYPE_TORCH_TO_NAME.get(dtype, str(dtype).replace("torch.", ""))


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 2D tensor of shape (num_items, embedding_dim) but got shape {tuple(tensor.shape)}."
        )
    return tensor


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    norms = tensor.norm(p=2, dim=1, keepdim=True).clamp_min(_EPS)
    return tensor / norms


def _squared_euclidean_distances(batch: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    batch_norm = (batch**2).sum(dim=1, keepdim=True)
    centroid_norm = (centroids**2).sum(dim=1).unsqueeze(0)
    distances = batch_norm + centroid_norm - 2.0 * batch @ centroids.t()
    return distances.clamp_min_(0.0)


@dataclass
class ResidualQuantizerResult:
    codebooks: List[torch.Tensor]
    assignments: torch.Tensor
    item_ids: List[str]
    normalize_residuals: bool
    storage_dtype: torch.dtype
    compute_dtype: torch.dtype

    def save(self, path: Path | str) -> None:
        serializable = {
            "codebooks": [codebook.cpu() for codebook in self.codebooks],
            "assignments": self.assignments.cpu(),
            "item_ids": list(self.item_ids),
            "normalize_residuals": bool(self.normalize_residuals),
            "storage_dtype": _dtype_to_name(self.storage_dtype),
            "compute_dtype": _dtype_to_name(self.compute_dtype),
        }
        torch.save(serializable, Path(path))

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        map_location: torch.device | str | None = None,
    ) -> "ResidualQuantizerResult":
        payload = torch.load(Path(path), map_location=map_location)
        storage_dtype = _resolve_dtype_config(payload.get("storage_dtype"), default=torch.float16)
        compute_dtype = _resolve_dtype_config(payload.get("compute_dtype"), default=torch.float32)
        codebooks = [
            tensor.clone().contiguous().to(dtype=compute_dtype)
            for tensor in payload["codebooks"]
        ]
        assignments = payload["assignments"].clone().contiguous()
        item_ids = list(payload["item_ids"])
        normalize_residuals = bool(payload.get("normalize_residuals", False))
        return cls(
            codebooks=codebooks,
            assignments=assignments,
            item_ids=item_ids,
            normalize_residuals=normalize_residuals,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
        )

    def semantic_id_tensor(self) -> torch.Tensor:
        return self.assignments.clone()

    def quantize(
        self,
        embeddings: torch.Tensor,
        *,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        embeddings = _ensure_2d(embeddings).contiguous()
        if embeddings.dtype != self.storage_dtype:
            embeddings = embeddings.to(self.storage_dtype)
        num_items, _ = embeddings.shape
        if num_items == 0:
            return torch.empty((0, len(self.codebooks)), dtype=torch.long, device=embeddings.device)
        normalized = _normalize_rows if self.normalize_residuals else (lambda x: x)
        codebooks = [
            codebook.to(device=embeddings.device, dtype=self.compute_dtype)
            for codebook in self.codebooks
        ]
        output = torch.empty((num_items, len(codebooks)), dtype=torch.long, device=embeddings.device)
        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            chunk = embeddings[start:end]
            if chunk.numel() == 0:
                continue
            residual = chunk.to(dtype=self.compute_dtype)
            residual = normalized(residual)
            for layer_idx, codebook in enumerate(codebooks):
                distances = _squared_euclidean_distances(residual, codebook)
                layer_assignments = torch.argmin(distances, dim=1)
                output[start:end, layer_idx] = layer_assignments
                residual = residual - codebook[layer_assignments]
                residual = normalized(residual)
        return output


class ResidualVectorQuantizer:
    def __init__(
        self,
        *,
        num_layers: int,
        codebook_size: int | Sequence[int],
        chunk_size: int = 65536,
        max_iterations: int = 25,
        tolerance: float = 1e-4,
        normalize_residuals: bool = True,
        seed: int | None = None,
        device: torch.device | str | None = None,
        use_mini_batch: bool = False,
        mini_batch_size: int | None = None,
        mini_batch_epochs: int | None = None,
        shuffle_mini_batches: bool = True,
        storage_dtype: torch.dtype | str | None = torch.float16,
        compute_dtype: torch.dtype | str | None = torch.float32,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if isinstance(codebook_size, int):
            codebook_sizes = [int(codebook_size)] * num_layers
        else:
            codebook_sizes = [int(value) for value in codebook_size]
        if len(codebook_sizes) != num_layers:
            raise ValueError(
                "Length of codebook_size sequence must match num_layers."
            )
        if any(size <= 0 for size in codebook_sizes):
            raise ValueError("All codebook sizes must be positive.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if mini_batch_size is not None and mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be a positive integer when provided.")
        if mini_batch_epochs is not None and mini_batch_epochs <= 0:
            raise ValueError("mini_batch_epochs must be a positive integer when provided.")
        self.num_layers = int(num_layers)
        self.codebook_sizes = codebook_sizes
        self.chunk_size = int(chunk_size)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.normalize_residuals = bool(normalize_residuals)
        self.seed = seed
        self.device = torch.device(device) if device is not None else None
        self.use_mini_batch = bool(use_mini_batch)
        self.mini_batch_size = int(mini_batch_size) if mini_batch_size is not None else self.chunk_size
        self.mini_batch_epochs = (
            int(mini_batch_epochs)
            if mini_batch_epochs is not None
            else self.max_iterations
        )
        self.shuffle_mini_batches = bool(shuffle_mini_batches)
        self.storage_dtype = _resolve_dtype_config(storage_dtype, default=torch.float16)
        self.compute_dtype = _resolve_dtype_config(compute_dtype, default=torch.float32)
        if not self.storage_dtype.is_floating_point:
            raise ValueError("storage_dtype must be a floating point dtype.")
        if not self.compute_dtype.is_floating_point:
            raise ValueError("compute_dtype must be a floating point dtype.")

    def fit(self, embeddings: torch.Tensor, item_ids: Iterable[str]) -> ResidualQuantizerResult:
        data = _ensure_2d(embeddings).contiguous()
        ids = list(item_ids)
        if data.shape[0] != len(ids):
            raise ValueError(
                "The number of item_ids must match the number of embeddings."
            )
        if data.shape[0] == 0:
            raise ValueError("Cannot fit quantizer with zero embeddings.")
        if min(self.codebook_sizes) > data.shape[0]:
            raise ValueError(
                "Each codebook size must be <= number of training embeddings."
            )

        device = self.device or data.device
        if data.device != device:
            data = data.to(device)

        if data.dtype != self.storage_dtype:
            data = data.to(self.storage_dtype)

        residual = data
        self._normalize_inplace(residual)

        codebooks: list[torch.Tensor] = []
        assignment_layers: list[torch.Tensor] = []
        for layer_idx, num_centroids in enumerate(self.codebook_sizes):
            centroids = self._initialise_centroids(residual, num_centroids, layer_idx)
            centroids = self._run_kmeans(residual, centroids, layer_idx)
            assignments = self._assign(residual, centroids)
            codebooks.append(centroids.detach().clone())
            assignment_layers.append(assignments.detach().clone())
            self._apply_residual_update(residual, centroids, assignments)
        assignment_tensor = torch.stack(assignment_layers, dim=1)
        return ResidualQuantizerResult(
            codebooks=[codebook.cpu() for codebook in codebooks],
            assignments=assignment_tensor.cpu(),
            item_ids=ids,
            normalize_residuals=self.normalize_residuals,
            storage_dtype=self.storage_dtype,
            compute_dtype=self.compute_dtype,
        )

    def _normalize_inplace(self, tensor: torch.Tensor) -> None:
        if not self.normalize_residuals:
            return
        for start in range(0, tensor.size(0), self.chunk_size):
            end = min(start + self.chunk_size, tensor.size(0))
            if start >= end:
                continue
            chunk = tensor[start:end]
            if chunk.numel() == 0:
                continue
            chunk_norm = chunk.to(dtype=self.compute_dtype)
            chunk_norm = _normalize_rows(chunk_norm)
            tensor[start:end] = chunk_norm.to(dtype=self.storage_dtype)

    def _apply_residual_update(
        self,
        residual: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
    ) -> None:
        for start in range(0, residual.size(0), self.chunk_size):
            end = min(start + self.chunk_size, residual.size(0))
            batch_assignments = assignments[start:end]
            if batch_assignments.numel() == 0:
                continue
            chunk = residual[start:end]
            chunk_compute = chunk.to(dtype=self.compute_dtype)
            chunk_compute = chunk_compute - centroids[batch_assignments]
            if self.normalize_residuals:
                chunk_compute = _normalize_rows(chunk_compute)
            residual[start:end] = chunk_compute.to(dtype=self.storage_dtype)

    def _initialise_centroids(
        self, data: torch.Tensor, num_centroids: int, layer_idx: int
    ) -> torch.Tensor:
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=data.device)
            generator.manual_seed(self.seed + layer_idx)
        indices = torch.randperm(data.size(0), device=data.device, generator=generator)[:num_centroids]
        return data[indices].to(dtype=self.compute_dtype).contiguous()

    def _run_kmeans(
        self,
        data: torch.Tensor,
        centroids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if self.use_mini_batch:
            return self._run_mini_batch_kmeans(data, centroids, layer_idx)
        return self._run_full_batch_kmeans(data, centroids, layer_idx)

    def _run_full_batch_kmeans(
        self,
        data: torch.Tensor,
        centroids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=data.device)
            generator.manual_seed(self.seed + layer_idx)
        centroids = centroids.to(dtype=self.compute_dtype)
        for _ in range(self.max_iterations):
            assignments = self._assign(data, centroids)
            new_centroids = self._update_centroids(
                data,
                assignments,
                centroids.shape[0],
                generator,
            )
            shift = torch.norm(centroids - new_centroids, dim=1).mean()
            centroids = new_centroids
            if shift <= self.tolerance:
                break
        assignments = self._assign(data, centroids)
        return self._update_centroids(
            data,
            assignments,
            centroids.shape[0],
            generator,
        )

    def _run_mini_batch_kmeans(
        self,
        data: torch.Tensor,
        centroids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=data.device)
            generator.manual_seed(self.seed + layer_idx)

        num_centroids = centroids.shape[0]
        num_points = data.size(0)
        if num_points == 0:
            return centroids

        batch_size = min(self.mini_batch_size, num_points)
        centroids = centroids.clone().to(dtype=self.compute_dtype)
        global_counts = torch.zeros(
            num_centroids, dtype=self.compute_dtype, device=data.device
        )

        for _ in range(self.mini_batch_epochs):
            previous = centroids.clone()
            if self.shuffle_mini_batches and num_points > batch_size:
                permutation = torch.randperm(
                    num_points, device=data.device, generator=generator
                )
            else:
                permutation = None

            for start in range(0, num_points, batch_size):
                end = min(start + batch_size, num_points)
                if permutation is None:
                    batch = data[start:end]
                else:
                    indices = permutation[start:end]
                    batch = data[indices]

                if batch.numel() == 0:
                    continue

                batch_compute = batch.to(device=centroids.device, dtype=self.compute_dtype)
                distances = _squared_euclidean_distances(batch_compute, centroids)
                assignments = torch.argmin(distances, dim=1)
                batch_counts = torch.bincount(
                    assignments, minlength=num_centroids
                ).to(self.compute_dtype)

                active = torch.nonzero(batch_counts > 0, as_tuple=False).squeeze(1)
                if active.numel() == 0:
                    continue

                batch_sums = torch.zeros_like(centroids)
                batch_sums.index_add_(0, assignments, batch_compute)

                global_counts.index_add_(0, active, batch_counts[active])
                total_counts = global_counts[active]
                target = batch_sums[active] / batch_counts[active].unsqueeze(1)
                eta = batch_counts[active] / total_counts
                centroids[active] = (
                    (1 - eta.unsqueeze(1)) * centroids[active]
                    + eta.unsqueeze(1) * target
                )

            shift = torch.norm(centroids - previous, dim=1).mean()
            if shift <= self.tolerance:
                break

        return centroids

    def _assign(self, data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        assignments = torch.empty(data.size(0), dtype=torch.long, device=data.device)
        for start in range(0, data.size(0), self.chunk_size):
            end = min(start + self.chunk_size, data.size(0))
            chunk = data[start:end]
            if chunk.numel() == 0:
                continue
            chunk = chunk.to(device=centroids.device, dtype=self.compute_dtype)
            distances = _squared_euclidean_distances(chunk, centroids)
            assignments[start:end] = torch.argmin(distances, dim=1)
        return assignments

    def _update_centroids(
        self,
        data: torch.Tensor,
        assignments: torch.Tensor,
        num_centroids: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        device = data.device
        feature_dim = data.size(1)
        sums = torch.zeros(num_centroids, feature_dim, dtype=self.compute_dtype, device=device)
        for start in range(0, data.size(0), self.chunk_size):
            end = min(start + self.chunk_size, data.size(0))
            if start >= end:
                continue
            chunk_assignments = assignments[start:end]
            if chunk_assignments.numel() == 0:
                continue
            chunk = data[start:end].to(dtype=self.compute_dtype)
            sums.index_add_(0, chunk_assignments, chunk)
        counts = torch.bincount(assignments, minlength=num_centroids).to(self.compute_dtype)
        empty = counts == 0
        if empty.any():
            if generator is None:
                generator = torch.Generator(device=data.device)
                generator.manual_seed(torch.seed())
            replacement_indices = torch.randint(
                0,
                data.size(0),
                (int(empty.sum()),),
                device=data.device,
                generator=generator,
            )
            replacements = data[replacement_indices].to(dtype=self.compute_dtype)
            sums[empty] = replacements
            counts[empty] = 1.0
        centroids = sums / counts.unsqueeze(1)
        return centroids.contiguous()
