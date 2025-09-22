from __future__ import annotations

import ast
import csv
import json
import numpy as np
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple, Union

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


def _log_gpu_memory_usage(prefix: str = "") -> None:
    """Log current GPU memory usage (best effort)."""

    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_reserved = torch.cuda.max_memory_reserved() / 1024**3
    log.info(
        "[%s] GPU Memory - Allocated: %.2fGB, Reserved: %.2fGB, Max Reserved: %.2fGB",
        prefix,
        allocated,
        reserved,
        max_reserved,
    )


@dataclass(frozen=True)
class ItemText:
    """Pair each item identifier with the text used for embedding."""

    item_id: str
    text: str


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _flatten_categories(
    categories: Sequence[Sequence[str]] | Sequence[str] | str | None,
) -> str:
    if categories is None:
        return ""
    if isinstance(categories, str):
        return _normalize_space(categories)
    flattened: List[str] = []
    for entry in categories:
        if isinstance(entry, str):
            cleaned = _normalize_space(entry)
            if cleaned:
                flattened.append(cleaned)
        else:
            branch = [segment for segment in entry if isinstance(segment, str) and segment]
            if branch:
                flattened.append(" > ".join(_normalize_space(seg) for seg in branch))
    return "; ".join(flattened)



def iter_amazon_metadata(meta_path: Path) -> Iterator[ItemText]:
    """Yield ItemText entries from an Amazon `meta_*.json` dump."""

    with meta_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError) as exc:
                log.warning("Unable to parse %s line %s: %s", meta_path, line_number, exc)
                continue

            asin_raw = record.get("asin")
            asin = str(asin_raw).strip() if asin_raw is not None else ""
            if not asin:
                continue

            parts: List[str] = []
            title = record.get("title")
            if isinstance(title, str) and title.strip():
                parts.append(f"title: {_normalize_space(title)}")

            categories = record.get("categories")
            categories_str = _flatten_categories(categories)
            if categories_str:
                parts.append(f"categories: {categories_str}")

            description = record.get("description")
            if isinstance(description, str) and description.strip():
                parts.append(f"description: {_normalize_space(description)}")

            if not parts:
                text = f"asin: {asin}"
            else:
                parts.insert(0, f"asin: {asin}")
                text = " [SEP] ".join(parts)
            yield ItemText(item_id=asin, text=text)


def iter_serendipity_movies(csv_path: Path) -> Iterator[ItemText]:
    """Yield ItemText entries from Serendipity 2018 `movies.csv`."""

    with csv_path.open("r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            movie_id_raw = row.get("movieId")
            movie_id = str(movie_id_raw).strip() if movie_id_raw is not None else ""
            if not movie_id:
                continue

            parts: List[str] = []
            for column in ("title", "releaseDate", "directedBy", "starring", "genres"):
                value = row.get(column)
                if value:
                    normalized = _normalize_space(value)
                    if normalized:
                        parts.append(f"{column}: {normalized}")

            if not parts:
                text = f"movieId: {movie_id}"
            else:
                text = " [SEP] ".join(parts)
            yield ItemText(item_id=movie_id, text=text)



_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
}

_SUPPORTED_OUTPUT_FORMATS = {"parquet"}


def _resolve_storage_dtype(value: Union[str, torch.dtype, None]) -> torch.dtype:
    if value is None:
        return torch.float16
    if isinstance(value, torch.dtype):
        return value
    key = value.lower().strip()
    try:
        resolved = _DTYPE_ALIASES[key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Unsupported storage dtype '{value}'. Allowed: {', '.join(sorted(_DTYPE_ALIASES))}."
        ) from exc
    if resolved not in (torch.float16, torch.float32):
        raise ValueError("storage dtype must be either float16 or float32 for parquet outputs.")
    return resolved


def _resolve_compute_dtype(
    value: Union[str, torch.dtype, None],
    *,
    default: torch.dtype,
) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    key = value.lower().strip()
    try:
        return _DTYPE_ALIASES[key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Unsupported compute dtype '{value}'. Allowed: {', '.join(sorted(_DTYPE_ALIASES))}."
        ) from exc


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _resolve_output_format(output_path: Path | None, explicit: str | None) -> str:
    if explicit is not None:
        fmt = explicit.lower().strip()
        if fmt not in _SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output_format '{explicit}'. Allowed: {', '.join(sorted(_SUPPORTED_OUTPUT_FORMATS))}."
            )
    if output_path is not None:
        suffix = output_path.suffix.lower()
        if suffix and suffix != ".parquet":
            raise ValueError(
                f"Embedding outputs must use '.parquet' extension, got '{output_path.suffix}'."
            )
    return "parquet"


class EmbeddingWriter:
    """Strategy for persisting embedding payloads."""

    def __init__(self, storage_dtype: torch.dtype) -> None:
        self.storage_dtype = storage_dtype

    def save_full(self, path: Path, ids: List[str], embeddings: torch.Tensor) -> None:
        self._save(path, ids, embeddings)

    def save_chunk(self, path: Path, ids: List[str], embeddings: torch.Tensor) -> None:
        self._save(path, ids, embeddings)

    def count_items(self, path: Path) -> int:
        raise NotImplementedError

    def _save(self, path: Path, ids: List[str], embeddings: torch.Tensor) -> None:
        raise NotImplementedError


class ParquetEmbeddingWriter(EmbeddingWriter):
    """Persist embeddings to Parquet using PyArrow."""

    def __init__(self, storage_dtype: torch.dtype) -> None:
        super().__init__(storage_dtype)
        self._pa, self._pq = self._require_pyarrow()

    @staticmethod
    def _require_pyarrow():  # pragma: no cover - optional runtime dependency
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pyarrow is required for Parquet outputs. Install it with `pip install pyarrow`."
            ) from exc
        return pa, pq

    def _save(self, path: Path, ids: List[str], embeddings: torch.Tensor) -> None:
        embedding_tensor = embeddings.to(self.storage_dtype).cpu().contiguous()
        embedding_dim = embedding_tensor.shape[1] if embedding_tensor.ndim == 2 else 0
        ids_array = self._pa.array(ids, type=self._pa.string())
        if embedding_dim == 0:
            vectors = self._pa.FixedSizeListArray.from_arrays(
                self._pa.array([], type=self._pa.float16() if self.storage_dtype == torch.float16 else self._pa.float32()),
                0,
            )
        else:
            value_type = self._pa.float16() if self.storage_dtype == torch.float16 else self._pa.float32()
            flattened = self._pa.array(embedding_tensor.numpy().reshape(-1), type=value_type)
            vectors = self._pa.FixedSizeListArray.from_arrays(flattened, embedding_dim)
        table = self._pa.table({"item_id": ids_array, "embedding": vectors})
        self._pq.write_table(table, path)

    def count_items(self, path: Path) -> int:
        return self._pq.read_metadata(path).num_rows


def load_parquet_embeddings(
    path: Path | str | Sequence[Path | str],
    *,
    target_dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor]:
    try:  # pragma: no cover - optional runtime dependency
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to load Parquet embeddings. Install it with `pip install pyarrow`."
        ) from exc

    dtype_map = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
    }
    try:
        np_dtype = dtype_map[target_dtype]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported target dtype {target_dtype} for parquet embeddings.") from exc

    def _load_single(parquet_path: Path) -> tuple[list[str], torch.Tensor]:
        table = pq.read_table(parquet_path)
        if "item_id" not in table.column_names or "embedding" not in table.column_names:
            raise KeyError(
                f"Embedding parquet {parquet_path} must contain 'item_id' and 'embedding' columns."
            )

        item_ids = [str(value) for value in table.column("item_id").to_pylist()]
        embedding_column = table.column("embedding")
        embedding_type = embedding_column.type
        if not isinstance(embedding_type, pa.FixedSizeListType):
            raise TypeError(
                f"Embedding column in {parquet_path} must be FixedSizeList but was {embedding_type!r}."
            )

        embedding_dim = embedding_type.list_size
        matrices: list[np.ndarray] = []
        for chunk in embedding_column.chunks:
            if len(chunk) == 0:
                continue

            values = chunk.values
            value_offset = chunk.offset * embedding_dim
            value_length = len(chunk) * embedding_dim
            if value_offset or len(values) != value_length:
                values = values.slice(value_offset, value_length)

            try:
                chunk_array = values.to_numpy(zero_copy_only=False)
                if chunk_array.size != value_length:
                    raise ValueError("values buffer size mismatch")
                chunk_matrix = chunk_array.reshape(len(chunk), embedding_dim).astype(
                    np_dtype,
                    copy=False,
                )
            except Exception:
                # For some Arrow builds, list values cannot be reshaped reliably. Fall back to
                # stacking the list entries (with an unavoidable copy) to guarantee correctness.
                chunk_lists = chunk.to_numpy(zero_copy_only=False)
                if len(chunk_lists) == 0:
                    continue
                chunk_matrix = np.stack(chunk_lists).astype(np_dtype, copy=False)

            matrices.append(chunk_matrix)

        if matrices:
            stacked = np.concatenate(matrices, axis=0)
        else:
            stacked = np.empty((0, embedding_dim), dtype=np_dtype)

        embeddings_tensor = torch.from_numpy(stacked).to(target_dtype).contiguous()
        return item_ids, embeddings_tensor

    def _resolve_paths(base_path: Path) -> list[Path]:
        if base_path.exists():
            if base_path.is_dir():
                candidates = sorted(p for p in base_path.glob("*.parquet") if p.is_file())
                if not candidates:
                    raise FileNotFoundError(
                        f"Directory {base_path} does not contain any parquet files."
                    )
                return candidates
            return [base_path]

        suffix = base_path.suffix or ".parquet"
        pattern = f"{base_path.stem}_part*{suffix}"
        candidates = sorted(base_path.parent.glob(pattern))
        candidates = [candidate for candidate in candidates if candidate.is_file()]
        if candidates:
            return candidates

        raise FileNotFoundError(
            "Parquet file {0} not found and no chunked files matching pattern '{1}' were discovered.".format(
                base_path,
                pattern,
            )
        )

    def _load_multiple(paths_to_load: list[Path]) -> tuple[list[str], torch.Tensor]:
        if not paths_to_load:
            raise FileNotFoundError("No parquet files were provided for loading embeddings.")

        all_ids: list[str] = []
        tensors: list[torch.Tensor] = []
        embedding_dim = 0

        for candidate in paths_to_load:
            item_ids, embeddings_tensor = _load_single(candidate)
            all_ids.extend(item_ids)
            if embeddings_tensor.ndim == 2:
                embedding_dim = embeddings_tensor.shape[1]
            if embeddings_tensor.shape[0] > 0:
                tensors.append(embeddings_tensor)

        if tensors:
            combined = torch.cat(tensors, dim=0)
        else:
            combined = torch.empty((0, embedding_dim), dtype=target_dtype)

        return all_ids, combined.contiguous()

    if isinstance(path, SequenceABC) and not isinstance(path, (str, bytes, Path)):
        resolved = [Path(entry) for entry in path]
        return _load_multiple(resolved)

    resolved_paths = _resolve_paths(Path(path))
    return _load_multiple(resolved_paths)


def _build_writer(storage_dtype: torch.dtype) -> EmbeddingWriter:
    return ParquetEmbeddingWriter(storage_dtype)


def _format_chunk_path(base_path: Path, index: int) -> Path:
    suffix = base_path.suffix
    stem = base_path.stem
    return base_path.with_name(f"{stem}_part{index:04d}{suffix}")


class ChunkTracker:
    """Manage chunked output, including resume support and bookkeeping."""

    def __init__(
        self,
        base_path: Path,
        chunk_size: int,
        writer: EmbeddingWriter,
        *,
        resume: bool,
    ) -> None:
        self._base_path = base_path
        self._chunk_size = chunk_size
        self._writer = writer

        self._pending_ids: List[str] = []
        self._pending_embeddings: List[torch.Tensor] = []
        self._pending_count = 0

        self._next_chunk_index = 1
        self.existing_chunks = 0
        self.existing_items = 0
        self.written_chunks = 0
        self.written_items = 0

        self._completion_flag_path = (
            self._base_path.parent / f"{self._base_path.name}.complete"
        )
        self._inprogress_flag_path = (
            self._base_path.parent / f"{self._base_path.name}.inprogress"
        )
        self._confirmed_chunks, self._confirmed_items = self._read_completion_metadata()
        self.already_completed = False

        self._clear_inprogress_flag()

        if resume:
            self._initialise_resume_state()
            self.already_completed = self._resume_state_is_complete()

        if not self.already_completed:
            self._mark_inprogress()


    def _initialise_resume_state(self) -> None:
        index = 1
        while True:
            chunk_path = _format_chunk_path(self._base_path, index)
            if not chunk_path.exists():
                break
            count = self._writer.count_items(chunk_path)
            if count == 0:
                log.warning("Existing chunk %s is empty; removing for regeneration.", chunk_path)
                chunk_path.unlink(missing_ok=True)
                self._confirmed_chunks = min(self._confirmed_chunks, index - 1)
                break
            if count < self._chunk_size:
                if index <= self._confirmed_chunks:
                    log.info(
                        "Detected final chunk %s with %s items (<%s) from completed run; keeping it.",
                        chunk_path,
                        count,
                        self._chunk_size,
                    )
                    self.existing_chunks += 1
                    self.existing_items += count
                    index += 1
                    break
                log.warning(
                    "Existing chunk %s has only %s items (<%s). Removing to resume safely.",
                    chunk_path,
                    count,
                    self._chunk_size,
                )
                chunk_path.unlink()
                self._confirmed_chunks = min(self._confirmed_chunks, index - 1)
                break
            self.existing_chunks += 1
            self.existing_items += count
            index += 1
        self._next_chunk_index = self.existing_chunks + 1
        if self.existing_chunks:
            log.info(
                "Resuming from %s completed chunk(s) totalling %s items.",
                self.existing_chunks,
                self.existing_items,
            )

    @property
    def skip_items(self) -> int:
        return self.existing_items


    def _resume_state_is_complete(self) -> bool:
        if not self._completion_flag_path.exists():
            return False
        if self._confirmed_chunks == 0 and self._confirmed_items == 0:
            return self.existing_chunks == 0 and self.existing_items == 0
        if self.existing_chunks < self._confirmed_chunks:
            return False
        if self.existing_items < self._confirmed_items:
            return False
        if self.existing_chunks > self._confirmed_chunks:
            return False
        return True

    @property
    def completion_flag_path(self) -> Path:
        return self._completion_flag_path

    
    def add_batch(self, ids: List[str], embeddings: torch.Tensor) -> None:
        if not ids:
            return
        start = 0
        total = len(ids)
        while start < total:
            capacity = self._chunk_size - self._pending_count
            if capacity <= 0:
                self._flush_pending()
                capacity = self._chunk_size - self._pending_count
                if capacity <= 0:  # pragma: no cover - defensive
                    raise RuntimeError("Unable to flush chunk buffer")
            end = min(start + capacity, total)
            self._pending_ids.extend(ids[start:end])
            self._pending_embeddings.append(embeddings[start:end])
            added = end - start
            self._pending_count += added
            start = end
            if self._pending_count >= self._chunk_size:
                self._flush_pending()

    def finalize(self) -> None:
        self._flush_pending(force=True)
        self._mark_completed()

    def summary(self) -> tuple[int, int, int]:
        total_chunks = self.existing_chunks + self.written_chunks
        total_items = self.existing_items + self.written_items
        return total_chunks, total_items, self.written_items

    
    def _flush_pending(self, force: bool = False) -> None:
        if not self._pending_ids:
            return
        if not force and self._pending_count < self._chunk_size:
            return

        embeddings_tensor = self._concatenate_pending()
        chunk_path = _format_chunk_path(self._base_path, self._next_chunk_index)
        self._writer.save_chunk(chunk_path, self._pending_ids, embeddings_tensor)
        self.written_chunks += 1
        self.written_items += embeddings_tensor.shape[0]
        log.info(
            "Saved chunk %04d with %s items to %s",
            self._next_chunk_index,
            embeddings_tensor.shape[0],
            chunk_path,
        )

        self._next_chunk_index += 1
        self._pending_ids.clear()
        self._pending_embeddings.clear()
        self._pending_count = 0

    def _concatenate_pending(self) -> torch.Tensor:
        if len(self._pending_embeddings) == 1:
            return self._pending_embeddings[0]
        if not self._pending_embeddings:
            return torch.empty((0, 0))
        return torch.cat(self._pending_embeddings, dim=0)

    def _mark_completed(self) -> None:
        total_chunks = self.existing_chunks + self.written_chunks
        total_items = self.existing_items + self.written_items
        self._clear_inprogress_flag()
        payload = {"chunks": total_chunks, "items": total_items}
        try:
            self._completion_flag_path.write_text(json.dumps(payload))
        except FileNotFoundError:
            self._completion_flag_path.parent.mkdir(parents=True, exist_ok=True)
            self._completion_flag_path.write_text(json.dumps(payload))
        self._confirmed_chunks = total_chunks
        self._confirmed_items = total_items
        self.already_completed = True

    def _read_completion_metadata(self) -> tuple[int, int]:
        if not self._completion_flag_path.exists():
            return 0, 0
        try:
            data = json.loads(self._completion_flag_path.read_text())
            chunks = int(data.get("chunks", 0))
            items = int(data.get("items", 0))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            log.warning(
                "Failed to parse completion metadata at %s (%s); ignoring it.",
                self._completion_flag_path,
                exc,
            )
            return 0, 0
        return chunks, items

    def _mark_inprogress(self) -> None:
        try:
            self._inprogress_flag_path.touch()
        except FileNotFoundError:
            self._inprogress_flag_path.parent.mkdir(parents=True, exist_ok=True)
            self._inprogress_flag_path.touch()

    def _clear_inprogress_flag(self) -> None:
        if self._inprogress_flag_path.exists():
            self._inprogress_flag_path.unlink(missing_ok=True)



def generate_item_embeddings(
    records: Iterable[ItemText],
    *,
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    batch_size: int = 8,
    device: str | None = None,
    normalize_embeddings: bool = True,
    output_path: Path | None = None,
    trust_remote_code: bool = True,
    use_flash_attention: bool = True,
    use_fp16: bool = True,
    padding_side: str = "left",
    truncate_dim: int | None = None,
    storage_dtype: Union[str, torch.dtype, None] = torch.float16,
    compute_dtype: Union[str, torch.dtype, None] = None,
    max_items_per_file: int | None = None,
    output_format: str | None = None,
    resume: bool = False,
) -> Tuple[List[str], torch.Tensor]:
    """Embed the provided records with Qwen's encoder via SentenceTransformers."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    storage_dtype_resolved = _resolve_storage_dtype(storage_dtype)
    default_compute_dtype = torch.float16 if use_fp16 else torch.float32
    compute_dtype_resolved = _resolve_compute_dtype(
        compute_dtype, default=default_compute_dtype
    )
    resolved_format = _resolve_output_format(output_path, output_format)
    writer = _build_writer(storage_dtype_resolved)

    chunk_tracker: ChunkTracker | None = None
    skip_items = 0
    if max_items_per_file is not None:
        if max_items_per_file <= 0:
            raise ValueError("max_items_per_file must be positive when provided")
        if output_path is None:
            raise ValueError("output_path must be provided when max_items_per_file is set")
        chunk_tracker = ChunkTracker(
            output_path,
            int(max_items_per_file),
            writer,
            resume=resume,
        )
        if chunk_tracker.already_completed:
            log.info(
                "Found completion marker at %s; skipping embedding regeneration.",
                chunk_tracker.completion_flag_path,
            )
            return [], torch.empty((0, 0), dtype=storage_dtype_resolved)
        skip_items = chunk_tracker.skip_items
        log.info(
            "Chunked output enabled: base=%s, max_items_per_file=%s, format=%s",
            output_path,
            max_items_per_file,
            resolved_format,
        )
    elif resume and output_path and output_path.exists():
        log.info("Output %s already exists; resume requested so skipping regeneration.", output_path)
        ids, embeddings_tensor = load_parquet_embeddings(output_path, target_dtype=storage_dtype_resolved)
        return ids, embeddings_tensor

    
    model_kwargs = {}
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        log.info("Using FlashAttention-2")
    if compute_dtype_resolved != torch.float32:
        model_kwargs["dtype"] = compute_dtype_resolved

    truncate_kwargs = {"truncate_dim": int(truncate_dim)} if truncate_dim else {}

    sentence_model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        model_kwargs=model_kwargs,
        **truncate_kwargs,
    )
    if use_flash_attention:
        log.info("FlashAttention-2 enabled successfully")
    if hasattr(sentence_model, "tokenizer") and sentence_model.tokenizer:
        sentence_model.tokenizer.padding_side = padding_side
        log.info("Set tokenizer padding_side to '%s'", padding_side)
    if compute_dtype_resolved != torch.float32:
        try:
            sentence_model = sentence_model.to(dtype=compute_dtype_resolved)
            log.info(
                "Enabled %s precision for the model",
                _dtype_to_str(compute_dtype_resolved),
            )
        except (TypeError, RuntimeError) as exc:
            log.warning(
                "Unable to convert model to %s precision; continuing with original dtype. %s",
                _dtype_to_str(compute_dtype_resolved),
                exc,
            )

    final_dim = sentence_model.get_sentence_embedding_dimension()
    if truncate_dim and final_dim != int(truncate_dim):
        log.warning(
            "Requested truncate_dim=%s but model reports %s. Proceeding with reported value.",
            truncate_dim,
            final_dim,
        )
    log.info(
        "Loaded sentence-transformer %s on %s with embedding dim %s",
        model_name,
        sentence_model.device,
        final_dim,
    )
    _log_gpu_memory_usage("After model loading")

    collected_ids: List[str] | None = [] if chunk_tracker is None else None
    collected_embeddings: List[torch.Tensor] | None = [] if chunk_tracker is None else None

    buffer_ids: List[str] = []
    buffer_texts: List[str] = []
    current_batch_size = batch_size

    def _persist_batch(ids: List[str], embeddings: torch.Tensor) -> None:
        if not ids:
            return
        if chunk_tracker is not None:
            chunk_tracker.add_batch(ids, embeddings)
        else:
            assert collected_ids is not None and collected_embeddings is not None
            collected_ids.extend(ids)
            collected_embeddings.append(embeddings)

    def flush_buffer() -> None:
        nonlocal current_batch_size
        if not buffer_texts:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        while True:
            try:
                with torch.inference_mode():
                    embeddings_np = sentence_model.encode(
                        buffer_texts,
                        batch_size=current_batch_size,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=normalize_embeddings,
                    )
                break
            except torch.cuda.OutOfMemoryError as exc:
                if current_batch_size == 1:
                    raise
                new_batch_size = max(1, current_batch_size // 2)
                log.warning(
                    "CUDA OOM detected during embedding (requested %d -> reducing batch_size to %d). %s",
                    current_batch_size,
                    new_batch_size,
                    exc,
                )
                current_batch_size = new_batch_size
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" not in message:
                    raise
                if current_batch_size == 1:
                    raise
                new_batch_size = max(1, current_batch_size // 2)
                log.warning(
                    "Runtime OOM during embedding (requested %d -> reducing batch_size to %d). %s",
                    current_batch_size,
                    new_batch_size,
                    exc,
                )
                current_batch_size = new_batch_size
                torch.cuda.empty_cache()

        embeddings = torch.from_numpy(embeddings_np)
        embeddings = embeddings.to(
            device=sentence_model.device,
            dtype=compute_dtype_resolved,
        )
        embeddings = embeddings.to("cpu", storage_dtype_resolved)

        _persist_batch(list(buffer_ids), embeddings)
        buffer_ids.clear()
        buffer_texts.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    if hasattr(records, "__len__"):
        try:
            total_items = len(records)  # type: ignore[arg-type]
        except TypeError:
            total_items = None
    else:
        total_items = None

    total_msg = f", total={total_items}" if total_items is not None else ""
    log.info(
        "Processing item texts for embedding generation (initial batch_size=%d%s)",
        batch_size,
        total_msg,
    )

    iterator = records if hasattr(records, "__iter__") else iter(records)
    processed_non_empty = 0
    progress = tqdm(iterator, desc="Processing items", total=total_items, unit="items")
    for item in progress:
        if not item.text.strip():
            continue
        processed_non_empty += 1
        if skip_items and processed_non_empty <= skip_items:
            continue
        buffer_ids.append(item.item_id)
        buffer_texts.append(item.text)
        if len(buffer_ids) >= current_batch_size:
            flush_buffer()

    flush_buffer()

    if chunk_tracker is not None:
        chunk_tracker.finalize()
        total_chunks, total_items_written, new_items = chunk_tracker.summary()
        log.info(
            "Chunked embedding generation complete: new_items=%s, total_items=%s, chunks=%s",
            new_items,
            total_items_written,
            total_chunks,
        )
        return [], torch.empty((0, final_dim), dtype=storage_dtype_resolved)

    if collected_embeddings:
        embeddings_tensor = torch.cat(collected_embeddings, dim=0)
    else:
        embeddings_tensor = torch.empty((0, final_dim), dtype=storage_dtype_resolved)

    final_dim = embeddings_tensor.shape[1] if embeddings_tensor.numel() > 0 else final_dim
    log.info(
        "Final embedding tensor shape: %s (dimension: %d, storage_dtype=%s)",
        embeddings_tensor.shape,
        final_dim,
        storage_dtype_resolved,
    )

    if output_path is not None:
        writer.save_full(output_path, collected_ids or [], embeddings_tensor)
        log.info("Saved %s item embeddings to %s", len(collected_ids or []), output_path)

    return collected_ids or [], embeddings_tensor
