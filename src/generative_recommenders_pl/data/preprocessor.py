from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRETRAIN_ROOT = PROJECT_ROOT / "tmp" / "pre_train"
DEFAULT_TUNING_ROOT = PROJECT_ROOT / "tmp" / "tuning"


def _format_sequence(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def _read_movie_ids(movies_path: Path) -> list[str]:
    movie_ids: list[str] = []
    seen: set[str] = set()

    with movies_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, raw_line in enumerate(handle):
            if not raw_line:
                continue
            first_token = raw_line.split(",", 1)[0].strip()
            if first_token.startswith("\ufeff"):
                first_token = first_token.replace("\ufeff", "", 1)
            first_token = first_token.strip('"').strip()
            if not first_token:
                continue
            if line_number == 0 and first_token.lower() == "movieid":
                continue
            if first_token not in seen:
                seen.add(first_token)
                movie_ids.append(first_token)

    if not movie_ids:
        raise ValueError(f"No movie ids could be read from {movies_path}")
    return movie_ids


def _resolve_reference_column(
    columns: Iterable[str],
    *,
    explicit: str | None,
    candidates: Iterable[str],
    label: str,
    reference_path: Path,
) -> str:
    available = list(columns)
    if explicit is not None:
        if explicit not in available:
            raise ValueError(
                f"Reference file {reference_path} is missing explicit {label} column "
                f"{explicit!r}; available columns: {available}"
            )
        return explicit
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(
        f"Reference file {reference_path} is missing a {label} column. "
        f"Tried {list(candidates)}; available columns: {available}"
    )


@dataclass(frozen=True)
class LeakageAuditSummary:
    source_before: int
    target_pairs: int
    removed_target_pairs: int
    removed_post_target_interactions: int
    ambiguous: int
    source_after: int


def _load_reference_interactions(
    reference_path: Path,
    *,
    user_column: str | None = None,
    item_column: str | None = None,
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference interactions file not found: {reference_path}")

    header = pd.read_csv(reference_path, nrows=0, encoding="utf-8-sig")
    user_col = _resolve_reference_column(
        header.columns,
        explicit=user_column,
        candidates=("user_id", "userId"),
        label="user",
        reference_path=reference_path,
    )
    item_col = _resolve_reference_column(
        header.columns,
        explicit=item_column,
        candidates=("item_id", "movieId", "itemId"),
        label="item",
        reference_path=reference_path,
    )
    timestamp_col = _resolve_reference_column(
        header.columns,
        explicit=timestamp_column,
        candidates=("timestamp", "unix_timestamp"),
        label="timestamp",
        reference_path=reference_path,
    )

    reference = pd.read_csv(
        reference_path,
        usecols=[user_col, item_col, timestamp_col],
        encoding="utf-8-sig",
    )
    reference.rename(
        columns={
            user_col: "user_id",
            item_col: "item_id",
            timestamp_col: "timestamp",
        },
        inplace=True,
    )
    reference["user_id"] = reference["user_id"].astype("object")
    reference["item_id"] = reference["item_id"].astype("object")
    reference["timestamp"] = pd.to_numeric(reference["timestamp"], errors="coerce")

    invalid_mask = (
        reference["timestamp"].isna()
        | reference["user_id"].isna()
        | reference["item_id"].isna()
    )
    if invalid_mask.any():
        dropped = int(invalid_mask.sum())
        log.warning(
            "Dropping %s invalid reference rows from %s before leakage filtering",
            dropped,
            reference_path,
        )
        reference = reference.loc[~invalid_mask].reset_index(drop=True)

    reference["user_id"] = reference["user_id"].map(lambda value: str(value).strip())
    reference["item_id"] = reference["item_id"].map(lambda value: str(value).strip())
    empty_mask = (reference["user_id"] == "") | (reference["item_id"] == "")
    if empty_mask.any():
        dropped = int(empty_mask.sum())
        log.warning(
            "Dropping %s empty-id reference rows from %s before leakage filtering",
            dropped,
            reference_path,
        )
        reference = reference.loc[~empty_mask].reset_index(drop=True)

    if reference.empty:
        raise ValueError(f"No valid reference rows could be loaded from {reference_path}")

    reference["timestamp"] = reference["timestamp"].astype("int64")
    reference["_normalized_user_id"] = reference["user_id"].str.casefold()
    reference["_normalized_item_id"] = reference["item_id"].str.casefold()
    return reference


def load_reference_interactions(
    reference_path: str | Path,
    *,
    user_column: str | None = None,
    item_column: str | None = None,
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    return _load_reference_interactions(
        Path(reference_path),
        user_column=user_column,
        item_column=item_column,
        timestamp_column=timestamp_column,
    )


def apply_source_target_leakage_filter(
    ratings: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    prefix: str = "eda",
    reference_name: str = "reference",
    remove_post_reference_interactions: bool = True,
    return_details: bool = False,
) -> tuple[pd.DataFrame, LeakageAuditSummary, dict[str, pd.DataFrame]] | tuple[
    pd.DataFrame, LeakageAuditSummary
]:
    if ratings.empty or reference.empty:
        summary = LeakageAuditSummary(
            source_before=int(ratings.shape[0]),
            target_pairs=int(
                reference[["_normalized_user_id", "_normalized_item_id"]]
                .drop_duplicates()
                .shape[0]
            )
            if not reference.empty
            else 0,
            removed_target_pairs=0,
            removed_post_target_interactions=0,
            ambiguous=0,
            source_after=int(ratings.shape[0]),
        )
        if return_details:
            empty = pd.DataFrame()
            return ratings.copy(), summary, {
                "removed_target_pairs": empty,
                "removed_post_target_interactions": empty,
                "ambiguous_cases": empty,
            }
        return ratings.copy(), summary

    working = ratings.copy()
    source_user = working["user_id"].astype(str).str.casefold()
    source_item = working["item_id"].astype(str).str.casefold()
    source_pairs = pd.MultiIndex.from_arrays([source_user, source_item])
    reference_pairs = pd.MultiIndex.from_frame(
        reference[["_normalized_user_id", "_normalized_item_id"]].drop_duplicates()
    )
    pair_overlap_mask = pd.Series(
        source_pairs.isin(reference_pairs),
        index=working.index,
        dtype=bool,
    )

    source_timestamps = pd.to_numeric(working["timestamp"], errors="coerce")
    reference_cutoffs = reference.groupby("_normalized_user_id", sort=False)[
        "timestamp"
    ].min()
    source_cutoffs = source_user.map(reference_cutoffs)
    ambiguous_timestamp_mask = source_cutoffs.notna() & (
        source_timestamps == source_cutoffs
    )
    post_reference_mask = source_cutoffs.notna() & (source_timestamps > source_cutoffs)
    if remove_post_reference_interactions:
        cutoff_mask = ambiguous_timestamp_mask | post_reference_mask
    else:
        cutoff_mask = pd.Series(False, index=working.index, dtype=bool)
    removal_mask = pair_overlap_mask | cutoff_mask

    removed_pairs = int(pair_overlap_mask.sum())
    removed_post_reference = int((post_reference_mask & ~pair_overlap_mask).sum())
    ambiguous_rows = int((ambiguous_timestamp_mask & ~pair_overlap_mask).sum())
    removed_total = int(removal_mask.sum())

    if removed_total:
        log.info(
            (
                "%s applied %s leakage guard: removed=%s "
                "(pair_overlap=%s, post_reference=%s, ambiguous_boundary=%s)"
            ),
            prefix,
            reference_name,
            removed_total,
            removed_pairs,
            removed_post_reference,
            ambiguous_rows,
        )

    cleaned = working.loc[~removal_mask].reset_index(drop=True)
    summary = LeakageAuditSummary(
        source_before=int(working.shape[0]),
        target_pairs=int(reference_pairs.shape[0]),
        removed_target_pairs=removed_pairs,
        removed_post_target_interactions=removed_post_reference,
        ambiguous=ambiguous_rows,
        source_after=int(cleaned.shape[0]),
    )
    if not return_details:
        return cleaned, summary

    details = {
        "removed_target_pairs": working.loc[pair_overlap_mask].reset_index(drop=True),
        "removed_post_target_interactions": working.loc[
            post_reference_mask & ~pair_overlap_mask
        ].reset_index(drop=True),
        "ambiguous_cases": working.loc[
            ambiguous_timestamp_mask & ~pair_overlap_mask
        ].reset_index(drop=True),
    }
    return cleaned, summary, details


def _apply_reference_leakage_filter(
    ratings: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    prefix: str,
    reference_name: str,
    remove_post_reference_interactions: bool = True,
) -> pd.DataFrame:
    cleaned, _summary = apply_source_target_leakage_filter(
        ratings,
        reference,
        prefix=prefix,
        reference_name=reference_name,
        remove_post_reference_interactions=remove_post_reference_interactions,
    )
    return cleaned


def _ensure_lookup_series(
    values: pd.Series,
    lookup_path: Path,
    *,
    normalized_col: str,
    original_col: str,
    initial_entries: Iterable[str] | None = None,
    key_normalizer: Callable[[str], str] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Map values to integer ids, updating the lookup file in-place if needed."""

    lookup_path.parent.mkdir(parents=True, exist_ok=True)

    if lookup_path.exists():
        lookup = pd.read_csv(lookup_path, dtype={normalized_col: "int64"})
        lookup[original_col] = lookup[original_col].astype(str)
    else:
        lookup = pd.DataFrame(columns=[normalized_col, original_col])

    def _normalize_key(raw_key: str) -> str:
        return key_normalizer(raw_key) if key_normalizer else raw_key

    existing_original = lookup[original_col].astype("object").map(str)
    existing_normalized = existing_original.map(_normalize_key)
    mapping: dict[str, int] = dict(
        zip(existing_normalized, lookup[normalized_col].astype("int64"))
    )

    next_id = int(lookup[normalized_col].max()) + 1 if not lookup.empty else 0
    updated = False

    def _add_entries(raw_series: pd.Series) -> None:
        nonlocal next_id, updated, lookup, mapping
        if raw_series.empty:
            return
        raw_series = raw_series.astype("object")
        normalized_series = raw_series.map(_normalize_key) if key_normalizer else raw_series
        mask = ~normalized_series.isin(mapping)
        if not mask.any():
            return
        new_normalized = normalized_series.loc[mask].drop_duplicates(keep="first")
        new_original = raw_series.loc[new_normalized.index].map(str)
        new_ids = range(next_id, next_id + len(new_normalized))
        next_id += len(new_normalized)
        mapping.update(zip(new_normalized.values, new_ids))
        new_lookup_rows = pd.DataFrame(
            {normalized_col: list(new_ids), original_col: new_original.values}
        )
        lookup = pd.concat([lookup, new_lookup_rows], ignore_index=True)
        updated = True

    if initial_entries is not None:
        initial_series = pd.Series(list(initial_entries), dtype="object").map(str)
        _add_entries(initial_series)

    value_series = values.astype("object").map(str)
    _add_entries(value_series)

    normalized_series = (
        value_series.map(_normalize_key)
        if key_normalizer is not None
        else value_series
    )
    normalized_values = normalized_series.map(mapping).astype("int64")

    if updated or not lookup_path.exists():
        lookup[normalized_col] = lookup[normalized_col].astype(int)
        lookup[original_col] = lookup[original_col].astype(str)
        lookup.sort_values(by=normalized_col, inplace=True)
        lookup.reset_index(drop=True, inplace=True)
        lookup.to_csv(lookup_path, index=False)

    return normalized_values, lookup


def _build_sequence_frame(
    ratings: pd.DataFrame,
    *,
    min_length: int = 1,
    extra_sequence_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if extra_sequence_columns:
        missing = [col for col in extra_sequence_columns if col not in ratings.columns]
        if missing:
            raise ValueError(f"Missing columns required for sequences: {missing}")

    ordered = ratings.sort_values(by=["user_id", "timestamp"], kind="mergesort")
    grouped = ordered.groupby("user_id", sort=False)

    agg_spec: dict[str, list] = {
        "item_id": list,
        "rating": list,
        "timestamp": list,
    }
    if extra_sequence_columns:
        for source in extra_sequence_columns:
            agg_spec[source] = list

    seq_df = grouped.agg(agg_spec).reset_index()
    seq_df.rename(
        columns={
            "item_id": "sequence_item_ids",
            "rating": "sequence_ratings",
            "timestamp": "sequence_timestamps",
        },
        inplace=True,
    )

    lengths = seq_df["sequence_item_ids"].apply(len)
    if min_length > 1:
        mask = lengths >= min_length
        seq_df = seq_df.loc[mask].reset_index(drop=True)
        lengths = lengths.loc[mask].reset_index(drop=True)

    seq_df["sequence_item_ids"] = seq_df["sequence_item_ids"].apply(_format_sequence)
    seq_df["sequence_ratings"] = seq_df["sequence_ratings"].apply(_format_sequence)
    seq_df["sequence_timestamps"] = seq_df["sequence_timestamps"].apply(_format_sequence)

    if extra_sequence_columns:
        for source, target in extra_sequence_columns.items():
            seq_df.rename(columns={source: target}, inplace=True)
            seq_df[target] = seq_df[target].apply(_format_sequence)

    return seq_df, lengths


class DataProcessor(abc.ABC):
    """Base preprocessor that materializes SASRec-style sequential data."""

    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
        *,
        output_root: str | Path | None = None,
        lookup_root: str | Path | None = None,
        output_dirname: str | None = None,
        lookup_dirname: str | None = None,
    ) -> None:
        self._prefix = prefix
        self._expected_num_unique_items = expected_num_unique_items
        self._expected_max_item_id = expected_max_item_id
        self._output_root = Path(output_root) if output_root is not None else DEFAULT_PRETRAIN_ROOT
        self._lookup_root = Path(lookup_root) if lookup_root is not None else self._output_root
        self._output_dirname = str(output_dirname) if output_dirname is not None else prefix
        self._lookup_dirname = str(lookup_dirname) if lookup_dirname is not None else prefix
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._lookup_root.mkdir(parents=True, exist_ok=True)

    def output_dir(self) -> Path:
        return self._output_root / self._output_dirname

    def lookup_dir(self) -> Path:
        return self._lookup_root / self._lookup_dirname

    def output_format_csv(self) -> str:
        return str(self.output_dir() / "sasrec_format.csv")

    def normalized_ratings_csv(self) -> str:
        return str(self.output_dir() / "ratings.csv")

    def user_lookup_csv(self) -> str:
        return str(self.lookup_dir() / "user_lookup.csv")

    def item_lookup_csv(self) -> str:
        return str(self.lookup_dir() / "item_lookup.csv")

    def ensure_output_dir(self) -> None:
        self.output_dir().mkdir(parents=True, exist_ok=True)

    def expected_num_unique_items(self) -> Optional[int]:
        self._ensure_item_statistics()
        return self._expected_num_unique_items

    def expected_max_item_id(self) -> Optional[int]:
        self._ensure_item_statistics()
        return self._expected_max_item_id

    def _ensure_item_statistics(self) -> None:
        if (
            self._expected_num_unique_items is not None
            and self._expected_max_item_id is not None
        ):
            return
        lookup_path = Path(self.item_lookup_csv())
        if not Path(lookup_path).exists():
            return
        try:
            lookup = pd.read_csv(lookup_path)
        except Exception as exc:  # pragma: no cover - informative fallback
            log.warning(
                "%s failed to load item lookup metadata: %s",
                self._prefix,
                exc,
            )
            return
        if self._expected_num_unique_items is None:
            self._expected_num_unique_items = int(lookup.shape[0])
        if (
            self._expected_max_item_id is None
            and "normalized_item_id" in lookup.columns
        ):
            self._expected_max_item_id = int(lookup["normalized_item_id"].max())

    @abc.abstractmethod
    def preprocess_rating(self) -> int:
        """Process source data and return the number of unique items."""


class MovielensDataProcessor(DataProcessor):
    def __init__(
        self,
        ratings_path: str | Path,
        prefix: str,
        *,
        movies_path: str | Path | None = None,
        min_sequence_length: int = 1,
        min_presence: int = 1,
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
        leakage_reference_path: str | Path | None = None,
        leakage_reference_user_column: str | None = None,
        leakage_reference_item_column: str | None = None,
        leakage_reference_timestamp_column: str | None = None,
        remove_post_reference_interactions: bool = True,
        output_root: str | Path | None = None,
        lookup_root: str | Path | None = None,
        output_dirname: str | None = None,
        lookup_dirname: str | None = None,
        extra_sequence_columns: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=expected_max_item_id,
            output_root=output_root,
            lookup_root=lookup_root,
            output_dirname=output_dirname,
            lookup_dirname=lookup_dirname,
        )
        self._ratings_path = Path(ratings_path)
        self._movies_path = Path(movies_path) if movies_path else None
        self._min_sequence_length = max(1, min_sequence_length)
        self._min_presence = max(1, min_presence)
        self._extra_sequence_columns = extra_sequence_columns or {}
        self._leakage_reference_path = (
            Path(leakage_reference_path) if leakage_reference_path else None
        )
        self._leakage_reference_user_column = leakage_reference_user_column
        self._leakage_reference_item_column = leakage_reference_item_column
        self._leakage_reference_timestamp_column = leakage_reference_timestamp_column
        self._remove_post_reference_interactions = bool(
            remove_post_reference_interactions
        )

    def _transform_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        return ratings

    def _user_lookup_initial_entries(self) -> Iterable[str] | None:
        return None

    def _item_lookup_initial_entries(self) -> Iterable[str] | None:
        return _read_movie_ids(self._movies_path) if self._movies_path else None

    def preprocess_rating(self) -> int:
        if not self._ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {self._ratings_path}")

        ratings = pd.read_csv(self._ratings_path)
        rename_map = {
            "userId": "user_id",
            "movieId": "item_id",
            "unix_timestamp": "timestamp",
        }
        ratings.rename(columns=rename_map, inplace=True)

        required = ["user_id", "item_id", "rating", "timestamp"]
        missing = set(required) - set(ratings.columns)
        if missing:
            raise ValueError(
                f"Missing required columns {missing} in ratings file {self._ratings_path}"
            )

        ratings = self._transform_ratings(ratings)
        ratings = ratings[required + list(self._extra_sequence_columns.keys())]

        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")
        if ratings["rating"].isnull().any():
            raise ValueError("Ratings column contains non-numeric values.")
        if ratings["timestamp"].isnull().any():
            raise ValueError("Timestamp column contains non-numeric values.")

        ratings["rating"] = ratings["rating"].astype(float)
        ratings["timestamp"] = ratings["timestamp"].astype("int64")

        if self._leakage_reference_path is not None:
            reference = _load_reference_interactions(
                self._leakage_reference_path,
                user_column=self._leakage_reference_user_column,
                item_column=self._leakage_reference_item_column,
                timestamp_column=self._leakage_reference_timestamp_column,
            )
            ratings = _apply_reference_leakage_filter(
                ratings,
                reference,
                prefix=self._prefix,
                reference_name=self._leakage_reference_path.name,
                remove_post_reference_interactions=self._remove_post_reference_interactions,
            )
            if ratings.empty:
                raise ValueError(
                    f"No ratings remaining after applying leakage reference filter from "
                    f"{self._leakage_reference_path}"
                )

        if self._min_presence > 1:
            initial_rows = ratings.shape[0]
            item_counts = ratings["item_id"].value_counts()
            user_counts = ratings["user_id"].value_counts()
            mask = (ratings["item_id"].map(item_counts) >= self._min_presence) & (
                ratings["user_id"].map(user_counts) >= self._min_presence
            )
            ratings = ratings.loc[mask].reset_index(drop=True)
            filtered_rows = ratings.shape[0]
            if filtered_rows == 0:
                raise ValueError(
                    f"No ratings remaining after applying min_presence={self._min_presence}"
                )
            if filtered_rows != initial_rows:
                log.info(
                    "%s applied min_presence=%s filter (%s -> %s rows)",
                    self._prefix,
                    self._min_presence,
                    initial_rows,
                    filtered_rows,
                )

        initial_user_entries = self._user_lookup_initial_entries()
        user_ids, user_lookup = _ensure_lookup_series(
            ratings["user_id"],
            Path(self.user_lookup_csv()),
            normalized_col="normalized_user_id",
            original_col="original_user_id",
            initial_entries=initial_user_entries,
            key_normalizer=str.casefold,
        )
        ratings["user_id"] = user_ids

        initial_movie_ids = self._item_lookup_initial_entries()
        item_ids, item_lookup = _ensure_lookup_series(
            ratings["item_id"],
            Path(self.item_lookup_csv()),
            normalized_col="normalized_item_id",
            original_col="original_item_id",
            initial_entries=initial_movie_ids,
            key_normalizer=str.casefold,
        )
        ratings["item_id"] = item_ids

        self.ensure_output_dir()

        num_items = item_lookup.shape[0]
        self._expected_num_unique_items = num_items
        self._expected_max_item_id = int(item_lookup["normalized_item_id"].max())

        seq_df, lengths = _build_sequence_frame(
            ratings,
            min_length=self._min_sequence_length,
            extra_sequence_columns=self._extra_sequence_columns,
        )
        if not lengths.empty:
            log.info(
                "%s sequence length stats min=%s max=%s mean=%.2f median=%.2f",
                self._prefix,
                lengths.min(),
                lengths.max(),
                lengths.mean(),
                lengths.median(),
            )
        else:
            log.warning("No sequences generated for %s", self._prefix)

        ratings.sort_values(by=["user_id", "timestamp"], inplace=True)
        ratings.reset_index(drop=True, inplace=True)

        ratings.to_csv(self.normalized_ratings_csv(), index=False)
        seq_df.reset_index().to_csv(self.output_format_csv(), index=False)
        user_lookup.to_csv(self.user_lookup_csv(), index=False)
        item_lookup.to_csv(self.item_lookup_csv(), index=False)

        log.info(
            "%s unique users=%s unique items=%s",
            self._prefix,
            user_lookup.shape[0],
            num_items,
        )
        return num_items


class SerendipityAnswersDataProcessor(MovielensDataProcessor):
    def __init__(
        self,
        ratings_path: str | Path,
        prefix: str = "serendipity-2018-answers",
        *,
        movies_path: str | Path | None = None,
        training_path: str | Path | None = None,
        min_sequence_length: int = 1,
        output_root: str | Path | None = None,
        lookup_root: str | Path | None = None,
        output_dirname: str | None = None,
        lookup_dirname: str | None = None,
        use_training_initial_entries: bool = True,
        use_movies_initial_entries: bool = True,
        serendipity_columns: Optional[list[str]] = None,
        min_presence: int = 1,
    ) -> None:
        self._ser_columns = serendipity_columns or [
            "s_ser_find",
            "s_ser_imp",
            "s_ser_rec",
            "m_ser_find",
            "m_ser_imp",
            "m_ser_rec",
        ]
        if training_path is None:
            raise ValueError(
                "SerendipityAnswersDataProcessor requires a training_path to build user lookups"
            )
        self._training_path = Path(training_path)
        self._cached_training_user_ids: list[str] | None = None
        self._use_training_initial_entries = bool(use_training_initial_entries)
        self._use_movies_initial_entries = bool(use_movies_initial_entries)
        super().__init__(
            ratings_path,
            prefix,
            movies_path=movies_path,
            min_sequence_length=min_sequence_length,
            output_root=output_root,
            lookup_root=lookup_root,
            output_dirname=output_dirname,
            lookup_dirname=lookup_dirname,
            min_presence=min_presence,
            extra_sequence_columns={"ser_label": "sequence_ser_label"},
        )

    def _user_lookup_initial_entries(self) -> Iterable[str] | None:
        if not self._use_training_initial_entries:
            return None
        if self._cached_training_user_ids is not None:
            return self._cached_training_user_ids
        if not self._training_path.exists():
            raise FileNotFoundError(
                f"Training file not found for Serendipity answers preprocessing: {self._training_path}"
            )
        try:
            training_users = pd.read_csv(
                self._training_path,
                usecols=["userId"],
            )
            column_name = "userId"
        except ValueError:
            try:
                training_users = pd.read_csv(
                    self._training_path,
                    usecols=["user_id"],
                )
                column_name = "user_id"
            except ValueError as secondary_exc:
                raise ValueError(
                    "Training file for Serendipity answers must contain a userId column"
                ) from secondary_exc
            else:
                log.warning(
                    "userId column missing from %s, using user_id instead",
                    self._training_path,
                )
        training_users[column_name] = training_users[column_name].astype(str)
        self._cached_training_user_ids = (
            training_users[column_name]
            .dropna()
            .drop_duplicates()
            .tolist()
        )
        return self._cached_training_user_ids

    def _item_lookup_initial_entries(self) -> Iterable[str] | None:
        if not self._use_movies_initial_entries:
            return None
        return super()._item_lookup_initial_entries()

    def _transform_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self._ser_columns if col not in ratings.columns]
        if missing:
            raise ValueError(
                f"Serendipity signal columns missing in answers file: {missing}"
            )
        ser_flags = ratings[self._ser_columns].applymap(
            lambda value: str(value).lower() == "true"
        )
        ser_label = ser_flags.any(axis=1).astype(int)
        ratings = ratings.copy()
        ratings["ser_label"] = ser_label
        return ratings


class AmazonDataProcessor(DataProcessor):
    def __init__(
        self,
        ratings_path: str | Path,
        serenlens_path: str | Path,
        prefix: str,
        *,
        min_presence: int = 5,
        min_sequence_length: int = 5,
        remove_post_reference_interactions: bool = True,
        expected_num_unique_items: Optional[int] = None,
        output_root: str | Path | None = None,
        lookup_root: str | Path | None = None,
        output_dirname: str | None = None,
        lookup_dirname: str | None = None,
        extra_sequence_columns: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=None,
            output_root=output_root,
            lookup_root=lookup_root,
            output_dirname=output_dirname,
            lookup_dirname=lookup_dirname,
        )
        self._ratings_path = Path(ratings_path)
        self._serenlens_path = Path(serenlens_path)
        self._min_presence = max(1, min_presence)
        self._min_sequence_length = max(1, min_sequence_length)
        self._extra_sequence_columns = extra_sequence_columns or {}
        self._remove_post_reference_interactions = bool(
            remove_post_reference_interactions
        )

    def preprocess_rating(self) -> int:
        if not self._ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {self._ratings_path}")
        if not self._serenlens_path.exists():
            raise FileNotFoundError(f"SerenLens file not found: {self._serenlens_path}")

        ratings = pd.read_csv(
            self._ratings_path,
            sep=",",
            names=["user_id", "item_id", "rating", "timestamp"],
            header=None,
            dtype={"user_id": str, "item_id": str, "rating": float, "timestamp": float},
            engine="python",
        )
        ratings.dropna(subset=["rating", "timestamp"], inplace=True)
        ratings["rating"] = ratings["rating"].astype(float)
        ratings["timestamp"] = ratings["timestamp"].astype("int64")
        ratings = _apply_reference_leakage_filter(
            ratings,
            _load_reference_interactions(self._serenlens_path),
            prefix=self._prefix,
            reference_name=self._serenlens_path.name,
            remove_post_reference_interactions=self._remove_post_reference_interactions,
        )
        if ratings.empty:
            raise ValueError(
                f"No ratings remaining after applying leakage reference filter from "
                f"{self._serenlens_path}"
            )

        item_counts = ratings["item_id"].value_counts()
        user_counts = ratings["user_id"].value_counts()
        ratings = ratings[
            (ratings["item_id"].map(item_counts) >= self._min_presence)
            & (ratings["user_id"].map(user_counts) >= self._min_presence)
        ]
        if ratings.empty:
            raise ValueError(
                f"No ratings remaining after applying min_presence={self._min_presence}"
            )

        ratings = ratings[["user_id", "item_id", "rating", "timestamp"]]

        user_ids, user_lookup = _ensure_lookup_series(
            ratings["user_id"],
            Path(self.user_lookup_csv()),
            normalized_col="normalized_user_id",
            original_col="original_user_id",
            key_normalizer=str.casefold,
        )
        ratings["user_id"] = user_ids

        item_ids, item_lookup = _ensure_lookup_series(
            ratings["item_id"],
            Path(self.item_lookup_csv()),
            normalized_col="normalized_item_id",
            original_col="original_item_id",
            key_normalizer=str.casefold,
        )
        ratings["item_id"] = item_ids

        self.ensure_output_dir()

        num_items = item_lookup.shape[0]
        self._expected_num_unique_items = num_items
        self._expected_max_item_id = int(item_lookup["normalized_item_id"].max())

        seq_df, lengths = _build_sequence_frame(
            ratings,
            min_length=self._min_sequence_length,
            extra_sequence_columns=self._extra_sequence_columns,
        )
        if not lengths.empty:
            log.info(
                "%s sequence length stats min=%s max=%s mean=%.2f median=%.2f",
                self._prefix,
                lengths.min(),
                lengths.max(),
                lengths.mean(),
                lengths.median(),
            )
        else:
            log.warning("No sequences generated for %s", self._prefix)

        ratings.sort_values(by=["user_id", "timestamp"], inplace=True)
        ratings.reset_index(drop=True, inplace=True)

        ratings.to_csv(self.normalized_ratings_csv(), index=False)
        seq_df.reset_index().to_csv(self.output_format_csv(), index=False)
        user_lookup.to_csv(self.user_lookup_csv(), index=False)
        item_lookup.to_csv(self.item_lookup_csv(), index=False)

        log.info(
            "%s final users=%s items=%s rows=%s",
            self._prefix,
            user_lookup.shape[0],
            num_items,
            ratings.shape[0],
        )
        return num_items


class SerenLensDataProcessor(DataProcessor):
    def __init__(
        self,
        ratings_path: str | Path,
        prefix: str,
        *,
        min_sequence_length: int = 1,
        output_root: str | Path | None = None,
        lookup_root: str | Path | None = None,
        output_dirname: str | None = None,
        lookup_dirname: str | None = None,
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=None,
            expected_max_item_id=None,
            output_root=output_root,
            lookup_root=lookup_root,
            output_dirname=output_dirname,
            lookup_dirname=lookup_dirname,
        )
        self._ratings_path = Path(ratings_path)
        self._min_sequence_length = max(1, min_sequence_length)

    def preprocess_rating(self) -> int:
        if not self._ratings_path.exists():
            raise FileNotFoundError(f"SerenLens file not found: {self._ratings_path}")

        ratings = pd.read_csv(
            self._ratings_path,
            usecols=["user_id", "item_id", "rating", "timestamp", "label"],
            engine="python",
        )
        ratings["user_id"] = ratings["user_id"].astype(str)
        ratings["item_id"] = ratings["item_id"].astype(str)
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")
        ratings["label"] = pd.to_numeric(ratings["label"], errors="coerce")

        invalid_mask = ratings[["rating", "timestamp", "label"]].isnull().any(axis=1)
        if invalid_mask.any():
            dropped = int(invalid_mask.sum())
            log.warning(
                "%s dropping %s SerenLens rows with invalid numeric fields",
                self._prefix,
                dropped,
            )
            ratings = ratings.loc[~invalid_mask].reset_index(drop=True)
            if ratings.empty:
                raise ValueError(
                    f"All rows removed from SerenLens ratings after filtering invalid numeric values for {self._prefix}"
                )

        ratings["rating"] = ratings["rating"].astype(float)
        ratings["timestamp"] = ratings["timestamp"].astype("int64")
        ratings["label"] = ratings["label"].astype(int)

        ratings = ratings[["user_id", "item_id", "rating", "timestamp", "label"]]

        user_ids, user_lookup = _ensure_lookup_series(
            ratings["user_id"],
            Path(self.user_lookup_csv()),
            normalized_col="normalized_user_id",
            original_col="original_user_id",
            key_normalizer=str.casefold,
        )
        ratings["user_id"] = user_ids

        item_ids, item_lookup = _ensure_lookup_series(
            ratings["item_id"],
            Path(self.item_lookup_csv()),
            normalized_col="normalized_item_id",
            original_col="original_item_id",
            key_normalizer=str.casefold,
        )
        ratings["item_id"] = item_ids

        self.ensure_output_dir()

        num_items = item_lookup.shape[0]
        self._expected_num_unique_items = num_items
        self._expected_max_item_id = int(item_lookup["normalized_item_id"].max())

        seq_df, lengths = _build_sequence_frame(
            ratings,
            min_length=self._min_sequence_length,
            extra_sequence_columns={"label": "sequence_ser_label"},
        )
        if not lengths.empty:
            log.info(
                "%s sequence length stats min=%s max=%s mean=%.2f median=%.2f",
                self._prefix,
                lengths.min(),
                lengths.max(),
                lengths.mean(),
                lengths.median(),
            )
        else:
            log.warning("No sequences generated for %s", self._prefix)

        ratings.sort_values(by=["user_id", "timestamp"], inplace=True)
        ratings.reset_index(drop=True, inplace=True)

        ratings.to_csv(self.normalized_ratings_csv(), index=False)
        seq_df.reset_index().to_csv(self.output_format_csv(), index=False)
        user_lookup.to_csv(self.user_lookup_csv(), index=False)
        item_lookup.to_csv(self.item_lookup_csv(), index=False)

        log.info(
            "%s final users=%s items=%s rows=%s",
            self._prefix,
            user_lookup.shape[0],
            num_items,
            ratings.shape[0],
        )
        return num_items


__all__ = [
    "DataProcessor",
    "LeakageAuditSummary",
    "MovielensDataProcessor",
    "SerendipityAnswersDataProcessor",
    "AmazonDataProcessor",
    "SerenLensDataProcessor",
    "apply_source_target_leakage_filter",
    "load_reference_interactions",
]
