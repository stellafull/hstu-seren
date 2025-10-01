from __future__ import annotations

import abc
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
    ) -> None:
        self._prefix = prefix
        self._expected_num_unique_items = expected_num_unique_items
        self._expected_max_item_id = expected_max_item_id
        self._output_root = Path(output_root) if output_root is not None else DEFAULT_PRETRAIN_ROOT
        self._lookup_root = Path(lookup_root) if lookup_root is not None else self._output_root
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._lookup_root.mkdir(parents=True, exist_ok=True)

    def output_dir(self) -> Path:
        return self._output_root / self._prefix

    def lookup_dir(self) -> Path:
        return self._lookup_root / self._prefix

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
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
        output_root: str | Path | None = None,
        lookup_root: str | Path | None = None,
        extra_sequence_columns: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=expected_max_item_id,
            output_root=output_root,
            lookup_root=lookup_root,
        )
        self._ratings_path = Path(ratings_path)
        self._movies_path = Path(movies_path) if movies_path else None
        self._min_sequence_length = max(1, min_sequence_length)
        self._extra_sequence_columns = extra_sequence_columns or {}

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
        serendipity_columns: Optional[list[str]] = None,
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
        super().__init__(
            ratings_path,
            prefix,
            movies_path=movies_path,
            min_sequence_length=min_sequence_length,
            output_root=output_root,
            lookup_root=lookup_root,
            extra_sequence_columns={"ser_label": "sequence_ser_label"},
        )

    def _user_lookup_initial_entries(self) -> Iterable[str] | None:
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
        expected_num_unique_items: Optional[int] = None,
        output_root: str | Path | None = None,
        lookup_root: str | Path | None = None,
        extra_sequence_columns: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=None,
            output_root=output_root,
            lookup_root=lookup_root,
        )
        self._ratings_path = Path(ratings_path)
        self._serenlens_path = Path(serenlens_path)
        self._min_presence = max(1, min_presence)
        self._min_sequence_length = max(1, min_sequence_length)
        self._extra_sequence_columns = extra_sequence_columns or {}

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
        serenlens = pd.read_csv(
            self._serenlens_path,
            usecols=["user_id", "item_id"],
            dtype={"user_id": str, "item_id": str},
            engine="python",
        )

        initial_rows = ratings.shape[0]

        ratings["_normalized_user_id"] = ratings["user_id"].str.casefold()
        ratings["_normalized_item_id"] = ratings["item_id"].str.casefold()

        serenlens_pairs = serenlens.copy()
        serenlens_pairs["_normalized_user_id"] = (
            serenlens_pairs["user_id"].str.casefold()
        )
        serenlens_pairs["_normalized_item_id"] = (
            serenlens_pairs["item_id"].str.casefold()
        )
        serenlens_pairs = serenlens_pairs.drop_duplicates(
            subset=["_normalized_user_id", "_normalized_item_id"]
        )
        serenlens_pairs["_serenlens"] = True
        ratings = ratings.merge(
            serenlens_pairs[
                ["_normalized_user_id", "_normalized_item_id", "_serenlens"]
            ],
            how="left",
            on=["_normalized_user_id", "_normalized_item_id"],
        )
        leak_rows = ratings["_serenlens"].fillna(False).sum()
        if leak_rows:
            log.info(
                "%s removing %s SerenLens overlaps (%s -> %s records)",
                self._prefix,
                int(leak_rows),
                initial_rows,
                initial_rows - int(leak_rows),
            )
        ratings = ratings[ratings["_serenlens"].isna()].drop(columns="_serenlens")
        ratings.drop(columns=["_normalized_user_id", "_normalized_item_id"], inplace=True)

        item_counts = ratings["item_id"].value_counts()
        user_counts = ratings["user_id"].value_counts()
        ratings = ratings[
            (ratings["item_id"].map(item_counts) >= self._min_presence)
            & (ratings["user_id"].map(user_counts) >= self._min_presence)
        ]

        ratings.dropna(subset=["rating", "timestamp"], inplace=True)
        ratings["rating"] = ratings["rating"].astype(float)
        ratings["timestamp"] = ratings["timestamp"].astype("int64")

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
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=None,
            expected_max_item_id=None,
            output_root=output_root,
            lookup_root=lookup_root,
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
    "MovielensDataProcessor",
    "SerendipityAnswersDataProcessor",
    "AmazonDataProcessor",
    "SerenLensDataProcessor",
]
