from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from generative_recommenders_pl.data.download import DEFAULT_ROOT, download_dataset
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

AMAZON_DOMAIN_CONFIG = {
    "books": {
        "dataset": "amzn_books_2015",
        "ratings": "ratings_Books.csv",
        "serenlens": "SerenLens_Books.csv",
        "meta": "meta_Books.json",
    },
    "movies": {
        "dataset": "amzn_mv_2015",
        "ratings": "ratings_Movies_and_TV.csv",
        "serenlens": "SerenLens_Movies.csv",
        "meta": "meta_Movies_and_TV.json",
    },
}

SERENDIPITY2018_SUBSETS = {
    "train": "training.csv",
    "answers": "answers.csv",
}



def _ensure_mapping(
    mapping_path: Path,
    raw_values: Iterable[object],
    *,
    encoded_col: str,
    raw_col: str,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    """Load or create an ID mapping stored as CSV."""
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    processed_values = [str(v) for v in raw_values if pd.notna(v)]
    unique_values = sorted(set(processed_values))

    if mapping_path.exists():
        df = pd.read_csv(mapping_path)
        df[raw_col] = df[raw_col].astype(str)
        mapping: Dict[str, int] = dict(zip(df[raw_col], df[encoded_col].astype(int)))
        missing = sorted(set(unique_values) - set(mapping.keys()))
        if missing:
            start_idx = max(mapping.values(), default=-1) + 1
            new_rows = pd.DataFrame(
                {encoded_col: range(start_idx, start_idx + len(missing)), raw_col: missing}
            )
            if extra_metadata:
                for key, value in extra_metadata.items():
                    new_rows[key] = value
            df = pd.concat([df, new_rows], ignore_index=True)
            df.to_csv(mapping_path, index=False)
            mapping.update(dict(zip(new_rows[raw_col], new_rows[encoded_col])))
        if extra_metadata:
            for key, value in extra_metadata.items():
                if key not in df.columns:
                    df[key] = value
                else:
                    df[key] = df[key].fillna(value)
            df.to_csv(mapping_path, index=False)
        return mapping

    data = {encoded_col: range(len(unique_values)), raw_col: unique_values}
    if extra_metadata:
        for key, value in extra_metadata.items():
            data[key] = [value] * len(unique_values)
    df = pd.DataFrame(data)
    df.to_csv(mapping_path, index=False)
    return dict(zip(df[raw_col], df[encoded_col]))


def _resolve_column(columns: Sequence[str], keywords: Sequence[str]) -> Optional[str]:
    lowered = {column.lower(): column for column in columns}
    for keyword in keywords:
        for column_lower, column in lowered.items():
            if keyword in column_lower:
                return column
    return None


def _sequence_frame(
    frame: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    rating_col: str,
    timestamp_col: str,
    min_length: int = 1,
    extra_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    ordered = frame.sort_values(by=[user_col, timestamp_col])
    grouped = ordered.groupby(user_col, sort=False)
    item_lists = grouped[item_col].apply(list)
    rating_lists = grouped[rating_col].apply(list)
    timestamp_lists = grouped[timestamp_col].apply(list)
    data = {
        "user_id": list(item_lists.index),
        "item_ids": item_lists.values,
        "ratings": rating_lists.values,
        "timestamps": timestamp_lists.values,
    }
    if extra_cols:
        for col in extra_cols:
            if col in frame.columns:
                data[col] = grouped[col].apply(list).values
    seq_df = pd.DataFrame(data)
    if min_length > 1:
        seq_df = seq_df[seq_df["item_ids"].apply(len) >= min_length]
    return seq_df


class DataProcessor:
    """Abstract base class for creating SASRec-ready datasets."""

    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int],
        expected_max_item_id: Optional[int],
    ) -> None:
        self._prefix = prefix
        self._expected_num_unique_items = expected_num_unique_items
        self._expected_max_item_id = expected_max_item_id

    @abc.abstractmethod
    def expected_num_unique_items(self) -> Optional[int]:
        return self._expected_num_unique_items

    @abc.abstractmethod
    def expected_max_item_id(self) -> Optional[int]:
        return self._expected_max_item_id

    @abc.abstractmethod
    def processed_item_csv(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess_rating(self) -> int:
        raise NotImplementedError

    def output_format_csv(self) -> str:
        return str(DEFAULT_ROOT / self._prefix / "sasrec_format.csv")

    def to_seq_data(
        self,
        ratings_data: pd.DataFrame,
        user_data: Optional[pd.DataFrame] = None,
        *,
        sequence_columns: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        if user_data is not None:
            ratings_data_transformed = ratings_data.join(
                user_data.set_index("user_id"), on="user_id"
            )
        else:
            ratings_data_transformed = ratings_data
        ratings_data_transformed.item_ids = ratings_data_transformed.item_ids.apply(
            lambda x: ",".join(str(v) for v in x)
        )
        ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
            lambda x: ",".join(str(v) for v in x)
        )
        ratings_data_transformed.timestamps = ratings_data_transformed.timestamps.apply(
            lambda x: ",".join(str(v) for v in x)
        )
        if sequence_columns:
            for column, _ in sequence_columns.items():
                if column in ratings_data_transformed.columns:
                    ratings_data_transformed[column] = ratings_data_transformed[column].apply(
                        lambda x: ",".join(str(v) for v in x)
                    )
        ratings_data_transformed.rename(
            columns={
                "item_ids": "sequence_item_ids",
                "ratings": "sequence_ratings",
                "timestamps": "sequence_timestamps",
            },
            inplace=True,
        )
        if sequence_columns:
            rename_map = {
                column: new_name
                for column, new_name in sequence_columns.items()
                if column in ratings_data_transformed.columns
            }
            if rename_map:
                ratings_data_transformed.rename(columns=rename_map, inplace=True)
        return ratings_data_transformed

    def file_exists(self, name: str) -> bool:
        return (Path(os.getcwd()) / name).is_file()


class SerendipityAmazonDataProcessor(DataProcessor):
    """Preprocess SerenLens-enhanced Amazon domains into SASRec format."""

    def __init__(
        self,
        domain: str,
        subset: str,
        *,
        root: str | Path = DEFAULT_ROOT,
        min_sequence_length: int = 1,
    ) -> None:
        domain_key = domain.lower()
        subset_key = subset.lower()
        if domain_key not in AMAZON_DOMAIN_CONFIG:
            raise ValueError(f"Unsupported domain '{domain}'. Choose from {list(AMAZON_DOMAIN_CONFIG)}")
        if subset_key not in {"ratings", "serenlens"}:
            raise ValueError("subset must be either 'ratings' or 'serenlens'")

        prefix = f"{domain_key}-{subset_key}"
        super().__init__(prefix, expected_num_unique_items=None, expected_max_item_id=None)
        self._domain = domain_key
        self._subset = subset_key
        self._root = Path(root)
        self._dataset_config = AMAZON_DOMAIN_CONFIG[domain_key]
        self._domain_output_root = self._root / "serendipity-aware" / domain_key
        self._domain_output_root.mkdir(parents=True, exist_ok=True)
        self._mapping_dir = self._domain_output_root / "mappings"
        self._user_mapping_path = self._mapping_dir / "user_mapping.csv"
        self._item_mapping_path = self._mapping_dir / "item_mapping.csv"
        self._min_sequence_length = min_sequence_length
        self._output_path = self._domain_output_root / f"{subset_key}_sasrec.csv"
        self._raw_output_path = self._domain_output_root / f"{subset_key}_sasrec_raw.csv"
        self.outputs: Dict[str, str] = {}

    def _load_mapping_stats(self) -> None:
        if self._expected_num_unique_items is not None and self._expected_max_item_id is not None:
            return
        if not self._item_mapping_path.exists():
            return
        mapping_df = pd.read_csv(self._item_mapping_path)
        if "item_id" not in mapping_df.columns:
            return
        self._expected_num_unique_items = int(mapping_df["item_id"].nunique())
        self._expected_max_item_id = int(mapping_df["item_id"].max())

    def expected_num_unique_items(self) -> Optional[int]:
        self._load_mapping_stats()
        return self._expected_num_unique_items

    def expected_max_item_id(self) -> Optional[int]:
        self._load_mapping_stats()
        return self._expected_max_item_id

    def processed_item_csv(self) -> str:
        return str(self._item_mapping_path)

    def output_format_csv(self) -> str:
        return str(self._output_path)

    def raw_output_format_csv(self) -> str:
        return str(self._raw_output_path)

    def preprocess_rating(self) -> int:
        download_dataset(self._dataset_config["dataset"], root=self._root)
        dataset_root = self._root / self._dataset_config["dataset"]
        ratings_path = dataset_root / self._dataset_config["ratings"]
        serenlens_path = dataset_root / self._dataset_config["serenlens"]
        if not ratings_path.exists():
            raise FileNotFoundError(f"Missing ratings file at {ratings_path}")
        if not serenlens_path.exists():
            raise FileNotFoundError(f"Missing SerenLens file at {serenlens_path}")

        ratings = pd.read_csv(
            ratings_path,
            names=["user_raw", "item_raw", "rating", "timestamp"],
        )
        serenlens = pd.read_csv(serenlens_path)
        serenlens_user_col = _resolve_column(serenlens.columns, ["user", "reviewer"])
        serenlens_item_col = _resolve_column(serenlens.columns, ["item", "asin"])
        serenlens_rating_col = _resolve_column(serenlens.columns, ["rating", "score"])
        serenlens_time_col = _resolve_column(serenlens.columns, ["time", "stamp"])
        if not serenlens_user_col or not serenlens_item_col:
            raise ValueError("Could not locate user or item columns in SerenLens dataset")
        if serenlens_rating_col is None:
            serenlens_rating_col = "rating"
            serenlens[serenlens_rating_col] = 1.0
        if serenlens_time_col is None:
            serenlens_time_col = "timestamp"
            serenlens[serenlens_time_col] = serenlens.groupby(serenlens_user_col, sort=False).cumcount()

        serenlens = serenlens.rename(
            columns={
                serenlens_user_col: "user_raw",
                serenlens_item_col: "item_raw",
                serenlens_rating_col: "rating",
                serenlens_time_col: "timestamp",
            }
        )
        selected_columns = ["user_raw", "item_raw", "rating", "timestamp"]
        if "label" in serenlens.columns:
            serenlens["label"] = pd.to_numeric(serenlens["label"], errors="coerce").fillna(0).astype(int)
            selected_columns.append("label")
        serenlens = serenlens[selected_columns]

        ratings["user_raw"] = ratings["user_raw"].astype(str).str.strip().str.upper()
        ratings["item_raw"] = ratings["item_raw"].astype(str).str.strip().str.upper()
        serenlens["user_raw"] = serenlens["user_raw"].astype(str).str.strip().str.upper()
        serenlens["item_raw"] = serenlens["item_raw"].astype(str).str.strip().str.upper()

        serenlens_pairs = serenlens[["user_raw", "item_raw"]].drop_duplicates()
        ratings = ratings.merge(
            serenlens_pairs.assign(_in_serenlens=True),
            on=["user_raw", "item_raw"],
            how="left",
        )
        filtered_ratings = ratings[ratings["_in_serenlens"].isna()].drop(columns="_in_serenlens")

        combined_users = pd.concat(
            [filtered_ratings["user_raw"], serenlens["user_raw"]], ignore_index=True
        )
        combined_items = pd.concat(
            [filtered_ratings["item_raw"], serenlens["item_raw"]], ignore_index=True
        )

        user_mapping = _ensure_mapping(
            self._user_mapping_path,
            combined_users.tolist(),
            encoded_col="user_id",
            raw_col="raw_user_id",
            extra_metadata={"domain": self._domain},
        )
        item_mapping = _ensure_mapping(
            self._item_mapping_path,
            combined_items.tolist(),
            encoded_col="item_id",
            raw_col="raw_item_id",
            extra_metadata={"domain": self._domain},
        )

        self._expected_num_unique_items = len(item_mapping)
        self._expected_max_item_id = max(item_mapping.values()) if item_mapping else None

        target_frame = filtered_ratings if self._subset == "ratings" else serenlens
        encoded = self._encode_for_sequences(target_frame, user_mapping, item_mapping)
        extra_cols = ["label"] if "label" in target_frame.columns else None
        sequences = _sequence_frame(
            encoded,
            user_col="user_id",
            item_col="item_id",
            rating_col="rating",
            timestamp_col="timestamp",
            min_length=self._min_sequence_length,
            extra_cols=extra_cols,
        )
        seq_map = {"label": "sequence_labels"} if extra_cols else None
        sasrec_ready = self.to_seq_data(sequences, sequence_columns=seq_map)
        sasrec_ready.to_csv(self.output_format_csv(), index=False)

        raw_sequences = _sequence_frame(
            target_frame,
            user_col="user_raw",
            item_col="item_raw",
            rating_col="rating",
            timestamp_col="timestamp",
            min_length=self._min_sequence_length,
            extra_cols=extra_cols,
        )
        raw_seq_map = {"label": "sequence_labels"} if extra_cols else None
        raw_ready = self.to_seq_data(raw_sequences, sequence_columns=raw_seq_map)
        rename_map = {
            "user_id": "raw_user_id",
            "sequence_item_ids": "raw_sequence_item_ids",
            "sequence_ratings": "raw_sequence_ratings",
            "sequence_timestamps": "raw_sequence_timestamps",
        }
        if raw_seq_map and "sequence_labels" in raw_ready.columns:
            rename_map["sequence_labels"] = "raw_sequence_labels"
        raw_ready.rename(columns=rename_map, inplace=True)
        raw_ready.to_csv(self.raw_output_format_csv(), index=False)

        self.outputs = {
            "sasrec": self.output_format_csv(),
            "user_mapping": str(self._user_mapping_path),
            "item_mapping": str(self._item_mapping_path),
            "sasrec_raw": self.raw_output_format_csv(),
        }
        return self._expected_num_unique_items or 0

    def _encode_for_sequences(
        self,
        frame: pd.DataFrame,
        user_mapping: Mapping[str, int],
        item_mapping: Mapping[str, int],
    ) -> pd.DataFrame:
        data = frame.copy()
        data["user_raw"] = data["user_raw"].astype(str)
        data["item_raw"] = data["item_raw"].astype(str)
        data["user_id"] = data["user_raw"].map(user_mapping)
        data["item_id"] = data["item_raw"].map(item_mapping)
        missing_mask = data["user_id"].isna() | data["item_id"].isna()
        if missing_mask.any():
            dropped = int(missing_mask.sum())
            log.warning(
                "%s/%s: dropping %s rows with unmapped IDs",
                self._domain,
                self._subset,
                dropped,
            )
            data = data.loc[~missing_mask]
        data["user_id"] = data["user_id"].astype(int)
        data["item_id"] = data["item_id"].astype(int)
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce").fillna(0.0)
        data["timestamp"] = (
            pd.to_numeric(data["timestamp"], errors="coerce").fillna(0).astype(int)
        )
        return data[["user_id", "item_id", "rating", "timestamp"]]


class Serendipity2018DataProcessor(DataProcessor):
    """Preprocess Serendipity 2018 challenge data into SASRec format."""

    def __init__(
        self,
        subset: str,
        *,
        root: str | Path = DEFAULT_ROOT,
        min_sequence_length: int = 1,
    ) -> None:
        subset_key = subset.lower()
        if subset_key not in SERENDIPITY2018_SUBSETS:
            raise ValueError(
                f"Unsupported subset '{subset}'. Choose from {list(SERENDIPITY2018_SUBSETS)}"
            )
        prefix = f"sac2018-{subset_key}"
        super().__init__(prefix, expected_num_unique_items=None, expected_max_item_id=None)
        self._subset = subset_key
        self._root = Path(root)
        self._min_sequence_length = min_sequence_length
        self._output_root = self._root / "serendipity-aware" / "sac2018"
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._mapping_dir = self._output_root / "mappings"
        self._user_mapping_path = self._mapping_dir / "user_mapping.csv"
        self._item_mapping_path = self._mapping_dir / "item_mapping.csv"
        self._output_path = self._output_root / f"{subset_key}_sasrec.csv"
        self._raw_output_path = self._output_root / f"{subset_key}_sasrec_raw.csv"
        self.outputs: Dict[str, str] = {}

    def _load_mapping_stats(self) -> None:
        if self._expected_num_unique_items is not None and self._expected_max_item_id is not None:
            return
        if not self._item_mapping_path.exists():
            return
        mapping_df = pd.read_csv(self._item_mapping_path)
        if "item_id" not in mapping_df.columns:
            return
        self._expected_num_unique_items = int(mapping_df["item_id"].nunique())
        self._expected_max_item_id = int(mapping_df["item_id"].max())

    def expected_num_unique_items(self) -> Optional[int]:
        self._load_mapping_stats()
        return self._expected_num_unique_items

    def expected_max_item_id(self) -> Optional[int]:
        self._load_mapping_stats()
        return self._expected_max_item_id

    def processed_item_csv(self) -> str:
        return str(self._item_mapping_path)

    def output_format_csv(self) -> str:
        return str(self._output_path)

    def raw_output_format_csv(self) -> str:
        return str(self._raw_output_path)

    def preprocess_rating(self) -> int:
        download_dataset("serendipity-2018", root=self._root)
        dataset_root = self._root / "serendipity-2018"

        subset_frames: Dict[str, pd.DataFrame] = {}
        for subset_key, filename in SERENDIPITY2018_SUBSETS.items():
            file_path = self._locate_dataset_file(dataset_root, filename)
            subset_frames[subset_key] = self._normalize_subset(pd.read_csv(file_path))

        combined_users = pd.concat(
            [frame["user_raw"] for frame in subset_frames.values()], ignore_index=True
        )
        combined_items = pd.concat(
            [frame["item_raw"] for frame in subset_frames.values()], ignore_index=True
        )

        user_mapping = _ensure_mapping(
            self._user_mapping_path,
            combined_users.tolist(),
            encoded_col="user_id",
            raw_col="raw_user_id",
            extra_metadata={"dataset": "serendipity-2018"},
        )
        item_mapping = _ensure_mapping(
            self._item_mapping_path,
            combined_items.tolist(),
            encoded_col="item_id",
            raw_col="raw_item_id",
            extra_metadata={"dataset": "serendipity-2018"},
        )

        self._expected_num_unique_items = len(item_mapping)
        self._expected_max_item_id = max(item_mapping.values()) if item_mapping else None

        target_frame = subset_frames[self._subset]
        encoded = self._encode_for_sequences(target_frame, user_mapping, item_mapping)
        sequences = _sequence_frame(
            encoded,
            user_col="user_id",
            item_col="item_id",
            rating_col="rating",
            timestamp_col="timestamp",
            min_length=self._min_sequence_length,
        )
        sasrec_ready = self.to_seq_data(sequences)
        sasrec_ready.to_csv(self.output_format_csv(), index=False)

        raw_sequences = _sequence_frame(
            target_frame,
            user_col="user_raw",
            item_col="item_raw",
            rating_col="rating",
            timestamp_col="timestamp",
            min_length=self._min_sequence_length,
        )
        raw_ready = self.to_seq_data(raw_sequences)
        raw_ready.rename(
            columns={
                "user_id": "raw_user_id",
                "sequence_item_ids": "raw_sequence_item_ids",
                "sequence_ratings": "raw_sequence_ratings",
                "sequence_timestamps": "raw_sequence_timestamps",
            },
            inplace=True,
        )
        raw_ready.to_csv(self.raw_output_format_csv(), index=False)

        self.outputs = {
            "sasrec": self.output_format_csv(),
            "user_mapping": str(self._user_mapping_path),
            "item_mapping": str(self._item_mapping_path),
            "sasrec_raw": self.raw_output_format_csv(),
        }
        return self._expected_num_unique_items or 0

    def _locate_dataset_file(self, root: Path, filename: str) -> Path:
        candidates = list(root.rglob(filename))
        if not candidates:
            raise FileNotFoundError(f"Could not find {filename} inside {root}")
        return candidates[0]

    def _normalize_subset(self, frame: pd.DataFrame) -> pd.DataFrame:
        user_col = _resolve_column(frame.columns, ["user", "uid"])
        item_col = _resolve_column(frame.columns, ["item", "movie", "iid", "sid"])
        rating_col = _resolve_column(frame.columns, ["rating", "score", "value", "relevance"])
        timestamp_col = _resolve_column(frame.columns, ["timestamp", "time", "unix"])
        if not user_col or not item_col:
            raise ValueError("Subset does not contain identifiable user/item columns")

        normalized = pd.DataFrame()
        normalized["user_raw"] = frame[user_col].astype(str)
        normalized["item_raw"] = frame[item_col].astype(str)
        if rating_col:
            normalized["rating"] = pd.to_numeric(frame[rating_col], errors="coerce").fillna(0.0)
        else:
            normalized["rating"] = 1.0
        if timestamp_col:
            timestamps = pd.to_numeric(frame[timestamp_col], errors="coerce")
            normalized["timestamp"] = timestamps
        else:
            normalized["timestamp"] = pd.NA
        normalized["timestamp"] = normalized.groupby("user_raw", sort=False)["timestamp"].apply(
            lambda col: col.fillna(pd.RangeIndex(len(col)))
        )
        normalized["timestamp"] = normalized["timestamp"].astype(int)
        return normalized[["user_raw", "item_raw", "rating", "timestamp"]]

    def _encode_for_sequences(
        self,
        frame: pd.DataFrame,
        user_mapping: Mapping[str, int],
        item_mapping: Mapping[str, int],
    ) -> pd.DataFrame:
        data = frame.copy()
        data["user_id"] = data["user_raw"].map(user_mapping)
        data["item_id"] = data["item_raw"].map(item_mapping)
        missing_mask = data["user_id"].isna() | data["item_id"].isna()
        if missing_mask.any():
            dropped = int(missing_mask.sum())
            log.warning("sac2018-%s: dropping %s rows with unmapped IDs", self._subset, dropped)
            data = data.loc[~missing_mask]
        data["user_id"] = data["user_id"].astype(int)
        data["item_id"] = data["item_id"].astype(int)
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce").fillna(0.0)
        data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce").fillna(0).astype(int)
        return data[["user_id", "item_id", "rating", "timestamp"]]


class SerendipityAllDataProcessor(DataProcessor):
    """Utility preprocessor that orchestrates all serendipity-aware datasets."""

    def __init__(self, *, root: str | Path = DEFAULT_ROOT, min_sequence_length: int = 1) -> None:
        super().__init__("serendipity-all", expected_num_unique_items=None, expected_max_item_id=None)
        self._root = Path(root)
        self._min_sequence_length = min_sequence_length
        self._output_path = self._root / "serendipity-all" / "summary.csv"
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._processors = [
            SerendipityAmazonDataProcessor(
                "books",
                "serenlens",
                root=self._root,
                min_sequence_length=self._min_sequence_length,
            ),
            SerendipityAmazonDataProcessor(
                "movies",
                "serenlens",
                root=self._root,
                min_sequence_length=self._min_sequence_length,
            ),
            Serendipity2018DataProcessor(
                "train",
                root=self._root,
                min_sequence_length=self._min_sequence_length,
            ),
            Serendipity2018DataProcessor(
                "answers",
                root=self._root,
                min_sequence_length=self._min_sequence_length,
            ),
        ]
        self.outputs: Dict[str, Dict[str, str]] = {}

    def expected_num_unique_items(self) -> Optional[int]:
        return None

    def expected_max_item_id(self) -> Optional[int]:
        return None

    def processed_item_csv(self) -> str:
        return str(self._output_path)

    def output_format_csv(self) -> str:
        return str(self._output_path)

    def preprocess_rating(self) -> int:
        summary_rows = []
        total_items = 0
        for processor in self._processors:
            log.info("Running preprocessor for %s", processor.output_format_csv())
            num_items = processor.preprocess_rating()
            total_items += num_items
            outputs = getattr(processor, "outputs", {})
            if outputs:
                self.outputs[processor._prefix] = outputs  # type: ignore[assignment]
            summary_rows.append(
                {
                    "prefix": processor._prefix,
                    "sasrec_path": outputs.get("sasrec", processor.output_format_csv()),
                    "sasrec_raw_path": outputs.get("sasrec_raw", ""),
                    "user_mapping_path": outputs.get("user_mapping", ""),
                    "item_mapping_path": outputs.get("item_mapping", processor.processed_item_csv()),
                    "num_items": num_items,
                }
            )
        summary_frame = pd.DataFrame(summary_rows)
        summary_frame.to_csv(self._output_path, index=False)
        self.outputs["summary"] = {"summary_path": str(self._output_path)}
        log.info("Completed preprocessing for all datasets. Summary saved to %s", self._output_path)
        return total_items
