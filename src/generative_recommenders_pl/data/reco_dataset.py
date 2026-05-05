import ast
import json
import bisect
import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig

from generative_recommenders_pl.data.preprocessor import DataProcessor
from generative_recommenders_pl.serenfree.geometry import prefix_surprise
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)


def load_data(ratings_file: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(ratings_file, pd.DataFrame):
        return ratings_file
    elif isinstance(ratings_file, str) and ratings_file.endswith(".csv"):
        return pd.read_csv(ratings_file, delimiter=",")
    elif isinstance(ratings_file, str) and ratings_file.endswith(".parquet"):
        return pd.read_parquet(ratings_file)
    else:
        raise ValueError("ratings_file must be a csv or parquet file.")


def save_data(ratings_frame: pd.DataFrame, output_file: str):
    if output_file.endswith(".csv"):
        ratings_frame.to_csv(output_file, index=False)
    else:
        raise ValueError("ratings_file must be a csv file.")


def parse_sequence_value(value: Any) -> list[Any]:
    """Parse stored sequence columns without executing input text."""

    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(text)
        elif text[0] == "(":
            parsed = ast.literal_eval(text)
        else:
            parsed = [part for part in text.split(",") if part != ""]
    elif hasattr(value, "tolist"):
        parsed = value.tolist()
    else:
        parsed = value
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


class RecoDataset(torch.utils.data.Dataset):
    """In reverse chronological order."""

    def __init__(
        self,
        ratings_file: str | pd.DataFrame,
        padding_length: int,
        ignore_last_n: int,  # used for creating train/valid/test sets
        shift_id_by: int = 0,
        chronological: bool = False,
        sample_ratio: float = 1.0,
        additional_columns: Optional[List[str]] = [],
    ) -> None:
        """
        Args:
            ratings_file: str or pd.DataFrame, path to the ratings file or DataFrame.
            padding_length: int, length to pad sequences to.
            ignore_last_n: int, number of last interactions to ignore (used for creating train/valid/test sets).
            shift_id_by: int, value to shift IDs by. Default is 0.
            chronological: bool, whether to sort interactions chronologically. Default is False.
            sample_ratio: float, ratio of data to sample. Default is 1.0 (use all data).
            additional_columns: Optional[List[str]], list of additional columns to include. Default is None.
        """
        super().__init__()

        self.ratings_frame: pd.DataFrame = load_data(ratings_file)
        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio
        self._additional_columns = additional_columns
        self.__additional_columns_check()

    def __additional_columns_check(self):
        if self._additional_columns:
            columns_status = []
            for column in self._additional_columns:
                # check the column exists and status, like type, max, min, etc.
                column_exists = column in self.ratings_frame.columns
                if not column_exists:
                    raise ValueError(
                        f"Column {column} does not exist in the ratings file."
                    )
                column_type = self.ratings_frame[column].dtype
                max_value = self.ratings_frame[column].max()
                min_value = self.ratings_frame[column].min()
                columns_status.append(
                    {
                        "column": column,
                        "type": column_type,
                        "max": max_value,
                        "min": min_value,
                    }
                )
            log.info(f"Additional columns status: {columns_status}")

    def __len__(self) -> int:
        return len(self.ratings_frame)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if idx in self._cache.keys():
            return self._cache[idx]
        sample = self.load_item(idx)
        self._cache[idx] = sample
        return sample

    def load_item(self, idx) -> Dict[str, torch.Tensor]:
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        def parse_as_list(x, ignore_last_n) -> List[int]:
            y_list = [int(float(v)) for v in parse_sequence_value(x)]
            if ignore_last_n > 0:
                # for training data creation
                y_list = y_list[:-ignore_last_n]
            return y_list

        def parse_int_list(
            x,
            target_len: int,
            ignore_last_n: int,
            shift_id_by: int,
            sampling_kept_mask: Optional[List[bool]],
        ) -> Tuple[List[int], int]:
            y = parse_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        if self._sample_ratio < 1.0:
            raw_length = len(parse_as_list(data.sequence_item_ids, self._ignore_last_n))
            sampling_kept_mask = (
                torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio
            ).tolist()
        else:
            sampling_kept_mask = None

        movie_history, movie_history_len = parse_int_list(
            data.sequence_item_ids,
            self._padding_length,
            self._ignore_last_n,
            shift_id_by=self._shift_id_by,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_history_ratings, ratings_len = parse_int_list(
            data.sequence_ratings,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_timestamps, timestamps_len = parse_int_list(
            data.sequence_timestamps,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        ser_sequence = None
        if "sequence_ser_label" in data:
            ser_sequence, ser_len = parse_int_list(
                data.sequence_ser_label,
                self._padding_length,
                self._ignore_last_n,
                0,
                sampling_kept_mask=sampling_kept_mask,
            )
            ser_sequence = [int(x) for x in ser_sequence]
        assert (
            movie_history_len == timestamps_len
        ), f"history len {movie_history_len} differs from timestamp len {timestamps_len}."
        assert (
            movie_history_len == ratings_len
        ), f"history len {movie_history_len} differs from ratings len {ratings_len}."
        if ser_sequence is not None:
            assert (
                movie_history_len == ser_len
            ), f"history len {movie_history_len} differs from ser len {ser_len}."

        def _truncate_or_pad_seq(
            y: List[int], target_len: int, chronological: bool
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            assert len(y) == target_len
            return y

        historical_ids = movie_history[1:]
        historical_ratings = movie_history_ratings[1:]
        historical_timestamps = movie_timestamps[1:]
        target_ids = movie_history[0]
        target_ratings = movie_history_ratings[0]
        target_timestamps = movie_timestamps[0]
        if self._chronological:
            historical_ids.reverse()
            historical_ratings.reverse()
            historical_timestamps.reverse()

        max_seq_len = self._padding_length - 1
        history_length = min(len(historical_ids), max_seq_len)
        historical_ids = _truncate_or_pad_seq(
            historical_ids,
            max_seq_len,
            self._chronological,
        )
        historical_ratings = _truncate_or_pad_seq(
            historical_ratings,
            max_seq_len,
            self._chronological,
        )
        historical_timestamps = _truncate_or_pad_seq(
            historical_timestamps,
            max_seq_len,
            self._chronological,
        )
        if ser_sequence is not None:
            historical_ser = ser_sequence[1:]
            target_ser_label = ser_sequence[0] if ser_sequence else 0
            if self._chronological:
                historical_ser.reverse()
            historical_ser = _truncate_or_pad_seq(
                historical_ser,
                max_seq_len,
                self._chronological,
            )
        else:
            historical_ser = None
            target_ser_label = 0
        ret = {
            "user_id": user_id,
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(
                historical_timestamps, dtype=torch.int64
            ),
            "history_lengths": history_length,
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
        }
        if historical_ser is not None:
            ret["historical_ser_labels"] = torch.tensor(
                historical_ser, dtype=torch.int64
            )
        ret["target_ser_label"] = torch.tensor(target_ser_label, dtype=torch.int64)

        for column in self._additional_columns:
            # currently we do not consider the sequence columns in the additional columns
            ret[column] = data[column]
        return ret


class LOOManifestEvalDataset(torch.utils.data.Dataset):
    """Evaluation dataset backed by frozen LOO_FULL_CATALOG manifest rows."""

    def __init__(
        self,
        ratings_file: str | pd.DataFrame,
        padding_length: int,
        shift_id_by: int = 0,
        chronological: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self._padding_length = int(padding_length)
        self._shift_id_by = int(shift_id_by)
        self._chronological = bool(chronological)
        self._materialize(load_data(ratings_file))

    def __len__(self) -> int:
        return int(self.target_ids.size(0))

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            "user_id": self.user_ids[idx],
            "historical_ids": self.historical_ids[idx],
            "historical_ratings": self.historical_ratings[idx],
            "historical_timestamps": self.historical_timestamps[idx],
            "history_lengths": self.history_lengths[idx],
            "target_ids": self.target_ids[idx],
            "target_ratings": self.target_ratings[idx],
            "target_timestamps": self.target_timestamps[idx],
            "target_ser_label": self.target_ser_labels[idx],
        }

    def _materialize(self, frame: pd.DataFrame) -> None:
        max_seq_len = self._padding_length - 1
        row_count = len(frame)
        self.user_ids: list[Any] = [None] * row_count
        self.historical_ids = torch.zeros((row_count, max_seq_len), dtype=torch.int64)
        self.historical_ratings = torch.zeros((row_count, max_seq_len), dtype=torch.int64)
        self.historical_timestamps = torch.zeros((row_count, max_seq_len), dtype=torch.int64)
        self.history_lengths = torch.zeros(row_count, dtype=torch.int64)
        self.target_ids = torch.empty(row_count, dtype=torch.int64)
        self.target_ratings = torch.empty(row_count, dtype=torch.int64)
        self.target_timestamps = torch.empty(row_count, dtype=torch.int64)
        self.target_ser_labels = torch.zeros(row_count, dtype=torch.int64)

        for row_idx, row in enumerate(frame.itertuples(index=False)):
            history_items = [
                int(float(x)) + self._shift_id_by
                for x in parse_sequence_value(row.history_items)
            ]
            history_ratings = [int(float(x)) for x in parse_sequence_value(row.history_ratings)]
            history_timestamps = [
                int(float(x)) for x in parse_sequence_value(row.history_timestamps)
            ]
            history_length = min(len(history_items), max_seq_len)
            if not self._chronological:
                history_items = list(reversed(history_items))[:max_seq_len]
                history_ratings = list(reversed(history_ratings))[:max_seq_len]
                history_timestamps = list(reversed(history_timestamps))[:max_seq_len]
            else:
                history_items = history_items[-max_seq_len:]
                history_ratings = history_ratings[-max_seq_len:]
                history_timestamps = history_timestamps[-max_seq_len:]

            self.user_ids[row_idx] = row.user_id
            self.history_lengths[row_idx] = history_length
            if history_items:
                length = len(history_items)
                self.historical_ids[row_idx, :length] = torch.tensor(
                    history_items,
                    dtype=torch.int64,
                )
                self.historical_ratings[row_idx, :length] = torch.tensor(
                    history_ratings,
                    dtype=torch.int64,
                )
                self.historical_timestamps[row_idx, :length] = torch.tensor(
                    history_timestamps,
                    dtype=torch.int64,
                )
            self.target_ids[row_idx] = int(float(row.target_item)) + self._shift_id_by
            self.target_ratings[row_idx] = int(float(row.target_rating))
            self.target_timestamps[row_idx] = int(float(row.target_timestamp))
            self.target_ser_labels[row_idx] = int(float(getattr(row, "target_ser_label", 0)))


class LOOManifestTrainDataset(torch.utils.data.Dataset):
    """User-level LOO train rows for next-transition policy training."""

    def __init__(
        self,
        ratings_file: str | pd.DataFrame,
        padding_length: int,
        shift_id_by: int = 0,
        chronological: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self._padding_length = int(padding_length)
        self._shift_id_by = int(shift_id_by)
        self._chronological = bool(chronological)
        self._materialize(load_data(ratings_file))

    def __len__(self) -> int:
        return int(self.target_ids.size(0))

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            "user_id": self.user_ids[idx],
            "historical_ids": self.historical_ids[idx],
            "historical_ratings": self.historical_ratings[idx],
            "historical_timestamps": self.historical_timestamps[idx],
            "history_lengths": self.history_lengths[idx],
            "target_ids": self.target_ids[idx],
            "target_ratings": self.target_ratings[idx],
            "target_timestamps": self.target_timestamps[idx],
        }

    def _materialize(self, frame: pd.DataFrame) -> None:
        max_seq_len = self._padding_length - 1
        row_count = len(frame)
        self.user_ids: list[Any] = [None] * row_count
        self.historical_ids = torch.zeros((row_count, max_seq_len), dtype=torch.int64)
        self.historical_ratings = torch.zeros((row_count, max_seq_len), dtype=torch.int64)
        self.historical_timestamps = torch.zeros((row_count, max_seq_len), dtype=torch.int64)
        self.history_lengths = torch.zeros(row_count, dtype=torch.int64)
        self.target_ids = torch.empty(row_count, dtype=torch.int64)
        self.target_ratings = torch.empty(row_count, dtype=torch.int64)
        self.target_timestamps = torch.empty(row_count, dtype=torch.int64)

        for row_idx, row in enumerate(frame.itertuples(index=False)):
            items = [int(float(x)) for x in parse_sequence_value(row.train_items)]
            ratings = [int(float(x)) for x in parse_sequence_value(row.train_ratings)]
            timestamps = [int(float(x)) for x in parse_sequence_value(row.train_timestamps)]
            n = min(len(items), len(ratings), len(timestamps))
            items, ratings, timestamps = items[:n], ratings[:n], timestamps[:n]
            if len(items) < 2:
                raise ValueError("LOOManifestTrainDataset rows need at least two train items")

            window_len = self._padding_length
            if not self._chronological:
                items = list(reversed(items))[:window_len]
                ratings = list(reversed(ratings))[:window_len]
                timestamps = list(reversed(timestamps))[:window_len]
            else:
                items = items[-window_len:]
                ratings = ratings[-window_len:]
                timestamps = timestamps[-window_len:]

            historical_items = [int(item) + self._shift_id_by for item in items[:-1]]
            historical_ratings = ratings[:-1]
            historical_timestamps = timestamps[:-1]
            history_length = min(len(historical_items), max_seq_len)

            self.user_ids[row_idx] = row.user_id
            self.history_lengths[row_idx] = history_length
            if historical_items:
                length = len(historical_items)
                self.historical_ids[row_idx, :length] = torch.tensor(
                    historical_items,
                    dtype=torch.int64,
                )
                self.historical_ratings[row_idx, :length] = torch.tensor(
                    historical_ratings,
                    dtype=torch.int64,
                )
                self.historical_timestamps[row_idx, :length] = torch.tensor(
                    historical_timestamps,
                    dtype=torch.int64,
                )
            self.target_ids[row_idx] = int(items[-1]) + self._shift_id_by
            self.target_ratings[row_idx] = int(float(ratings[-1]))
            self.target_timestamps[row_idx] = int(float(timestamps[-1]))



class FutureWindowTargetDataset(torch.utils.data.Dataset):
    """Dataset over V2 future-window target rows.

    Rows are produced by ``tools/build_future_window_targets.py`` and contain a
    user-position context plus multi-positive I/A SID sets. This is the data
    bridge that prevents V2 training from silently falling back to the legacy
    history-window acceptable proxy.
    """

    def __init__(
        self,
        ratings_file: str | pd.DataFrame,
        padding_length: int,
        max_i_targets: int = 3,
        max_a_targets: int = 32,
        shift_id_by: int = 0,
        sid_columns: int | None = None,
        chronological: bool = True,
        emit_rank_from_future_targets: bool = False,
        rank_geometry_levels: str = "adaptive_semantic_non_dedup",
        rank_geometry_recency_decay: float = 0.85,
        rank_geometry_epsilon: float = 1e-6,
        rank_geometry_history_len: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.ratings_frame = load_data(ratings_file)
        # RecoDataModule passes max_sequence_length + 1 because RecoDataset uses
        # one slot for the target and the rest for history. Future target rows
        # already store R_item separately, so only history receives padding.
        self._padding_length = int(padding_length)
        self._history_padding_length = max(self._padding_length - 1, 1)
        self.max_i_targets = int(max_i_targets)
        self.max_a_targets = int(max_a_targets)
        self._shift_id_by = int(shift_id_by)
        self.sid_columns = int(sid_columns or self._infer_sid_columns())
        self._chronological = bool(chronological)
        self.emit_rank_from_future_targets = bool(emit_rank_from_future_targets)
        self.rank_geometry_levels = rank_geometry_levels
        self.rank_geometry_recency_decay = float(rank_geometry_recency_decay)
        self.rank_geometry_epsilon = float(rank_geometry_epsilon)
        self.rank_geometry_history_len = (
            None if rank_geometry_history_len is None else int(rank_geometry_history_len)
        )

    def __len__(self) -> int:
        return len(self.ratings_frame)

    @staticmethod
    def _loads(value):
        if value is None:
            return []
        if isinstance(value, float) and pd.isna(value):
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            return json.loads(text)
        if hasattr(value, "tolist"):
            return value.tolist()
        return value if isinstance(value, list) else []

    def _infer_sid_columns(self) -> int:
        for column in ("R_sid", "pos_sid", "I_sids", "A_sids", "neg_sids", "train_sids"):
            if column not in self.ratings_frame.columns:
                continue
            for value in self.ratings_frame[column].tolist():
                parsed = self._loads(value)
                if not parsed:
                    continue
                if column in {"R_sid", "pos_sid"}:
                    return len(parsed)
                first = parsed[0] if isinstance(parsed[0], list) else parsed
                if first:
                    return len(first)
        return 4

    def _pad_1d(self, values: list[int | float], dtype=torch.int64) -> torch.Tensor:
        values = list(values)
        if len(values) > self._history_padding_length:
            values = values[-self._history_padding_length:] if self._chronological else values[: self._history_padding_length]
        if len(values) < self._history_padding_length:
            values = values + [0] * (self._history_padding_length - len(values))
        return torch.tensor(values, dtype=dtype)

    def _pad_sids(self, values, max_targets: int) -> torch.Tensor:
        parsed = self._loads(values)
        out = torch.zeros((max_targets, self.sid_columns), dtype=torch.int64)
        for idx, sid in enumerate(parsed[:max_targets]):
            if not sid:
                continue
            sid = [int(token) for token in sid[: self.sid_columns]]
            out[idx, : len(sid)] = torch.tensor(sid, dtype=torch.int64)
        return out

    def _all_sids(self, values) -> torch.Tensor:
        parsed = self._loads(values)
        if not parsed:
            return torch.zeros((0, self.sid_columns), dtype=torch.int64)
        out = torch.zeros((len(parsed), self.sid_columns), dtype=torch.int64)
        for idx, sid in enumerate(parsed):
            if not sid:
                continue
            sid = [int(token) for token in sid[: self.sid_columns]]
            out[idx, : len(sid)] = torch.tensor(sid, dtype=torch.int64)
        return out

    def _row_value(self, row: pd.Series, name: str, default=0):
        return row[name] if name in row.index else default

    def _rank_candidates(
        self,
        a_items: list[int],
        a_sids: torch.Tensor,
        i_items: list[int],
        i_sids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def dedup(items: list[int], sids: torch.Tensor, limit: int, blocked: set[int] | None = None) -> torch.Tensor:
            blocked = blocked or set()
            rows = []
            seen: set[int] = set()
            for item, sid in zip(items, sids):
                item = int(item)
                if item in seen or item in blocked or not bool((sid != 0).any().item()):
                    continue
                seen.add(item)
                rows.append(sid)
                if len(rows) >= limit:
                    break
            if not rows:
                return torch.zeros((0, self.sid_columns), dtype=torch.int64)
            return torch.stack(rows).to(torch.int64)

        i_item_set = {int(item) for item in i_items}
        pos = dedup(a_items, a_sids, self.max_a_targets, blocked=i_item_set)
        neg = dedup(i_items, i_sids, self.max_i_targets)
        width = self.max_a_targets + self.max_i_targets
        candidates = torch.zeros((width, self.sid_columns), dtype=torch.int64)
        positive_mask = torch.zeros(width, dtype=torch.bool)
        if pos.numel():
            candidates[: pos.size(0)] = pos
            positive_mask[: pos.size(0)] = True
        if neg.numel():
            start = int(pos.size(0))
            candidates[start : start + neg.size(0)] = neg
        return candidates, positive_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.ratings_frame.iloc[idx]
        history_items = [int(x) for x in self._loads(row.history_items)]
        history_ratings = [int(float(x)) for x in self._loads(row.history_ratings)]
        history_timestamps = [int(float(x)) for x in self._loads(row.history_timestamps)]
        history_length = min(len(history_items), self._history_padding_length)
        i_items = [int(x) for x in self._loads(self._row_value(row, "I_items", "[]"))]
        a_items = [int(x) for x in self._loads(self._row_value(row, "A_items", "[]"))]
        i_sids = self._pad_sids(self._row_value(row, "I_sids", "[]"), self.max_i_targets)
        a_sids = self._pad_sids(self._row_value(row, "A_sids", "[]"), self.max_a_targets)
        r_item = self._row_value(row, "R_item", self._row_value(row, "pos_item_id", 0))
        r_rating = self._row_value(row, "R_rating", self._row_value(row, "pos_rating", 1.0))
        timestamp_t = self._row_value(row, "timestamp_t", 0)
        ret = {
            "user_id": row.user_id,
            "historical_ids": self._pad_1d(
                [item + self._shift_id_by for item in history_items],
                torch.int64,
            ),
            "historical_ratings": self._pad_1d(history_ratings, torch.int64),
            "historical_timestamps": self._pad_1d(history_timestamps, torch.int64),
            "history_lengths": torch.tensor(history_length, dtype=torch.int64),
            "target_ids": torch.tensor(int(r_item) + self._shift_id_by, dtype=torch.int64),
            "target_ratings": torch.tensor(int(float(r_rating)), dtype=torch.int64),
            "target_timestamps": torch.tensor(int(float(timestamp_t)), dtype=torch.int64),
            "I_sids": i_sids,
            "A_sids": a_sids,
        }
        if self.emit_rank_from_future_targets:
            if "history_sids" not in row.index:
                raise ValueError(
                    "FutureWindowTargetDataset needs history_sids to emit rank_geometry; "
                    "rebuild future targets with tools/build_future_window_targets.py."
                )
            rank_sids, positive_mask = self._rank_candidates(
                a_items,
                self._all_sids(self._row_value(row, "A_sids", "[]")),
                i_items,
                self._all_sids(self._row_value(row, "I_sids", "[]")),
            )
            history_sids = self._pad_sids(row.history_sids, max(len(history_items), 1))
            valid_history_sids = history_sids[(history_sids != 0).all(dim=1)]
            ret["rank_candidate_sids"] = rank_sids
            ret["rank_geometry"] = prefix_surprise(
                valid_history_sids,
                rank_sids,
                levels=self.rank_geometry_levels,
                recency_decay=self.rank_geometry_recency_decay,
                epsilon=self.rank_geometry_epsilon,
                max_history=self.rank_geometry_history_len,
            ).to(torch.float32)
            ret["rank_positive_mask"] = positive_mask
        return ret


class DynamicFutureWindowTargetDataset(torch.utils.data.Dataset):
    """Dynamically constructs sparse V2 future-window targets from LOO train rows."""

    def __init__(
        self,
        ratings_file: str | pd.DataFrame,
        padding_length: int,
        max_i_targets: int = 3,
        max_a_targets: int = 32,
        shift_id_by: int = 0,
        sid_columns: int | None = None,
        chronological: bool = True,
        emit_rank_from_future_targets: bool = False,
        imminent_window: int = 3,
        acceptable_min_gap: int = 4,
        acceptable_window: int = 50,
        rating_positive_threshold: float = 4.0,
        exclude_seen: bool = True,
        rank_geometry_levels: str = "adaptive_semantic_non_dedup",
        rank_geometry_recency_decay: float = 0.85,
        rank_geometry_epsilon: float = 1e-6,
        rank_geometry_history_len: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.ratings_frame = load_data(ratings_file)
        self._padding_length = int(padding_length)
        self._history_padding_length = max(self._padding_length - 1, 1)
        self.max_i_targets = int(max_i_targets)
        self.max_a_targets = int(max_a_targets)
        self._shift_id_by = int(shift_id_by)
        self.sid_columns = int(sid_columns or self._infer_sid_columns())
        self._chronological = bool(chronological)
        self.emit_rank_from_future_targets = bool(emit_rank_from_future_targets)
        self.imminent_window = int(imminent_window)
        self.acceptable_min_gap = int(acceptable_min_gap)
        self.acceptable_window = int(acceptable_window)
        self.rating_positive_threshold = float(rating_positive_threshold)
        self.exclude_seen = bool(exclude_seen)
        self.rank_geometry_levels = rank_geometry_levels
        self.rank_geometry_recency_decay = float(rank_geometry_recency_decay)
        self.rank_geometry_epsilon = float(rank_geometry_epsilon)
        self.rank_geometry_history_len = (
            None if rank_geometry_history_len is None else int(rank_geometry_history_len)
        )
        lengths = []
        for value in self.ratings_frame["train_items"].tolist():
            lengths.append(len(self._loads(value)))
        self._sequence_lengths = lengths
        self._cum_positions: list[int] = []
        total = 0
        for length in lengths:
            total += max(int(length) - 1, 0)
            self._cum_positions.append(total)

    @staticmethod
    def _loads(value):
        return FutureWindowTargetDataset._loads(value)

    def _infer_sid_columns(self) -> int:
        if "train_sids" not in self.ratings_frame.columns:
            return 4
        for value in self.ratings_frame["train_sids"].tolist():
            parsed = self._loads(value)
            if parsed and parsed[0]:
                return len(parsed[0])
        return 4

    def __len__(self) -> int:
        return self._cum_positions[-1] if self._cum_positions else 0

    def _position(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        row_idx = bisect.bisect_right(self._cum_positions, idx)
        prev = 0 if row_idx == 0 else self._cum_positions[row_idx - 1]
        return row_idx, idx - prev

    def _pad_1d(self, values: list[int | float], dtype=torch.int64) -> torch.Tensor:
        values = list(values)
        if len(values) > self._history_padding_length:
            values = values[-self._history_padding_length:] if self._chronological else values[: self._history_padding_length]
        if len(values) < self._history_padding_length:
            values = values + [0] * (self._history_padding_length - len(values))
        return torch.tensor(values, dtype=dtype)

    def _sids_tensor(self, sids: list[list[int]], max_targets: int) -> torch.Tensor:
        out = torch.zeros((max_targets, self.sid_columns), dtype=torch.int64)
        for idx, sid in enumerate(sids[:max_targets]):
            if not sid:
                continue
            sid = [int(token) for token in sid[: self.sid_columns]]
            out[idx, : len(sid)] = torch.tensor(sid, dtype=torch.int64)
        return out

    def _targets(
        self,
        items: list[int],
        ratings: list[float],
        sids: list[list[int]],
        position_t: int,
    ) -> tuple[list[tuple[int, int, float, list[int]]], list[tuple[int, int, float, list[int]]]]:
        seen = set(items[: position_t + 1])
        imminent: list[tuple[int, int, float, list[int]]] = []
        acceptable: list[tuple[int, int, float, list[int]]] = []
        for delta in range(1, self.imminent_window + 1):
            pos = position_t + delta
            if pos >= len(items):
                break
            if self.exclude_seen and items[pos] in seen:
                continue
            imminent.append((items[pos], delta, ratings[pos], sids[pos]))
        a_start = max(self.acceptable_min_gap, self.imminent_window + 1)
        for delta in range(a_start, self.acceptable_window + 1):
            pos = position_t + delta
            if pos >= len(items):
                break
            if ratings[pos] < self.rating_positive_threshold:
                continue
            if self.exclude_seen and items[pos] in seen:
                continue
            acceptable.append((items[pos], delta, ratings[pos], sids[pos]))
        imminent = self._dedup(imminent, key=lambda row: row[1])[: self.max_i_targets]
        blocked = {int(row[0]) for row in imminent}
        acceptable = [
            row
            for row in self._dedup(acceptable, key=lambda row: (-row[2], row[1]))
            if int(row[0]) not in blocked
        ][: self.max_a_targets]
        return imminent, acceptable

    @staticmethod
    def _dedup(items: list[tuple[int, int, float, list[int]]], key) -> list[tuple[int, int, float, list[int]]]:
        out = []
        seen: set[int] = set()
        for row in sorted(items, key=key):
            item = int(row[0])
            if item in seen:
                continue
            seen.add(item)
            out.append(row)
        return out

    def _rank_candidates(
        self,
        acceptable: list[tuple[int, int, float, list[int]]],
        imminent: list[tuple[int, int, float, list[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i_items = {int(row[0]) for row in imminent}
        candidates = torch.zeros((self.max_a_targets + self.max_i_targets, self.sid_columns), dtype=torch.int64)
        positive_mask = torch.zeros(self.max_a_targets + self.max_i_targets, dtype=torch.bool)
        cursor = 0
        for row in acceptable:
            if int(row[0]) in i_items or not row[3]:
                continue
            candidates[cursor] = self._sids_tensor([row[3]], 1)[0]
            positive_mask[cursor] = True
            cursor += 1
            if cursor >= self.max_a_targets:
                break
        for row in imminent:
            if cursor >= candidates.size(0):
                break
            if not row[3]:
                continue
            candidates[cursor] = self._sids_tensor([row[3]], 1)[0]
            cursor += 1
        return candidates, positive_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx, position_t = self._position(int(idx))
        row = self.ratings_frame.iloc[row_idx]
        items = [int(float(x)) for x in self._loads(row.train_items)]
        ratings = [float(x) for x in self._loads(row.train_ratings)]
        timestamps = [int(float(x)) for x in self._loads(row.train_timestamps)]
        sids = self._loads(row.train_sids) if "train_sids" in row.index else [[] for _ in items]
        n = min(len(items), len(ratings), len(timestamps), len(sids))
        items, ratings, timestamps, sids = items[:n], ratings[:n], timestamps[:n], sids[:n]
        history_items = items[: position_t + 1]
        history_ratings = ratings[: position_t + 1]
        history_timestamps = timestamps[: position_t + 1]
        history_sids = sids[: position_t + 1]
        target_pos = position_t + 1
        imminent, acceptable = self._targets(items, ratings, sids, position_t)
        i_sids = self._sids_tensor([row[3] for row in imminent], self.max_i_targets)
        a_sids = self._sids_tensor([row[3] for row in acceptable], self.max_a_targets)
        ret = {
            "user_id": row.user_id,
            "historical_ids": self._pad_1d([item + self._shift_id_by for item in history_items], torch.int64),
            "historical_ratings": self._pad_1d([int(float(x)) for x in history_ratings], torch.int64),
            "historical_timestamps": self._pad_1d(history_timestamps, torch.int64),
            "history_lengths": torch.tensor(min(len(history_items), self._history_padding_length), dtype=torch.int64),
            "target_ids": torch.tensor(int(items[target_pos]) + self._shift_id_by, dtype=torch.int64),
            "target_ratings": torch.tensor(int(float(ratings[target_pos])), dtype=torch.int64),
            "target_timestamps": torch.tensor(int(float(timestamps[position_t])), dtype=torch.int64),
            "I_sids": i_sids,
            "A_sids": a_sids,
        }
        if self.emit_rank_from_future_targets:
            rank_sids, positive_mask = self._rank_candidates(acceptable, imminent)
            history_sid_tensor = self._sids_tensor(history_sids, max(len(history_sids), 1))
            valid_history_sids = history_sid_tensor[(history_sid_tensor != 0).all(dim=1)]
            ret["rank_candidate_sids"] = rank_sids
            ret["rank_geometry"] = prefix_surprise(
                valid_history_sids,
                rank_sids,
                levels=self.rank_geometry_levels,
                recency_decay=self.rank_geometry_recency_decay,
                epsilon=self.rank_geometry_epsilon,
                max_history=self.rank_geometry_history_len,
            ).to(torch.float32)
            ret["rank_positive_mask"] = positive_mask
        return ret

class RecoDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_preprocessor: DataProcessor,
        train_dataset: RecoDataset | DictConfig,
        val_dataset: RecoDataset | DictConfig,
        test_dataset: RecoDataset | DictConfig,
        max_sequence_length: int,
        chronological: bool,
        positional_sampling_ratio: float,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        prefetch_factor: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        semantic_id_prefix: str | None = None,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.dataset_name = dataset_name
        self.data_preprocessor: DataProcessor = (
            hydra.utils.instantiate(data_preprocessor)
            if isinstance(data_preprocessor, DictConfig)
            else data_preprocessor
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.max_sequence_length = max_sequence_length
        self.chronological = chronological
        self.positional_sampling_ratio = positional_sampling_ratio
        self.batch_size = batch_size
        if num_workers is None:
            cpu_count = os.cpu_count() or 0
            num_workers = 0 if cpu_count < 2 else max(cpu_count // 4, 1)
        self.num_workers = int(num_workers)
        self.prefetch = prefetch_factor if self.num_workers > 0 else None
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.semantic_id_prefix = semantic_id_prefix
        self.__init_item_ids()

    def __init_item_ids(self):
        if self.dataset_name == "ml-1m" or self.dataset_name == "ml-20m":
            items = pd.read_csv(
                self.data_preprocessor.processed_item_csv(), delimiter=","
            )
            max_jagged_dimension = 16
            max_item_id = self.data_preprocessor.expected_max_item_id()

            # Initialize dictionaries for lengths and values
            lengths = {
                i: torch.zeros((max_item_id + 1,), dtype=torch.int64) for i in range(3)
            }
            values = {
                i: torch.zeros(
                    (max_item_id + 1, max_jagged_dimension), dtype=torch.int64
                )
                for i in range(3)
            }

            # Define max index ranges for each feature type
            max_ind_ranges = [63, 16383, 511]

            all_item_ids = []
            for df_index, row in items.iterrows():
                movie_id = int(row["movie_id"])
                genres = row["genres"].split("|")
                titles = row["cleaned_title"].split(" ")
                years = [row["year"]]

                # Process each feature type
                for i, feature_set in enumerate([genres, titles, years]):
                    feature_vector = [hash(x) % max_ind_ranges[i] for x in feature_set]
                    lengths[i][movie_id] = min(
                        len(feature_vector), max_jagged_dimension
                    )
                    for j, value in enumerate(feature_vector[:max_jagged_dimension]):
                        values[i][movie_id][j] = value

                all_item_ids.append(movie_id)
            self.all_item_ids = all_item_ids
            self.max_item_id = max_item_id
        else:
            expected_items = self.data_preprocessor.expected_num_unique_items()
            if expected_items is None:
                raise ValueError(
                    "Data preprocessor could not determine the number of unique items. "
                    "Ensure the dataset is prepared (e.g., `make prepare_data data=%s`)."
                    % self.dataset_name
                )
            self.all_item_ids = [x + 1 for x in range(expected_items)]
            self.max_item_id = expected_items

    def instantiate_dataset(self, dataset: RecoDataset | DictConfig) -> RecoDataset:
        if isinstance(dataset, DictConfig):
            kwargs = {}
            if "padding_length" not in dataset:
                kwargs["padding_length"] = self.max_sequence_length + 1
            if "chronological" not in dataset:
                kwargs["chronological"] = self.chronological
            if "position_sampling_ratio" not in dataset:
                kwargs["sample_ratio"] = self.positional_sampling_ratio
            # preload the data for shared dataset
            ratings_file = (
                dataset.pop("ratings_file")
                if "ratings_file" in dataset
                else self.data_preprocessor.output_format_csv()
            )
            ratings_file = load_data(ratings_file)
            return hydra.utils.instantiate(dataset, ratings_file=ratings_file, **kwargs)
        else:
            return dataset

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.instantiate_dataset(self.train_dataset)
            self.val_dataset = self.instantiate_dataset(self.val_dataset)

        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = self.instantiate_dataset(self.test_dataset)

    def train_dataloader(self):
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.prefetch is not None:
            kwargs["prefetch_factor"] = self.prefetch
        if self.persistent_workers:
            kwargs["persistent_workers"] = True
        return torch.utils.data.DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self):
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.prefetch is not None:
            kwargs["prefetch_factor"] = self.prefetch
        if self.persistent_workers:
            kwargs["persistent_workers"] = True
        return torch.utils.data.DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self):
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.prefetch is not None:
            kwargs["prefetch_factor"] = self.prefetch
        if self.persistent_workers:
            kwargs["persistent_workers"] = True
        return torch.utils.data.DataLoader(self.test_dataset, **kwargs)

    def predict_dataloader(self):
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.prefetch is not None:
            kwargs["prefetch_factor"] = self.prefetch
        if self.persistent_workers:
            kwargs["persistent_workers"] = True
        return torch.utils.data.DataLoader(self.test_dataset, **kwargs)

    def save_predictions(self, output_file: str, predictions: dict):
        """Save the predictions to a file.

        It adds the predictions to the ratings_frame in the test dataset
        since it is used for prediction and saves it to a file. And it
        expects the predictions to be a dictionary of list / numpy arrays,
        which has the same length and order as the test dataset.

        Args:
            output_file: str, path to the output file.
            predictions: dict, predictions to save.
        """
        ratings_frame = self.test_dataset.ratings_frame
        for key, value in predictions.items():
            ratings_frame[key] = value
        save_data(ratings_frame, output_file)
