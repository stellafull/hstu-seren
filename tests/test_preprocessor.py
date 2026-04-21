from pathlib import Path

import pandas as pd

from generative_recommenders_pl.data.preprocessor import (
    AmazonDataProcessor,
    MovielensDataProcessor,
)


def _restore_rows(processor) -> set[tuple[str, str, int]]:
    ratings = pd.read_csv(processor.normalized_ratings_csv())
    user_lookup = pd.read_csv(processor.user_lookup_csv())
    item_lookup = pd.read_csv(processor.item_lookup_csv())
    user_map = dict(
        zip(user_lookup["normalized_user_id"], user_lookup["original_user_id"])
    )
    item_map = dict(
        zip(item_lookup["normalized_item_id"], item_lookup["original_item_id"])
    )
    return {
        (user_map[row.user_id], item_map[row.item_id], int(row.timestamp))
        for row in ratings.itertuples(index=False)
    }


def _write_movies_catalog(path: Path, movie_ids: list[str]) -> None:
    lines = ["movieId,title\n"]
    lines.extend(f"{movie_id},{movie_id}\n" for movie_id in movie_ids)
    path.write_text("".join(lines), encoding="utf-8")


def test_movielens_source_preprocessing_filters_target_overlap_and_future_rows(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.csv"
    answers_path = tmp_path / "answers.csv"
    movies_path = tmp_path / "movies.csv"

    pd.DataFrame(
        [
            {"userId": "UserA", "movieId": "M1", "rating": 5.0, "timestamp": 100},
            {"userId": "UserA", "movieId": "M2", "rating": 4.0, "timestamp": 150},
            {"userId": "UserA", "movieId": "M3", "rating": 4.0, "timestamp": 220},
            {"userId": "UserB", "movieId": "M4", "rating": 3.0, "timestamp": 50},
            {"userId": "UserB", "movieId": "M5", "rating": 3.0, "timestamp": 110},
            {"userId": "UserB", "movieId": "M7", "rating": 4.0, "timestamp": 120},
            {"userId": "UserC", "movieId": "M6", "rating": 4.0, "timestamp": 70},
        ]
    ).to_csv(training_path, index=False)
    pd.DataFrame(
        [
            {"userId": "usera", "movieId": "M2", "rating": 5.0, "timestamp": 150},
            {"userId": "USERB", "movieId": "M9", "rating": 4.0, "timestamp": 110},
        ]
    ).to_csv(answers_path, index=False)
    _write_movies_catalog(
        movies_path,
        ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"],
    )

    processor = MovielensDataProcessor(
        ratings_path=training_path,
        leakage_reference_path=answers_path,
        movies_path=movies_path,
        prefix="ml-local-gate",
        output_root=tmp_path / "out",
        lookup_root=tmp_path / "lookups",
        min_presence=1,
        min_sequence_length=1,
    )

    num_items = processor.preprocess_rating()

    assert num_items == 9
    assert _restore_rows(processor) == {
        ("UserA", "M1", 100),
        ("UserB", "M4", 50),
        ("UserC", "M6", 70),
    }

    seq_df = pd.read_csv(processor.output_format_csv())
    assert set(seq_df["sequence_timestamps"]) == {"100", "50", "70"}


def test_amazon_source_preprocessing_filters_target_overlap_and_boundary_rows(
    tmp_path: Path,
) -> None:
    ratings_path = tmp_path / "ratings.csv"
    serenlens_path = tmp_path / "serenlens.csv"

    ratings_path.write_text(
        "\n".join(
            [
                "usera,item1,5.0,10",
                "usera,item2,4.0,20",
                "usera,item3,4.0,25",
                "userb,item4,4.0,5",
                "userb,item5,4.0,15",
                "userc,item6,4.0,30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "user_id": "USERA",
                "item_id": "item2",
                "timestamp": 20,
                "review": "x",
                "rating": 5.0,
                "label": 1,
            },
            {
                "user_id": "userb",
                "item_id": "item9",
                "timestamp": 15,
                "review": "y",
                "rating": 4.0,
                "label": 0,
            },
            {
                "user_id": "userc",
                "item_id": "item10",
                "timestamp": 10,
                "review": "z",
                "rating": 3.0,
                "label": 0,
            },
        ]
    ).to_csv(serenlens_path, index=False)

    processor = AmazonDataProcessor(
        ratings_path=ratings_path,
        serenlens_path=serenlens_path,
        prefix="amazon-local-gate",
        output_root=tmp_path / "out",
        lookup_root=tmp_path / "lookups",
        min_presence=1,
        min_sequence_length=1,
    )

    num_items = processor.preprocess_rating()

    assert num_items == 2
    assert _restore_rows(processor) == {
        ("usera", "item1", 10),
        ("userb", "item4", 5),
    }

