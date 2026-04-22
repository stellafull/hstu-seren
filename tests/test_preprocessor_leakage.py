import pandas as pd

from generative_recommenders_pl.data.preprocessor import (
    apply_source_target_leakage_filter,
    load_reference_interactions,
)


def test_apply_source_target_leakage_filter_removes_pairs_and_post_target_rows():
    ratings = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "item_id": ["i1", "i2", "i3", "i4"],
            "rating": [5.0, 4.0, 3.0, 5.0],
            "timestamp": [10, 20, 35, 5],
        }
    )
    reference = pd.DataFrame(
        {
            "user_id": ["u1", "u3"],
            "item_id": ["i2", "i5"],
            "timestamp": [30, 7],
        }
    )
    reference["_normalized_user_id"] = reference["user_id"].str.casefold()
    reference["_normalized_item_id"] = reference["item_id"].str.casefold()

    cleaned, summary, details = apply_source_target_leakage_filter(
        ratings,
        reference,
        prefix="test",
        reference_name="target.csv",
        return_details=True,
    )

    assert cleaned[["user_id", "item_id"]].values.tolist() == [["u1", "i1"], ["u2", "i4"]]
    assert summary.source_before == 4
    assert summary.target_pairs == 2
    assert summary.removed_target_pairs == 1
    assert summary.removed_post_target_interactions == 1
    assert summary.ambiguous == 0
    assert summary.source_after == 2
    assert details["removed_target_pairs"]["item_id"].tolist() == ["i2"]
    assert details["removed_post_target_interactions"]["item_id"].tolist() == ["i3"]


def test_load_reference_interactions_supports_bom_userid_columns(tmp_path):
    reference_path = tmp_path / "answers.csv"
    reference_path.write_text(
        "\ufeffuserId,movieId,timestamp\n1,10,100\n2,11,101\n",
        encoding="utf-8",
    )

    frame = load_reference_interactions(reference_path)

    assert frame[["user_id", "item_id", "timestamp"]].values.tolist() == [
        ["1", "10", 100],
        ["2", "11", 101],
    ]
    assert set(frame.columns) >= {
        "user_id",
        "item_id",
        "timestamp",
        "_normalized_user_id",
        "_normalized_item_id",
    }
