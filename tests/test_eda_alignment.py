import math

import pandas as pd

from EDA.local_gate import summarize_alignment_scores


def test_summarize_alignment_scores_ignores_nan_positive_rows() -> None:
    scores = pd.DataFrame(
        {
            "y_ser": [1, 1, 1, 0],
            "F_alignment_rank_pct": [0.2, math.nan, 0.6, math.nan],
        }
    )

    summary = summarize_alignment_scores(scores, domain_label="Books", sample_size=50)

    row = summary.iloc[0]
    assert row["domain"] == "Books"
    assert row["sample_size"] == 2
    assert row["median_rank_pct"] == 0.4
    assert row["mean_rank_pct"] == 0.4


def test_summarize_alignment_scores_returns_empty_when_no_valid_positive_rows() -> None:
    scores = pd.DataFrame(
        {
            "y_ser": [1, 1, 0],
            "F_alignment_rank_pct": [math.nan, math.nan, 0.3],
        }
    )

    summary = summarize_alignment_scores(scores, domain_label="Books", sample_size=50)

    row = summary.iloc[0]
    assert row["domain"] == "Books"
    assert row["sample_size"] == 0
    assert math.isnan(row["median_rank_pct"])
    assert math.isnan(row["mean_rank_pct"])
