from datetime import date, timedelta

import pandas as pd

from src.features.temporal_features import compute_temporal_features


def _series(values, start, step=6):
    rows = []
    for i, v in enumerate(values):
        rows.append({"lab_date": start + timedelta(days=i * step),
                     "normalized_lab_name": "potassium", "value": v})
    return pd.DataFrame(rows)


def test_rising_trend_features():
    drug_start = date(2024, 1, 10)
    # two pre-drug baseline points then a rising post-drug trend
    pre = _series([4.2, 4.3], drug_start - timedelta(days=14), step=7)
    post = _series([4.5, 4.9, 5.3, 5.6], drug_start, step=6)
    series = pd.concat([pre, post], ignore_index=True)
    cutoff = post["lab_date"].max()

    f = compute_temporal_features(series, drug_start, cutoff, direction=+1)
    assert f["baseline_value"] == 4.25            # median of pre-drug points
    assert f["current_value"] == 5.6
    assert f["delta_value"] > 0
    assert f["slope_14d"] > 0                       # rising
    assert f["consecutive_abnormal_trend"] >= 2     # sustained rise
    assert f["trend_started_after_drug"] == 1


def test_no_data_returns_safe_defaults():
    drug_start = date(2024, 1, 10)
    empty = pd.DataFrame(columns=["lab_date", "normalized_lab_name", "value"])
    f = compute_temporal_features(empty, drug_start, drug_start, direction=+1)
    assert f["current_value"] is None
    assert f["slope_7d"] == 0.0
