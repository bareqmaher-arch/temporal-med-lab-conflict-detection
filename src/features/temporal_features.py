"""Temporal feature extraction for a single (patient, lab) series.

All features are computed using only readings at or before an ``observation cutoff``
so the feature matrix never leaks future (label-window) information.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import SLOPE_WINDOWS
from src.preprocessing.build_timeline import compute_baseline


def _days(d: date, ref: date) -> int:
    return (pd.Timestamp(d) - pd.Timestamp(ref)).days


def _slope_per_day(sub: pd.DataFrame, ref: date) -> float:
    """Linear-regression slope (units/day) of value vs days-since-ref."""
    if len(sub) < 2:
        return 0.0
    x = np.array([_days(d, ref) for d in sub["lab_date"]], dtype=float)
    y = sub["value"].to_numpy(dtype=float)
    if np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def compute_temporal_features(
    series: pd.DataFrame, drug_start: date, cutoff: date, direction: int,
    dose_change_date: date | None = None,
) -> dict:
    """direction: +1 if rising is dangerous, -1 if falling is dangerous."""
    s = series[series["lab_date"] <= cutoff].sort_values("lab_date")
    baseline = compute_baseline(series, drug_start)
    feats = {
        "baseline_value": baseline,
        "current_value": None, "delta_value": None, "percent_change": None,
        "min_in_window": None, "max_in_window": None, "variability": None,
        "acceleration": 0.0, "consecutive_abnormal_trend": 0,
        "days_since_drug_start": _days(cutoff, drug_start),
        "days_since_dose_change": (_days(cutoff, dose_change_date)
                                   if dose_change_date is not None else -1),
        "trend_started_after_drug": 0,
    }
    for w in SLOPE_WINDOWS:
        feats[f"slope_{w}d"] = 0.0
    if s.empty:
        return feats

    post = s[s["lab_date"] >= drug_start]
    work = post if not post.empty else s
    current = float(work["value"].iloc[-1])
    feats["current_value"] = current
    feats["min_in_window"] = float(work["value"].min())
    feats["max_in_window"] = float(work["value"].max())
    feats["variability"] = float(work["value"].std(ddof=0))

    if baseline is not None:
        feats["delta_value"] = current - baseline
        feats["percent_change"] = (current - baseline) / baseline if baseline else 0.0

    # slopes over windows ending at cutoff
    for w in SLOPE_WINDOWS:
        win = work[work["lab_date"] >= (cutoff - timedelta(days=w))]
        feats[f"slope_{w}d"] = _slope_per_day(win, drug_start)

    # acceleration: recent slope minus earlier slope
    feats["acceleration"] = feats["slope_7d"] - feats["slope_30d"]

    # consecutive readings moving in the danger direction (post-drug)
    vals = work["value"].to_numpy(dtype=float)
    count = 0
    for i in range(len(vals) - 1, 0, -1):
        if direction * (vals[i] - vals[i - 1]) > 0:
            count += 1
        else:
            break
    feats["consecutive_abnormal_trend"] = count

    # did the danger-direction trend begin after the drug?
    if baseline is not None and len(post) >= 1:
        feats["trend_started_after_drug"] = int(direction * (current - baseline) > 0)

    return feats
