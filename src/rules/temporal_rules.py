"""Temporal, baseline-aware rules.

Fires earlier than the static rule (gate is set before the hard threshold) but only on
a *sustained* trend, so a single transient out-of-range spike does not trigger it — this
is the mechanism behind both earlier detection and fewer false alerts.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.config import Scenario
from src.features.temporal_features import compute_temporal_features
from src.preprocessing.build_timeline import labs_for

_DANGER_DIRECTION = {"potassium": +1, "creatinine": +1, "inr": +1,
                     "egfr": -1, "sodium": -1, "alt": +1}

MIN_CONSECUTIVE = 2  # a sustained trend, not a one-off spike


def temporal_fires(scenario: Scenario, feats: dict) -> bool:
    current = feats.get("current_value")
    if current is None:
        return False
    direction = _DANGER_DIRECTION[scenario.primary_lab]

    if scenario.delta_direction == "increase":
        delta = feats.get("delta_value")
        change_ok = delta is not None and delta >= scenario.delta_threshold
        current_ok = current >= scenario.temporal_current_gate
    else:  # decrease (relative for eGFR)
        pct = feats.get("percent_change")
        change_ok = pct is not None and pct <= -scenario.delta_threshold
        current_ok = current <= scenario.temporal_current_gate

    sustained = (feats.get("consecutive_abnormal_trend", 0) >= MIN_CONSECUTIVE
                 or feats.get(f"slope_{scenario.temporal_window_days}d", 0) * direction > 0
                 and feats.get("consecutive_abnormal_trend", 0) >= 1)
    return bool(change_ok and current_ok and sustained)


def first_temporal_alert_date(scenario: Scenario, labs: pd.DataFrame,
                              patient: pd.Series, drug_start: date,
                              dose_change_date: date | None = None) -> date | None:
    """First post-drug reading at which the temporal rule fires."""
    series = labs_for(labs, patient["patient_id"], scenario.primary_lab)
    direction = _DANGER_DIRECTION[scenario.primary_lab]
    post = series[series["lab_date"] >= drug_start]
    for _, row in post.iterrows():
        cutoff = row["lab_date"]
        feats = compute_temporal_features(series, drug_start, cutoff, direction,
                                          dose_change_date)
        if temporal_fires(scenario, feats):
            return cutoff
    return None
