"""Experiment 3: early detection / time-to-alert.

Among true-positive patients where both methods eventually alert, how many days earlier
does the temporal rule fire than the static threshold? Also reports alert-burden change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SCENARIOS
from src.experiments.common import build_alert_table


def run(patients, medications, labs) -> pd.DataFrame:
    out = []
    for key in SCENARIOS:
        at = build_alert_table(key, patients, medications, labs)
        if at.empty:
            continue
        pos = at[at["label"] == 1]
        both = pos.dropna(subset=["days_to_static", "days_to_temporal"])
        days_earlier = (both["days_to_static"] - both["days_to_temporal"])

        static_alerts = at["static_alert"].sum()
        temporal_alerts = at["temporal_alert"].sum()
        reduction = (100 * (static_alerts - temporal_alerts) / static_alerts
                     if static_alerts else 0.0)
        out.append({
            "scenario": SCENARIOS[key].name,
            "n_positive": int(len(pos)),
            "n_both_alerted": int(len(both)),
            "mean_static_alert_day": round(float(both["days_to_static"].mean()), 1)
            if len(both) else np.nan,
            "mean_temporal_alert_day": round(float(both["days_to_temporal"].mean()), 1)
            if len(both) else np.nan,
            "mean_days_earlier": round(float(days_earlier.mean()), 1) if len(both) else np.nan,
            "median_days_earlier": round(float(days_earlier.median()), 1) if len(both) else np.nan,
            "alert_reduction_pct": round(reduction, 1),
        })
    return pd.DataFrame(out)
