"""Experiment 1: static-threshold rules vs temporal rules."""
from __future__ import annotations

import pandas as pd

from src.config import SCENARIOS
from src.experiments.common import build_alert_table
from src.models.evaluate import binary_metrics


def _rule_row(scenario_key, method, alerts, y, fired):
    n = len(y)
    m = binary_metrics(y, fired)
    alerts_per_100 = 100 * fired.sum() / n if n else 0
    false_alerts = ((fired == 1) & (y == 0)).sum()
    false_per_100 = 100 * false_alerts / n if n else 0
    return {
        "scenario": SCENARIOS[scenario_key].name, "method": method,
        "sensitivity": m["sensitivity"], "specificity": m["specificity"],
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "false_alert_rate": round(false_per_100 / 100, 4),
        "alerts_per_100": round(alerts_per_100, 1),
        "false_alerts_per_100": round(false_per_100, 1),
    }


def run(patients, medications, labs) -> pd.DataFrame:
    out = []
    for key in SCENARIOS:
        at = build_alert_table(key, patients, medications, labs)
        if at.empty:
            continue
        y = at["label"]
        out.append(_rule_row(key, "Static threshold", at, y, at["static_alert"]))
        out.append(_rule_row(key, "Temporal rule", at, y, at["temporal_alert"]))
    return pd.DataFrame(out)
