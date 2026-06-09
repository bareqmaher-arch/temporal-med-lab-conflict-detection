"""Conventional static-threshold rules (the baseline for comparison)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.config import Scenario
from src.preprocessing.build_timeline import labs_for


def static_fires(scenario: Scenario, current_value: float | None) -> bool:
    if current_value is None:
        return False
    if scenario.static_direction == "above":
        return current_value > scenario.static_threshold
    return current_value < scenario.static_threshold


def first_static_alert_date(scenario: Scenario, labs: pd.DataFrame,
                            patient_id: int, drug_start: date) -> date | None:
    """First post-drug reading of the primary lab that crosses the static threshold."""
    s = labs_for(labs, patient_id, scenario.primary_lab)
    s = s[s["lab_date"] >= drug_start]
    for _, row in s.iterrows():
        if static_fires(scenario, float(row["value"])):
            return row["lab_date"]
    return None
