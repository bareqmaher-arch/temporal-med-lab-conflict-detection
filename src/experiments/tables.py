"""Descriptive manuscript tables (Table 1 scenarios, Table 2 feature categories)."""
from __future__ import annotations

import pandas as pd

from src.config import SCENARIOS


def table1_scenarios(knowledge: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SCENARIOS.values():
        k = knowledge[(knowledge["drug_class"] == s.drug_class)
                      & (knowledge["lab_name"] == s.primary_lab)]
        risk_type = k["risk_type"].iloc[0] if not k.empty else ""
        pattern = (f"{s.delta_direction} of >= {s.delta_threshold} within "
                   f"{s.temporal_window_days}d, current "
                   f"{'>=' if s.static_direction == 'above' else '<='} {s.temporal_current_gate}")
        rows.append({
            "Drug/Class": f"{s.drug_class} ({', '.join(s.drug_names[:3])}...)",
            "Laboratory Test": ", ".join(s.labs),
            "Risk Type": risk_type.replace("_", " "),
            "Temporal Pattern": pattern,
            "Example Alert": f"{s.name}",
        })
    return pd.DataFrame(rows)


def table2_feature_categories() -> pd.DataFrame:
    data = [
        ("Level / last value", "current_value, baseline_value, min/max in window",
         "Where the patient is now vs their own baseline"),
        ("Change from baseline", "delta_value, percent_change",
         "Patient-specific deviation, not population reference only"),
        ("Temporal slope", "slope_7d, slope_14d, slope_30d, acceleration",
         "Speed and acceleration of change over time"),
        ("Trend persistence", "consecutive_abnormal_trend, trend_started_after_drug",
         "Whether a danger-direction trend is sustained and post-drug"),
        ("Exposure timing", "days_since_drug_start, days_since_dose_change",
         "Temporal link between the drug and the lab change"),
        ("Renal / hepatic", "egfr, renal_impairment, hepatic_impairment",
         "Organ-function vulnerability modifying risk"),
        ("Demographic", "age, age_risk",
         "Age-related susceptibility"),
        ("Polypharmacy", "polypharmacy_count, high_risk_medication_count",
         "Cumulative and overlapping medication risk"),
        ("Drug-lab knowledge", "drug_risk_strength",
         "Strength of the known drug-lab relationship"),
    ]
    return pd.DataFrame(data, columns=["Feature Category", "Examples", "Clinical Meaning"])
