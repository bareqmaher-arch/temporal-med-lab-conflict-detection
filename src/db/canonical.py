"""Canonical table contract.

Every data loader (synthetic, CSV, future MIMIC-IV) MUST return a `CanonicalTables`
object whose DataFrames carry exactly these columns. This contract is what makes the
data source swappable: the entire downstream pipeline depends only on these names.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PATIENT_COLUMNS = [
    "patient_id", "age", "sex", "comorbidities", "admission_date",
    "discharge_date", "diagnosis_codes", "renal_disease_status",
    "liver_disease_status",
]

MEDICATION_COLUMNS = [
    "medication_id", "patient_id", "drug_name", "normalized_drug_name",
    "drug_class", "start_date", "stop_date", "dose", "route", "frequency",
    "dose_change_date",
]

LAB_COLUMNS = [
    "lab_id", "patient_id", "lab_name", "normalized_lab_name", "value",
    "unit", "reference_range_low", "reference_range_high", "lab_date",
]

KNOWLEDGE_COLUMNS = [
    "rule_id", "drug_class", "drug_name", "lab_name", "expected_direction",
    "risk_type", "severity_weight", "time_window_days", "threshold_value",
    "delta_threshold", "clinical_note",
]

ALERT_COLUMNS = [
    "alert_id", "patient_id", "medication_id", "lab_id", "alert_type",
    "risk_score", "risk_level", "explanation", "suggested_action",
    "alert_date", "model_type", "clinician_feedback",
]

FEATURE_COLUMNS = [
    "patient_id", "medication_id", "lab_name", "baseline_value",
    "current_value", "delta_value", "percent_change", "slope_7d",
    "slope_14d", "slope_30d", "days_since_drug_start", "age", "egfr",
    "comorbidity_count", "polypharmacy_count", "label",
]

DATE_COLUMNS = {
    "patients": ["admission_date", "discharge_date"],
    "medications": ["start_date", "stop_date", "dose_change_date"],
    "labs": ["lab_date"],
}


@dataclass
class CanonicalTables:
    """The four input tables every loader must produce."""

    patients: pd.DataFrame
    medications: pd.DataFrame
    labs: pd.DataFrame
    knowledge: pd.DataFrame

    def validate(self) -> "CanonicalTables":
        _check(self.patients, PATIENT_COLUMNS, "patients")
        _check(self.medications, MEDICATION_COLUMNS, "medications")
        _check(self.labs, LAB_COLUMNS, "labs")
        _check(self.knowledge, KNOWLEDGE_COLUMNS, "knowledge")
        return self


def _check(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Canonical table '{name}' is missing columns: {missing}")
