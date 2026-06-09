"""Patient-level risk-modifier features (renal/hepatic/age/polypharmacy)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.config import (
    AGE_RISK_THRESHOLD, HIGH_RISK_DRUG_CLASSES, POLYPHARMACY_THRESHOLD,
    RENAL_IMPAIRMENT_EGFR,
)
from src.preprocessing.build_timeline import labs_for, meds_for_patient


def latest_egfr(labs, patient_id: int, cutoff: date) -> float | None:
    s = labs_for(labs, patient_id, "egfr")
    s = s[s["lab_date"] <= cutoff]
    return float(s["value"].iloc[-1]) if not s.empty else None


def compute_risk_features(patient: pd.Series, medications,
                          labs, cutoff: date) -> dict:
    """Compute patient-level risk modifiers.

    `medications` and `labs` accept either DataFrames or precomputed indices
    (see `index_labs` / `index_medications_by_pid`). On MIMIC-IV the indexed
    form is required — without it every call would re-scan the full table.
    """
    pid = patient["patient_id"]
    egfr = latest_egfr(labs, pid, cutoff)
    renal = bool(patient.get("renal_disease_status", 0)) or (
        egfr is not None and egfr < RENAL_IMPAIRMENT_EGFR)
    hepatic = bool(patient.get("liver_disease_status", 0))

    comorbs = str(patient.get("comorbidities", "") or "")
    comorbidity_count = len([c for c in comorbs.split(";") if c.strip()])

    pmeds = meds_for_patient(medications, pid)
    pmeds = pmeds[pmeds["start_date"] <= cutoff]
    polypharmacy_count = int(pmeds["normalized_drug_name"].nunique())
    high_risk_med_count = int(pmeds[pmeds["drug_class"].isin(HIGH_RISK_DRUG_CLASSES)]
                              ["normalized_drug_name"].nunique())

    age = int(patient.get("age", 0))
    return {
        "age": age,
        "age_risk": int(age >= AGE_RISK_THRESHOLD),
        "egfr": egfr if egfr is not None else np.nan,
        "renal_impairment": int(renal),
        "hepatic_impairment": int(hepatic),
        "comorbidity_count": comorbidity_count,
        "polypharmacy_count": polypharmacy_count,
        "polypharmacy_flag": int(polypharmacy_count >= POLYPHARMACY_THRESHOLD),
        "high_risk_medication_count": high_risk_med_count,
    }
