"""Build the labeled feature matrix per scenario.

One row per exposed patient: temporal features measured at the *early* observation
cutoff, joined to patient-level risk features, with a **sustained-outcome** label
evaluated over the full label window. Sustained labels (>=2 qualifying readings) keep
transient single-reading spikes as negatives, which is what makes the false-alert
comparison fair.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import OBSERVATION_WINDOW_DAYS, SCENARIOS, Scenario
from src.features.risk_features import compute_risk_features
from src.features.temporal_features import compute_temporal_features
from src.preprocessing.build_timeline import (
    compute_baseline, index_labs, index_medications_by_pid, labs_for,
    meds_for_patient,
)

_DANGER_DIRECTION = {"potassium": +1, "creatinine": +1, "inr": +1,
                     "egfr": -1, "sodium": -1, "alt": +1}

ML_FEATURE_COLUMNS = [
    "baseline_value", "current_value", "delta_value", "percent_change",
    "slope_7d", "slope_14d", "slope_30d", "acceleration", "variability",
    "min_in_window", "max_in_window", "consecutive_abnormal_trend",
    "days_since_drug_start", "days_since_dose_change", "trend_started_after_drug",
    "age", "age_risk", "egfr", "renal_impairment", "hepatic_impairment",
    "comorbidity_count", "polypharmacy_count", "high_risk_medication_count",
    "drug_risk_strength",
]


def _index_medication(medications, patient_id: int, scenario: Scenario):
    """Earliest exposure to the scenario's drug class for one patient.

    Accepts either the raw medications DataFrame or a `{pid -> df}` dict index;
    the dict path makes iterating MIMIC-scale cohorts O(N) instead of O(N*M).
    """
    meds = meds_for_patient(medications, patient_id)
    meds = meds[meds["drug_class"] == scenario.drug_class]
    if meds.empty:
        return None
    return meds.sort_values("start_date").iloc[0]


def _sustained_label(scenario: Scenario, series: pd.DataFrame, drug_start: date) -> int:
    """Sustained adverse outcome within the label window (transient spikes excluded)."""
    window_end = drug_start + timedelta(days=scenario.label_window_days)
    win = series[(series["lab_date"] >= drug_start) & (series["lab_date"] <= window_end)]
    if win.empty:
        return 0
    baseline = compute_baseline(series, drug_start)
    vals = win["value"].to_numpy(dtype=float)

    if scenario.primary_lab == "potassium":
        qualifying = np.sum((vals >= 5.0) & ((vals - (baseline or vals.min())) >= 0.4))
        severe = np.sum(vals >= 5.5)
        return int(qualifying >= 2 or severe >= 2)
    if scenario.primary_lab == "inr":
        qualifying = np.sum(vals >= 3.5)
        severe = np.sum(vals >= 4.0)
        return int(qualifying >= 2 or severe >= 2)
    if scenario.primary_lab == "egfr":
        if baseline is None or baseline == 0:
            return int(np.sum(vals < 30) >= 2)
        rel = (baseline - vals) / baseline
        qualifying = np.sum((rel >= 0.20) & (vals < 45))
        severe = np.sum(vals < 30)
        return int(qualifying >= 2 or severe >= 2)
    return 0


def build_scenario_dataset(scenario_key: str, patients: pd.DataFrame,
                           medications, labs,
                           knowledge: pd.DataFrame) -> pd.DataFrame:
    """Build the labeled feature matrix for one scenario.

    `medications` and `labs` accept either raw DataFrames or precomputed indices
    (see `index_labs` / `index_medications_by_pid`). The indexed form is required
    for MIMIC-scale cohorts — the function builds indices itself if given the
    raw frames, so callers don't have to.
    """
    scenario = SCENARIOS[scenario_key]
    direction = _DANGER_DIRECTION[scenario.primary_lab]
    obs_days = OBSERVATION_WINDOW_DAYS[scenario_key]
    severity = float(knowledge[(knowledge["drug_class"] == scenario.drug_class)
                               & (knowledge["lab_name"] == scenario.primary_lab)]
                     ["severity_weight"].max())
    severity = severity if not np.isnan(severity) else 0.5

    # Build indices once if the caller passed raw DataFrames. The O(N*L) inner
    # filter was the MIMIC-IV bottleneck (hours -> seconds after this change).
    labs_idx = labs if isinstance(labs, dict) else index_labs(labs)
    meds_idx = (medications if isinstance(medications, dict)
                else index_medications_by_pid(medications))

    rows = []
    n_patients = len(patients)
    progress_every = max(1, n_patients // 20)  # ~5% step
    for i, (_, patient) in enumerate(patients.iterrows()):
        if i and i % progress_every == 0:
            print(f"    [{scenario_key}] {i}/{n_patients} patients processed", flush=True)
        med = _index_medication(meds_idx, patient["patient_id"], scenario)
        if med is None:
            continue
        drug_start = med["start_date"]
        dose_change = med.get("dose_change_date")
        dose_change = dose_change if pd.notna(dose_change) else None
        cutoff = drug_start + timedelta(days=obs_days)

        series = labs_for(labs_idx, patient["patient_id"], scenario.primary_lab)
        tf = compute_temporal_features(series, drug_start, cutoff, direction, dose_change)
        rf = compute_risk_features(patient, meds_idx, labs_idx, cutoff)

        row = {"patient_id": patient["patient_id"], "medication_id": med["medication_id"],
               "scenario": scenario_key, "lab_name": scenario.primary_lab,
               "drug_name": med["drug_name"], "drug_risk_strength": severity,
               "drug_start": drug_start}
        row.update(tf)
        row.update(rf)
        row["label"] = _sustained_label(scenario, series, drug_start)
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in ML_FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_all_datasets(patients, medications, labs, knowledge) -> dict[str, pd.DataFrame]:
    return {k: build_scenario_dataset(k, patients, medications, labs, knowledge)
            for k in SCENARIOS}
