"""Per-patient timeline construction and patient-specific baseline extraction."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

BASELINE_LOOKBACK_DAYS = 180

# Empty fallback DataFrame returned when a (patient, lab) pair has no rows in the
# precomputed index. Defined once so the hot path doesn't allocate per miss.
_EMPTY_LAB_DF = pd.DataFrame(
    columns=["patient_id", "normalized_lab_name", "value", "lab_date", "unit"]
)


def index_labs(labs: pd.DataFrame) -> dict[tuple[int, str], pd.DataFrame]:
    """Build an O(1) lookup index: {(patient_id, normalized_lab_name) -> sorted df}.

    Used by experiments to avoid an O(N_patients * N_lab_rows) scan when iterating
    patients. On synthetic data the cost is negligible; on MIMIC-IV this turns a
    multi-hour stall into a few seconds.
    """
    if labs is None or labs.empty:
        return {}
    sorted_labs = labs.sort_values("lab_date")
    return {
        key: g for key, g in sorted_labs.groupby(
            ["patient_id", "normalized_lab_name"], sort=False
        )
    }


def index_medications_by_pid(medications: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Build {patient_id -> medications_df} index for the same reason as labs."""
    if medications is None or medications.empty:
        return {}
    return {pid: g for pid, g in medications.groupby("patient_id", sort=False)}


def labs_for(labs, patient_id: int, lab_name: str) -> pd.DataFrame:
    """Return labs for one (patient, lab) sorted by date.

    `labs` may be either:
      - a canonical labs DataFrame (slow path: linear filter per call), or
      - a precomputed dict produced by `index_labs(labs)` (fast O(1) path).

    Callers that iterate over many patients should build the index once and pass
    it in — this is what keeps the experiments tractable on MIMIC-IV.
    """
    if isinstance(labs, dict):
        return labs.get((patient_id, lab_name), _EMPTY_LAB_DF)
    df = labs[(labs["patient_id"] == patient_id)
              & (labs["normalized_lab_name"] == lab_name)]
    return df.sort_values("lab_date")


_EMPTY_MED_DF = pd.DataFrame(columns=[
    "medication_id", "patient_id", "drug_name", "normalized_drug_name",
    "drug_class", "start_date", "stop_date", "dose", "route",
    "frequency", "dose_change_date",
])


def meds_for_patient(medications, patient_id: int) -> pd.DataFrame:
    """Return all medication rows for one patient. Accepts DataFrame or dict index."""
    if isinstance(medications, dict):
        return medications.get(patient_id, _EMPTY_MED_DF)
    return medications[medications["patient_id"] == patient_id]


def compute_baseline(series: pd.DataFrame, drug_start: date,
                     lookback_days: int = BASELINE_LOOKBACK_DAYS) -> float | None:
    """Patient-specific baseline = median of pre-drug readings within the lookback.

    Falls back to the earliest post-start reading if no pre-drug data exists.
    """
    if series.empty:
        return None
    window_start = drug_start - timedelta(days=lookback_days)
    pre = series[(series["lab_date"] < drug_start) & (series["lab_date"] >= window_start)]
    if not pre.empty:
        return float(np.median(pre["value"]))
    post = series[series["lab_date"] >= drug_start]
    return float(post["value"].iloc[0]) if not post.empty else None


def build_patient_timeline(patient_id: int, patients: pd.DataFrame,
                           medications: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """Tidy event table (medication starts + lab readings) for the dashboard."""
    events = []
    meds = medications[medications["patient_id"] == patient_id]
    for _, m in meds.iterrows():
        events.append({
            "date": m["start_date"], "kind": "medication",
            "name": m["drug_name"], "detail": f"{m['drug_class']} start (dose {m['dose']})",
            "value": np.nan,
        })
        if pd.notna(m.get("dose_change_date")):
            events.append({
                "date": m["dose_change_date"], "kind": "dose_change",
                "name": m["drug_name"], "detail": "dose change", "value": np.nan,
            })
    plabs = labs[labs["patient_id"] == patient_id]
    for _, l in plabs.iterrows():
        events.append({
            "date": l["lab_date"], "kind": "lab",
            "name": l["normalized_lab_name"], "detail": l["unit"],
            "value": l["value"],
        })
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)
