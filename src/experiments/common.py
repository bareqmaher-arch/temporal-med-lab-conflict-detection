"""Shared experiment utilities: per-patient alert timing for rule comparison."""
from __future__ import annotations

import multiprocessing as _mp
from datetime import timedelta

import pandas as pd

from src.config import N_JOBS, PARALLEL_PATIENTS, SCENARIOS
from src.models.dataset import _index_medication, _sustained_label
from src.preprocessing.build_timeline import (
    index_labs, index_medications_by_pid, labs_for,
)
from src.rules.baseline_rules import first_static_alert_date
from src.rules.temporal_rules import first_temporal_alert_date


def _alert_row_for_patient(patient, scenario, labs_idx, meds_idx):
    """Compute one alert-table row for one patient. Returns None if not exposed."""
    med = _index_medication(meds_idx, patient["patient_id"], scenario)
    if med is None:
        return None
    drug_start = med["start_date"]
    dose_change = med.get("dose_change_date")
    dose_change = dose_change if pd.notna(dose_change) else None
    horizon_end = drug_start + timedelta(days=scenario.label_window_days)

    series = labs_for(labs_idx, patient["patient_id"], scenario.primary_lab)
    label = _sustained_label(scenario, series, drug_start)

    s_date = first_static_alert_date(scenario, labs_idx, patient["patient_id"], drug_start)
    t_date = first_temporal_alert_date(scenario, labs_idx, patient, drug_start, dose_change)

    s_in = s_date is not None and s_date <= horizon_end
    t_in = t_date is not None and t_date <= horizon_end
    return {
        "patient_id": patient["patient_id"], "label": label,
        "static_alert": int(s_in), "temporal_alert": int(t_in),
        "static_date": s_date if s_in else None,
        "temporal_date": t_date if t_in else None,
        "days_to_static": (s_date - drug_start).days if s_in else None,
        "days_to_temporal": (t_date - drug_start).days if t_in else None,
    }


def _alert_rows_for_chunk(patient_chunk, scenario, labs_idx, meds_idx):
    """Process a contiguous slice of patients on one worker (one pickling round-trip)."""
    out = []
    for _, patient in patient_chunk.iterrows():
        row = _alert_row_for_patient(patient, scenario, labs_idx, meds_idx)
        if row is not None:
            out.append(row)
    return out


def build_alert_table(scenario_key: str, patients: pd.DataFrame,
                      medications, labs) -> pd.DataFrame:
    """One row per exposed patient with static/temporal alert dates + label.

    Alerts are counted within the scenario's label window so the rule comparison and
    the time-to-alert analysis share the same horizon.

    `medications` and `labs` may be raw DataFrames or precomputed indices. The
    function builds indices itself if given raw frames — required for MIMIC-IV
    where the naive per-patient filter is O(N_patients * N_lab_rows).

    On a multi-core machine the loop is parallelised across cores via joblib
    (controlled by `PARALLEL_PATIENTS` / `N_JOBS` in config). Patients are
    chunked so each worker pickles the labs/meds indices at most once.
    """
    scenario = SCENARIOS[scenario_key]
    labs_idx = labs if isinstance(labs, dict) else index_labs(labs)
    meds_idx = (medications if isinstance(medications, dict)
                else index_medications_by_pid(medications))

    n_patients = len(patients)
    cores = _mp.cpu_count() if N_JOBS in (-1, 0) else min(N_JOBS, _mp.cpu_count())
    use_parallel = PARALLEL_PATIENTS and cores > 1 and n_patients >= 200

    if not use_parallel:
        rows = []
        progress_every = max(1, n_patients // 20)
        for i, (_, patient) in enumerate(patients.iterrows()):
            if i and i % progress_every == 0:
                print(f"    [{scenario_key}] {i}/{n_patients} patients processed",
                      flush=True)
            row = _alert_row_for_patient(patient, scenario, labs_idx, meds_idx)
            if row is not None:
                rows.append(row)
        return pd.DataFrame(rows)

    # Parallel path: threading backend, NOT processes. On Windows, joblib's
    # loky/process backend re-pickles the labs_idx dict (~100s of MB) for every
    # chunk, which dominates runtime and pins CPU at ~10% utilisation.
    # pandas/numpy release the GIL during the hot inner work (DataFrame slicing,
    # numeric ops), so threads scale almost linearly here without any pickle
    # overhead — workers share the labs_idx via shared memory.
    # Override via JOBLIB_BACKEND env var if a future scenario benefits from
    # processes (e.g. a CPython-only inner loop).
    import os as _os
    backend = _os.environ.get("JOBLIB_BACKEND", "threading")

    # Larger chunks with fewer total chunks = less scheduling overhead.
    # Aim for ~2 chunks per worker so straggler tasks don't dominate.
    n_chunks = max(cores * 2, cores)
    chunk_size = max(1, (n_patients + n_chunks - 1) // n_chunks)
    chunks = [patients.iloc[i:i + chunk_size]
              for i in range(0, n_patients, chunk_size)]
    print(f"    [{scenario_key}] parallel: {len(chunks)} chunks across "
          f"{cores} workers (backend={backend}), ~{chunk_size} patients each",
          flush=True)

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=cores, backend=backend, verbose=0)(
        delayed(_alert_rows_for_chunk)(chunk, scenario, labs_idx, meds_idx)
        for chunk in chunks
    )
    rows = [r for chunk_out in results for r in chunk_out]
    return pd.DataFrame(rows)
