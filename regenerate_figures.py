"""Regenerate paper figures (Fig 1-5) from the cached cohort + models.

Why this exists
---------------
The full `run_pipeline.py` takes ~3 hours on the 122k-patient MIMIC-IV cohort
because it re-loads raw CSVs, rebuilds indices, runs all four experiments,
trains every model, and re-generates alerts. After a code change to the
figure-generation routines (e.g. fixing a truncated legend or a bad x-axis
window), we only need to re-render the PNGs — none of the heavy work needs
to repeat.

What this script does
---------------------
1.  Load canonical tables (patients / medications / labs / knowledge) straight
    from `cohort.db` — instant, no MIMIC raw-CSV parsing.
2.  Rebuild the `trained` dict by re-running Experiment 2 only (ML models).
    This is the only thing the figures consume that isn't on disk in a
    figure-ready form. On 122k patients the ML step is the fast part of the
    pipeline (~5-15 min depending on cores) because the per-patient feature
    extraction is already O(N) thanks to the index-based optimisations.
3.  Call `figures.generate_all(...)` to write Fig 1-5 to `paper/figures/`.

Usage:
    # Same env as run_pipeline.py — DATA_SOURCE controls which DB is read.
    $env:DATA_SOURCE = "mimic"   # (PowerShell)
    python regenerate_figures.py

Skip-models flag (Fig 1 + Fig 2 only — no ML retraining):
    python regenerate_figures.py --skip-models
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd
from sqlalchemy import create_engine

# Windows consoles default to cp1252; scenario names contain non-ASCII (e.g. "→").
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import DB_URL, FIGURES_DIR, print_perf_summary
from src.experiments import exp2_ml_models, figures
from src.preprocessing.build_timeline import (
    index_labs, index_medications_by_pid,
)


def load_canonical_from_db():
    """Read the canonical tables back out of cohort.db as pandas DataFrames.

    These are the same tables `store_canonical()` wrote at the end of stage [3/8]
    of the main pipeline, so no information is lost vs. re-running the loader.
    """
    print(f"  reading from {DB_URL}")
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        patients = pd.read_sql("SELECT * FROM patients", conn)
        medications = pd.read_sql("SELECT * FROM medications", conn)
        labs = pd.read_sql("SELECT * FROM labs", conn)
        knowledge = pd.read_sql("SELECT * FROM medication_lab_risk_knowledge", conn)

    # Date columns come back as strings from SQLite — restore datetime dtype so
    # downstream code (timeline indexing, slope calc) works without changes.
    for df, cols in [
        (patients, ["admission_date", "discharge_date"]),
        (medications, ["start_date", "stop_date", "dose_change_date"]),
        (labs, ["lab_date"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    print(f"  loaded: patients={len(patients):,}  meds={len(medications):,}  "
          f"labs={len(labs):,}  rules={len(knowledge)}")
    return patients, medications, labs, knowledge


def main(skip_models: bool = False):
    print_perf_summary()
    t0 = time.time()

    print("[1/3] Loading canonical tables from cohort.db ...")
    patients, medications, labs, knowledge = load_canonical_from_db()

    # Build the same shared indices the main pipeline uses — saves ~30s on full
    # MIMIC by not re-scanning the labs frame per scenario.
    print("[2/3] Building shared indices ...")
    labs_idx = index_labs(labs)
    meds_idx = index_medications_by_pid(medications)

    trained = None
    if not skip_models:
        print("[3/3a] Re-running Experiment 2 (ML training) to rebuild "
              "the `trained` dict needed by Fig 3-5 ...")
        _, trained = exp2_ml_models.run(patients, meds_idx, labs_idx, knowledge)
    else:
        print("[3/3a] --skip-models set: Fig 3-5 will be skipped (no ML retraining).")

    print("[3/3b] Generating figures ...")
    if skip_models:
        # Fig 1 + Fig 2 only — Fig 1 is data-free, Fig 2 just needs the cohort.
        out = {
            "fig1": figures.fig1_architecture(),
            "fig2": figures.fig2_patient_timeline(patients, medications, labs),
        }
    else:
        out = figures.generate_all(patients, medications, labs, trained)

    print(f"\nDone in {time.time() - t0:.1f}s. Figures saved to {FIGURES_DIR}:")
    for k, v in out.items():
        print(f"  {k} -> {v.name if v else 'skipped'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-models", action="store_true",
                    help="Skip ML retraining; only regenerate Fig 1 + Fig 2.")
    main(**vars(ap.parse_args()))
