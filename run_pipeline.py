"""End-to-end pipeline: ingest -> preprocess -> features -> rules + ML -> experiments
-> figures/tables -> alerts. One command reproduces every manuscript artifact.

Usage:
    python run_pipeline.py            # use existing synthetic data (generate if missing)
    python run_pipeline.py --regen    # regenerate the synthetic cohort first
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

# Windows consoles default to cp1252; scenario names contain non-ASCII (e.g. "->").
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import (
    DATA_SOURCE, MIMIC_DIR, OUTPUTS_DIR, SCENARIOS, SYNTHETIC_DIR, TABLES_DIR,
    print_perf_summary,
)
from src.db.store import store_alerts, store_canonical
from src.experiments import (
    exp1_rules_vs_temporal, exp2_ml_models, exp3_time_to_alert,
    exp4_explainability_eval, figures, tables,
)
from src.ingestion.csv_loader import CSVLoader
from src.ingestion.mimic_loader import MIMICLoader
from src.ingestion.synthetic_generator import generate_and_save
from src.models.predict import load_model
from src.preprocessing.build_timeline import index_labs, index_medications_by_pid
from src.preprocessing.clean_labs import clean_labs
from src.preprocessing.clean_medications import clean_medications
from src.risk.assess import assess_patient


def _save_table(df: pd.DataFrame, name: str):
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  table -> {path.name}")


def build_alerts(patients, medications, labs, knowledge) -> pd.DataFrame:
    """Generate one alert record per patient/scenario that fires (rule or ML)."""
    # Build indices once for the whole pass. Without this, MIMIC-scale cohorts
    # would re-scan the full meds/labs tables for every (patient, scenario) pair.
    labs_idx = index_labs(labs)
    meds_idx = index_medications_by_pid(medications)
    rows = []
    aid = 1
    for key in SCENARIOS:
        model = load_model(key, "primary")
        n_patients = len(patients)
        progress_every = max(1, n_patients // 20)
        for i, (_, patient) in enumerate(patients.iterrows()):
            if i and i % progress_every == 0:
                print(f"    [alerts {key}] {i}/{n_patients}", flush=True)
            a = assess_patient(key, patient, meds_idx, labs_idx, knowledge, model,
                               explain_shap=False)
            if a is None:
                continue
            fired = a["temporal_alert"] or a["static_alert"] or (
                a["ml_probability"] is not None and a["ml_probability"] >= 0.5)
            if not fired:
                continue
            model_type = "temporal+ml" if a["temporal_alert"] else (
                "ml" if (a["ml_probability"] or 0) >= 0.5 else "static")
            rows.append({
                "alert_id": aid, "patient_id": a["patient_id"],
                "medication_id": a["features"].get("medication_id"),
                "lab_id": None, "alert_type": a["risk_type"],
                "risk_score": a["risk_score"], "risk_level": a["risk_level"],
                "explanation": a["explanation"], "suggested_action": a["suggested_action"],
                "alert_date": None, "model_type": model_type, "clinician_feedback": None,
            })
            aid += 1
    return pd.DataFrame(rows)


def main(regen: bool = False):
    print_perf_summary()
    if DATA_SOURCE == "mimic":
        print(f"[1/8] Loading MIMIC-IV from {MIMIC_DIR} ...")
        t = MIMICLoader(MIMIC_DIR).load()
    else:
        if regen or not (SYNTHETIC_DIR / "labs.csv").exists():
            print("[1/8] Generating synthetic cohort ...")
            generate_and_save()
        else:
            print("[1/8] Using existing synthetic cohort")
        t = CSVLoader().load()

    print("[2/8] Cleaning ...")
    labs = clean_labs(t.labs)
    meds = clean_medications(t.medications)
    patients = t.patients
    knowledge = t.knowledge

    print("[3/8] Persisting canonical tables to DB ...")
    t.labs, t.medications = labs, meds
    store_canonical(t)

    print("[4/8] Experiment 1: rules vs temporal ...")
    exp1 = exp1_rules_vs_temporal.run(patients, meds, labs)
    _save_table(exp1, "exp1_rules_vs_temporal")

    print("[5/8] Experiment 2: ML models (training) ...")
    exp2_df, trained = exp2_ml_models.run(patients, meds, labs, knowledge)
    _save_table(exp2_df, "table3_model_comparison")

    print("[6/8] Experiments 3 & 4 ...")
    exp3 = exp3_time_to_alert.run(patients, meds, labs)
    _save_table(exp3, "table4_early_detection")
    examples, questionnaire = exp4_explainability_eval.run(patients, meds, labs, knowledge)
    _save_table(examples, "table5_example_explanations")
    questionnaire.to_csv(OUTPUTS_DIR / "explainability_questionnaire.csv", index=False)

    print("[7/8] Descriptive tables + figures ...")
    _save_table(tables.table1_scenarios(knowledge), "table1_scenarios")
    _save_table(tables.table2_feature_categories(), "table2_feature_categories")
    figs = figures.generate_all(patients, meds, labs, trained)
    for k, v in figs.items():
        print(f"  figure {k} -> {v.name if v else 'skipped'}")

    print("[8/8] Generating + storing alerts ...")
    alerts = build_alerts(patients, meds, labs, knowledge)
    store_alerts(alerts)
    print(f"  {len(alerts)} alerts stored")

    metrics = {
        "exp1_rules_vs_temporal": exp1.to_dict(orient="records"),
        "exp2_model_comparison": exp2_df.to_dict(orient="records"),
        "exp3_early_detection": exp3.to_dict(orient="records"),
        "n_patients": int(len(patients)), "n_alerts": int(len(alerts)),
    }
    with open(OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print(f"\nDone. Metrics -> {OUTPUTS_DIR / 'metrics.json'}")
    print("\n=== Experiment 1 (rules vs temporal) ===")
    print(exp1.to_string(index=False))
    print("\n=== Experiment 3 (early detection) ===")
    print(exp3.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate synthetic data")
    main(**vars(ap.parse_args()))
