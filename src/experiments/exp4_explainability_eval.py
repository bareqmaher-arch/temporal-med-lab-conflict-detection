"""Experiment 4: explainability evaluation bundle.

Selects representative high-risk alerts, generates their clinical explanations, and
writes a clinician-rating questionnaire (clarity / usefulness / actionability / trust /
ignore-likelihood) plus an example-explanations table for the manuscript.
"""
from __future__ import annotations

import pandas as pd

from src.config import SCENARIOS
from src.models.predict import load_model
from src.preprocessing.build_timeline import index_labs, index_medications_by_pid
from src.risk.assess import assess_patient

RATING_COLUMNS = [
    "clarity_1to5", "usefulness_1to5", "actionability_1to5",
    "trust_1to5", "ignore_likelihood_1to5", "necessary_or_excessive",
]


def run(patients, medications, labs, knowledge, per_scenario: int = 4):
    # Build indices once — assess_patient runs per-patient and re-filtering the
    # full labs/meds tables each call would be O(N_patients * data_size) and
    # stall on MIMIC-IV.
    labs_idx = labs if isinstance(labs, dict) else index_labs(labs)
    meds_idx = (medications if isinstance(medications, dict)
                else index_medications_by_pid(medications))

    examples, questionnaire = [], []
    for key in SCENARIOS:
        model = load_model(key, "primary")
        # rank quickly without SHAP, then explain only the picked few with SHAP
        ranked = []
        n_patients = len(patients)
        progress_every = max(1, n_patients // 20)
        for i, (_, patient) in enumerate(patients.iterrows()):
            if i and i % progress_every == 0:
                print(f"    [exp4 {key}] {i}/{n_patients} ranked", flush=True)
            a = assess_patient(key, patient, meds_idx, labs_idx, knowledge, model,
                               explain_shap=False)
            if a is not None:
                ranked.append((a["risk_score"], a["temporal_alert"], patient))
        ranked.sort(key=lambda r: (r[1], r[0]), reverse=True)
        picked = [
            assess_patient(key, patient, meds_idx, labs_idx, knowledge, model,
                           explain_shap=True)
            for _, _, patient in ranked[:per_scenario]
        ]

        for a in picked:
            examples.append({
                "case": f"{SCENARIOS[key].name} (patient {a['patient_id']})",
                "drug": a["drug"], "lab": a["lab"],
                "baseline": a["features"].get("baseline_value"),
                "current": a["features"].get("current_value"),
                "risk_level": a["risk_level"], "risk_score": a["risk_score"],
                "explanation": a["explanation"],
                "suggested_action": a["suggested_action"],
            })
            q = {"alert_id": f"{key}_{a['patient_id']}",
                 "scenario": SCENARIOS[key].name,
                 "explanation": a["explanation"],
                 "suggested_action": a["suggested_action"]}
            for c in RATING_COLUMNS:
                q[c] = ""
            questionnaire.append(q)

    return pd.DataFrame(examples), pd.DataFrame(questionnaire)
