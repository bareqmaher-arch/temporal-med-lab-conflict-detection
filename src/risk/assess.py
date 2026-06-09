"""Single-patient assessment used by experiments, API, and dashboard.

Combines temporal + risk features, the risk score, both rule verdicts, the optional ML
probability, and the clinical explanation into one object.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import OBSERVATION_WINDOW_DAYS, SCENARIOS, Scenario
from src.explainability.clinical_explanation_generator import generate_explanation
from src.features.risk_features import compute_risk_features
from src.features.temporal_features import compute_temporal_features
from src.models.dataset import _index_medication
from src.preprocessing.build_timeline import labs_for
from src.risk.risk_score import compute_risk_score
from src.rules.baseline_rules import static_fires
from src.rules.temporal_rules import temporal_fires

_DANGER_DIRECTION = {"potassium": +1, "creatinine": +1, "inr": +1,
                     "egfr": -1, "sodium": -1, "alt": +1}


def _knowledge_lookup(knowledge: pd.DataFrame, scenario: Scenario):
    rows = knowledge[(knowledge["drug_class"] == scenario.drug_class)
                     & (knowledge["lab_name"] == scenario.primary_lab)]
    if rows.empty:
        return 0.5, scenario.primary_lab
    r = rows.iloc[0]
    sev = float(r["severity_weight"]) if not pd.isna(r["severity_weight"]) else 0.5
    return sev, str(r["risk_type"])


def compute_patient_features(scenario: Scenario, patient: pd.Series,
                             medications: pd.DataFrame, labs: pd.DataFrame,
                             severity: float, cutoff=None) -> dict | None:
    med = _index_medication(medications, patient["patient_id"], scenario)
    if med is None:
        return None
    drug_start = med["start_date"]
    dose_change = med.get("dose_change_date")
    dose_change = dose_change if pd.notna(dose_change) else None
    if cutoff is None:
        cutoff = drug_start + timedelta(days=OBSERVATION_WINDOW_DAYS[scenario.key])

    series = labs_for(labs, patient["patient_id"], scenario.primary_lab)
    direction = _DANGER_DIRECTION[scenario.primary_lab]
    feats = compute_temporal_features(series, drug_start, cutoff, direction, dose_change)
    feats.update(compute_risk_features(patient, medications, labs, cutoff))
    feats["drug_risk_strength"] = severity
    feats["drug_name"] = med["drug_name"]
    feats["medication_id"] = int(med["medication_id"])
    feats["drug_start"] = drug_start
    feats["cutoff"] = cutoff
    return feats


def assess_patient(scenario_key: str, patient: pd.Series, medications: pd.DataFrame,
                   labs: pd.DataFrame, knowledge: pd.DataFrame,
                   model=None, cutoff=None, explain_shap: bool = True) -> dict | None:
    scenario = SCENARIOS[scenario_key]
    severity, risk_type = _knowledge_lookup(knowledge, scenario)
    feats = compute_patient_features(scenario, patient, medications, labs, severity, cutoff)
    if feats is None:
        return None

    direction = _DANGER_DIRECTION[scenario.primary_lab]
    rs = compute_risk_score(feats, severity, direction, scenario.primary_lab)
    static = static_fires(scenario, feats.get("current_value"))
    temporal = temporal_fires(scenario, feats)

    ml_proba, shap_top = None, []
    if model is not None:
        from src.models.dataset import ML_FEATURE_COLUMNS
        from src.models.predict import predict_proba_row
        ml_proba = predict_proba_row(model, feats)
        if explain_shap:
            from src.explainability.shap_explainer import top_features_for_instance
            X_row = pd.DataFrame([{c: feats.get(c, np.nan) for c in ML_FEATURE_COLUMNS}])
            shap_top = top_features_for_instance(model, X_row, ML_FEATURE_COLUMNS)

    expl = generate_explanation(scenario, feats["drug_name"], feats,
                                rs["risk_level"], rs["risk_score"], risk_type, shap_top)
    return {
        "scenario": scenario_key, "scenario_name": scenario.name,
        "patient_id": int(patient["patient_id"]), "risk_type": risk_type,
        "risk_score": rs["risk_score"], "risk_level": rs["risk_level"],
        "risk_components": rs["components"],
        "static_alert": bool(static), "temporal_alert": bool(temporal),
        "ml_probability": ml_proba, "shap_top": shap_top,
        "features": {k: v for k, v in feats.items() if k not in ("drug_start", "cutoff")},
        **expl,
    }
