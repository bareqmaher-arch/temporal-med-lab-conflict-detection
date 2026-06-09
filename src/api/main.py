"""FastAPI service exposing patients, timelines, labs, medications, alerts, and the
temporal risk model.

Run: uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.app_data import get_data, get_model
from src.config import OUTPUTS_DIR, SCENARIOS
from src.preprocessing.build_timeline import build_patient_timeline
from src.risk.assess import assess_patient

app = FastAPI(title="Temporal Medication–Lab Conflict Detection",
              description="Explainable early-warning CDS research prototype.",
              version="0.1.0")


class PredictRequest(BaseModel):
    patient_id: int
    scenario: str


def _patient_row(pid: int):
    data = get_data()
    rows = data["patients"][data["patients"]["patient_id"] == pid]
    if rows.empty:
        raise HTTPException(404, f"patient {pid} not found")
    return rows.iloc[0]


@app.get("/")
def root():
    return {"service": "temporal-med-lab-conflict-detector",
            "scenarios": {k: s.name for k, s in SCENARIOS.items()}}


@app.get("/patients")
def list_patients(limit: int = 100):
    df = get_data()["patients"].head(limit)
    return json.loads(df.to_json(orient="records"))


@app.get("/patients/{pid}/timeline")
def patient_timeline(pid: int):
    d = get_data()
    _patient_row(pid)
    tl = build_patient_timeline(pid, d["patients"], d["medications"], d["labs"])
    return json.loads(tl.to_json(orient="records"))


@app.get("/patients/{pid}/labs")
def patient_labs(pid: int):
    d = get_data()
    df = d["labs"][d["labs"]["patient_id"] == pid]
    return json.loads(df.to_json(orient="records"))


@app.get("/patients/{pid}/medications")
def patient_medications(pid: int):
    d = get_data()
    df = d["medications"][d["medications"]["patient_id"] == pid]
    return json.loads(df.to_json(orient="records"))


@app.get("/patients/{pid}/alerts")
def patient_alerts(pid: int):
    d = get_data()
    patient = _patient_row(pid)
    out = []
    for key in SCENARIOS:
        a = assess_patient(key, patient, d["medications"], d["labs"], d["knowledge"],
                           get_model(key), explain_shap=False)
        if a is None:
            continue
        if a["temporal_alert"] or a["static_alert"] or (a["ml_probability"] or 0) >= 0.5:
            out.append(a)
    return out


@app.post("/predict-risk")
def predict_risk(req: PredictRequest):
    if req.scenario not in SCENARIOS:
        raise HTTPException(400, f"unknown scenario '{req.scenario}'")
    d = get_data()
    patient = _patient_row(req.patient_id)
    a = assess_patient(req.scenario, patient, d["medications"], d["labs"],
                       d["knowledge"], get_model(req.scenario), explain_shap=False)
    if a is None:
        raise HTTPException(404, "patient not exposed to this scenario's drug class")
    return {"patient_id": req.patient_id, "scenario": req.scenario,
            "risk_score": a["risk_score"], "risk_level": a["risk_level"],
            "ml_probability": a["ml_probability"],
            "static_alert": a["static_alert"], "temporal_alert": a["temporal_alert"]}


@app.post("/explain-alert")
def explain_alert(req: PredictRequest):
    d = get_data()
    patient = _patient_row(req.patient_id)
    a = assess_patient(req.scenario, patient, d["medications"], d["labs"],
                       d["knowledge"], get_model(req.scenario), explain_shap=True)
    if a is None:
        raise HTTPException(404, "patient not exposed to this scenario's drug class")
    return a


@app.get("/model-performance")
def model_performance():
    path = OUTPUTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "run run_pipeline.py first to produce metrics.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
