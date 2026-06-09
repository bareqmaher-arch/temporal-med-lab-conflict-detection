"""Risk scoring: 0-100 weighted composite + Low/Moderate/High/Critical bands."""
from __future__ import annotations

import numpy as np

from src.config import REFERENCE_RANGES, RISK_BANDS, RISK_WEIGHTS

# per-lab normalisation scales for slope (units/day) and delta-from-baseline
_SLOPE_SCALE = {"potassium": 0.04, "creatinine": 0.03, "egfr": 1.5,
                "inr": 0.10, "sodium": 0.5, "alt": 3.0}
_DELTA_SCALE = {"potassium": 1.0, "creatinine": 1.0, "egfr": 30.0,
                "inr": 2.0, "sodium": 8.0, "alt": 100.0}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _abnormality(lab_name: str, current: float | None, direction: int) -> float:
    if current is None:
        return 0.0
    lo, hi = REFERENCE_RANGES.get(lab_name, (None, None))
    if direction > 0 and hi:                 # rising is dangerous
        return _clip01((current - hi) / (0.5 * hi))
    if direction < 0 and lo:                 # falling is dangerous
        return _clip01((lo - current) / (0.5 * lo))
    return 0.0


def compute_risk_score(feats: dict, knowledge_severity: float, direction: int,
                       lab_name: str) -> dict:
    current = feats.get("current_value")
    delta = feats.get("delta_value") or 0.0
    pct = feats.get("percent_change") or 0.0
    slope = feats.get("slope_7d") or feats.get("slope_14d") or 0.0

    comp = {
        "lab_abnormality_severity": _abnormality(lab_name, current, direction),
        "temporal_slope": _clip01(abs(slope) / _SLOPE_SCALE.get(lab_name, 1.0)),
        "delta_from_baseline": (
            _clip01(abs(pct) / 0.4) if lab_name == "egfr"
            else _clip01(abs(delta) / _DELTA_SCALE.get(lab_name, 1.0))
        ),
        "drug_risk_strength": _clip01(knowledge_severity),
        "renal_hepatic_vulnerability": _clip01(
            0.6 * feats.get("renal_impairment", 0) + 0.4 * feats.get("hepatic_impairment", 0)
        ),
        "age_factor": _clip01((feats.get("age", 0) - 50) / 40),
        "polypharmacy_factor": _clip01(feats.get("polypharmacy_count", 0) / 10),
        "consecutive_trend_factor": _clip01(feats.get("consecutive_abnormal_trend", 0) / 4),
    }

    # only count danger-direction movement; resolving trends should not score high
    if direction * delta < 0 and lab_name != "egfr":
        comp["delta_from_baseline"] = 0.0

    raw = sum(RISK_WEIGHTS[k] * comp[k] for k in RISK_WEIGHTS)
    score = float(np.clip(100.0 * raw, 0, 100))
    return {"risk_score": round(score, 1), "risk_level": band_for(score),
            "components": {k: round(v, 3) for k, v in comp.items()}}


def band_for(score: float) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= score <= hi:
            return label
    return "Critical" if score > 100 else "Low"
