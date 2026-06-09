"""Turn a scored alert into a clinically readable explanation + review suggestion.

Deliberately template-based and transparent: the explanation cites the drug, lab,
baseline->current change, time window, relation to drug start, and the main risk
factor. It suggests *review*, never a treatment decision.
"""
from __future__ import annotations

from src.config import SAFETY_STATEMENT, Scenario

_LAB_LABEL = {"potassium": "Potassium", "creatinine": "Creatinine", "egfr": "eGFR",
              "inr": "INR", "sodium": "Sodium", "alt": "ALT"}
_LAB_UNIT = {"potassium": "mmol/L", "creatinine": "mg/dL", "egfr": "mL/min/1.73m2",
             "inr": "", "sodium": "mmol/L", "alt": "U/L"}

_SUGGESTED_ACTION = {
    "hyperkalemia": "Review ACE inhibitor/ARB (or potassium-sparing) dose, repeat "
                    "potassium, and assess renal function and other potassium-raising drugs.",
    "bleeding": "Review warfarin dose, recheck INR, and assess bleeding risk and "
                "interacting medications.",
    "renal_safety": "Reassess metformin dose/continuation per renal function, and "
                    "recheck creatinine and eGFR.",
    "renal_decline": "Review renal function, reconsider the implicated drug dose, and "
                     "recheck creatinine and eGFR.",
    "renal_impairment": "Review NSAID exposure and renal function, and recheck "
                        "creatinine and eGFR.",
    "hepatotoxicity": "Review hepatic function and recheck transaminases.",
    "hyponatremia": "Review sodium, assess fluid status and contributing drugs.",
    "hypokalemia": "Review diuretic dose and potassium replacement, and recheck potassium.",
}


def _fmt(value, unit):
    if value is None:
        return "n/a"
    return f"{value:.2f} {unit}".strip()


def generate_explanation(scenario: Scenario, drug_name: str, feats: dict,
                         risk_level: str, risk_score: float,
                         risk_type: str, shap_top: list | None = None) -> dict:
    lab = scenario.primary_lab
    label = _LAB_LABEL.get(lab, lab)
    unit = _LAB_UNIT.get(lab, "")
    baseline = feats.get("baseline_value")
    current = feats.get("current_value")
    delta = feats.get("delta_value")
    pct = feats.get("percent_change")
    days = feats.get("days_since_drug_start")

    direction_word = "rose" if (delta or 0) > 0 else "fell"
    if lab == "egfr" and pct is not None:
        change_phrase = (f"{label} declined by {abs(pct) * 100:.0f}% "
                         f"(from {_fmt(baseline, unit)} to {_fmt(current, unit)})")
    else:
        change_phrase = (f"{label} {direction_word} from {_fmt(baseline, unit)} "
                         f"to {_fmt(current, unit)}")

    modifiers = []
    if feats.get("renal_impairment"):
        egfr = feats.get("egfr")
        modifiers.append(f"reduced renal function (eGFR {egfr:.0f})" if egfr == egfr
                         else "reduced renal function")
    if feats.get("hepatic_impairment"):
        modifiers.append("hepatic impairment")
    if feats.get("age_risk"):
        modifiers.append(f"older age ({feats.get('age')})")
    if feats.get("polypharmacy_count", 0) >= 5:
        modifiers.append(f"polypharmacy ({feats.get('polypharmacy_count')} medications)")
    modifier_phrase = (" The patient also has " + ", ".join(modifiers) + ", which raises the risk."
                       if modifiers else "")

    explanation = (
        f"{change_phrase} within {days} days of {drug_name.title()} initiation, "
        f"consistent with possible {risk_type.replace('_', ' ')}.{modifier_phrase}"
    )
    if shap_top:
        drivers = ", ".join(f"{n} ({v:+.2f})" for n, v in shap_top[:3])
        explanation += f" Top model drivers: {drivers}."

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "drug": drug_name.title(),
        "lab": label,
        "explanation": explanation,
        "suggested_action": _SUGGESTED_ACTION.get(risk_type, "Clinical review recommended."),
        "safety_note": SAFETY_STATEMENT,
    }
