"""Medication cleaning: name normalisation + drug_class backfill."""
from __future__ import annotations

import pandas as pd

# Minimal drug -> class map; extend as needed (MIMIC will need a larger dictionary).
DRUG_CLASS_MAP = {
    "lisinopril": "acei_arb", "enalapril": "acei_arb", "ramipril": "acei_arb",
    "losartan": "acei_arb", "valsartan": "acei_arb",
    "warfarin": "anticoagulant",
    "metformin": "biguanide",
    "ibuprofen": "nsaid", "naproxen": "nsaid", "diclofenac": "nsaid",
    "furosemide": "diuretic", "hydrochlorothiazide": "diuretic",
    "spironolactone": "potassium_sparing_diuretic",
    "atorvastatin": "statin", "simvastatin": "statin",
    "sertraline": "ssri", "fluoxetine": "ssri",
}


def normalize_drug_name(name: str) -> str:
    return str(name).strip().lower()


def clean_medications(medications: pd.DataFrame) -> pd.DataFrame:
    df = medications.copy()
    df["normalized_drug_name"] = df["drug_name"].map(normalize_drug_name)

    # backfill drug_class from the map where missing/unknown
    needs_class = df["drug_class"].isna() | (df["drug_class"].astype(str).str.strip() == "")
    df.loc[needs_class, "drug_class"] = df.loc[needs_class, "normalized_drug_name"].map(DRUG_CLASS_MAP)
    df["drug_class"] = df["drug_class"].fillna("other")

    for col in ("start_date", "stop_date", "dose_change_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    df = df.dropna(subset=["start_date"])
    return df.sort_values(["patient_id", "start_date"]).reset_index(drop=True)
