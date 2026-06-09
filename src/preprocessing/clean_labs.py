"""Lab cleaning: name + unit harmonisation, outlier removal, reference ranges."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import REFERENCE_RANGES

# Map common name variants to the canonical lab name.
LAB_NAME_MAP = {
    "k": "potassium", "k+": "potassium", "potassium": "potassium",
    "serum potassium": "potassium", "potassium, serum": "potassium",
    "cr": "creatinine", "creat": "creatinine", "creatinine": "creatinine",
    "egfr": "egfr", "gfr": "egfr", "estimated gfr": "egfr",
    "inr": "inr", "pt-inr": "inr",
    "na": "sodium", "na+": "sodium", "sodium": "sodium",
    "alt": "alt", "sgpt": "alt", "alanine aminotransferase": "alt",
}

# Physiologically plausible bounds; values outside are dropped as errors.
PLAUSIBLE_BOUNDS = {
    "potassium": (1.5, 9.5), "creatinine": (0.1, 20.0), "egfr": (1, 200),
    "inr": (0.5, 15.0), "sodium": (100, 180), "alt": (1, 5000),
}


def normalize_lab_name(name: str) -> str:
    return LAB_NAME_MAP.get(str(name).strip().lower(), str(name).strip().lower())


def clean_labs(labs: pd.DataFrame) -> pd.DataFrame:
    df = labs.copy()
    df["normalized_lab_name"] = df["lab_name"].map(normalize_lab_name)

    # fill reference ranges from config where missing
    for lab, (lo, hi) in REFERENCE_RANGES.items():
        mask = df["normalized_lab_name"] == lab
        df.loc[mask & df["reference_range_low"].isna(), "reference_range_low"] = lo
        df.loc[mask & df["reference_range_high"].isna(), "reference_range_high"] = hi

    # drop implausible values (data-entry errors)
    keep = pd.Series(True, index=df.index)
    for lab, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        mask = df["normalized_lab_name"] == lab
        keep &= ~(mask & ((df["value"] < lo) | (df["value"] > hi)))
    df = df[keep].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value", "lab_date"])
    df["lab_date"] = pd.to_datetime(df["lab_date"]).dt.date
    return df.sort_values(["patient_id", "normalized_lab_name", "lab_date"]).reset_index(drop=True)
