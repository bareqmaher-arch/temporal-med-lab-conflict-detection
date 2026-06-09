"""Persist / load trained models and score single feature rows (used by the API)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR
from src.models.dataset import ML_FEATURE_COLUMNS

MODELS_DIR = OUTPUTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, scenario_key: str, name: str = "primary") -> Path:
    path = MODELS_DIR / f"{scenario_key}__{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    return path


def load_model(scenario_key: str, name: str = "primary"):
    path = MODELS_DIR / f"{scenario_key}__{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def predict_proba_row(model, feats: dict) -> float:
    X = pd.DataFrame([{c: feats.get(c, np.nan) for c in ML_FEATURE_COLUMNS}])
    X = X.apply(pd.to_numeric, errors="coerce")
    return float(model.predict_proba(X)[:, 1][0])
