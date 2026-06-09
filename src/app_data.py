"""Shared, cached data access for the API and dashboard."""
from __future__ import annotations

from functools import lru_cache

from src.ingestion.csv_loader import CSVLoader
from src.preprocessing.clean_labs import clean_labs
from src.preprocessing.clean_medications import clean_medications


@lru_cache(maxsize=1)
def get_data():
    t = CSVLoader().load()
    return {
        "patients": t.patients,
        "medications": clean_medications(t.medications),
        "labs": clean_labs(t.labs),
        "knowledge": t.knowledge,
    }


@lru_cache(maxsize=8)
def get_model(scenario_key: str):
    from src.models.predict import load_model
    return load_model(scenario_key, "primary")
