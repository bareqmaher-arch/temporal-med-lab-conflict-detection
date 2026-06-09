import pandas as pd
import pytest

from src.db.canonical import (
    LAB_COLUMNS, MEDICATION_COLUMNS, PATIENT_COLUMNS, CanonicalTables,
)
from src.ingestion.synthetic_generator import SyntheticLoader


def test_synthetic_loader_returns_valid_canonical_tables():
    tables = SyntheticLoader(n_patients=30, seed=1).load()
    assert set(PATIENT_COLUMNS).issubset(tables.patients.columns)
    assert set(MEDICATION_COLUMNS).issubset(tables.medications.columns)
    assert set(LAB_COLUMNS).issubset(tables.labs.columns)
    assert len(tables.patients) == 30
    assert len(tables.labs) > 0
    # validate() must not raise on well-formed tables
    tables.validate()


def test_validate_catches_missing_columns():
    bad = CanonicalTables(
        patients=pd.DataFrame({"patient_id": [1]}),
        medications=pd.DataFrame(),
        labs=pd.DataFrame(),
        knowledge=pd.DataFrame(),
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_each_patient_has_index_drug():
    tables = SyntheticLoader(n_patients=15, seed=2).load()
    pids_with_meds = set(tables.medications["patient_id"])
    assert set(tables.patients["patient_id"]).issubset(pids_with_meds)
