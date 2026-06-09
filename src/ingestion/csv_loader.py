"""Load canonical tables from CSV files on disk."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import SYNTHETIC_DIR
from src.db.canonical import DATE_COLUMNS, CanonicalTables
from src.ingestion.base import AbstractLoader


class CSVLoader(AbstractLoader):
    def __init__(self, directory: Path | str = SYNTHETIC_DIR):
        self.directory = Path(directory)

    def load(self) -> CanonicalTables:
        patients = pd.read_csv(self.directory / "patients.csv")
        medications = pd.read_csv(self.directory / "medications.csv")
        labs = pd.read_csv(self.directory / "labs.csv")

        for col in DATE_COLUMNS["patients"]:
            patients[col] = pd.to_datetime(patients[col], errors="coerce").dt.date
        for col in DATE_COLUMNS["medications"]:
            medications[col] = pd.to_datetime(medications[col], errors="coerce").dt.date
        for col in DATE_COLUMNS["labs"]:
            labs[col] = pd.to_datetime(labs[col], errors="coerce").dt.date

        tables = CanonicalTables(
            patients=patients, medications=medications,
            labs=labs, knowledge=self.load_knowledge(),
        )
        return tables.validate()
