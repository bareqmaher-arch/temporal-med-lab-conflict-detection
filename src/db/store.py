"""Persist canonical tables and generated alerts to the SQLite database."""
from __future__ import annotations

import pandas as pd

from src.db.schema import get_engine, init_db


def store_canonical(tables, db_url: str | None = None) -> None:
    engine = init_db(db_url) if db_url else init_db()
    with engine.begin() as conn:
        tables.patients.to_sql("patients", conn, if_exists="replace", index=False)
        tables.medications.to_sql("medications", conn, if_exists="replace", index=False)
        tables.labs.to_sql("labs", conn, if_exists="replace", index=False)
        tables.knowledge.to_sql("medication_lab_risk_knowledge", conn,
                                if_exists="replace", index=False)


def store_alerts(alerts: pd.DataFrame, db_url: str | None = None) -> None:
    engine = get_engine(db_url) if db_url else get_engine()
    with engine.begin() as conn:
        alerts.to_sql("alerts", conn, if_exists="replace", index=False)
