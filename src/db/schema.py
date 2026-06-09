"""SQLAlchemy ORM for the canonical schema.

SQLite for the proof-of-concept; switching to PostgreSQL is a DB_URL change only.
The ORM mirrors the pandas contract in ``canonical.py``.
"""
from __future__ import annotations

from sqlalchemy import (
    Date, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config import DB_URL


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    age: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(8))
    comorbidities: Mapped[str] = mapped_column(Text, default="")
    admission_date: Mapped[str] = mapped_column(Date, nullable=True)
    discharge_date: Mapped[str] = mapped_column(Date, nullable=True)
    diagnosis_codes: Mapped[str] = mapped_column(Text, default="")
    renal_disease_status: Mapped[int] = mapped_column(Integer, default=0)
    liver_disease_status: Mapped[int] = mapped_column(Integer, default=0)


class Medication(Base):
    __tablename__ = "medications"

    medication_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), index=True)
    drug_name: Mapped[str] = mapped_column(String(128))
    normalized_drug_name: Mapped[str] = mapped_column(String(128), index=True)
    drug_class: Mapped[str] = mapped_column(String(64), index=True)
    start_date: Mapped[str] = mapped_column(Date)
    stop_date: Mapped[str] = mapped_column(Date, nullable=True)
    dose: Mapped[float] = mapped_column(Float, nullable=True)
    route: Mapped[str] = mapped_column(String(32), default="PO")
    frequency: Mapped[str] = mapped_column(String(32), default="")
    dose_change_date: Mapped[str] = mapped_column(Date, nullable=True)


class Lab(Base):
    __tablename__ = "labs"

    lab_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), index=True)
    lab_name: Mapped[str] = mapped_column(String(64))
    normalized_lab_name: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32), default="")
    reference_range_low: Mapped[float] = mapped_column(Float, nullable=True)
    reference_range_high: Mapped[float] = mapped_column(Float, nullable=True)
    lab_date: Mapped[str] = mapped_column(Date, index=True)


class MedicationLabRiskKnowledge(Base):
    __tablename__ = "medication_lab_risk_knowledge"

    rule_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    drug_class: Mapped[str] = mapped_column(String(64), index=True)
    drug_name: Mapped[str] = mapped_column(String(128), default="")
    lab_name: Mapped[str] = mapped_column(String(64), index=True)
    expected_direction: Mapped[str] = mapped_column(String(16))
    risk_type: Mapped[str] = mapped_column(String(64))
    severity_weight: Mapped[float] = mapped_column(Float)
    time_window_days: Mapped[int] = mapped_column(Integer)
    threshold_value: Mapped[float] = mapped_column(Float, nullable=True)
    delta_threshold: Mapped[float] = mapped_column(Float, nullable=True)
    clinical_note: Mapped[str] = mapped_column(Text, default="")


class Alert(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), index=True)
    medication_id: Mapped[int] = mapped_column(Integer, nullable=True)
    lab_id: Mapped[int] = mapped_column(Integer, nullable=True)
    alert_type: Mapped[str] = mapped_column(String(64))
    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    explanation: Mapped[str] = mapped_column(Text)
    suggested_action: Mapped[str] = mapped_column(Text)
    alert_date: Mapped[str] = mapped_column(Date, nullable=True)
    model_type: Mapped[str] = mapped_column(String(32))
    clinician_feedback: Mapped[str] = mapped_column(Text, nullable=True)


class ModelFeature(Base):
    __tablename__ = "model_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(Integer, index=True)
    medication_id: Mapped[int] = mapped_column(Integer)
    lab_name: Mapped[str] = mapped_column(String(64))
    baseline_value: Mapped[float] = mapped_column(Float, nullable=True)
    current_value: Mapped[float] = mapped_column(Float, nullable=True)
    delta_value: Mapped[float] = mapped_column(Float, nullable=True)
    percent_change: Mapped[float] = mapped_column(Float, nullable=True)
    slope_7d: Mapped[float] = mapped_column(Float, nullable=True)
    slope_14d: Mapped[float] = mapped_column(Float, nullable=True)
    slope_30d: Mapped[float] = mapped_column(Float, nullable=True)
    days_since_drug_start: Mapped[int] = mapped_column(Integer, nullable=True)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    egfr: Mapped[float] = mapped_column(Float, nullable=True)
    comorbidity_count: Mapped[int] = mapped_column(Integer, nullable=True)
    polypharmacy_count: Mapped[int] = mapped_column(Integer, nullable=True)
    label: Mapped[int] = mapped_column(Integer, nullable=True)


def get_engine(db_url: str = DB_URL):
    return create_engine(db_url, future=True)


def init_db(db_url: str = DB_URL):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_url: str = DB_URL):
    engine = get_engine(db_url)
    return sessionmaker(bind=engine, future=True)()
