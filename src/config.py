"""Central configuration: paths, scenarios, risk weights, thresholds.

Everything tunable for the study lives here so experiments and ablations can be
run by editing one file. Risk-score weights and rule thresholds are deliberately
centralized and documented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
MIMIC_DIR = RAW_DIR / "mimiciv"          # populated by scripts/download_mimic.py
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PAPER_DIR = PROJECT_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"

# --------------------------------------------------------------------------- #
# Data source switch — change this single line to swap the entire pipeline
# from synthetic to MIMIC-IV (no other code changes required).
# Valid values: "synthetic" | "mimic"
# --------------------------------------------------------------------------- #
import os as _os
DATA_SOURCE = _os.environ.get("DATA_SOURCE", "synthetic").lower()

# Optional cap on the MIMIC-IV cohort. Useful for smoke-testing the full pipeline
# before running on the entire (~30k-patient) cohort. Set via environment, e.g.
# `$env:MIMIC_MAX_PATIENTS = "500"`. Leave unset / empty for the full cohort.
_mmp = _os.environ.get("MIMIC_MAX_PATIENTS", "").strip()
MIMIC_MAX_PATIENTS: int | None = int(_mmp) if _mmp.isdigit() and int(_mmp) > 0 else None

# --------------------------------------------------------------------------- #
# Performance knobs — defaults are tuned for a multi-core desktop (Ryzen 9 +
# 32 GB RAM). All overridable via environment variables for laptops / cloud.
# --------------------------------------------------------------------------- #
def _env_int(name: str, default: int) -> int:
    v = _os.environ.get(name, "").strip()
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = _os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


# -1 means "use every logical core". On Ryzen 9 (12c/24t) this saturates CPU.
N_JOBS = _env_int("N_JOBS", -1)

# Parallelise the per-patient experiment loops across cores. The serial path is
# fine on synthetic data and small cohorts; on full MIMIC-IV (~120k patients)
# this is what makes the desktop a real win over the laptop.
# Set PARALLEL_PATIENTS=0 to force serial (useful for debugging / tracebacks).
PARALLEL_PATIENTS = _env_bool("PARALLEL_PATIENTS", True)

# Use the XGBoost GPU device when available (`device="cuda"`). Falls back to
# CPU automatically inside train_model if CUDA / a compatible XGBoost build is
# missing. RTX 2070 Super is plenty for our feature matrix size.
XGBOOST_USE_GPU = _env_bool("XGBOOST_USE_GPU", False)

# pyarrow CSV engine is 2-4x faster than the C engine on large gzipped files,
# but only if pyarrow is installed. Auto-detected at runtime.
try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False
USE_PYARROW_CSV = _env_bool("USE_PYARROW_CSV", _HAS_PYARROW)


def print_perf_summary() -> None:
    """One-line banner showing the active performance settings.

    Called from `run_pipeline.py` so the user can confirm that the desktop's
    high-throughput knobs are on (or see what's left to enable).
    """
    import multiprocessing as _mp

    cores = _mp.cpu_count()
    used = cores if N_JOBS in (-1, 0) else min(N_JOBS, cores)
    try:
        from xgboost import XGBClassifier  # noqa: F401
        xgb = "yes"
    except Exception:
        xgb = "no"
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
        lgbm = "yes"
    except Exception:
        lgbm = "no"
    print(
        "Perf: "
        f"cores_total={cores}, cores_used={used}, "
        f"xgboost={xgb} (gpu={'on' if XGBOOST_USE_GPU else 'off'}, hist=on), "
        f"lightgbm={lgbm}, pyarrow={_HAS_PYARROW}, "
        f"parallel_patients={PARALLEL_PATIENTS}, "
        f"mimic_max_patients={MIMIC_MAX_PATIENTS or 'full'}",
        flush=True,
    )

for _d in (RAW_DIR, SYNTHETIC_DIR, PROCESSED_DIR, KNOWLEDGE_DIR,
           OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

KNOWLEDGE_CSV = KNOWLEDGE_DIR / "drug_lab_rules.csv"
DB_URL = f"sqlite:///{(DATA_DIR / 'cohort.db').as_posix()}"

RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Population reference ranges (used only as a fallback; the system prefers the
# patient-specific baseline). Values are typical adult ranges.
# --------------------------------------------------------------------------- #
REFERENCE_RANGES = {
    "potassium": (3.5, 5.1),      # mmol/L
    "creatinine": (0.6, 1.3),     # mg/dL
    "egfr": (90.0, 120.0),        # mL/min/1.73m2 (lower bound is the clinically relevant one)
    "inr": (0.8, 1.2),            # unitless (therapeutic on warfarin is 2.0-3.0)
    "sodium": (135.0, 145.0),     # mmol/L
    "alt": (7.0, 56.0),           # U/L
}

# Canonical units we harmonise everything to.
CANONICAL_UNITS = {
    "potassium": "mmol/L",
    "creatinine": "mg/dL",
    "egfr": "mL/min/1.73m2",
    "inr": "ratio",
    "sodium": "mmol/L",
    "alt": "U/L",
}

# --------------------------------------------------------------------------- #
# Clinical scenarios (first version: three primary scenarios)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    drug_class: str
    drug_names: tuple[str, ...]
    labs: tuple[str, ...]
    primary_lab: str
    # outcome / label parameters
    label_window_days: int
    # static-threshold (baseline rule) trigger on the primary lab
    static_threshold: float
    static_direction: str            # "above" or "below"
    # temporal-rule parameters on the primary lab
    delta_threshold: float           # absolute change from baseline that matters
    delta_direction: str             # "increase" or "decrease"
    temporal_current_gate: float     # current value must cross this for a temporal alert
    temporal_window_days: int


SCENARIOS: dict[str, Scenario] = {
    "ace_potassium": Scenario(
        key="ace_potassium",
        name="ACE inhibitor / ARB → Hyperkalemia",
        drug_class="acei_arb",
        drug_names=("lisinopril", "enalapril", "ramipril", "losartan", "valsartan"),
        labs=("potassium", "creatinine", "egfr"),
        primary_lab="potassium",
        label_window_days=30,
        static_threshold=5.3,
        static_direction="above",
        delta_threshold=0.5,
        delta_direction="increase",
        temporal_current_gate=5.0,
        temporal_window_days=30,
    ),
    "warfarin_inr": Scenario(
        key="warfarin_inr",
        name="Warfarin → Elevated INR / Bleeding risk",
        drug_class="anticoagulant",
        drug_names=("warfarin",),
        labs=("inr",),
        primary_lab="inr",
        label_window_days=30,
        static_threshold=4.0,
        static_direction="above",
        delta_threshold=1.3,
        delta_direction="increase",
        temporal_current_gate=3.5,
        temporal_window_days=14,
    ),
    "metformin_egfr": Scenario(
        key="metformin_egfr",
        name="Metformin → Renal function decline",
        drug_class="biguanide",
        drug_names=("metformin",),
        labs=("egfr", "creatinine"),
        primary_lab="egfr",
        label_window_days=60,
        static_threshold=30.0,
        static_direction="below",
        delta_threshold=0.20,            # interpreted as 20% relative decline
        delta_direction="decrease",
        temporal_current_gate=45.0,
        temporal_window_days=60,
    ),
}

# High-risk drug classes used for the polypharmacy / high-risk-med risk factor.
HIGH_RISK_DRUG_CLASSES = {
    "acei_arb", "anticoagulant", "biguanide", "nsaid", "diuretic",
    "statin", "potassium_sparing_diuretic", "ssri",
}

# --------------------------------------------------------------------------- #
# Risk-score weights (0-100 composite). Tunable / ablatable.
# --------------------------------------------------------------------------- #
RISK_WEIGHTS = {
    "lab_abnormality_severity": 0.22,
    "temporal_slope": 0.18,
    "delta_from_baseline": 0.18,
    "drug_risk_strength": 0.14,
    "renal_hepatic_vulnerability": 0.12,
    "age_factor": 0.06,
    "polypharmacy_factor": 0.05,
    "consecutive_trend_factor": 0.05,
}

RISK_BANDS = [
    (0, 30, "Low"),
    (31, 60, "Moderate"),
    (61, 80, "High"),
    (81, 100, "Critical"),
]

# Feature-engineering windows (days).
SLOPE_WINDOWS = (7, 14, 30)

# Early-prediction observation window per scenario: features use only labs within
# [drug_start, drug_start + observation_days]; the label is evaluated over the longer
# label_window_days. observation < label is what makes this an *early* detection task.
OBSERVATION_WINDOW_DAYS = {
    "ace_potassium": 14,
    "warfarin_inr": 10,
    "metformin_egfr": 30,
}

# Renal / hepatic vulnerability thresholds.
RENAL_IMPAIRMENT_EGFR = 60.0
SEVERE_RENAL_EGFR = 30.0
AGE_RISK_THRESHOLD = 65
POLYPHARMACY_THRESHOLD = 5

SAFETY_STATEMENT = (
    "This system is intended to support, not replace, clinical judgment. "
    "It provides early risk signals that require review by qualified "
    "healthcare professionals."
)
