"""MIMIC-IV adapter -> canonical tables.

Implements the source-agnostic ingestion contract for MIMIC-IV (PhysioNet
credentialed dataset). Reads the seven `hosp/` files we actually need, applies
a cohort filter, normalises drug names and lab names/units, derives eGFR via
CKD-EPI 2021 from creatinine, derives renal/hepatic flags from ICD codes, and
returns the four canonical DataFrames.

Tested against MIMIC-IV v2.2 and v3.1 — the hosp/ schema is identical between
them for the columns we use (itemids, drug strings, ICD codes, anchor dates).

Designed so that switching the data source from synthetic to MIMIC requires
only changing the loader in `config.py` (no downstream code changes).

PhysioNet citation:
    Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely
    accessible electronic health record dataset. Sci Data 10, 1 (2023).

Time alignment (MIMIC-IV anchor mechanism):
    MIMIC-IV shifts every subject's calendar timeline by a per-subject random
    offset (anchor_year). All dates inside a subject's record are internally
    consistent — drug starttime, labevents charttime, admittime all share the
    same offset — so day-level temporal features (slopes, days_since_start)
    are unaffected. We treat the shifted timestamps as ground truth.

Run cost note:
    labevents.csv.gz is ~2 GB compressed (~50 GB uncompressed, ~432 M rows).
    We filter by `itemid` *while streaming* in chunks to keep memory bounded.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.canonical import CanonicalTables
from src.ingestion.base import AbstractLoader

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lab item IDs in MIMIC-IV v3.1 (verified against d_labitems.csv.gz).
# Each canonical lab maps to one or more itemids; we coalesce serum-preferred.
# --------------------------------------------------------------------------- #
LAB_ITEMIDS: dict[str, list[int]] = {
    "potassium":  [50971, 52610, 50822, 52452],   # Serum, BG, Whole-Blood
    "creatinine": [50912, 52546],                 # Serum, Whole-Blood
    "inr":        [51237, 51675],                 # INR (PT-derived)
    "sodium":     [50983, 52623],
    "alt":        [50861, 52458],
}
# Reverse map: itemid -> canonical lab name
ITEMID_TO_LAB: dict[int, str] = {
    iid: name for name, iids in LAB_ITEMIDS.items() for iid in iids
}
ALL_LAB_ITEMIDS: list[int] = sorted(ITEMID_TO_LAB.keys())

# Canonical unit per lab (we drop rows in clearly-wrong units).
LAB_UNITS: dict[str, set[str]] = {
    "potassium":  {"meq/l", "mmol/l"},
    "creatinine": {"mg/dl"},
    "inr":        {"", "ratio", "inr"},   # often blank in MIMIC
    "sodium":     {"meq/l", "mmol/l"},
    "alt":        {"iu/l", "u/l"},
}


# --------------------------------------------------------------------------- #
# Drug-name patterns (case-insensitive substring match on `prescriptions.drug`).
# These cover the common brand and generic strings observed in MIMIC-IV.
# --------------------------------------------------------------------------- #
DRUG_PATTERNS: list[tuple[str, str, str]] = [
    # (regex pattern, normalized_drug_name, drug_class)
    (r"\blisinopril\b",      "lisinopril",   "acei_arb"),
    (r"\benalapril\b",       "enalapril",    "acei_arb"),
    (r"\bramipril\b",        "ramipril",     "acei_arb"),
    (r"\bcaptopril\b",       "captopril",    "acei_arb"),
    (r"\bbenazepril\b",      "benazepril",   "acei_arb"),
    (r"\bquinapril\b",       "quinapril",    "acei_arb"),
    (r"\bfosinopril\b",      "fosinopril",   "acei_arb"),
    (r"\bperindopril\b",     "perindopril",  "acei_arb"),
    (r"\btrandolapril\b",    "trandolapril", "acei_arb"),
    (r"\bmoexipril\b",       "moexipril",    "acei_arb"),
    (r"\blosartan\b",        "losartan",     "acei_arb"),
    (r"\bvalsartan\b",       "valsartan",    "acei_arb"),
    (r"\birbesartan\b",      "irbesartan",   "acei_arb"),
    (r"\bolmesartan\b",      "olmesartan",   "acei_arb"),
    (r"\bcandesartan\b",     "candesartan",  "acei_arb"),
    (r"\btelmisartan\b",     "telmisartan",  "acei_arb"),
    (r"\bazilsartan\b",      "azilsartan",   "acei_arb"),
    (r"\beprosartan\b",      "eprosartan",   "acei_arb"),
    (r"\bwarfarin\b|\bcoumadin\b|\bjantoven\b", "warfarin",  "anticoagulant"),
    (r"\bmetformin\b|\bglucophage\b|\bglumetza\b|\bfortamet\b|\briomet\b",
     "metformin", "biguanide"),
    # Co-meds that affect the risk model (polypharmacy / nephrotoxic etc.)
    (r"\bspironolactone\b|\beplerenone\b",    "spironolactone", "potassium_sparing_diuretic"),
    (r"\bfurosemide\b|\blasix\b|\bbumetanide\b|\btorsemide\b",
     "furosemide", "diuretic"),
    (r"\bibuprofen\b|\bnaproxen\b|\bdiclofenac\b|\bketorolac\b|\bcelecoxib\b|\bmeloxicam\b|\bindomethacin\b",
     "nsaid", "nsaid"),
    (r"\batorvastatin\b|\bsimvastatin\b|\brosuvastatin\b|\bpravastatin\b|\blovastatin\b",
     "statin", "statin"),
    (r"\bsertraline\b|\bfluoxetine\b|\bparoxetine\b|\bescitalopram\b|\bcitalopram\b",
     "ssri", "ssri"),
]


# --------------------------------------------------------------------------- #
# ICD-9 / ICD-10 codes for organ-failure flags.
# --------------------------------------------------------------------------- #
def _icd_renal(code: str, version: int) -> bool:
    """CKD or AKI."""
    if not isinstance(code, str):
        return False
    c = code.upper().replace(".", "")
    if version == 10:
        return c.startswith("N17") or c.startswith("N18") or c.startswith("N19")
    if version == 9:
        return c.startswith("584") or c.startswith("585") or c.startswith("586")
    return False


def _icd_hepatic(code: str, version: int) -> bool:
    """Liver disease (chronic, cirrhosis, failure)."""
    if not isinstance(code, str):
        return False
    c = code.upper().replace(".", "")
    if version == 10:
        return any(c.startswith(p) for p in ("K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77"))
    if version == 9:
        return c.startswith("570") or c.startswith("571") or c.startswith("572") or c.startswith("573")
    return False


# --------------------------------------------------------------------------- #
# eGFR via CKD-EPI 2021 (race-neutral) — Inker LA et al., NEJM 2021.
# --------------------------------------------------------------------------- #
def ckd_epi_2021(scr_mg_dl: float, age: float, sex: str) -> float:
    """Return eGFR (mL/min/1.73 m^2). sex in {'M','F'}."""
    if scr_mg_dl is None or np.isnan(scr_mg_dl) or scr_mg_dl <= 0:
        return np.nan
    if sex == "F":
        kappa, alpha, sex_mult = 0.7, -0.241, 1.012
    else:
        kappa, alpha, sex_mult = 0.9, -0.302, 1.0
    ratio = scr_mg_dl / kappa
    egfr = (
        142.0
        * (min(ratio, 1.0) ** alpha)
        * (max(ratio, 1.0) ** -1.200)
        * (0.9938 ** age)
        * sex_mult
    )
    return float(egfr)


# --------------------------------------------------------------------------- #
# Compiled regex for drug matching (one pass per row).
# --------------------------------------------------------------------------- #
_DRUG_REGEXES = [(re.compile(p, re.IGNORECASE), n, c) for p, n, c in DRUG_PATTERNS]

def normalize_drug(drug_str: str) -> tuple[str | None, str | None]:
    """Return (normalized_name, drug_class) or (None, None) if not interesting."""
    if not isinstance(drug_str, str):
        return None, None
    for rx, name, cls in _DRUG_REGEXES:
        if rx.search(drug_str):
            return name, cls
    return None, None


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
@dataclass
class MIMICLoader(AbstractLoader):
    """Read MIMIC-IV v3.1 hosp/ subset and emit canonical tables.

    Parameters
    ----------
    mimic_dir : Path
        Directory containing the unpacked MIMIC-IV files. We expect
        `mimic_dir/hosp/{patients,admissions,labevents,d_labitems,
        prescriptions,diagnoses_icd,d_icd_diagnoses}.csv.gz`.
    chunksize : int
        Row chunk size for streaming labevents/prescriptions.
    """
    mimic_dir: Path
    # 1M rows = ~150 MB resident. On 32 GB machines we can comfortably push to
    # 5M for ~2-3x faster streaming of labevents (the 432M-row bottleneck).
    # Override via `MIMIC_CHUNKSIZE` env var if memory is tight.
    chunksize: int = field(
        default_factory=lambda: int(
            __import__("os").environ.get("MIMIC_CHUNKSIZE", "5000000")
        )
    )
    _meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mimic_dir = Path(self.mimic_dir)
        self.hosp = self.mimic_dir / "hosp"
        if not self.hosp.exists():
            raise FileNotFoundError(
                f"MIMIC hosp/ folder not found at {self.hosp}. "
                "Run scripts/download_mimic.py first."
            )

    # ------------------------------------------------------------------ #
    def load(self) -> CanonicalTables:
        from src.config import MIMIC_MAX_PATIENTS

        log.info("Loading MIMIC-IV from %s", self.mimic_dir)
        print(f"  reading patients ...", flush=True)
        patients_raw = self._read_patients()
        print(f"    -> {len(patients_raw):,} patients", flush=True)

        print(f"  reading admissions ...", flush=True)
        admissions  = self._read_admissions()

        print(f"  reading diagnoses (ICD) ...", flush=True)
        icd_long    = self._read_diagnoses()      # subject_id -> flags

        print(f"  streaming prescriptions (filtering to scenario drugs) ...", flush=True)
        prescriptions = self._read_prescriptions_cohort()    # streams + filters
        cohort_ids = set(prescriptions["subject_id"].unique().tolist())
        log.info("Cohort size after drug filter: %d subjects", len(cohort_ids))
        print(f"    -> {len(cohort_ids):,} patients in scenario cohort", flush=True)

        # Optional cap for smoke-testing the pipeline on a subset. The cap is
        # applied here so all downstream tables (prescriptions, labs) are
        # filtered consistently to the same patient subset.
        if MIMIC_MAX_PATIENTS is not None and len(cohort_ids) > MIMIC_MAX_PATIENTS:
            cohort_ids = set(list(cohort_ids)[:MIMIC_MAX_PATIENTS])
            prescriptions = prescriptions[
                prescriptions["subject_id"].isin(cohort_ids)
            ].reset_index(drop=True)
            print(f"  MIMIC_MAX_PATIENTS={MIMIC_MAX_PATIENTS} -> capped to "
                  f"{len(cohort_ids):,} patients for this run", flush=True)

        print(f"  streaming labevents for cohort ...", flush=True)
        labs_creat = self._read_labs_for_cohort(cohort_ids, lab_itemids=ALL_LAB_ITEMIDS)
        log.info("Lab rows for cohort: %d", len(labs_creat))
        print(f"    -> {len(labs_creat):,} lab rows kept", flush=True)

        patients_canonical = self._build_patients(
            patients_raw, admissions, icd_long, cohort_ids
        )
        medications_canonical = self._build_medications(prescriptions)
        labs_canonical = self._build_labs(labs_creat, patients_canonical)

        knowledge = AbstractLoader.load_knowledge()

        tables = CanonicalTables(
            patients=patients_canonical,
            medications=medications_canonical,
            labs=labs_canonical,
            knowledge=knowledge,
        ).validate()

        self._meta = {
            "n_patients": len(tables.patients),
            "n_medications": len(tables.medications),
            "n_labs": len(tables.labs),
            "source": "MIMIC-IV v3.1",
            "loaded_at": datetime.utcnow().isoformat(),
        }
        log.info("MIMIC cohort: %s", self._meta)
        return tables

    # ------------------------------------------------------------------ #
    # Readers
    # ------------------------------------------------------------------ #
    def _read_patients(self) -> pd.DataFrame:
        p = self.hosp / "patients.csv.gz"
        df = pd.read_csv(
            p,
            usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
            dtype={"subject_id": "int64", "gender": "category",
                   "anchor_age": "int32", "anchor_year": "int32"},
            parse_dates=["dod"],
        )
        log.info("Read patients: %d rows", len(df))
        return df

    def _read_admissions(self) -> pd.DataFrame:
        p = self.hosp / "admissions.csv.gz"
        df = pd.read_csv(
            p,
            usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
            dtype={"subject_id": "int64", "hadm_id": "int64"},
            parse_dates=["admittime", "dischtime"],
        )
        # First admission per subject (used for canonical admission_date)
        first = (
            df.sort_values("admittime")
              .groupby("subject_id", as_index=False)
              .agg(admission_date=("admittime", "min"),
                   discharge_date=("dischtime", "max"))
        )
        log.info("Read admissions: %d -> %d unique subjects", len(df), len(first))
        return first

    def _read_diagnoses(self) -> pd.DataFrame:
        """Return per-subject flags + comorbidity_count + diagnosis_codes string."""
        diag = pd.read_csv(
            self.hosp / "diagnoses_icd.csv.gz",
            usecols=["subject_id", "icd_code", "icd_version"],
            dtype={"subject_id": "int64", "icd_code": "string", "icd_version": "int8"},
        )
        diag["renal"] = [
            _icd_renal(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])
        ]
        diag["hepatic"] = [
            _icd_hepatic(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])
        ]
        agg = (
            diag.groupby("subject_id", as_index=False)
                .agg(renal_disease_status=("renal", "any"),
                     liver_disease_status=("hepatic", "any"),
                     comorbidity_count=("icd_code", "nunique"),
                     diagnosis_codes=("icd_code", lambda s: ";".join(sorted(set(s.dropna().astype(str))))[:500]))
        )
        log.info("Read diagnoses: aggregated to %d subjects", len(agg))
        return agg

    def _read_prescriptions_cohort(self) -> pd.DataFrame:
        """Stream prescriptions, keep only rows whose drug matches our patterns."""
        p = self.hosp / "prescriptions.csv.gz"
        usecols = ["subject_id", "pharmacy_id", "starttime", "stoptime",
                   "drug", "dose_val_rx", "dose_unit_rx", "route"]
        keep_rows: list[pd.DataFrame] = []
        n_seen = 0
        for chunk in pd.read_csv(
            p, usecols=usecols, chunksize=self.chunksize,
            dtype={"subject_id": "int64", "drug": "string",
                   "dose_val_rx": "string", "dose_unit_rx": "string",
                   "route": "string"},
            parse_dates=["starttime", "stoptime"],
        ):
            n_seen += len(chunk)
            # Vectorised: try each regex and keep the first hit per row.
            chunk["normalized_drug_name"] = pd.NA
            chunk["drug_class"] = pd.NA
            drugs_lower = chunk["drug"].str.lower().fillna("")
            for rx, name, cls in _DRUG_REGEXES:
                hit = drugs_lower.str.contains(rx.pattern, regex=True, na=False)
                hit &= chunk["normalized_drug_name"].isna()
                chunk.loc[hit, "normalized_drug_name"] = name
                chunk.loc[hit, "drug_class"] = cls
            kept = chunk[chunk["normalized_drug_name"].notna()].copy()
            if not kept.empty:
                keep_rows.append(kept)
        log.info("Prescriptions: scanned %d rows, kept %d",
                 n_seen, sum(len(k) for k in keep_rows))
        out = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame()
        return out

    def _read_labs_for_cohort(
        self, cohort_ids: set[int], lab_itemids: list[int]
    ) -> pd.DataFrame:
        """Stream labevents.csv.gz keeping only (cohort, itemid) rows."""
        p = self.hosp / "labevents.csv.gz"
        usecols = ["labevent_id", "subject_id", "itemid", "charttime",
                   "valuenum", "valueuom", "ref_range_lower", "ref_range_upper"]
        itemid_set = set(lab_itemids)
        keep_rows: list[pd.DataFrame] = []
        n_seen = 0
        for chunk in pd.read_csv(
            p, usecols=usecols, chunksize=self.chunksize,
            dtype={"labevent_id": "int64", "subject_id": "int64",
                   "itemid": "int32", "valuenum": "float32",
                   "valueuom": "string",
                   "ref_range_lower": "float32", "ref_range_upper": "float32"},
            parse_dates=["charttime"],
        ):
            n_seen += len(chunk)
            mask = chunk["itemid"].isin(itemid_set) & chunk["subject_id"].isin(cohort_ids)
            kept = chunk[mask & chunk["valuenum"].notna()].copy()
            if not kept.empty:
                keep_rows.append(kept)
        log.info("Labevents: scanned %d rows, kept %d for cohort",
                 n_seen, sum(len(k) for k in keep_rows))
        return pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Builders -> canonical schema
    # ------------------------------------------------------------------ #
    def _build_patients(
        self,
        patients_raw: pd.DataFrame,
        admissions: pd.DataFrame,
        diagnoses_agg: pd.DataFrame,
        cohort_ids: set[int],
    ) -> pd.DataFrame:
        df = patients_raw[patients_raw["subject_id"].isin(cohort_ids)].copy()
        df = df.merge(admissions, on="subject_id", how="left")
        df = df.merge(diagnoses_agg, on="subject_id", how="left")
        df["renal_disease_status"] = df["renal_disease_status"].fillna(False).astype(int)
        df["liver_disease_status"] = df["liver_disease_status"].fillna(False).astype(int)
        df["comorbidity_count"] = df["comorbidity_count"].fillna(0).astype(int)
        df["diagnosis_codes"] = df["diagnosis_codes"].fillna("")
        df["sex"] = df["gender"].astype(str).str.upper().str[0]
        df = df.rename(columns={"subject_id": "patient_id", "anchor_age": "age"})
        df["comorbidities"] = df["diagnosis_codes"]  # short alias used elsewhere
        canonical = df[[
            "patient_id", "age", "sex", "comorbidities", "admission_date",
            "discharge_date", "diagnosis_codes", "renal_disease_status",
            "liver_disease_status",
        ]]
        return canonical.reset_index(drop=True)

    def _build_medications(self, presc: pd.DataFrame) -> pd.DataFrame:
        if presc.empty:
            return pd.DataFrame(columns=[
                "medication_id", "patient_id", "drug_name", "normalized_drug_name",
                "drug_class", "start_date", "stop_date", "dose", "route",
                "frequency", "dose_change_date",
            ])
        df = presc.copy()
        df = df.rename(columns={
            "subject_id": "patient_id",
            "drug": "drug_name",
            "starttime": "start_date",
            "stoptime": "stop_date",
            "dose_val_rx": "dose",
            "route": "route",
        })
        # Collapse repeat prescriptions of the same drug for the same patient into
        # a single exposure episode (earliest start, latest stop) — preserves the
        # temporal-from-drug-start semantics used downstream.
        df = (
            df.sort_values(["patient_id", "normalized_drug_name", "start_date"])
              .groupby(["patient_id", "normalized_drug_name", "drug_class"],
                       as_index=False)
              .agg(drug_name=("drug_name", "first"),
                   start_date=("start_date", "min"),
                   stop_date=("stop_date", "max"),
                   dose=("dose", "first"),
                   route=("route", "first"))
        )
        df["frequency"] = ""
        df["dose_change_date"] = pd.NaT
        df.insert(0, "medication_id", np.arange(1, len(df) + 1))
        canonical = df[[
            "medication_id", "patient_id", "drug_name", "normalized_drug_name",
            "drug_class", "start_date", "stop_date", "dose", "route",
            "frequency", "dose_change_date",
        ]]
        return canonical.reset_index(drop=True)

    def _build_labs(
        self, lab_rows: pd.DataFrame, patients_canonical: pd.DataFrame
    ) -> pd.DataFrame:
        if lab_rows.empty:
            return pd.DataFrame(columns=[
                "lab_id", "patient_id", "lab_name", "normalized_lab_name",
                "value", "unit", "reference_range_low", "reference_range_high",
                "lab_date",
            ])
        df = lab_rows.copy()
        df["normalized_lab_name"] = df["itemid"].map(ITEMID_TO_LAB)
        df = df[df["normalized_lab_name"].notna()].copy()
        # Unit sanity filter (drop blatantly wrong-unit rows).
        unit_lower = df["valueuom"].fillna("").str.lower()
        ok = pd.Series(False, index=df.index)
        for name, valid in LAB_UNITS.items():
            mask = (df["normalized_lab_name"] == name) & unit_lower.isin(valid)
            ok |= mask
        df = df[ok].copy()
        df = df.rename(columns={
            "subject_id": "patient_id",
            "valuenum": "value",
            "valueuom": "unit",
            "charttime": "lab_date",
            "ref_range_lower": "reference_range_low",
            "ref_range_upper": "reference_range_high",
            "labevent_id": "lab_id",
        })
        df["lab_name"] = df["normalized_lab_name"]
        canonical_cols = [
            "lab_id", "patient_id", "lab_name", "normalized_lab_name", "value",
            "unit", "reference_range_low", "reference_range_high", "lab_date",
        ]
        labs_main = df[canonical_cols].reset_index(drop=True)

        # Derive eGFR rows from creatinine using CKD-EPI 2021.
        egfr_rows = self._derive_egfr_rows(labs_main, patients_canonical)
        if not egfr_rows.empty:
            labs_main = pd.concat([labs_main, egfr_rows], ignore_index=True)

        labs_main["lab_id"] = np.arange(1, len(labs_main) + 1)
        return labs_main

    def _derive_egfr_rows(
        self, labs: pd.DataFrame, patients: pd.DataFrame
    ) -> pd.DataFrame:
        creat = labs[labs["normalized_lab_name"] == "creatinine"]
        if creat.empty:
            return pd.DataFrame(columns=labs.columns)
        meta = patients[["patient_id", "age", "sex"]].set_index("patient_id")
        merged = creat.join(meta, on="patient_id")
        merged = merged.dropna(subset=["value", "age", "sex"])
        egfr_vals = [
            ckd_epi_2021(v, a, s) for v, a, s in
            zip(merged["value"], merged["age"], merged["sex"])
        ]
        out = pd.DataFrame({
            "lab_id": -1,            # reassigned by caller
            "patient_id": merged["patient_id"].values,
            "lab_name": "egfr",
            "normalized_lab_name": "egfr",
            "value": egfr_vals,
            "unit": "mL/min/1.73m2",
            "reference_range_low": 60.0,
            "reference_range_high": 120.0,
            "lab_date": merged["lab_date"].values,
        })
        out = out[out["value"].notna()]
        return out
