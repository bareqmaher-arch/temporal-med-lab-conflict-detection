"""Synthetic EHR generator (proof-of-concept).

Generates a cohort with **encoded causal structure**: a fraction of exposed patients
develop a *progressive* drug-associated lab drift (true positives); others are stable
or show a *transient* out-of-range spike that resolves (confounded negatives). The
transient-spike negatives are what let a single static threshold fire falsely while a
temporal rule (which needs a sustained trend) does not — so the temporal advantage is
real, not baked-in.

Effect magnitudes follow published-order-of-magnitude pharmacology (e.g. ACEi raising
K+ ~0.3-0.8 mmol/L over weeks). This is a proof-of-concept only.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, REFERENCE_RANGES, SCENARIOS
from src.db.canonical import CanonicalTables
from src.ingestion.base import AbstractLoader

ANCHOR_DATE = date(2024, 1, 1)
COMORBIDITY_POOL = [
    "hypertension", "diabetes", "heart_failure", "chronic_kidney_disease",
    "atrial_fibrillation", "ischemic_heart_disease", "copd", "hyperlipidemia",
]
EXTRA_DRUGS = [
    ("atorvastatin", "statin"), ("amlodipine", "ccb"),
    ("furosemide", "diuretic"), ("aspirin", "antiplatelet"),
    ("omeprazole", "ppi"), ("metoprolol", "beta_blocker"),
    ("sertraline", "ssri"), ("ibuprofen", "nsaid"),
]

# direction in which the primary lab moves toward danger, per scenario primary lab
_DANGER_DIRECTION = {
    "potassium": +1, "creatinine": +1, "inr": +1,
    "egfr": -1, "sodium": -1, "alt": +1,
}


class SyntheticLoader(AbstractLoader):
    def __init__(
        self,
        n_patients: int = 900,
        positive_rate: float = 0.34,
        transient_negative_rate: float = 0.22,
        follow_up_days: int = 75,
        sample_every_days: int = 6,
        seed: int = RANDOM_SEED,
    ):
        self.n_patients = n_patients
        self.positive_rate = positive_rate
        self.transient_negative_rate = transient_negative_rate
        self.follow_up_days = follow_up_days
        self.sample_every_days = sample_every_days
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def load(self) -> CanonicalTables:
        patients, medications, labs = [], [], []
        med_id = 1
        lab_id = 1
        scenario_keys = list(SCENARIOS)

        for pid in range(1, self.n_patients + 1):
            scenario = SCENARIOS[scenario_keys[(pid - 1) % len(scenario_keys)]]
            case_type = self._draw_case_type()

            patient, renal_impaired, liver_impaired = self._make_patient(pid, scenario, case_type)
            patients.append(patient)

            drug_start = ANCHOR_DATE + timedelta(days=int(self.rng.integers(5, 20)))
            drug_name = str(self.rng.choice(scenario.drug_names))
            dose_change = (
                drug_start + timedelta(days=int(self.rng.integers(15, 35)))
                if self.rng.random() < 0.3 else None
            )
            medications.append({
                "medication_id": med_id, "patient_id": pid,
                "drug_name": drug_name, "normalized_drug_name": drug_name.lower(),
                "drug_class": scenario.drug_class, "start_date": drug_start,
                "stop_date": None, "dose": float(self.rng.choice([5, 10, 20, 40])),
                "route": "PO", "frequency": "OD",
                "dose_change_date": dose_change,
            })
            index_med_id = med_id
            med_id += 1

            # polypharmacy: a few unrelated drugs
            for dname, dclass in self._sample_extra_drugs():
                medications.append({
                    "medication_id": med_id, "patient_id": pid,
                    "drug_name": dname, "normalized_drug_name": dname,
                    "drug_class": dclass,
                    "start_date": drug_start - timedelta(days=int(self.rng.integers(0, 30))),
                    "stop_date": None, "dose": float(self.rng.choice([5, 10, 25, 50])),
                    "route": "PO", "frequency": "OD", "dose_change_date": None,
                })
                med_id += 1

            # labs for the scenario's labs
            for lab_name in scenario.labs:
                series = self._make_lab_series(
                    lab_name, scenario, case_type, renal_impaired, drug_start, index_med_id,
                )
                for d, val in series:
                    lo, hi = REFERENCE_RANGES[lab_name]
                    labs.append({
                        "lab_id": lab_id, "patient_id": pid, "lab_name": lab_name,
                        "normalized_lab_name": lab_name, "value": round(float(val), 3),
                        "unit": "", "reference_range_low": lo, "reference_range_high": hi,
                        "lab_date": d,
                    })
                    lab_id += 1

        tables = CanonicalTables(
            patients=pd.DataFrame(patients),
            medications=pd.DataFrame(medications),
            labs=pd.DataFrame(labs),
            knowledge=self.load_knowledge(),
        )
        return tables.validate()

    # ------------------------------------------------------------------ #
    def _draw_case_type(self) -> str:
        r = self.rng.random()
        if r < self.positive_rate:
            return "positive"
        if r < self.positive_rate + self.transient_negative_rate:
            return "transient"
        return "stable"

    def _make_patient(self, pid: int, scenario, case_type: str):
        age = int(np.clip(self.rng.normal(66, 14), 28, 95))
        sex = str(self.rng.choice(["M", "F"]))
        # renal impairment more likely in positive cases / older patients
        p_renal = 0.25 + 0.25 * (case_type == "positive") + 0.15 * (age >= 70)
        renal_impaired = self.rng.random() < p_renal
        liver_impaired = self.rng.random() < 0.08
        comorbs = list(self.rng.choice(
            COMORBIDITY_POOL, size=int(self.rng.integers(0, 4)), replace=False))
        if renal_impaired and "chronic_kidney_disease" not in comorbs:
            comorbs.append("chronic_kidney_disease")
        return (
            {
                "patient_id": pid, "age": age, "sex": sex,
                "comorbidities": ";".join(comorbs),
                "admission_date": ANCHOR_DATE,
                "discharge_date": ANCHOR_DATE + timedelta(days=self.follow_up_days),
                "diagnosis_codes": "",
                "renal_disease_status": int(renal_impaired),
                "liver_disease_status": int(liver_impaired),
            },
            renal_impaired,
            liver_impaired,
        )

    def _sample_extra_drugs(self):
        k = int(self.rng.integers(0, 6))
        if k == 0:
            return []
        idx = self.rng.choice(len(EXTRA_DRUGS), size=k, replace=False)
        return [EXTRA_DRUGS[i] for i in idx]

    # ------------------------------------------------------------------ #
    def _make_lab_series(self, lab_name, scenario, case_type, renal_impaired,
                         drug_start, med_id):
        """Return list of (date, value)."""
        direction = _DANGER_DIRECTION[lab_name]
        baseline = self._baseline_for(lab_name, renal_impaired)
        is_primary = lab_name == scenario.primary_lab
        noise = self._noise_for(lab_name)

        # pre-drug baseline readings
        out = []
        for offset in (-14, -7):
            out.append((drug_start + timedelta(days=offset),
                        baseline + self.rng.normal(0, noise)))

        # total drift magnitude toward danger for a positive progressive case
        drift = self._drift_for(lab_name, scenario) if (is_primary and case_type == "positive") else 0.0
        # secondary labs in positive cases drift mildly too
        if not is_primary and case_type == "positive":
            drift = 0.45 * self._drift_for(lab_name, scenario)

        # the drug effect manifests *within* the scenario's label window, then plateaus,
        # so a positive case crosses into danger during the prediction horizon.
        manifest_day = self.rng.uniform(0.5, 1.0) * scenario.label_window_days

        days = list(range(self.sample_every_days, self.follow_up_days + 1, self.sample_every_days))
        spike_day = self.rng.choice(days) if (is_primary and case_type == "transient") else None

        for d in days:
            frac = min(1.0, d / manifest_day) if manifest_day > 0 else 1.0
            val = baseline + direction * drift * frac + self.rng.normal(0, noise)
            if d == spike_day:
                # a single transient excursion past the static threshold that resolves
                val = self._spike_value(lab_name, scenario)
            out.append((drug_start + timedelta(days=int(d)), val))

        return [(d, self._clip(lab_name, v)) for d, v in out]

    # --- per-lab parameterisation ------------------------------------- #
    def _baseline_for(self, lab_name, renal_impaired):
        if lab_name == "potassium":
            return self.rng.normal(4.2, 0.25)
        if lab_name == "creatinine":
            return self.rng.normal(1.4 if renal_impaired else 0.95, 0.15)
        if lab_name == "egfr":
            return self.rng.normal(48 if renal_impaired else 78, 8)
        if lab_name == "inr":
            return self.rng.normal(2.4, 0.25)   # therapeutic on warfarin
        if lab_name == "sodium":
            return self.rng.normal(139, 2)
        if lab_name == "alt":
            return self.rng.normal(28, 6)
        return self.rng.normal(1.0, 0.1)

    def _noise_for(self, lab_name):
        return {
            "potassium": 0.12, "creatinine": 0.06, "egfr": 2.5,
            "inr": 0.18, "sodium": 1.2, "alt": 4.0,
        }.get(lab_name, 0.1)

    def _drift_for(self, lab_name, scenario):
        if lab_name == "potassium":
            return self.rng.uniform(1.0, 1.7)
        if lab_name == "creatinine":
            return self.rng.uniform(0.5, 1.1)
        if lab_name == "egfr":
            return self.rng.uniform(28, 48)        # absolute decline
        if lab_name == "inr":
            return self.rng.uniform(1.7, 2.8)
        return self.rng.uniform(0.5, 1.0)

    def _spike_value(self, lab_name, scenario):
        """A transient excursion just past the static threshold."""
        if scenario.static_direction == "above":
            return scenario.static_threshold + abs(self.rng.normal(0.3, 0.15))
        return scenario.static_threshold - abs(self.rng.normal(3, 1.5))

    def _clip(self, lab_name, v):
        bounds = {
            "potassium": (2.5, 8.0), "creatinine": (0.3, 8.0),
            "egfr": (5, 130), "inr": (0.8, 9.0), "sodium": (115, 155),
            "alt": (5, 600),
        }
        lo, hi = bounds.get(lab_name, (0, 1e6))
        return float(np.clip(v, lo, hi))


def generate_and_save(**kwargs) -> CanonicalTables:
    """Generate a cohort and persist canonical tables to data/synthetic."""
    from src.config import SYNTHETIC_DIR

    tables = SyntheticLoader(**kwargs).load()
    tables.patients.to_csv(SYNTHETIC_DIR / "patients.csv", index=False)
    tables.medications.to_csv(SYNTHETIC_DIR / "medications.csv", index=False)
    tables.labs.to_csv(SYNTHETIC_DIR / "labs.csv", index=False)
    return tables


if __name__ == "__main__":
    t = generate_and_save()
    print(f"patients={len(t.patients)} medications={len(t.medications)} labs={len(t.labs)}")
