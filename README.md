# Explainable Temporal Medication–Laboratory Conflict Detection


A research prototype that flags medication-related laboratory risks **earlier** than
fixed-threshold rules by reading each patient against **their own pre-drug baseline**,
scoring the risk on a transparent 0–100 scale, and explaining every alert in language a
prescriber can act on. Evaluated end-to-end on **122,166 MIMIC-IV patients**.

> **Safety statement.** This system is intended to **support, not replace**, clinical
> judgement. It produces early risk signals that require review by qualified healthcare
> professionals. It does not diagnose, prescribe, or stop medications.

---

## What the system does

Conventional drug-safety alerts fire on a single fixed cut-off (e.g. `K⁺ > 5.3 mmol/L`).
They ignore the patient's own baseline, the direction of travel, and the time relationship
to the drug — which is why clinicians override most of them.

This system replaces that logic with four pillars:

1. **Temporal trend analysis** — 7/14/30-day slope, change-from-baseline, acceleration,
   days-since-drug-start, trend persistence.
2. **Patient-specific baseline** — the median of pre-drug readings within a 180-day
   look-back, used to detect within-patient drift even when values never leave the
   population reference range.
3. **Explainable alerts** — every alert carries a clinically readable narrative
   (drug, lab, baseline → current, time window, risk factor, review suggestion) and,
   for the learned models, SHAP attributions.
4. **Alert-burden reduction** — fires only when a known drug–lab relationship, a
   sustained baseline-relative change, and a plausible clinical risk co-occur.

### Scenarios studied

| # | Drug class | Lab outcome | Risk |
|---|---|---|---|
| S1 | ACE inhibitors / ARBs | Potassium, Creatinine, eGFR | Hyperkalemia, renal decline |
| S2 | Warfarin | INR | Bleeding / instability |
| S3 | Metformin | eGFR, Creatinine | Renal safety / lactic acidosis |

---

## Headline results on MIMIC-IV v2.2

On the full 122,166-patient cohort the pipeline produced 6,693 alerts across the three
scenarios. The full Methods, Discussion, and Limitations live in the accompanying
manuscript (currently in submission); this repository hosts the code, figures, and
aggregate result tables that the manuscript draws on.

| Scenario | XGBoost AUROC | Static→Temporal alert burden | Best case-level lead |
|---|---|---|---|
| ACE inhibitor / ARB → potassium | 0.955 | **18.1% fewer alerts** (11.5 → 9.4 per 100 patients) | 29 days |
| Warfarin → INR | 0.928 | Trade-off: sensitivity preserved, specificity falls | — |
| Metformin → eGFR | 0.974 | Both detectors limited by sparse positive labels (n=342) | — |

Brier scores ranged from 0.014 to 0.050 across the primary models, well within the
range usually considered well-calibrated.

The headline lesson — reported in the manuscript and in the §6.1 *Scope of utility*
discussion — is that the temporal layer helps most where the underlying physiology
drifts gradually and reliably (RAAS-driven hyperkalemia) and less where the outcome is
abrupt (warfarin/INR) or label-sparse (metformin/eGFR).

---

## Evidence of execution

The two screenshots below are saved under `docs/screenshots/` and record the
actual MIMIC-IV pipeline run that produced the numbers reported in the manuscript.
They are intended as informal reproducibility evidence — every value in the tables
they show can also be regenerated from a clean clone by running
`python run_pipeline.py --source mimic` against a credentialed MIMIC-IV copy.

| # | What it shows |
|---|---|
| [`01_pipeline_start_and_training.jpg`](docs/screenshots/01_pipeline_start_and_training.jpg) | Environment configuration, MIMIC ingest (299,712 → 122,166 patients in scenario cohort, 10,772,233 lab rows), Experiment 1 parallel rules, and Experiment 2 XGBoost training progress. |
| [`02_pipeline_results_and_tables.jpg`](docs/screenshots/02_pipeline_results_and_tables.jpg) | Alert generation (6,693 alerts stored), Experiment 1 static-vs-temporal results, and Experiment 3 early-detection summary. |

## Repository layout

```
src/
  config.py                # paths, scenarios, risk weights, thresholds
  db/                      # canonical schema (SQLAlchemy) + pandas contract
  ingestion/               # synthetic generator, CSV loader, MIMIC-IV loader
  preprocessing/           # cleaning, unit harmonisation, timeline + baseline
  features/                # temporal + risk feature engineering
  rules/                   # static baseline rules + sustained-trend temporal rules
  risk/                    # transparent 0–100 risk score
  models/                  # LogReg / RandomForest / XGBoost / LightGBM + evaluation
  explainability/          # clinical-narrative generator + SHAP
  experiments/             # Exp 1–4 + figure / table generation
  api/                     # FastAPI service
  dashboard/               # Streamlit research dashboard
paper/
  build_manuscript.js      # rebuilds the local manuscript from CSV tables + figures
  refresh_manuscript_data.py
  figures/                 # Fig 1–5 PNGs (publication-ready)
  tables/                  # Tables 1–5 CSVs (anonymised — see DUA note below)
scripts/
  download_mimic.py        # convenience script for credentialed download
  prepare_public_artifacts.py
                           # anonymises subject_ids in shipped CSVs
tests/                     # pytest suite
run_pipeline.py            # end-to-end: ingest → features → rules + ML → tables
regenerate_figures.py      # rebuild figures without retraining
```

---

## Installation

Tested on Python 3.10 / 3.11 / 3.12 (Windows, macOS, Linux).

```bash
git clone https://github.com/bareqmaher-arch/temporal-med-lab-conflict-detection.git
cd temporal-med-lab-conflict-detection
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate
pip install -r requirements.txt
```

The manuscript builder is a Node script; install it once if you intend to rebuild
the `.docx`:

```bash
npm install -g docx
```

---

## Running the pipeline

### 1. On synthetic data (no credentials required)

```bash
python run_pipeline.py
```

Generates a 900-patient synthetic cohort, runs the full pipeline, and writes
aggregate metrics to `outputs/metrics.json`, figures to `paper/figures/`, and
tables to `paper/tables/`. Useful for verifying the install and for trying the
dashboard.

### 2. On MIMIC-IV (credentialed access required)

You will need a PhysioNet account with completed CITI training and a signed Data
Use Agreement for MIMIC-IV v2.2. **We do not redistribute any MIMIC files. Place
your own credentialed copy of the `hosp` module under `data/raw/mimiciv/hosp/`.**

```
data/raw/mimiciv/hosp/
  patients.csv.gz
  admissions.csv.gz
  prescriptions.csv.gz
  labevents.csv.gz
  d_labitems.csv.gz
  diagnoses_icd.csv.gz
  d_icd_diagnoses.csv.gz
```

Then run the pipeline against MIMIC:

```bash
python run_pipeline.py --source mimic
```

This regenerates all figures and tables in `paper/figures/` and `paper/tables/`.
The aggregate metrics are written to `outputs/metrics.json`. Patient-level outputs
are written under `outputs/` and `data/cohort.db` and **must not be committed**
back to git — these paths are protected by `.gitignore`.

### Dashboard and API

```bash
streamlit run src/dashboard/app.py     # research dashboard
uvicorn src.api.main:app --reload      # REST API
```

---

## Rebuilding the manuscript

After a fresh pipeline run the tables and figures are already in sync. To
regenerate the local manuscript from them (the `.docx` is kept off this repo
until journal acceptance):

```bash
NODE_PATH="$(npm root -g)" node paper/build_manuscript.js
```

(On Windows PowerShell: `$env:NODE_PATH = npm root -g; node paper/build_manuscript.js`.)

Every numeric value in the Results section traces back to a CSV in
`paper/tables/` and ultimately to the run that produced `outputs/metrics.json`.

---

## Reproducibility checklist

| Claim in the paper | Trace back through |
|---|---|
| 122,166 patients, 6,693 alerts | `outputs/metrics.json` |
| Static vs temporal rule numbers | `paper/tables/exp1_rules_vs_temporal.csv` |
| AUROC / AUPRC / Brier per model | `paper/tables/table3_model_comparison.csv` |
| Time-to-alert and burden reduction | `paper/tables/table4_early_detection.csv` |
| Example explanations | `paper/tables/table5_example_explanations.csv` (subject_ids anonymised) |
| Figures 1–5 | `paper/figures/fig*.png` |

Random seeds are fixed in `src/config.py`. A clean clone + the steps above should
reproduce every published number bit-for-bit on the same MIMIC release.

---

## MIMIC-IV Data Use Agreement compliance

This repository is published under the open-source obligation in §9 of the
PhysioNet Credentialed Health Data Use Agreement. To stay compliant we follow
three rules that are enforced both by `.gitignore` and by
`scripts/prepare_public_artifacts.py`:

1. **No raw or processed MIMIC files are committed.** `data/raw/`,
   `data/processed/`, `data/cohort.db`, `outputs/explainability_questionnaire.csv`
   (original form), and the pipeline run-log are all excluded.
2. **Subject_ids are anonymised in the shipped result CSVs.** The 12 worked
   examples in `paper/tables/table5_example_explanations.csv` and the matching
   `outputs/explainability_questionnaire.csv` use neutral `Case N` labels. The
   originals stay under `_local_only/` on the developer's machine, also
   excluded from git.
3. **One subject_id is referenced in the prose case study** (patient 14866589 in
   §5.3 and Figure 2), which is permitted by the DUA and is the accepted norm in
   MIMIC publications.

If you discover that a file with patient-level data has slipped through, please
open an issue and we will rewrite the offending history immediately.

---

## Citation

If you use this code or its findings, please cite the manuscript (in submission):

```bibtex
@article{khudhair2026etmldetector,
  title   = {Explainable Temporal Medication–Laboratory Conflict Detection for
             Early Prevention of Adverse Drug Events: an evaluation on 122,166
             MIMIC-IV patients},
  author  = {Bareq Maher Khudhair and Karrar Maher Khudhair},
  journal = {Manuscript in submission},
  year    = {2026},
  note    = {Code: https://github.com/bareqmaher-arch/temporal-med-lab-conflict-detection}
}
```

Please also cite the underlying data resource:

> Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible
> electronic health record dataset. *Sci Data*. 2023;10(1):1.

---

## License

The source code is released under the MIT License (see `LICENSE.txt`). This
licence covers code only — it does **not** grant you access to MIMIC-IV, which
remains governed by the PhysioNet DUA and is not redistributed here.

---

## Acknowledgements

Built at the Department of Medical Informatics, Imam Al-Kadhum University
College (IKU), Baghdad, Iraq. The work depends on MIMIC-IV from the MIT
Laboratory for Computational Physiology and on the open-source Python and Node
ecosystems.
