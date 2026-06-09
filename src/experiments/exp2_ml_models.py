"""Experiment 2: temporal ML model comparison."""
from __future__ import annotations

import pandas as pd

from src.config import SCENARIOS
from src.models.dataset import build_scenario_dataset
from src.models.predict import save_model
from src.models.train_model import train_scenario_models
from src.preprocessing.build_timeline import index_labs, index_medications_by_pid


def run(patients, medications, labs, knowledge):
    """Return (comparison_df, trained) where trained[scenario] holds models + test data."""
    # Share one index across all scenarios — saves rebuilding the lookup three
    # times when running on the full MIMIC-IV cohort.
    labs_idx = labs if isinstance(labs, dict) else index_labs(labs)
    meds_idx = (medications if isinstance(medications, dict)
                else index_medications_by_pid(medications))
    rows, trained = [], {}
    for key in SCENARIOS:
        df = build_scenario_dataset(key, patients, meds_idx, labs_idx, knowledge)
        out = train_scenario_models(df)
        trained[key] = {"dataset": df, **out}

        results = out["results"]
        # Skip-and-report path: degenerate cohort returned a sentinel result
        # with model=None. Record the metrics row so the manuscript still has
        # an entry, but don't try to save a non-existent model.
        usable = {n: r for n, r in results.items() if r.get("model") is not None}
        if not usable:
            trained[key]["primary_name"] = None
            for name, r in results.items():
                m = r["metrics"]
                rows.append({
                    "scenario": SCENARIOS[key].name, "model": name,
                    "auroc": m["auroc"], "auprc": m["auprc"], "f1": m["f1"],
                    "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                    "brier": m["brier"], "primary": False,
                })
            continue

        # pick primary model: prefer XGBoost, else best AUROC
        primary = ("XGBoost" if "XGBoost" in usable
                   else max(usable, key=lambda n: usable[n]["metrics"]["auroc"]))
        trained[key]["primary_name"] = primary
        save_model(usable[primary]["model"], key, "primary")

        for name, r in usable.items():
            m = r["metrics"]
            rows.append({
                "scenario": SCENARIOS[key].name, "model": name,
                "auroc": m["auroc"], "auprc": m["auprc"], "f1": m["f1"],
                "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                "brier": m["brier"], "primary": name == primary,
            })
    return pd.DataFrame(rows), trained
