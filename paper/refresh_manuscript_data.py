"""Rebuild paper/tables/manuscript_data.json from the latest CSVs + metrics.json.

Why this exists
---------------
manuscript_data.json was originally created from the synthetic 900-patient
proof-of-concept run. After we re-ran the pipeline on the full 122,166-patient
MIMIC-IV cohort, the CSVs in paper/tables/ were refreshed but the consolidated
JSON used by build_manuscript.js was not. This helper keeps them in sync.

It only touches the result tables (exp1, table3, table4, table5). The static
config tables (table1, table2) are taken from their CSVs too, so nothing in
build_manuscript.js needs to keep its own copy.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TABLES = ROOT / "tables"

# Force UTF-8 on Windows consoles so the arrow in scenario names doesn't crash
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _read_csv(name: str) -> list[dict]:
    with (TABLES / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cast(rows: list[dict], int_cols=(), float_cols=()) -> list[dict]:
    out = []
    for r in rows:
        rr = dict(r)
        for c in int_cols:
            if c in rr and rr[c] != "":
                rr[c] = int(rr[c])
        for c in float_cols:
            if c in rr and rr[c] != "":
                try:
                    rr[c] = float(rr[c])
                except ValueError:
                    pass
        # booleans encoded as "True"/"False"
        for k, v in list(rr.items()):
            if v in ("True", "False"):
                rr[k] = (v == "True")
        out.append(rr)
    return out


metrics = json.loads((ROOT.parent / "outputs" / "metrics.json").read_text(encoding="utf-8"))

bundle = {
    "table1_scenarios": _read_csv("table1_scenarios.csv"),
    "table2_feature_categories": _read_csv("table2_feature_categories.csv"),
    "exp1_rules_vs_temporal": _cast(
        _read_csv("exp1_rules_vs_temporal.csv"),
        float_cols=("sensitivity", "specificity", "precision", "recall", "f1",
                    "false_alert_rate", "alerts_per_100", "false_alerts_per_100"),
    ),
    "table3_model_comparison": _cast(
        _read_csv("table3_model_comparison.csv"),
        float_cols=("auroc", "auprc", "f1", "sensitivity", "specificity", "brier"),
    ),
    "table4_early_detection": _cast(
        _read_csv("table4_early_detection.csv"),
        int_cols=("n_positive", "n_both_alerted"),
        float_cols=("mean_static_alert_day", "mean_temporal_alert_day",
                    "mean_days_earlier", "median_days_earlier", "alert_reduction_pct"),
    ),
    "table5_example_explanations": _cast(
        _read_csv("table5_example_explanations.csv"),
        float_cols=("baseline", "current", "risk_score"),
    ),
    # Cohort-level numbers carried through from the pipeline run
    "n_patients": metrics["n_patients"],
    "n_alerts": metrics["n_alerts"],
}

out_path = TABLES / "manuscript_data.json"
out_path.write_text(json.dumps(bundle, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Rewrote {out_path}  ({len(bundle['table5_example_explanations'])} explanations, "
      f"n_patients={bundle['n_patients']:,}, n_alerts={bundle['n_alerts']:,})")
