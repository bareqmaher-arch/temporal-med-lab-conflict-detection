"""Prepare public-safe versions of the result artifacts for the open-source repo.

Why this exists
---------------
The MIMIC-IV DUA prohibits redistributing patient-level data even when
the identifiers are surrogate. Three of our generated files contain the
12 MIMIC subject_ids that were chosen as worked examples for Table 7 of
the manuscript, paired with their lab values and explanations:

  - paper/tables/table5_example_explanations.csv
  - paper/tables/manuscript_data.json
  - outputs/explainability_questionnaire.csv

Mentioning ONE subject_id in narrative prose for a case study is the
accepted norm in MIMIC publications (and we keep that in the manuscript
for patient 14866589 in Figure 2). But shipping a machine-readable CSV
that joins 12 subject_ids to their full clinical timeline is closer to
"data sharing" than "publication," so we replace those IDs with neutral
"Case N" labels before committing.

The originals stay on disk under `_local_only/` (excluded by .gitignore),
so re-running the pipeline still produces the real CSVs locally.

Run:
    python scripts/prepare_public_artifacts.py
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
LOCAL_BACKUP = ROOT / "_local_only" / "originals_with_real_ids"
LOCAL_BACKUP.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ROOT / "paper" / "tables" / "table5_example_explanations.csv",
    ROOT / "paper" / "tables" / "manuscript_data.json",
    ROOT / "outputs" / "explainability_questionnaire.csv",
]

# MIMIC-IV subject_ids are 8-digit integers starting with 1.
# We collect every unique ID across the three files and assign each
# a stable Case label so the same patient gets the same Case-N across
# all three artifacts.
ID_RE = re.compile(r"\b1[0-9]{7}\b")


def backup_then_collect_ids() -> dict[str, str]:
    """Copy originals to _local_only/, collect every unique ID."""
    seen: set[str] = set()
    for path in TARGETS:
        if not path.exists():
            print(f"  [skip] {path.name} not present")
            continue
        # Backup original
        backup = LOCAL_BACKUP / path.name
        shutil.copy2(path, backup)
        print(f"  [backup] {path.name} -> _local_only/originals_with_real_ids/")
        # Collect IDs
        text = path.read_text(encoding="utf-8")
        for m in ID_RE.findall(text):
            seen.add(m)
    # Stable order: numerical sort -> Case 1, 2, 3 ...
    ordered = sorted(seen, key=int)
    mapping = {pid: f"Case {i+1}" for i, pid in enumerate(ordered)}
    print(f"  Collected {len(ordered)} unique MIMIC subject_ids")
    return mapping


def anonymize_file(path: Path, mapping: dict[str, str]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    # Replace longest first so we don't get partial overlaps (all 8 digits,
    # but the principle is right)
    for pid in sorted(mapping, key=len, reverse=True):
        text = text.replace(pid, mapping[pid])
    path.write_text(text, encoding="utf-8")
    print(f"  [anon] {path.relative_to(ROOT)}")


def main() -> int:
    print("Step 1: backup originals and collect subject_ids")
    mapping = backup_then_collect_ids()
    if not mapping:
        print("  No subject_ids found - nothing to do.")
        return 0
    print()
    print("Step 2: anonymize in-place")
    for path in TARGETS:
        anonymize_file(path, mapping)
    print()
    print("Step 3: write the mapping (LOCAL ONLY) so we can reverse if needed")
    map_path = LOCAL_BACKUP / "id_map.json"
    map_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"  Mapping saved: {map_path.relative_to(ROOT)}")
    print()
    print("Done. The three artifacts now use 'Case N' labels. The originals")
    print("are backed up under _local_only/, which is excluded by .gitignore.")
    print("Re-running the pipeline locally will regenerate the real CSVs;")
    print("re-run this script before committing if that happens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
