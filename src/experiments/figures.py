"""Generate manuscript figures (Fig 1-5) into paper/figures/."""
from __future__ import annotations

from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import FIGURES_DIR, SCENARIOS
from src.experiments.common import build_alert_table
from src.preprocessing.build_timeline import compute_baseline, labs_for
from src.risk.risk_score import compute_risk_score


def fig1_architecture():
    blocks = ["EHR /\nSynthetic data", "Preprocessing\n+ Timeline", "Temporal\nFeatures",
              "Conflict Detection\n(Rules + ML)", "Risk Scoring", "Explanation", "Dashboard"]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    x = 0
    for i, b in enumerate(blocks):
        ax.add_patch(plt.Rectangle((x, 0), 1.6, 1.2, fill=True,
                                   facecolor="#dce8f5", edgecolor="#2c6"))
        ax.text(x + 0.8, 0.6, b, ha="center", va="center", fontsize=9)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + 1.85, 0.6), xytext=(x + 1.6, 0.6),
                        arrowprops=dict(arrowstyle="->"))
        x += 1.85
    ax.set_xlim(-0.1, x); ax.set_ylim(-0.2, 1.4); ax.axis("off")
    ax.set_title("Figure 1. System architecture", fontsize=11)
    p = FIGURES_DIR / "fig1_architecture.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig2_patient_timeline(patients, medications, labs, sample_size: int = 5000):
    """Pick a positive case where the temporal rule beats the static threshold.

    Selection logic (in order of preference):
      1. Patient where temporal alert fired *clearly earlier* than the static one
         (largest `days_to_static - days_to_temporal` gap) — the publication-quality
         storytelling case.
      2. Fall back to any temporal-beats-static case if no clear winner exists.
      3. Fall back to any positive case (worst case, may show coincident alerts).

    The x-axis is also zoomed to the clinically relevant window so the reader
    sees the drug-start → alert dynamics rather than years of unrelated history.

    Performance: We only need ONE good patient for the figure, so we sample a
    subset of the cohort (default 5000) instead of running the alert table over
    all 122k MIMIC patients. On a laptop the full pass takes 30+ minutes;
    sampling brings it down to 1-2 minutes with no loss of figure quality.
    The sampling is deterministic (seeded) so the figure is reproducible.
    """
    scenario = SCENARIOS["ace_potassium"]
    # Restrict to patients actually exposed to the drug class before sampling —
    # otherwise most samples are negative cases that yield no alert-table row.
    meds_df = (medications if not isinstance(medications, dict)
               else pd.concat(medications.values(), ignore_index=True))
    exposed_ids = set(meds_df.loc[meds_df["drug_class"] == scenario.drug_class,
                                  "patient_id"].unique())
    pts = patients[patients["patient_id"].isin(exposed_ids)]
    if len(pts) > sample_size:
        pts = pts.sample(n=sample_size, random_state=42)
    print(f"    [fig2] sampling {len(pts):,} ACE/ARB-exposed patients "
          f"(of {len(exposed_ids):,} total) for candidate selection",
          flush=True)
    at = build_alert_table("ace_potassium", pts, medications, labs)
    cand = at[(at["label"] == 1) & at["days_to_static"].notna()
              & at["days_to_temporal"].notna()].copy()
    # Order candidates by how decisively temporal beats static.
    cand["gap"] = cand["days_to_static"] - cand["days_to_temporal"]
    # Require a real gap (≥ 2 days) — otherwise the figure tells no story.
    decisive = cand[cand["gap"] >= 2].sort_values("gap", ascending=False)
    if not decisive.empty:
        row = decisive.iloc[0]
    elif not cand.empty:
        row = cand.sort_values("gap", ascending=False).iloc[0]
    else:
        pos = at[at["label"] == 1]
        if pos.empty:
            return None
        row = pos.iloc[0]
    pid = int(row["patient_id"])
    med = medications[(medications["patient_id"] == pid)
                      & (medications["drug_class"] == scenario.drug_class)].iloc[0]
    drug_start = med["start_date"]
    s = labs_for(labs, pid, "potassium")
    days = np.array([(pd.Timestamp(d) - pd.Timestamp(drug_start)).days
                     for d in s["lab_date"]])
    values = np.asarray(s["value"])

    # ── Axes: clip to the CLINICAL window first, then tighten to the data ──
    # MIMIC patients often have years of labs before/after a given drug
    # exposure (we saw a span of -1100 to +1700 days in one case).  A naïve
    # "hug the data" approach blows up the x-axis and squeezes the actual
    # alert story into a ~1% strip.  So we do this in two stages:
    #   (1) hard-clip to the clinically relevant window around drug start
    #   (2) within that window, pull the axes in to the actual data range
    #       so we don't waste space on the pre-drug-start padding either.
    clinical_lo = -30.0
    clinical_hi = float(scenario.label_window_days) + 20.0  # e.g. 30+20 = 50
    mask = (days >= clinical_lo) & (days <= clinical_hi)
    days_in = days[mask]
    values_in = values[mask]

    alert_marks = [float(d) for d in
                   (row.get("days_to_temporal"), row.get("days_to_static"))
                   if pd.notna(d)]
    if len(days_in) > 0:
        data_lo = float(np.min(days_in))
        data_hi = float(np.max(days_in))
    else:
        data_lo, data_hi = clinical_lo, clinical_hi
    # Always show at least the [-3, +5-past-last-alert] context, never wider
    # than the clinical window.
    x_lo = max(min(data_lo - 2.0, -3.0), clinical_lo)
    must_show_hi = max([data_hi] + alert_marks) if alert_marks else data_hi
    x_hi = min(must_show_hi + 5.0, clinical_hi)

    # ── Line-break across sparse measurement gaps ──────────────────────────
    # MIMIC labs can be days apart; a straight line across the gap visually
    # implies a measured trend that isn't there.  Inserting NaN tells
    # matplotlib to lift the pen between samples while still drawing markers.
    MAX_GAP_DAYS = 7
    plot_days = days_in.astype(float).copy()
    plot_vals = values_in.astype(float).copy()
    if len(plot_days) > 1:
        gaps = np.diff(plot_days)
        break_idx = np.where(gaps > MAX_GAP_DAYS)[0]
        for i in reversed(break_idx):
            plot_days = np.insert(plot_days, i + 1, np.nan)
            plot_vals = np.insert(plot_vals, i + 1, np.nan)

    # ── Y-range with head-room for annotations *inside* the plot ───────────
    y_data_min = float(np.nanmin(values_in)) if len(values_in) else 4.0
    y_data_max = float(np.nanmax(values_in)) if len(values_in) else 5.5
    y_lo = min(y_data_min, scenario.temporal_current_gate) - 0.35
    y_hi = max(y_data_max, scenario.static_threshold) + 0.65

    # ── Style palette (color-blind safe, NEJM-leaning) ─────────────────────
    C_DATA      = "#1f4e79"  # deep blue for measurements (good contrast)
    C_STATIC    = "#c0392b"  # crimson for static threshold / alert
    C_TEMPORAL  = "#e67e22"  # warm orange for temporal gate / alert
    C_DRUGSTART = "#1e8449"  # forest green for drug start
    C_TEXT      = "#2c3e50"  # dark slate for axis/title text

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Layer 1 — soft background grid
    ax.grid(True, color="#dadce0", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Layer 2 — reference threshold lines (under the data)
    ax.axhline(scenario.static_threshold, ls=(0, (6, 3)), color=C_STATIC,
               lw=1.6, zorder=2,
               label=f"Static threshold ({scenario.static_threshold} mmol/L)")
    ax.axhline(scenario.temporal_current_gate, ls=(0, (1, 2)), color=C_TEMPORAL,
               lw=1.6, zorder=2,
               label=f"Temporal gate ({scenario.temporal_current_gate} mmol/L)")

    # Layer 3 — event marker lines (drug start + alerts)
    ax.axvline(0, color=C_DRUGSTART, lw=2.2, zorder=3,
               label="Drug start (day 0)")
    if pd.notna(row["days_to_temporal"]):
        ax.axvline(row["days_to_temporal"], color=C_TEMPORAL,
                   lw=1.8, alpha=0.85, zorder=3)
    if pd.notna(row["days_to_static"]):
        ax.axvline(row["days_to_static"], color=C_STATIC,
                   lw=1.8, alpha=0.7, zorder=3)

    # Layer 4 — connecting line (dimmed; the markers carry the data)
    ax.plot(plot_days, plot_vals, "-", color=C_DATA, lw=1.4,
            alpha=0.55, zorder=4)

    # Layer 5 — measurements: white edge keeps them visible even when sitting
    # exactly on a threshold line (e.g. K=5.3 on the static threshold).
    ax.scatter(days_in, values_in, color=C_DATA, s=55,
               edgecolors="white", linewidths=1.2, zorder=10,
               label="Potassium measurement")

    # Layer 6 — annotation badges (boxed labels for the two alert events)
    # Boxed text reads cleanly against the grid AND keeps the colour-coding
    # consistent with the corresponding vertical line.
    badge = dict(boxstyle="round,pad=0.35", facecolor="white",
                 edgecolor="none", alpha=0.92)
    if pd.notna(row["days_to_temporal"]):
        d = int(row["days_to_temporal"])
        ax.annotate(f"Temporal alert\nday {d}",
                    xy=(row["days_to_temporal"], y_hi - 0.18),
                    xytext=(8, 0), textcoords="offset points",
                    ha="left", va="top",
                    color=C_TEMPORAL, fontsize=9.5, fontweight="bold",
                    bbox={**badge, "edgecolor": C_TEMPORAL, "linewidth": 1.0},
                    zorder=15)
    if pd.notna(row["days_to_static"]):
        d = int(row["days_to_static"])
        ax.annotate(f"Static alert\nday {d}",
                    xy=(row["days_to_static"], y_hi - 0.18),
                    xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="top",
                    color=C_STATIC, fontsize=9.5, fontweight="bold",
                    bbox={**badge, "edgecolor": C_STATIC, "linewidth": 1.0},
                    zorder=15)

    # ── Title with the headline result baked in ────────────────────────────
    gap_days = (int(row["days_to_static"] - row["days_to_temporal"])
                if pd.notna(row.get("days_to_static"))
                and pd.notna(row.get("days_to_temporal")) else None)
    subtitle = (f" — temporal alert {gap_days} day(s) earlier"
                if gap_days and gap_days > 0 else "")
    ax.set_title(f"Figure 2. Patient timeline (patient {pid}){subtitle}",
                 fontsize=11.5, color=C_TEXT, pad=10)
    ax.set_xlabel("Days since drug start", fontsize=11, color=C_TEXT)
    ax.set_ylabel("Potassium (mmol/L)", fontsize=11, color=C_TEXT)
    ax.tick_params(colors=C_TEXT, labelsize=10)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Spines: keep only the bottom + left for a cleaner look
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#666666")
        ax.spines[side].set_linewidth(0.8)

    # Legend placement: the gap between day-9 (last early lab) and day-30
    # (static alert) is consistently empty in this kind of timeline, so we
    # park the legend there to avoid covering any measurements.  Using a
    # bbox_to_anchor relative to the axes guarantees it stays in that
    # empty band even if matplotlib's "best" auto-placement disagrees.
    leg = ax.legend(fontsize=9, loc="center", bbox_to_anchor=(0.55, 0.28),
                    frameon=True, framealpha=0.95, edgecolor="#cccccc",
                    ncol=1)
    leg.get_frame().set_linewidth(0.6)

    p = FIGURES_DIR / "fig2_patient_timeline.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig3_risk_components(trained):
    """Risk-score component breakdown for a high-risk example."""
    key = "ace_potassium"
    df = trained[key]["dataset"]
    pos = df[df["label"] == 1]
    if pos.empty:
        pos = df
    row = pos.iloc[0].to_dict()
    rs = compute_risk_score(row, row.get("drug_risk_strength", 0.5), +1, "potassium")
    comp = rs["components"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(comp.keys()), list(comp.values()), color="#4c78a8")
    ax.set_xlabel("Normalised contribution (0-1)")
    ax.set_title(f"Figure 3. Risk-score components (score = {rs['risk_score']}, "
                 f"{rs['risk_level']})")
    p = FIGURES_DIR / "fig3_risk_components.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig4_model_performance(trained):
    """ROC + Precision-Recall curves. Uses concise scenario labels (no truncation)
    and reports AUROC / AUPRC inline in each legend entry."""
    # Short, publication-friendly labels — the prior 22-char truncation was
    # cutting "INR" off "Warfarin → Elevated INR ..." which looked unprofessional.
    short_labels = {
        "ace_potassium": "ACE/ARB → Hyperkalemia",
        "warfarin_inr":  "Warfarin → INR ↑",
        "metformin_egfr": "Metformin → eGFR ↓",
    }
    from sklearn.metrics import auc, average_precision_score

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for key, t in trained.items():
        if t is None:
            continue
        name = t.get("primary_name")
        if name is None or name not in t.get("results", {}):
            continue
        proba = t["results"][name].get("proba")
        if proba is None:
            continue
        y = t["y_test"]
        if len(np.unique(y)) < 2:
            continue
        label_base = short_labels.get(key, SCENARIOS[key].name)
        fpr, tpr, _ = roc_curve(y, proba)
        auroc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2,
                 label=f"{label_base} ({name}, AUROC={auroc:.3f})")
        prec, rec, _ = precision_recall_curve(y, proba)
        auprc = average_precision_score(y, proba)
        ax2.plot(rec, prec, lw=2,
                 label=f"{label_base} (AUPRC={auprc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax1.set_xlabel("False positive rate", fontsize=11)
    ax1.set_ylabel("True positive rate", fontsize=11)
    ax1.set_title("Figure 4a. ROC curves (XGBoost)", fontsize=11)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Figure 4b. Precision-Recall (XGBoost)", fontsize=11)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(alpha=0.3)
    p = FIGURES_DIR / "fig4_model_performance.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def fig5_shap(trained):
    from src.explainability.shap_explainer import save_global_summary
    key = "ace_potassium"
    t = trained[key]
    model = t["results"][t["primary_name"]]["model"]
    p = FIGURES_DIR / "fig5_shap_summary.png"
    return save_global_summary(model, t["X_test"], t["feature_names"], p)


def generate_all(patients, medications, labs, trained):
    return {
        "fig1": fig1_architecture(),
        "fig2": fig2_patient_timeline(patients, medications, labs),
        "fig3": fig3_risk_components(trained),
        "fig4": fig4_model_performance(trained),
        "fig5": fig5_shap(trained),
    }
