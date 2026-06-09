"""SHAP explanations for the ML model (global summary + per-instance top features)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _unwrap(model, X: pd.DataFrame):
    """Return (tree_estimator, transformed_X) for SHAP TreeExplainer."""
    if isinstance(model, Pipeline):
        Xt = X
        for _, step in model.steps[:-1]:
            Xt = step.transform(Xt)
        return model.steps[-1][1], Xt
    return model, X


def compute_shap(model, X: pd.DataFrame):
    import shap
    est, Xt = _unwrap(model, X)
    explainer = shap.TreeExplainer(est)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):          # some versions return per-class lists
        values = values[1]
    return np.asarray(values), Xt


def save_global_summary(model, X: pd.DataFrame, feature_names, path: Path) -> Path | None:
    """Save a publication-quality SHAP global summary (beeswarm) plot.

    The previous version called `plt.figure()` *and* passed `plot_size=` to
    `shap.summary_plot()` — SHAP creates its own figure when `plot_size` is set,
    so the pre-created blank figure was the one being saved, leaving a tiny
    300px-wide thumbnail. We now let SHAP own the figure and grab the active
    one for sizing/DPI control.
    """
    try:
        import shap
        values, Xt = compute_shap(model, X)
        # Let SHAP build the figure at a generous size — matches the other
        # manuscript figures (~10 inches wide @ 200 dpi).
        shap.summary_plot(
            values,
            features=np.asarray(Xt),
            feature_names=feature_names,
            show=False,
            plot_size=(11, 7),
        )
        fig = plt.gcf()
        fig.suptitle("Figure 5. SHAP global feature importance (XGBoost — ACE/ARB → K⁺)",
                     fontsize=11, y=1.02)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close("all")
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[shap] global summary skipped: {exc}")
        return None


def top_features_for_instance(model, X_row: pd.DataFrame, feature_names, k: int = 5):
    """Return list of (feature, shap_value) sorted by absolute contribution."""
    try:
        values, _ = compute_shap(model, X_row)
        row = values[0] if values.ndim == 2 else values
        order = np.argsort(np.abs(row))[::-1][:k]
        return [(feature_names[i], float(row[i])) for i in order]
    except Exception as exc:  # pragma: no cover
        print(f"[shap] instance explanation skipped: {exc}")
        return []
