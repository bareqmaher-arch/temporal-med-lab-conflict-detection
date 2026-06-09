"""Classification metrics for model and rule evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score,
)


def _safe_auc(y_true, score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def binary_metrics(y_true, y_pred) -> dict:
    """Sensitivity/specificity/precision/recall/F1 from hard predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "precision": round(prec, 4), "recall": round(sens, 4),
        "f1": round(f1, 4), "accuracy": round(acc, 4),
    }


def evaluate_predictions(y_true, proba, pred) -> dict:
    m = binary_metrics(y_true, pred)
    m["auroc"] = round(_safe_auc(y_true, proba), 4)
    try:
        m["auprc"] = round(float(average_precision_score(y_true, proba)), 4)
        m["brier"] = round(float(brier_score_loss(y_true, proba)), 4)
    except Exception:
        m["auprc"] = float("nan")
        m["brier"] = float("nan")
    return m
