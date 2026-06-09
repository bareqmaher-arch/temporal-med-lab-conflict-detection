"""Train and compare ML models per scenario (LogReg, RF, XGBoost, LightGBM)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import N_JOBS, RANDOM_SEED, XGBOOST_USE_GPU
from src.models.dataset import ML_FEATURE_COLUMNS
from src.models.evaluate import evaluate_predictions

warnings.filterwarnings("ignore")


def _make_models():
    models = {
        "LogisticRegression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            # saga solver converges fast on standardised features and handles
            # large N gracefully. n_jobs was removed in sklearn 1.8 — the
            # solver itself is single-threaded but the surrounding pipeline
            # (joblib outer parallelism) saturates cores at the experiment level.
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       solver="saga", random_state=RANDOM_SEED)),
        ]),
        "RandomForest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=RANDOM_SEED, n_jobs=N_JOBS)),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        xgb_kwargs = dict(
            n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_SEED,
            n_jobs=N_JOBS, tree_method="hist",  # 5-10x faster than "exact"
        )
        if XGBOOST_USE_GPU:
            # XGBoost 2.x prefers `device="cuda"`; older builds use the deprecated
            # `tree_method="gpu_hist"`. Try the new API first.
            try:
                models["XGBoost"] = XGBClassifier(device="cuda", **xgb_kwargs)
                # Quick GPU sanity check happens at .fit() — if CUDA isn't
                # available XGBoost falls back to CPU with a warning, which is
                # the behaviour we want.
            except TypeError:
                xgb_kwargs["tree_method"] = "gpu_hist"
                models["XGBoost"] = XGBClassifier(**xgb_kwargs)
        else:
            models["XGBoost"] = XGBClassifier(**xgb_kwargs)
    except Exception:  # pragma: no cover
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, random_state=RANDOM_SEED, n_jobs=N_JOBS, verbose=-1,
        )
    except Exception:  # pragma: no cover
        pass
    return models


def train_scenario_models(df: pd.DataFrame, test_size: float = 0.3):
    X = df[ML_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(int).to_numpy()

    # Degenerate-cohort guard: small smoke-test runs (e.g. MIMIC_MAX_PATIENTS=500)
    # sometimes contain zero positives for the rarer scenarios. Classifiers cannot
    # be trained on a single-class target — return a sentinel result so the
    # pipeline finishes and the user sees a clear message instead of a crash.
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 2 or n_neg < 2:
        print(f"    [train] SKIPPING ML training: only {n_pos} positives / "
              f"{n_neg} negatives in {len(y)} samples. "
              f"Increase MIMIC_MAX_PATIENTS for meaningful ML metrics.", flush=True)
        empty_metrics = {
            "auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
            "sensitivity": float("nan"), "specificity": float("nan"),
            "brier": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "accuracy": float("nan"), "n_pos": n_pos, "n_neg": n_neg,
            "note": "insufficient class balance — increase cohort size",
        }
        return {
            "results": {"LogisticRegression": {
                "model": None, "metrics": empty_metrics,
                "proba": np.array([]), "pred": np.array([]),
            }},
            "X_test": X.iloc[:0], "y_test": np.array([], dtype=int),
            "test_index": np.array([], dtype=int),
            "feature_names": ML_FEATURE_COLUMNS,
            "X_train": X.iloc[:0], "y_train": np.array([], dtype=int),
            "skipped_reason": f"only_one_class (pos={n_pos}, neg={n_neg})",
        }

    stratify = y if (y.sum() >= 2 and (len(y) - y.sum()) >= 2) else None
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=test_size, random_state=RANDOM_SEED, stratify=stratify)

    results = {}
    for name, model in _make_models().items():
        try:
            model.fit(X_tr, y_tr)
        except ValueError as e:
            # Individual model can still fail (e.g. XGBoost on tiny test sets) —
            # log and continue with the other models rather than aborting the run.
            print(f"    [train] {name} failed: {e}", flush=True)
            continue
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "model": model,
            "metrics": evaluate_predictions(y_te, proba, pred),
            "proba": proba, "pred": pred,
        }
    return {
        "results": results, "X_test": X_te, "y_test": y_te,
        "test_index": np.array(idx_te), "feature_names": ML_FEATURE_COLUMNS,
        "X_train": X_tr, "y_train": y_tr,
    }
