"""Metrics and scoring helpers shared by all models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.protocol import fold_masks

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def metrics(y_true, y_pred):
    """MAE, RMSE (ug/m^3) and R^2 on the original scale. Inputs must contain no NaN."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("NaN in y_true or y_pred: score only on the common eval set")
    return {"n": len(y_true),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}


def score_by_split(y, preds, split, mask, splits):
    """Metrics per (model, split) on the common eval set. `preds` maps name -> Series."""
    rows = []
    for name, p in preds.items():
        for s in splits:
            m = mask & (split == s)
            rows.append({"model": name, "split": s, **metrics(y[m], p[m])})
    return pd.DataFrame(rows)


def score_by_fold(y, preds, mask, folds):
    """Metrics per (model, fold) on the common eval set of each fold's test window."""
    rows = []
    for f in folds:
        _, te = fold_masks(y.index, f)
        m = te & mask
        for name, p in preds.items():
            rows.append({"model": name, "fold": f["fold"], "test_start": f["test_start"].date(),
                         "test_end": f["test_end"].date(), **metrics(y[m], p[m])})
    return pd.DataFrame(rows)
