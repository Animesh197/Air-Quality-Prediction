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


def diebold_mariano(e1, e2, loss="squared", max_lag=24):
    """Diebold-Mariano test of equal predictive accuracy (Diebold & Mariano, 1995).

    d_t = L(e1_t) - L(e2_t). The variance of mean(d) uses a Newey-West (Bartlett)
    long-run variance with `max_lag` lags, because hourly loss differentials are
    autocorrelated. Negative statistic => model 1 has lower loss.
    Returns (DM statistic, two-sided p-value from the normal approximation).
    """
    from scipy.stats import norm
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    f = np.square if loss == "squared" else np.abs
    d = f(e1) - f(e2)
    n, dc = len(d), d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, max_lag + 1):
        lrv += 2 * (1 - k / (max_lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    return float(stat), float(2 * norm.sf(abs(stat)))


def score_by_split(y, preds, split, mask, splits):
    """Metrics per (model, split) on the common eval set. `preds` maps name -> Series."""
    rows = []
    for name, p in preds.items():
        for s in splits:
            m = mask & (split == s)
            rows.append({"model": name, "split": s, **metrics(y[m], p[m])})
    return pd.DataFrame(rows)


def backtest(make_model, X, y, mask, folds):
    """Expanding-window backtest: refit `make_model()` per fold on targets before the fold.

    Training uses every row with an observed target up to the fold's `train_end`
    (the model decides which of those it can use); scoring uses the common eval
    set inside the fold's test window. Returns (per-fold metrics, predictions).
    """
    rows, preds = [], []
    for f in folds:
        train, test = fold_masks(X.index, f)
        model = make_model().fit(X[train], y[train])
        p = model.predict(X[test & mask])
        preds.append(p)
        rows.append({"fold": f["fold"], "test_start": f["test_start"].date(), "test_end": f["test_end"].date(),
                     "n_train": model.n_train_, **metrics(y[test & mask], p)})
    return pd.DataFrame(rows), pd.concat(preds)


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
