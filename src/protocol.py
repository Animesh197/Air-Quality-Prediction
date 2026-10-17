"""Forecasting protocol: target, splits, evaluation mask and backtest folds.

Setup (fixed in Stage 04, before any model is fitted):
- Issuance time t: the end of hour t, once all hour-t observations are available.
- Target: PM2.5 at hour t + HORIZON (ug/m^3). One sample per issuance time t.
- Inputs may use only observations at hours <= t (no future weather).
- A sample belongs to the split that contains its *target* hour, so no
  training label ever lies inside the validation or test period.
"""

import numpy as np
import pandas as pd

HORIZON = 1          # hours ahead
SEASONAL_PERIOD = 24  # daily cycle (EDA Stage 03); used by seasonal-naive

# Inclusive target-time boundaries. Whole calendar years keep every season in each split.
SPLITS = {
    "train": ("2010-01-01", "2012-12-31 23:00"),
    "val":   ("2013-01-01", "2013-12-31 23:00"),
    "test":  ("2014-01-01", "2014-12-31 23:00"),
}


def make_target(pm, horizon=HORIZON):
    """Target aligned to issuance time t: y[t] = pm[t + horizon]."""
    return pm.shift(-horizon).rename(f"pm25_t+{horizon}")


def target_time(index, horizon=HORIZON):
    """Timestamp of the target hour for each issuance time."""
    return index + pd.Timedelta(hours=horizon)


def assign_split(index, horizon=HORIZON):
    """Label each issuance time with the split containing its target hour (NaN if none)."""
    tt = target_time(index, horizon)
    labels = pd.Series(np.nan, index=index, dtype="object", name="split")
    for name, (start, end) in SPLITS.items():
        labels[(tt >= pd.Timestamp(start)) & (tt <= pd.Timestamp(end))] = name
    return labels


def eval_mask(pm, horizon=HORIZON, period=SEASONAL_PERIOD):
    """Common evaluation timestamps: the target and both naive-baseline inputs are observed.

    Persistence uses pm[t]; seasonal-naive uses pm[t + horizon - period].
    Every model is scored on exactly these timestamps so comparisons are fair.
    """
    y = make_target(pm, horizon)
    return (y.notna() & pm.notna() & pm.shift(period - horizon).notna()).rename("eval")


def backtest_folds(start, end, freq="MS", horizon=HORIZON):
    """Expanding-window folds over target times in [start, end].

    Each fold tests one period (default: one calendar month) and may train on
    all samples whose target time is earlier than that period's start.
    Returns a list of dicts with `train_end` (last allowed target time) and
    `test_start`/`test_end` (target times).
    """
    starts = pd.date_range(start, end, freq=freq)
    folds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] - pd.Timedelta(hours=1) if i + 1 < len(starts) else pd.Timestamp(end)
        folds.append({"fold": i + 1, "train_end": s - pd.Timedelta(hours=1),
                      "test_start": s, "test_end": e})
    return folds


def fold_masks(index, fold, horizon=HORIZON):
    """Boolean train/test masks over issuance times for one backtest fold."""
    tt = pd.Series(target_time(index, horizon), index=index)
    train = tt <= fold["train_end"]
    test = (tt >= fold["test_start"]) & (tt <= fold["test_end"])
    return train, test
