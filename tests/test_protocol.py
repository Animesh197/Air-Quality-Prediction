import pandas as pd
import numpy as np

from src.protocol import (
    HORIZON,
    SEASONAL_PERIOD,
    SPLITS,
    make_target,
    target_time,
    assign_split,
    eval_mask,
    backtest_folds,
    fold_masks,
)


def test_make_target_and_target_time():
    idx = pd.date_range("2012-01-01 00:00", periods=5, freq="h")
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)

    y = make_target(s, horizon=1)
    assert y.name == "pm25_t+1"
    assert y.iloc[0] == 20.0
    assert y.iloc[3] == 50.0
    assert np.isnan(y.iloc[4])

    tt = target_time(idx, horizon=1)
    assert tt[0] == pd.Timestamp("2012-01-01 01:00")
    assert tt[-1] == pd.Timestamp("2012-01-01 05:00")


def test_assign_split_boundaries():
    idx = pd.to_datetime([
        "2009-12-31 23:00",  # target is 2010-01-01 00:00 -> train
        "2012-12-31 22:00",  # target is 2012-12-31 23:00 -> train
        "2012-12-31 23:00",  # target is 2013-01-01 00:00 -> val
        "2013-12-31 22:00",  # target is 2013-12-31 23:00 -> val
        "2013-12-31 23:00",  # target is 2014-01-01 00:00 -> test
        "2014-12-31 22:00",  # target is 2014-12-31 23:00 -> test
        "2014-12-31 23:00",  # target is 2015-01-01 00:00 -> outside
    ])
    labels = assign_split(idx, horizon=1)
    assert labels.iloc[0] == "train"
    assert labels.iloc[1] == "train"
    assert labels.iloc[2] == "val"
    assert labels.iloc[3] == "val"
    assert labels.iloc[4] == "test"
    assert labels.iloc[5] == "test"
    assert pd.isna(labels.iloc[6])


def test_eval_mask():
    idx = pd.date_range("2013-01-01 00:00", periods=50, freq="h")
    pm = pd.Series(10.0, index=idx)
    # Target requires next hour; seasonal-naive requires t - 23 hours
    mask = eval_mask(pm, horizon=1, period=24)
    # The first 23 hours should be False because pm(t-23) is unavailable
    assert not mask.iloc[0:23].any()
    # At t=23 (hour 23), pm(t-23) = pm(0) is available, target pm(24) is available
    assert mask.iloc[23]
    # The last hour target t+1 is unavailable, so False
    assert not mask.iloc[-1]


def test_backtest_folds():
    folds = backtest_folds("2014-01-01", "2014-04-30 23:00", freq="MS")
    assert len(folds) == 4
    # First fold
    assert folds[0]["fold"] == 1
    assert folds[0]["train_end"] == pd.Timestamp("2013-12-31 23:00")
    assert folds[0]["test_start"] == pd.Timestamp("2014-01-01 00:00")
    assert folds[0]["test_end"] == pd.Timestamp("2014-01-31 23:00")
    # Expanding nature: fold 2 trains up through fold 1 test
    assert folds[1]["train_end"] == pd.Timestamp("2014-01-31 23:00")
