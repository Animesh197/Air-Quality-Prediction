import pandas as pd
from src.baselines import persistence, seasonal_naive


def test_persistence_baseline():
    idx = pd.date_range("2012-01-01 00:00", periods=5, freq="h")
    pm = pd.Series([10.0, 25.0, 30.0, 15.0, 40.0], index=idx)
    pred = persistence(pm)
    assert pred.name == "persistence"
    pd.testing.assert_series_equal(pred, pm.rename("persistence"))


def test_seasonal_naive_baseline():
    idx = pd.date_range("2012-01-01 00:00", periods=48, freq="h")
    pm = pd.Series(range(48), index=idx, dtype=float)
    # Target is t+1; seasonal-naive uses pm(t - 23)
    pred = seasonal_naive(pm, period=24, horizon=1)
    assert pred.name == "seasonal_naive_24h"
    # At t = 23 (hour 23), pred should be pm(23 - 23) = pm(0) = 0.0
    assert pred.iloc[23] == 0.0
    # At t = 24 (hour 24), pred should be pm(24 - 23) = pm(1) = 1.0
    assert pred.iloc[24] == 1.0
