import numpy as np
import pandas as pd
from src.features import build_features, build_dataset
from src.data import load_raw


def test_build_features_columns_and_shape():
    df = load_raw(verify_hash=False).iloc[:200].copy()
    X = build_features(df)
    assert len(X) == len(df)
    assert "pm_lag0" in X.columns
    assert "pm_lag23" in X.columns
    assert "dewp_spread" in X.columns
    assert "hour_sin" in X.columns
    assert "month_cos" in X.columns
    assert "wind_speed" in X.columns
    # All 37 expected features
    assert X.shape[1] == 37


def test_no_future_leakage():
    """Future perturbation test: modifying pm25 or weather at t+5 must NOT change features at t."""
    df1 = load_raw(verify_hash=False).iloc[:100].copy()
    df2 = df1.copy()

    # Alter future data at index 50 onwards
    df2.iloc[50:, df2.columns.get_loc("pm25")] = 999.0
    df2.iloc[50:, df2.columns.get_loc("temp")] = 45.0

    X1 = build_features(df1)
    X2 = build_features(df2)

    # All features up to row 49 must be strictly identical
    pd.testing.assert_frame_equal(X1.iloc[:50], X2.iloc[:50])


def test_calendar_features_periodicity():
    df = load_raw(verify_hash=False).iloc[:48].copy()
    X = build_features(df)
    # sin^2 + cos^2 should equal 1 for hour encodings
    hour_trig = X["hour_sin"] ** 2 + X["hour_cos"] ** 2
    np.testing.assert_allclose(hour_trig, 1.0, rtol=1e-5)

    month_trig = X["month_sin"] ** 2 + X["month_cos"] ** 2
    np.testing.assert_allclose(month_trig, 1.0, rtol=1e-5)


def test_build_dataset():
    df = load_raw(verify_hash=False).iloc[:200].copy()
    X, y, split, mask = build_dataset(df)
    assert len(X) == len(y) == len(split) == len(mask) == 200
    assert y.name == "pm25_t+1"
