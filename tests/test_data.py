import pytest
import pandas as pd
import numpy as np

from src.data import RAW_PATH, RAW_SHA256, file_sha256, load_raw, validate, missing_runs, RENAME, WIND_DIRS


def test_file_sha256_matches_recorded_hash():
    assert file_sha256(RAW_PATH) == RAW_SHA256


def test_load_raw_structure():
    df = load_raw(RAW_PATH, verify_hash=True)
    assert len(df) == 43824
    assert df.index.name == "timestamp"
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert list(df.columns) == list(RENAME.values())
    assert list(df["wind_dir"].cat.categories) == WIND_DIRS


def test_validate_passes_on_raw_data():
    df = load_raw(RAW_PATH, verify_hash=False)
    assert validate(df) is True


def test_validate_detects_duplicate_index():
    df = load_raw(RAW_PATH, verify_hash=False).iloc[:10].copy()
    df.index = [df.index[0]] * len(df)
    with pytest.raises(AssertionError, match="duplicate timestamps"):
        validate(df)


def test_validate_detects_negative_pm25():
    df = load_raw(RAW_PATH, verify_hash=False).iloc[:10].copy()
    df.loc[df.index[0], "pm25"] = -5.0
    with pytest.raises(AssertionError, match="negative PM2.5"):
        validate(df)


def test_validate_detects_invalid_pressure():
    df = load_raw(RAW_PATH, verify_hash=False).iloc[:10].copy()
    df.loc[df.index[0], "pres"] = 800.0
    with pytest.raises(AssertionError, match="pressure outside 900-1100 hPa"):
        validate(df)


def test_missing_runs():
    idx = pd.date_range("2020-01-01 00:00", periods=6, freq="h")
    s = pd.Series([10.0, np.nan, np.nan, 20.0, np.nan, 30.0], index=idx)
    runs = missing_runs(s)
    assert len(runs) == 2
    assert runs.loc[0, "hours"] == 2
    assert runs.loc[0, "start"] == pd.Timestamp("2020-01-01 01:00")
    assert runs.loc[0, "end"] == pd.Timestamp("2020-01-01 02:00")
    assert runs.loc[1, "hours"] == 1
    assert runs.loc[1, "start"] == pd.Timestamp("2020-01-01 04:00")
