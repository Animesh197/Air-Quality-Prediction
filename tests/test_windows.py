import numpy as np
import pandas as pd
from src.data import load_raw
from src.windows import (
    SCALE_CHANNELS,
    build_channels,
    build_sequence_dataset,
    make_windows,
    standardize,
)


def test_sequence_dataset_shapes_and_alignment():
    df = load_raw(verify_hash=False).iloc[:500].copy()
    X, y, split, mask, stats = build_sequence_dataset(df, lookback=24)
    # 3D window tensor: (samples, lookback, channels); 14 channels per hour.
    assert X.ndim == 3
    assert X.shape[1] == 24
    assert X.shape[2] == 14
    # Every aligned array has the same length and shares the issuance index.
    assert len(X) == len(y) == len(split) == len(mask)
    assert mask.index.name == "timestamp"
    assert set(SCALE_CHANNELS).issubset(stats.keys())


def test_no_future_leakage_in_windows():
    """Corrupting pm25 at a future hour must not change any earlier input window.

    This encodes the leakage claim made in the report (phase2.tex, Section IV):
    the window for issuance time t ends at hour t, so a change at t+1 can only
    affect windows that include t+1, never an earlier one, and it changes that
    earlier window's *target* instead.
    """
    df1 = load_raw(verify_hash=False).iloc[:200].copy()
    df2 = df1.copy()
    r = 150                                   # a future hour to corrupt
    df2.iloc[r, df2.columns.get_loc("pm25")] = 9999.0

    X1, y1, idx1 = make_windows(build_channels(df1), df1["pm25"], lookback=24)
    X2, y2, idx2 = make_windows(build_channels(df2), df2["pm25"], lookback=24)

    # Issuance time of window i is row i+23; windows ending before row r are those
    # with i+23 < r, i.e. i < r-23 = 127. Their inputs must be byte-identical.
    k = r - 23
    np.testing.assert_array_equal(X1[:k], X2[:k])
    assert idx1.equals(idx2)

    # The window whose target hour is exactly r (i = r-24 = 126) sees only its
    # target change, to the corrupted value.
    i_target = r - 24
    assert np.isclose(y2[i_target], 9999.0)
    assert not np.isclose(y1[i_target], y2[i_target])


def test_standardize_uses_training_statistics_only():
    """Scaling stats come from the training hours; future rows never enter them."""
    channels = build_channels(load_raw(verify_hash=False).iloc[:200].copy())
    train_hours = pd.Series(False, index=channels.index)
    train_hours.iloc[:150] = True

    scaled, stats = standardize(channels, train_hours)

    # Weather channels have no NaNs, so after scaling the training rows are ~N(0,1).
    for c in ["temp", "pres", "dewp"]:
        mu, sd = stats[c]
        assert np.isclose(mu, channels.loc[train_hours, c].mean())
        assert np.isclose(sd, channels.loc[train_hours, c].std())
        np.testing.assert_allclose(scaled.loc[train_hours, c].mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(scaled.loc[train_hours, c].std(), 1.0, atol=1e-6)

    # Changing only the non-training (future) rows must leave the stats unchanged.
    future = channels.copy()
    future.iloc[150:, future.columns.get_loc("temp")] = 999.0
    _, stats2 = standardize(future, train_hours)
    assert np.isclose(stats["temp"][0], stats2["temp"][0])
    assert np.isclose(stats["temp"][1], stats2["temp"][1])


def test_window_covers_lookback_hours_up_to_issuance():
    """X[i] is exactly the lookback rows of the channels ending at the issuance hour."""
    channels = build_channels(load_raw(verify_hash=False).iloc[:60].copy())
    X, y, idx = make_windows(channels, channels["pm25"] * 0, lookback=12)
    values = channels.to_numpy(dtype="float32")
    # First window = channel rows 0..11; its issuance time is row 11.
    np.testing.assert_array_equal(X[0], values[0:12])
    assert idx[0] == channels.index[11]
