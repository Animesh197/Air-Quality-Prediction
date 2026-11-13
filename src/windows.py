"""Windowed sequences for the LSTM forecaster (Stage 09).

The LSTM reads the last LOOKBACK hours of per-hour channels ending at the
issuance time t and predicts PM2.5 at t + 1 - the same target as Phase 1. The
task is kept identical to the ML phase so the two can be scored on the same
protocol and eval set; only the inputs change (raw hourly history instead of the
hand-made lag/rolling features).

An LSTM cannot read NaN, so pm25 is forward-filled and a missing flag is added.
The weather columns have no missing values (see src.data.validate). Continuous
channels are standardised with training-period statistics only.
"""

import numpy as np
import pandas as pd

from src.data import WIND_DIRS
from src.protocol import assign_split, eval_mask, make_target

LOOKBACK = 24   # hours of history fed to the LSTM (tuned later on 2013)

# Channels standardised with training stats; the 0/1 flags and one-hots are left as is.
SCALE_CHANNELS = ["pm25", "dewp", "temp", "pres", "wind_speed", "snow_cum", "rain_cum"]


def build_channels(df):
    """Per-hour input channels for the LSTM, indexed by hour t."""
    out = pd.DataFrame(index=df.index)

    out["pm25"] = df["pm25"].ffill()                 # carry the last reading forward
    out["pm25_missing"] = df["pm25"].isna().astype(int)   # 1 where pm25 was imputed

    out["dewp"], out["temp"], out["pres"] = df["dewp"], df["temp"], df["pres"]

    # wind_cum is cumulated speed since the direction last changed: decode to hourly speed
    same_dir = df["wind_dir"].eq(df["wind_dir"].shift())
    out["wind_speed"] = df["wind_cum"].diff().where(same_dir, df["wind_cum"])
    for d in WIND_DIRS:
        out[f"wind_{d}"] = (df["wind_dir"] == d).astype(int)

    out["snow_cum"], out["rain_cum"] = df["snow_cum"], df["rain_cum"]

    # time of day as a smooth cycle so each window carries the daily rhythm
    out["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    return out


def standardize(channels, train_hours):
    """Standardise SCALE_CHANNELS using training-hour statistics only.

    Returns the scaled channels and the (mean, std) used, so the exact transform
    can be reapplied inside each backtest fold. Leading pm25 NaNs (before the
    first reading) are filled with the training mean, i.e. 0 after scaling.
    """
    out = channels.copy()
    stats = {}
    for c in SCALE_CHANNELS:
        mu = channels.loc[train_hours, c].mean()
        sd = channels.loc[train_hours, c].std()
        sd = sd if sd and sd > 0 else 1.0
        out[c] = (channels[c].fillna(mu) - mu) / sd
        stats[c] = (float(mu), float(sd))
    return out, stats


def make_windows(channels, pm, lookback=LOOKBACK):
    """Stack the LOOKBACK-hour windows into a 3D tensor.

    X[i] holds hours t-lookback+1 .. t (inclusive) and y[i] = pm(t+1), where t is
    the issuance time. The first lookback-1 hours have no full history and are
    dropped. Rows whose target is NaN are kept and left to the caller's mask, so
    the index still lines up with the Phase 1 tables.
    """
    values = channels.to_numpy(dtype="float32")
    n = len(values) - lookback + 1
    X = np.empty((n, lookback, values.shape[1]), dtype="float32")
    for i in range(n):
        X[i] = values[i:i + lookback]

    index = channels.index[lookback - 1:]
    y = make_target(pm).to_numpy(dtype="float32")[lookback - 1:]
    return X, y, pd.DatetimeIndex(index, name="timestamp")


def build_sequence_dataset(df, lookback=LOOKBACK):
    """Full LSTM dataset aligned on issuance time t.

    Returns (X, y, split, mask, stats): the scaled window tensor, the t+1 target,
    split labels and the Phase 1 common eval mask (same hours as every other
    model), plus the scaling stats. Scaling uses training-period stats only.
    """
    channels = build_channels(df)
    train_hours = assign_split(df.index) == "train"
    scaled, stats = standardize(channels, train_hours)

    X, y, index = make_windows(scaled, df["pm25"], lookback)
    split = assign_split(index)
    mask = eval_mask(df["pm25"]).reindex(index)
    return X, y, split, mask, stats
