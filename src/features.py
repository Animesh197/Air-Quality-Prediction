"""Feature engineering for the t+1 PM2.5 forecast (Stage 05).

Every feature for issuance time t uses only rows at hours <= t, except the
calendar features, which describe the target hour t + HORIZON (known in advance).
Missing PM2.5 inputs stay NaN; nothing is interpolated or imputed.
"""

import numpy as np
import pandas as pd

from src.data import WIND_DIRS
from src.protocol import HORIZON, assign_split, eval_mask, make_target

PM_LAGS = [0, 1, 2, 3, 6, 12, 23]   # pm(t - k); 23 = same hour as the target, one day earlier
ROLL_WINDOWS = [3, 6, 12, 24]       # trailing windows ending at t (inclusive)


def _pm_features(pm):
    out = {f"pm_lag{k}": pm.shift(k) for k in PM_LAGS}
    out["pm_diff1"] = pm - pm.shift(1)
    for w in ROLL_WINDOWS:
        # Trailing window over hours t-w+1 .. t; require at least half the window observed
        r = pm.rolling(w, min_periods=max(2, w // 2))
        out[f"pm_mean{w}"] = r.mean()
        out[f"pm_std{w}"] = r.std()
    # Hours since PM2.5 was last observed (0 if observed at t): tells the model how stale the lags are
    last_obs = pd.Series(pm.index.where(pm.notna()), index=pm.index).ffill()
    out["pm_hours_since_obs"] = (pm.index - last_obs).dt.total_seconds() / 3600
    return pd.DataFrame(out, index=pm.index)


def _weather_features(df):
    out = pd.DataFrame(index=df.index)
    out["dewp"], out["temp"], out["pres"] = df["dewp"], df["temp"], df["pres"]
    out["dewp_spread"] = df["temp"] - df["dewp"]        # small spread = humid, stagnant air
    out["pres_diff3"] = df["pres"] - df["pres"].shift(3)  # pressure tendency (fronts)
    out["dewp_diff3"] = df["dewp"] - df["dewp"].shift(3)

    # wind_cum is cumulated speed since the direction last changed: decode to hourly speed
    same_dir = df["wind_dir"].eq(df["wind_dir"].shift())
    out["wind_speed"] = df["wind_cum"].diff().where(same_dir, df["wind_cum"])
    out["wind_cum"] = df["wind_cum"]
    run_id = (~same_dir).cumsum()
    out["wind_dir_hours"] = df.groupby(run_id).cumcount() + 1   # hours the current direction has persisted
    for d in WIND_DIRS:
        out[f"wind_{d}"] = (df["wind_dir"] == d).astype(int)

    out["snow_cum"], out["rain_cum"] = df["snow_cum"], df["rain_cum"]
    return out


def _calendar_features(index, horizon=HORIZON):
    tt = index + pd.Timedelta(hours=horizon)   # calendar of the *target* hour
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * tt.hour / 24), "hour_cos": np.cos(2 * np.pi * tt.hour / 24),
        "month_sin": np.sin(2 * np.pi * (tt.month - 1) / 12), "month_cos": np.cos(2 * np.pi * (tt.month - 1) / 12),
        "dow": tt.dayofweek,
    }, index=index)


def build_features(df):
    """Return the feature matrix X indexed by issuance time t."""
    return pd.concat([_pm_features(df["pm25"]), _weather_features(df), _calendar_features(df.index)], axis=1)


def build_dataset(df):
    """Return X, y, split labels and the common eval mask, all aligned on issuance time t."""
    X = build_features(df)
    y = make_target(df["pm25"])
    return X, y, assign_split(df.index), eval_mask(df["pm25"])
