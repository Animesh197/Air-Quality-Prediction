"""Load and validate the raw UCI Beijing PM2.5 hourly dataset.

The raw CSV is never modified. `load_raw` returns a tidy, hourly-indexed copy;
`validate` enforces hard invariants; `missing_runs` summarises gaps.
Missing values are deliberately left as NaN (imputation is decided in Stage 04).
"""

from pathlib import Path
import hashlib

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
RAW_SHA256 = "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c"

RAW_COLUMNS = ["No", "year", "month", "day", "hour", "pm2.5",
               "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]

# Raw name -> project name (raw names contain '.' or are terse)
RENAME = {
    "pm2.5": "pm25",      # PM2.5 concentration, ug/m^3 (US Embassy)
    "DEWP": "dewp",       # dew point, deg C (Capital Airport)
    "TEMP": "temp",       # temperature, deg C
    "PRES": "pres",       # pressure, hPa
    "cbwd": "wind_dir",   # combined wind direction: NE, NW, SE, cv
    "Iws": "wind_cum",    # cumulated wind speed (m/s) since direction last changed
    "Is": "snow_cum",     # cumulated hours of snow
    "Ir": "rain_cum",     # cumulated hours of rain
}

WIND_DIRS = ["NE", "NW", "SE", "cv"]


def file_sha256(path=RAW_PATH):
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path=RAW_PATH, verify_hash=True):
    """Load the raw CSV into an hourly DataFrame indexed by `timestamp`.

    Timestamps are built from year/month/day/hour and kept timezone-naive
    (assumed Beijing local time, UTC+8, no DST). The `No` row counter and the
    separate date-part columns are dropped; calendar features are derived from
    the index later.
    """
    path = Path(path)
    if verify_hash and file_sha256(path) != RAW_SHA256:
        raise ValueError(f"{path} does not match the recorded raw-data SHA-256")

    raw = pd.read_csv(path, na_values=["NA"], keep_default_na=False)
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(raw.columns)}")

    df = raw.rename(columns=RENAME)
    df.index = pd.DatetimeIndex(
        pd.to_datetime(raw[["year", "month", "day", "hour"]]), name="timestamp")
    df = df.drop(columns=["No", "year", "month", "day", "hour"])
    df["wind_dir"] = pd.Categorical(df["wind_dir"], categories=WIND_DIRS)
    return df


def validate(df):
    """Raise AssertionError if any hard data invariant is violated."""
    idx = df.index
    assert idx.is_unique, "duplicate timestamps"
    assert idx.is_monotonic_increasing, "timestamps not sorted"
    expected = pd.date_range(idx[0], idx[-1], freq="h")
    assert len(idx) == len(expected) and (idx == expected).all(), "index is not a complete hourly range"

    assert list(df.columns) == list(RENAME.values()), "unexpected columns"
    nan_cols = df.columns[df.isna().any()].tolist()
    assert nan_cols in ([], ["pm25"]), f"unexpected missing values in {nan_cols}"

    assert (df["pm25"].dropna() >= 0).all(), "negative PM2.5"
    assert df["pres"].between(900, 1100).all(), "pressure outside 900-1100 hPa"
    assert df["temp"].between(-50, 50).all(), "temperature outside -50..50 C"
    assert df["dewp"].between(-60, 40).all(), "dew point outside -60..40 C"
    for c in ["wind_cum", "snow_cum", "rain_cum"]:
        assert (df[c] >= 0).all(), f"negative {c}"
    return True


def missing_runs(s):
    """Return one row per run of consecutive NaNs in `s`: start, end, hours."""
    na = s.isna()
    run_id = (na != na.shift()).cumsum()
    runs = (s.index.to_series()[na]
            .groupby(run_id[na])
            .agg(start="min", end="max", hours="size")
            .reset_index(drop=True))
    return runs
