# Air Quality Forecasting (Project 19)

Short-horizon (initially 1-hour-ahead) PM2.5 forecasting for Beijing using historical air-quality and weather data. Phase 1 compares persistence and seasonal-naive baselines against XGBoost; Phase 2 (LSTM) is not started.

## Dataset
- File: `data/raw/dataset.csv` (read-only raw copy; SHA-256 `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`).
- Content matches the UCI "Beijing PM2.5 Data" set (US Embassy PM2.5 + Beijing Capital Airport weather, hourly, 2010-01-01 to 2014-12-31, 43,824 rows). Original download source not independently verified.
- Columns: `No, year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir`.

## Setup (macOS, Python 3.11)
```bash
brew install libomp                      # OpenMP runtime required by xgboost on macOS
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name air-quality-py311 --display-name "Python 3.11 (air-quality)"
```
Run notebooks with `.venv/bin/jupyter lab` and select the **Python 3.11 (air-quality)** kernel.

## Project structure
```
data/raw/           raw dataset (never modified)
data/processed/     regenerable intermediate data (git-ignored)
notebooks/          stage notebooks (data quality, EDA, ...)
src/                reusable loading / feature / evaluation code
results/figures/    saved plots
results/metrics/    saved metric tables
requirements.txt    pinned Phase 1 dependencies
```

## Results
No models have been run yet.

## Limitations
- Dataset provenance verified by content only, not by an official checksum.
