# Air Quality Forecasting (Project 19)

Short-horizon (initially 1-hour-ahead) PM2.5 forecasting for Beijing using historical air-quality and weather data. Phase 1 compares persistence and seasonal-naive baselines against XGBoost; Phase 2 (LSTM) is not started.

## Dataset
- File: `data/raw/dataset.csv` (read-only raw copy; SHA-256 `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`).
- Content matches the UCI "Beijing PM2.5 Data" set: Chen, S. (2015), UCI Machine Learning Repository, doi:[10.24432/C5JS49](https://doi.org/10.24432/C5JS49), CC BY 4.0. PM2.5 from the US Embassy in Beijing; weather from Beijing Capital International Airport. Hourly, 2010-01-01 00:00 to 2014-12-31 23:00, 43,824 rows. Original download source not independently verified.
- Introductory paper: Liang, X. et al. (2015), "Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating", *Proc. R. Soc. A*.

### Data dictionary (as returned by `src.data.load_raw`)
| Column | Raw name | Type | Unit | Description |
|---|---|---|---|---|
| `timestamp` (index) | year, month, day, hour | datetime | — | Hourly; assumed Beijing local time (UTC+8, no DST) |
| `pm25` | pm2.5 | float | µg/m³ | PM2.5 concentration; 2,067 missing (4.7%) |
| `dewp` | DEWP | int | °C | Dew point |
| `temp` | TEMP | float | °C | Temperature |
| `pres` | PRES | float | hPa | Pressure |
| `wind_dir` | cbwd | category | — | Combined wind direction: NE, NW, SE, cv |
| `wind_cum` | Iws | float | m/s | Cumulated wind speed; resets when `wind_dir` changes |
| `snow_cum` | Is | int | hours | Cumulated hours of snow |
| `rain_cum` | Ir | int | hours | Cumulated hours of rain |

The raw `No` column (row number) is dropped.

## Setup (macOS, Python 3.11)
```bash
brew install libomp                      # OpenMP runtime required by xgboost on macOS
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name air-quality-py311 --display-name "Python 3.11 (air-quality)"
```
Run notebooks with `.venv/bin/jupyter lab` and select the **Python 3.11 (air-quality)** kernel.

## Execution
Run in order from `notebooks/`:
1. `01_data_quality.ipynb`: loads and validates the raw data (integrity hash, timestamps, missingness, suspicious values).
2. `02_eda.ipynb`: exploratory analysis; writes figures to `results/figures/eda_*.png`.
3. `03_protocol.ipynb`: forecasting protocol (target, splits, backtest folds, missing-value policy) with leakage checks.
4. `04_features.ipynb`: builds the 37-feature matrix and runs alignment and leakage tests.
5. `05_baselines.ipynb`: persistence and seasonal-naive baselines on train/validation; writes `results/predictions/` and `results/metrics/`.
6. `06_xgboost.ipynb`: XGBoost target formulation, hyperparameter search on 2013 folds, feature importance (≈ 2.5 min).

Headless: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`

## Project structure
```
data/raw/           raw dataset (never modified)
data/processed/     regenerable intermediate data (git-ignored)
notebooks/          stage notebooks (data quality, EDA, ...)
src/                reusable code (data.py: load/validate; protocol.py: target/splits/folds;
                    features.py: feature matrix; baselines.py; models.py: XGBoost;
                    evaluate.py: metrics and backtesting;
                    plot_style.py: figure style)
results/figures/    saved plots
results/metrics/    saved metric tables
requirements.txt    pinned Phase 1 dependencies
```

## Forecasting protocol
1-hour-ahead PM2.5, issued at the end of hour *t* using only data up to *t*. Splits by target hour: train 2010–2012, validation 2013, test 2014. Defined in `src/protocol.py`.

## Results
Validation 2013, common eval set (test year not yet scored). XGBoost predictions come from the expanding-window quarterly folds:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence | 13.18 | 24.87 | 0.936 |
| Seasonal-naive 24 h | 76.08 | 110.18 | −0.260 |
| XGBoost (change target, tuned on 2013 folds) | 12.26 | 22.53 | 0.947 |

## Limitations
- Dataset provenance verified by content only, not by an official checksum.
- PM2.5 has gaps of up to 155 hours, and more missing data in 2010–2012 than in 2013–2014.
- `wind_cum`, `snow_cum` and `rain_cum` are cumulative, not hourly readings; `snow_cum`/`rain_cum` contain a few skipped increments.
- On validation, XGBoost does not beat persistence for the most extreme hours (true PM2.5 > 250 µg/m³).
