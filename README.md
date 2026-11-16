   # Air Quality Forecasting (Project 19)

One-hour-ahead PM2.5 forecasting for Beijing from historical air-quality and weather data. **Phase 1 (machine learning)** compares persistence and seasonal-naive baselines with XGBoost under a leakage-controlled, chronological protocol. **Phase 2 (deep learning)** adds an LSTM trained on raw hourly windows and an XGBoost–LSTM hybrid, evaluated under the *same* protocol and eval hours so all models are directly comparable.

## Dataset
- File: `data/raw/dataset.csv` (read-only raw copy; SHA-256 `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`, checked on every load).
- Content matches the UCI "Beijing PM2.5 Data" set: Chen, S. (2015), UCI Machine Learning Repository, doi:[10.24432/C5JS49](https://doi.org/10.24432/C5JS49), CC BY 4.0. PM2.5 from the US Embassy in Beijing; weather from Beijing Capital International Airport. Hourly, 2010-01-01 00:00 to 2014-12-31 23:00, 43,824 rows. The original download source was not independently verified.
- Introductory paper: Liang, X. et al. (2015), "Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating", *Proc. R. Soc. A* 471: 20150257, doi:[10.1098/rspa.2015.0257](https://doi.org/10.1098/rspa.2015.0257).

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
uv pip install --python .venv/bin/python -r requirements.txt     # direct dependencies
# or, for the exact verified environment:  -r requirements-lock.txt
.venv/bin/python -m ipykernel install --user --name air-quality-py311 --display-name "Python 3.11 (air-quality)"
```
Run notebooks interactively with `.venv/bin/jupyter lab` and the **Python 3.11 (air-quality)** kernel.

The Phase 2 LSTM uses PyTorch (`torch`, pinned in `requirements.txt`); the deep-learning notebooks run on the Apple GPU (Metal/MPS) when available and fall back to CPU automatically, so no extra setup is needed. GPU training is not perfectly deterministic, so the Phase 2 numbers can vary by a small amount between runs.

## Execution
Run the notebooks in order:

| Notebook | Stage | Purpose |
|---|---|---|
| `01_data_quality.ipynb` | 02 | Load and validate raw data (hash, timestamps, missingness, suspicious values) |
| `02_eda.ipynb` | 03 | Exploratory analysis; figures `eda_*` |
| `03_protocol.ipynb` | 04 | Target, splits, backtest folds, missing-value policy, leakage checks |
| `04_features.ipynb` | 05 | 37-feature matrix; alignment and future-perturbation leakage tests |
| `05_baselines.ipynb` | 06 | Persistence and seasonal-naive on train/validation |
| `06_xgboost.ipynb` | 07 | Target formulation, hyperparameter search on 2013 folds, feature importance (≈ 2.5 min) |
| `07_backtest.ipynb` | 08 | Single 2014 test evaluation, Diebold–Mariano test, error analysis |
| `08_dl_windows.ipynb` | 09 | Build 24-hour windowed input channels for the LSTM; future-perturbation leakage check; figure `dl_01` |
| `09_lstm.ipynb` | 10 | LSTM tuning (12-configuration ablation over lookback × hidden × layers) on 2013; figure `dl_02` |
| `10_lstm_backtest.ipynb` | 11 | Frozen-config monthly expanding-window backtest on 2014; Diebold–Mariano; figures `dl_03`–`dl_05` |
| `11_hybrid.ipynb` | 12 | XGBoost–LSTM convex blend (weight fixed on 2013, frozen for 2014); figure `dl_06` |

Phase 1 (notebooks 01–07) reproduces headless in ≈ 3.5 min. The Phase 2 notebooks (08–11) add PyTorch training and take longer. Run everything in order:
```bash
for nb in notebooks/*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

### CLI Pipeline Runner
You can also run the end-to-end forecasting pipeline or individual stages directly from the command line without Jupyter:
```bash
python3 -m src.pipeline --stage all         # Run data validation, baselines & XGBoost
python3 -m src.pipeline --stage validate    # Validate raw data integrity and missingness
python3 -m src.pipeline --stage baselines   # Evaluate persistence & seasonal-naive baselines
python3 -m src.pipeline --stage xgboost     # Fit and score XGBoost on validation set
python3 -m src.pipeline --stage lstm        # Fit the frozen-config LSTM and score it on validation
python3 -m src.pipeline --stage hybrid      # Blend XGBoost + LSTM on validation (fits the blend weight)
```
The `baselines`/`xgboost`/`lstm`/`hybrid` stages are quick 2013-validation checks that reuse the frozen configurations and print metric tables; they write no files. The full 2014 backtests live in notebooks 07/10/11. The `lstm` and `hybrid` stages are explicit opt-in (not part of `all`) and require PyTorch.

### Automated Testing
Run the complete unit and integration test suite:
```bash
pytest tests -v
```

**Reproducibility check:** in a freshly built environment from `requirements.txt`, running the seven Phase 1 notebooks (01–07) regenerated every corresponding file in `results/metrics/` and `results/predictions/` byte for byte, plus their 19 figures. The Phase 2 notebooks (08–11) are not byte-for-byte reproducible because GPU (MPS) training is not perfectly deterministic; their metrics and the `dl_*` figures can vary by a small amount between runs.

## Project structure
```
data/raw/               raw dataset (never modified)
data/processed/         regenerable intermediate data (git-ignored; currently unused)
notebooks/              stage notebooks 01–11 (01–07 Phase 1, 08–11 Phase 2)
src/                    data.py (load/validate), protocol.py (target/splits/folds),
                        features.py, baselines.py, models.py (XGBoost),
                        windows.py (LSTM input sequences), dl_models.py (LSTM),
                        hybrid.py (XGBoost–LSTM blend),
                        evaluate.py (metrics, backtest, Diebold–Mariano),
                        pipeline.py (CLI runner), plot_style.py
tests/                  automated unit and integration test suite (pytest)
results/figures/        plots (eda_*, xgb_*, test_* for Phase 1; dl_* for Phase 2)
results/metrics/        metric tables plus the selected XGBoost, LSTM and hybrid configs (JSON)
results/predictions/    saved predictions (baselines, XGBoost, LSTM and hybrid test 2014)
report/                 IEEE reports (main.tex Phase 1, phase2.tex Phase 2), references.bib, Beamer slides (slides.tex)
requirements.txt        direct dependencies (pinned)
requirements-lock.txt   full pinned environment
```

## Forecasting protocol
- **Task:** PM2.5 at hour *t + 1*, forecast issued at the end of hour *t* using only data up to *t* (no future weather).
- **Splits** (by target hour): train 2010–2012, validation 2013, test 2014.
- **Tuning:** 4 quarterly expanding-window folds in 2013.
- **Test:** 12 monthly expanding-window refits in 2014 with the frozen configuration.
- **Scoring:** all models on the same hours (target, pm(t) and pm(t−23) observed).

Defined in `src/protocol.py`.

## Results

### Phase 1 (machine learning): XGBoost vs. baselines
1-hour-ahead PM2.5 (µg/m³), common eval set. The 2014 test year was scored once, with the configuration frozen after tuning on 2013:

| Model | Val 2013 MAE / RMSE / R² | **Test 2014 MAE / RMSE / R²** |
|---|---|---|
| Persistence | 13.18 / 24.87 / 0.936 | 11.93 / 22.03 / 0.944 |
| Seasonal-naive 24 h | 76.08 / 110.18 / −0.260 | 67.30 / 98.75 / −0.121 |
| XGBoost (change target) | 12.26 / 22.53 / 0.947 | **10.96 / 19.96 / 0.954** |

- On the test year, XGBoost reduces MAE by 8.1% and RMSE by 9.4% relative to persistence and is better in all 12 months.
- The difference is significant: Diebold–Mariano p = 3.4 × 10⁻⁷ (squared loss).
- The gain comes from moderate levels and from anticipating falls in PM2.5.

Full tables are in `results/metrics/`.

### Phase 2 (deep learning): LSTM and hybrid
Same task, protocol and common eval hours as Phase 1. The LSTM (24-hour window, 2 layers, 64 hidden units, change target) was tuned on 2013 (validation MAE 12.47 µg/m³) and the XGBoost–LSTM hybrid weight (0.63 XGBoost / 0.37 LSTM) was fixed on 2013; both were then frozen and scored once on 2014:

| Model | **Test 2014 MAE / RMSE / R²** |
|---|---|
| Persistence | 11.93 / 22.03 / 0.944 |
| XGBoost (change target) | 10.96 / 19.96 / 0.954 |
| LSTM | 11.28 / 20.49 / 0.952 |
| **Hybrid (0.63 XGB + 0.37 LSTM)** | **10.90 / 19.89 / 0.955** |

- The LSTM beats persistence significantly (Diebold–Mariano p ≈ 3 × 10⁻⁸, squared loss) but does not overtake the tuned XGBoost (DM p ≈ 2 × 10⁻⁵, absolute loss).
- The hybrid gives the best overall test scores and improves most at high pollution, but its edge over XGBoost alone is small and not statistically significant (DM p = 0.55 squared, 0.07 absolute).
- No model solves the extreme band (> 250 µg/m³): all three have test RMSE near 42 µg/m³.

Phase 2 tables are in `results/metrics/` (`lstm_*`, `hybrid_*`); the full write-up is `report/phase2.tex`.

## Report and slides
`report/main.tex` (Phase 1), `report/phase2.tex` (Phase 2) and `report/slides.tex` (Beamer) all read figures from `../results/figures/`. Both reports are IEEEtran. Compile from `report/`:
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex phase2 && bibtex phase2 && pdflatex phase2 && pdflatex phase2
pdflatex slides && pdflatex slides
```
LaTeX is not installed in the development environment, so the documents have **not been compiled here**; references, figure paths and structure were checked programmatically. On Overleaf, upload `report/` together with `results/figures/` and keep the relative path.

## Limitations
- Dataset provenance verified by content only, not by an official checksum.
- PM2.5 has gaps of up to 155 hours, and more missing data in 2010–2012 than in 2013–2014.
- `wind_cum`, `snow_cum` and `rain_cum` are cumulative, not hourly readings; `snow_cum`/`rain_cum` contain a few skipped increments.
- XGBoost does not beat persistence on the most extreme hours (true PM2.5 > 250 µg/m³; test RMSE 42.0 vs 41.6) and under-predicts them. It also fails to anticipate sudden large rises.
- Single station and single test year; inputs use only observed weather (no weather forecasts).
- The Diebold–Mariano p-values are approximate: the test slightly over-rejected (≈ 8% at a nominal 5%) in a simulation with autocorrelated errors.
- Phase 2: the LSTM was tuned on a single 2013 validation year (not Phase 1's four folds) to keep deep-learning training affordable, and GPU (MPS) training is not perfectly deterministic, so its numbers can shift slightly between runs.
- Phase 2: the hybrid's improvement over XGBoost alone is not statistically significant, and learning from raw history did not overtake the hand-engineered features at this one-hour horizon.
