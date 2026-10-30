"""End-to-end command-line pipeline for Air Quality Forecasting.

Allows running validation, feature generation, baseline models, and XGBoost
forecasting directly from the terminal without Jupyter notebooks.

Usage:
    python3 -m src.pipeline --stage all
    python3 -m src.pipeline --stage validate
    python3 -m src.pipeline --stage baselines
    python3 -m src.pipeline --stage xgboost
"""

import argparse
import sys
import time
import pandas as pd

from src.data import RAW_PATH, load_raw, validate, missing_runs
from src.protocol import assign_split, eval_mask, make_target
from src.features import build_features
from src.baselines import persistence, seasonal_naive
from src.evaluate import metrics, directional_accuracy, symmetric_mape, diebold_mariano
from src.models import XGBForecaster


def run_validation(data_path=RAW_PATH, verbose=True):
    """Load and validate the raw dataset."""
    if verbose:
        print("=" * 60)
        print("STAGE: DATA VALIDATION")
        print("=" * 60)
    t0 = time.time()
    df = load_raw(data_path, verify_hash=True)
    validate(df)
    runs = missing_runs(df["pm25"])
    elapsed = time.time() - t0

    if verbose:
        print(f"Dataset loaded from: {data_path}")
        print(f"Total hourly rows:   {len(df):,}")
        print(f"Date range:          {df.index[0]} to {df.index[-1]}")
        missing_count = df["pm25"].isna().sum()
        print(f"Missing PM2.5:       {missing_count:,} ({missing_count / len(df) * 100:.2f}%)")
        print(f"Missing run count:   {len(runs):,} (longest run: {runs['hours'].max()} hours)")
        print(f"Validation passed successfully in {elapsed:.2f}s.\n")
    return df


def run_features(df, verbose=True):
    """Extract and validate the feature matrix."""
    if verbose:
        print("=" * 60)
        print("STAGE: FEATURE ENGINEERING")
        print("=" * 60)
    t0 = time.time()
    X = build_features(df)
    elapsed = time.time() - t0
    if verbose:
        print(f"Engineered {X.shape[1]} features across {X.shape[0]:,} timestamps.")
        print(f"Feature columns: {', '.join(X.columns[:8])}... (total {len(X.columns)})")
        print(f"Feature matrix built in {elapsed:.2f}s.\n")
    return X


def evaluate_model(name, y_true, y_pred, y_ref):
    """Compute standard and directional metrics for a prediction series."""
    m = metrics(y_true, y_pred)
    m["MDA (%)"] = directional_accuracy(y_true, y_pred, y_ref) * 100.0
    m["sMAPE (%)"] = symmetric_mape(y_true, y_pred)
    m["Model"] = name
    return m


def run_baselines(df, verbose=True):
    """Evaluate naive baselines (persistence & seasonal naive) on validation split."""
    if verbose:
        print("=" * 60)
        print("STAGE: BASELINE EVALUATION")
        print("=" * 60)
    y = make_target(df["pm25"])
    splits = assign_split(df.index)
    mask = eval_mask(df["pm25"])

    val_mask = mask & (splits == "val")
    y_val = y[val_mask]
    pm_val = df.loc[val_mask, "pm25"]

    pred_persist = persistence(df["pm25"])[val_mask]
    pred_snaive = seasonal_naive(df["pm25"])[val_mask]

    res_persist = evaluate_model("Persistence", y_val, pred_persist, pm_val)
    res_snaive = evaluate_model("Seasonal Naive (24h)", y_val, pred_snaive, pm_val)

    table = pd.DataFrame([res_persist, res_snaive])[["Model", "n", "MAE", "RMSE", "R2", "MDA (%)", "sMAPE (%)"]]
    if verbose:
        print("Validation 2013 Performance (Common Evaluation Set):")
        print(table.to_string(index=False))
        print()
    return table


def run_xgboost(df, X=None, verbose=True):
    """Train XGBForecaster (delta formulation) and evaluate on 2013 validation split."""
    if verbose:
        print("=" * 60)
        print("STAGE: XGBOOST TRAINING & EVALUATION")
        print("=" * 60)
    if X is None:
        X = build_features(df)

    y = make_target(df["pm25"])
    splits = assign_split(df.index)
    mask = eval_mask(df["pm25"])

    train_rows = (splits == "train")
    val_rows = mask & (splits == "val")

    if verbose:
        print("Fitting XGBForecaster(formulation='delta') on 2010-2012 training data...")
    t0 = time.time()
    model = XGBForecaster(formulation="delta")
    model.fit(X[train_rows], y[train_rows])
    fit_time = time.time() - t0

    preds_xgb = model.predict(X[val_rows])
    preds_persist = persistence(df["pm25"])[val_rows]
    y_val = y[val_rows]
    pm_val = df.loc[val_rows, "pm25"]

    res_xgb = evaluate_model("XGBoost (delta)", y_val, preds_xgb, pm_val)
    res_persist = evaluate_model("Persistence", y_val, preds_persist, pm_val)

    # Diebold-Mariano test between XGBoost and persistence
    e_xgb = y_val.to_numpy() - preds_xgb.to_numpy()
    e_persist = y_val.to_numpy() - preds_persist.to_numpy()
    dm_stat, p_val = diebold_mariano(e_xgb, e_persist, loss="squared")

    table = pd.DataFrame([res_persist, res_xgb])[["Model", "n", "MAE", "RMSE", "R2", "MDA (%)", "sMAPE (%)"]]
    if verbose:
        print(f"Trained on {model.n_train_:,} rows in {fit_time:.2f}s.")
        print("\nValidation 2013 Performance:")
        print(table.to_string(index=False))
        print(f"\nDiebold-Mariano test (squared loss): stat = {dm_stat:.3f}, p-value = {p_val:.2e}")
        mae_imp = (res_persist["MAE"] - res_xgb["MAE"]) / res_persist["MAE"] * 100.0
        rmse_imp = (res_persist["RMSE"] - res_xgb["RMSE"]) / res_persist["RMSE"] * 100.0
        print(f"XGBoost improvement over persistence: MAE {mae_imp:+.2f}%, RMSE {rmse_imp:+.2f}%\n")
    return table


def main():
    parser = argparse.ArgumentParser(description="Air Quality Forecasting CLI Pipeline")
    parser.add_argument(
        "--stage",
        choices=["validate", "features", "baselines", "xgboost", "all"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--data-path",
        default=RAW_PATH,
        help="Path to raw dataset CSV",
    )
    args = parser.parse_args()

    df = run_validation(args.data_path, verbose=True)

    if args.stage == "validate":
        sys.exit(0)

    X = run_features(df, verbose=True)
    if args.stage == "features":
        sys.exit(0)

    if args.stage in ("baselines", "all"):
        run_baselines(df, verbose=True)

    if args.stage in ("xgboost", "all"):
        run_xgboost(df, X=X, verbose=True)


if __name__ == "__main__":
    main()
