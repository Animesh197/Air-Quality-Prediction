"""XGBoost forecaster for the t+1 target (Stage 07).

`formulation` sets what the trees learn:
- "level": y = pm(t+1)
- "log":   y = log1p(pm(t+1)), back-transformed with expm1
- "delta": y = pm(t+1) - pm(t), then pm(t) is added back. Trees are piecewise
  constant, so learning the change avoids having to approximate "copy pm(t)"
  with many splits.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 0
BASE_PARAMS = {"tree_method": "hist", "objective": "reg:squarederror", "random_state": SEED, "n_jobs": -1}


class XGBForecaster:
    def __init__(self, formulation="delta", **params):
        if formulation not in ("level", "log", "delta"):
            raise ValueError(formulation)
        self.formulation = formulation
        self.params = {**BASE_PARAMS, **params}
        self.model = xgb.XGBRegressor(**self.params)

    def _train_rows(self, X, y):
        rows = y.notna()
        if self.formulation == "delta":
            rows &= X["pm_lag0"].notna()   # the change is undefined without pm(t)
        return rows

    def fit(self, X, y):
        rows = self._train_rows(X, y)
        Xt, yt = X[rows], y[rows]
        if self.formulation == "log":
            yt = np.log1p(yt)
        elif self.formulation == "delta":
            yt = yt - Xt["pm_lag0"]
        self.model.fit(Xt, yt)
        self.n_train_ = int(rows.sum())
        return self

    def predict(self, X):
        p = self.model.predict(X)
        if self.formulation == "log":
            p = np.expm1(p)
        elif self.formulation == "delta":
            p = p + X["pm_lag0"].to_numpy()
        return pd.Series(p, index=X.index, name="xgboost")
