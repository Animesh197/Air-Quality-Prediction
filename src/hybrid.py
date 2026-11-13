"""Hybrid of the XGBoost and LSTM forecasts (Stage 12).

The two models make partly different errors, so a convex blend

    pred = w * xgb + (1 - w) * lstm

can have lower error than either one alone. The single weight w is chosen on the
2013 validation year (least squares) and then frozen for the 2014 test, so nothing
from the test is used to build the hybrid. One parameter keeps it hard to overfit.
"""

import numpy as np


def best_blend_weight(y, xgb, lstm):
    """Weight w in [0, 1] minimising the squared error of w*xgb + (1-w)*lstm."""
    y, xgb, lstm = (np.asarray(a, dtype=float) for a in (y, xgb, lstm))
    d = xgb - lstm
    denom = float(d @ d)
    if denom == 0.0:
        return 0.5
    w = float((d @ (y - lstm)) / denom)
    return min(1.0, max(0.0, w))          # keep it a convex combination


def blend(xgb, lstm, w):
    """Convex blend of the two forecasts with weight w on XGBoost."""
    return w * np.asarray(xgb, dtype=float) + (1.0 - w) * np.asarray(lstm, dtype=float)
