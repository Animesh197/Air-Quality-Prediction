"""Naive baselines for the t+1 forecast (Stage 06). No fitted parameters.

Both return predictions indexed by issuance time t, aligned with `protocol.make_target`.
"""

from src.protocol import HORIZON, SEASONAL_PERIOD


def persistence(pm, horizon=HORIZON):
    """y_hat(t + h) = pm(t): the latest observation carries forward."""
    return pm.rename("persistence")


def seasonal_naive(pm, period=SEASONAL_PERIOD, horizon=HORIZON):
    """y_hat(t + h) = pm(t + h - period): the same hour one season (day) earlier."""
    return pm.shift(period - horizon).rename(f"seasonal_naive_{period}h")
