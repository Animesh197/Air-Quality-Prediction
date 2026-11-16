import numpy as np
import pytest
from src.hybrid import best_blend_weight, blend


def test_best_weight_recovers_known_mix():
    """If y is exactly 0.7*xgb + 0.3*lstm, the fitted weight should be 0.7."""
    rng = np.random.default_rng(0)
    xgb = rng.normal(size=500)
    lstm = rng.normal(size=500)
    y = 0.7 * xgb + 0.3 * lstm
    w = best_blend_weight(y, xgb, lstm)
    assert np.isclose(w, 0.7, atol=1e-6)


def test_weight_clamped_above_one():
    """Unconstrained least squares would give w = 2; it must be clamped to 1.0."""
    xgb = np.array([1.0, 2.0, 3.0])
    lstm = np.array([0.0, 0.0, 0.0])
    y = 2.0 * xgb                      # best fit pulls w to 2
    assert best_blend_weight(y, xgb, lstm) == 1.0


def test_weight_clamped_below_zero():
    """Unconstrained least squares would give w = -1; it must be clamped to 0.0."""
    xgb = np.array([1.0, 2.0, 3.0])
    lstm = np.array([0.0, 0.0, 0.0])
    y = -1.0 * xgb                     # best fit pulls w to -1
    assert best_blend_weight(y, xgb, lstm) == 0.0


def test_identical_forecasts_fall_back_to_half():
    """When xgb == lstm the denominator is zero; the weight falls back to 0.5."""
    xgb = np.array([5.0, 6.0, 7.0])
    assert best_blend_weight([1.0, 2.0, 3.0], xgb, xgb) == 0.5


def test_blend_is_convex_combination():
    xgb = np.array([10.0, 20.0, 30.0])
    lstm = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(blend(xgb, lstm, 0.63), 0.63 * xgb)
    # endpoints
    np.testing.assert_allclose(blend(xgb, lstm, 1.0), xgb)
    np.testing.assert_allclose(blend(xgb, lstm, 0.0), lstm)


def test_blend_never_worse_than_both_parts():
    """The least-squares blend cannot have larger squared error than the better part."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    xgb = y + rng.normal(scale=1.0, size=400)
    lstm = y + rng.normal(scale=1.5, size=400)
    w = best_blend_weight(y, xgb, lstm)
    mse = lambda p: np.mean((y - p) ** 2)
    assert mse(blend(xgb, lstm, w)) <= min(mse(xgb), mse(lstm)) + 1e-9
