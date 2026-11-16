import numpy as np
import pytest
from src.dl_models import LSTMForecaster


def test_invalid_formulation_rejected():
    with pytest.raises(AssertionError):
        LSTMForecaster(formulation="bogus")


def test_base_term_distinguishes_delta_and_level():
    """delta adds pm(t) back after prediction; level adds nothing."""
    X = np.zeros((5, 3, 2), dtype="float32")
    pm_t = np.arange(5, dtype="float32")

    delta = LSTMForecaster(formulation="delta")
    np.testing.assert_array_equal(delta._base(X, pm_t), pm_t)

    level = LSTMForecaster(formulation="level")
    np.testing.assert_array_equal(level._base(X, pm_t), np.zeros(5, dtype="float32"))


def _toy_data(n=80, lookback=4, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, lookback, channels)).astype("float32")
    pm_t = rng.normal(loc=50.0, scale=5.0, size=n).astype("float32")
    # target is a small change on top of pm(t), the regime the delta form targets
    y = (pm_t + rng.normal(scale=1.0, size=n)).astype("float32")
    return X, y, pm_t


def test_fit_predict_runs_and_returns_finite_shape():
    X, y, pm_t = _toy_data()
    model = LSTMForecaster(formulation="delta", hidden=8, layers=1,
                           epochs=3, batch_size=16, seed=0)
    model.fit(X, y, pm_t)
    preds = model.predict(X, pm_t)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_level_formulation_also_runs():
    X, y, pm_t = _toy_data()
    model = LSTMForecaster(formulation="level", hidden=8, layers=1,
                           epochs=3, batch_size=16, seed=0)
    model.fit(X, y)                       # level ignores pm_t
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))
