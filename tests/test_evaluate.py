import numpy as np
import pytest
from src.evaluate import metrics, diebold_mariano, directional_accuracy, symmetric_mape


def test_metrics_calculation():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    res = metrics(y_true, y_pred)
    assert res["n"] == 3
    # MAE = (|2| + |-2| + 0) / 3 = 4 / 3
    assert np.isclose(res["MAE"], 4.0 / 3.0)
    # RMSE = sqrt((4 + 4 + 0) / 3) = sqrt(8 / 3)
    assert np.isclose(res["RMSE"], np.sqrt(8.0 / 3.0))
    assert res["R2"] > 0


def test_metrics_raises_on_nan():
    with pytest.raises(ValueError, match="NaN in y_true or y_pred"):
        metrics([1.0, np.nan], [1.0, 2.0])


def test_diebold_mariano_identical():
    # If loss differentials are zero, DM stat should be 0 and p-value 1.0
    e1 = np.ones(100)
    e2 = np.ones(100)
    stat, pval = diebold_mariano(e1, e2, loss="squared", max_lag=5)
    assert np.isclose(stat, 0.0)
    assert np.isclose(pval, 1.0)


def test_diebold_mariano_significant_difference():
    # Model 1 has much smaller errors than Model 2
    np.random.seed(42)
    e1 = np.random.normal(0, 1, 200)
    e2 = np.random.normal(0, 10, 200)
    stat, pval = diebold_mariano(e1, e2, loss="squared", max_lag=5)
    # Negative statistic indicates model 1 has lower loss
    assert stat < 0
    assert pval < 0.001


def test_directional_accuracy():
    # y_ref is pm(t)
    y_ref = np.array([50.0, 50.0, 50.0, 50.0])
    # true changes: +10, -10, 0, +5
    y_true = np.array([60.0, 40.0, 50.0, 55.0])
    # pred changes: +5 (correct), -5 (correct), +2 (incorrect), -2 (incorrect)
    y_pred = np.array([55.0, 45.0, 52.0, 48.0])

    mda = directional_accuracy(y_true, y_pred, y_ref)
    # 2 out of 4 correct -> 0.5
    assert np.isclose(mda, 0.5)


def test_symmetric_mape():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([100.0, 200.0])
    assert np.isclose(symmetric_mape(y_true, y_pred), 0.0)

    y_pred_err = np.array([110.0, 190.0])
    smape = symmetric_mape(y_true, y_pred_err)
    assert 0 < smape < 100
