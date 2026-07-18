import numpy as np
import pytest
from gridsight.models.common.metrics import (
    pinball_loss,
    mean_pinball,
    coverage,
    crossing_rate,
    skill_vs_baseline,
    report,
)

def test_pinball_loss_exact_matches():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y_true, y_pred, 0.5) == 0.0
    assert pinball_loss(y_true, y_pred, 0.1) == 0.0
    assert pinball_loss(y_true, y_pred, 0.9) == 0.0

def test_pinball_loss_underprediction():
    y_true = np.array([10.0])
    y_pred = np.array([8.0])  # Diff = 2.0 (underpredicted)
    # q * diff = 0.9 * 2.0 = 1.8
    assert pytest.approx(pinball_loss(y_true, y_pred, 0.9)) == 1.8
    # q * diff = 0.1 * 2.0 = 0.2
    assert pytest.approx(pinball_loss(y_true, y_pred, 0.1)) == 0.2

def test_pinball_loss_overprediction():
    y_true = np.array([10.0])
    y_pred = np.array([12.0])  # Diff = -2.0 (overpredicted)
    # (q - 1.0) * diff = -0.1 * -2.0 = 0.2
    assert pytest.approx(pinball_loss(y_true, y_pred, 0.9)) == 0.2
    # (q - 1.0) * diff = -0.9 * -2.0 = 1.8
    assert pytest.approx(pinball_loss(y_true, y_pred, 0.1)) == 1.8

def test_mean_pinball():
    y_true = np.array([10.0])
    preds = {
        0.1: np.array([8.0]),  # loss = 0.2
        0.5: np.array([10.0]), # loss = 0.0
        0.9: np.array([12.0])  # loss = 0.2
    }
    # Mean of [0.2, 0.0, 0.2] = 0.4 / 3 = 0.13333333333333333
    assert pytest.approx(mean_pinball(y_true, preds)) == 0.4 / 3.0

def test_coverage():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    hi = np.array([3.5, 3.5, 3.5, 3.5, 3.5])
    # 2.0 and 3.0 are inside [1.5, 3.5] -> 2 out of 5 -> 0.4 coverage
    assert coverage(y_true, lo, hi) == 0.4

def test_crossing_rate():
    # Normal case: q10 <= q50 <= q90
    normal = {
        0.1: np.array([1.0, 2.0]),
        0.5: np.array([1.5, 2.5]),
        0.9: np.array([2.0, 3.0])
    }
    assert crossing_rate(normal) == 0.0

    # Crossing case: q10 > q50 at index 1
    crossed = {
        0.1: np.array([1.0, 3.0]),
        0.5: np.array([1.5, 2.5]),
        0.9: np.array([2.0, 3.5])
    }
    assert crossing_rate(crossed) == 0.5

def test_skill_vs_baseline():
    y_true = np.array([10.0, 20.0])
    q50_pred = np.array([11.0, 19.0]) # MAE = 1.0
    baseline_pred = np.array([12.0, 18.0]) # MAE = 2.0
    # Skill = 1 - 1.0/2.0 = 0.5
    assert skill_vs_baseline(y_true, q50_pred, baseline_pred) == 0.5

    # Zero baseline case
    assert np.isnan(skill_vs_baseline(y_true, q50_pred, y_true))

def test_report():
    y_true = np.array([10.0, 20.0])
    preds = {
        0.1: np.array([9.0, 18.0]),
        0.5: np.array([10.0, 20.0]),
        0.9: np.array([11.0, 22.0])
    }
    rep = report(y_true, preds)
    assert "pinball_q10" in rep
    assert "pinball_q50" in rep
    assert "pinball_q90" in rep
    assert "mean_pinball" in rep
    assert "coverage_10-90" in rep
    assert rep["coverage_10-90"] == 1.0
    assert rep["crossing_rate"] == 0.0
