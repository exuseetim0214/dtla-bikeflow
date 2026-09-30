import numpy as np
import pandas as pd

from bikeflow.backtest import retrain_due
from bikeflow.drift import population_stability_index, should_retrain


def test_psi_is_zero_for_same_distribution() -> None:
    values = np.arange(100)
    assert population_stability_index(values, values) == 0.0


def test_retraining_trigger_reports_reasons() -> None:
    result = should_retrain(reference_mae=1.0, recent_mae=1.4, feature_psi=0.25)
    assert result["trigger"] is True
    assert result["reasons"] == ["rolling_mae", "feature_psi"]


def test_retraining_schedule() -> None:
    previous = pd.Timestamp("2026-01-26")
    current = pd.Timestamp("2026-02-02")
    assert retrain_due("fixed", current, previous) is False
    assert retrain_due("weekly", current, previous) is True
    assert retrain_due("monthly", current, previous) is True
    assert retrain_due("drift_triggered", current, previous, drift_pending=True) is True
