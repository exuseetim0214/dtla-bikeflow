import math

import pytest

from bikeflow.metrics import precision_recall_at_k, regression_metrics


def test_regression_metrics() -> None:
    result = regression_metrics([0, 1, 3], [0, 2, 2])
    assert result["mae"] == pytest.approx(2 / 3)
    assert result["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert result["wape"] == pytest.approx(0.5)


def test_precision_recall_at_k() -> None:
    result = precision_recall_at_k([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], 2)
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0
