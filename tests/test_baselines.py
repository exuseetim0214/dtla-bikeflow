import pandas as pd

from bikeflow.baselines import last_week_prediction, seasonal_mean_prediction


def test_seasonal_mean_ignores_missing_values() -> None:
    frame = pd.DataFrame({"departures_lag_336": [1.0, None], "departures_lag_672": [3.0, 2.0]})
    assert seasonal_mean_prediction(frame).tolist() == [2.0, 2.0]
    assert last_week_prediction(frame).tolist() == [1.0, 0.0]
