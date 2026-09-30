import pandas as pd

from bikeflow.prepare import build_station_panel


def test_panel_targets_only_use_future_bins() -> None:
    bins = pd.date_range("2026-01-01", periods=5, freq="30min")
    index = pd.MultiIndex.from_product([[3005], bins], names=["station_id", "bin_start"])
    departures = pd.Series([1, 2, 3, 4, 5], index=index)
    arrivals = pd.Series([0, 1, 0, 1, 0], index=index)
    station = pd.Series(
        {
            "latitude": 34.0,
            "longitude": -118.2,
            "go_live_date": pd.Timestamp("2020-01-01"),
        }
    )
    frame = build_station_panel(
        3005,
        station,
        departures,
        arrivals,
        (bins[0], bins[-1]),
        frequency="30min",
        lags=[1, 2],
        rolling_windows=[2],
        horizons=[1, 2],
    )
    assert frame.loc[0, "departures_lag_1"] == 1
    assert pd.isna(frame.loc[0, "departures_lag_2"])
    assert frame.loc[0, "target_departures_30m"] == 2
    assert frame.loc[0, "target_departures_60m"] == 5


def test_panel_supports_arrival_only_station() -> None:
    bins = pd.date_range("2026-01-01", periods=3, freq="30min")
    empty_index = pd.MultiIndex.from_arrays([[], []], names=["station_id", "bin_start"])
    departures = pd.Series([], index=empty_index, dtype="int32")
    arrival_index = pd.MultiIndex.from_product([[4592], bins], names=["station_id", "bin_start"])
    arrivals = pd.Series([1, 2, 1], index=arrival_index)
    station = pd.Series(
        {
            "latitude": 34.0,
            "longitude": -118.2,
            "go_live_date": pd.Timestamp("2020-01-01"),
        }
    )
    frame = build_station_panel(
        4592,
        station,
        departures,
        arrivals,
        (bins[0], bins[-1]),
        frequency="30min",
        lags=[1],
        rolling_windows=[2],
        horizons=[1],
    )
    assert frame["departures_lag_1"].sum() == 0
    assert frame["arrivals_lag_1"].sum() == 4
