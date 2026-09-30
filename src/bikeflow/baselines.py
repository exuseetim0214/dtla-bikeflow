from __future__ import annotations

import pandas as pd


def seasonal_mean_prediction(
    frame: pd.DataFrame,
    *,
    prefix: str = "departures",
    lags: tuple[int, ...] = (336, 672, 1008, 1344),
) -> pd.Series:
    columns = [f"{prefix}_lag_{lag}" for lag in lags if f"{prefix}_lag_{lag}" in frame]
    if not columns:
        raise ValueError(f"No seasonal lag columns available for {prefix}")
    return frame[columns].mean(axis=1, skipna=True).fillna(0).clip(lower=0)


def last_week_prediction(frame: pd.DataFrame, *, prefix: str = "departures") -> pd.Series:
    column = f"{prefix}_lag_336"
    if column not in frame:
        raise ValueError(f"Missing required feature: {column}")
    return frame[column].fillna(0).clip(lower=0)
