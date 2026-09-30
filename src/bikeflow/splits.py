from __future__ import annotations

import pandas as pd


def assign_time_split(frame: pd.DataFrame, config: dict[str, str]) -> pd.Series:
    timestamp = pd.to_datetime(frame["forecast_time"], errors="coerce")
    result = pd.Series("excluded", index=frame.index, dtype="string")
    ranges = {
        "train": (config["train_start"], config["train_end"]),
        "validation": (config["validation_start"], config["validation_end"]),
        "test": (config["test_start"], config["test_end"]),
    }
    for name, (start, end) in ranges.items():
        result.loc[timestamp.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")] = (
            name
        )
    return result
