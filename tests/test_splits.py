import pandas as pd

from bikeflow.splits import assign_time_split


def test_assign_time_split() -> None:
    frame = pd.DataFrame(
        {"forecast_time": pd.to_datetime(["2025-12-01", "2026-02-01", "2026-05-01"])}
    )
    config = {
        "train_start": "2024-04-01",
        "train_end": "2025-12-31 23:59:59",
        "validation_start": "2026-01-01",
        "validation_end": "2026-03-31 23:59:59",
        "test_start": "2026-04-01",
        "test_end": "2026-06-30 23:59:59",
    }
    assert assign_time_split(frame, config).tolist() == ["train", "validation", "test"]
