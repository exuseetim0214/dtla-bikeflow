from datetime import UTC, datetime

import pandas as pd

from bikeflow.status import add_future_availability_labels, normalize_geojson


def test_normalize_geojson() -> None:
    payload = {
        "features": [
            {
                "geometry": {"coordinates": [-118.2, 34.0]},
                "properties": {
                    "kioskId": 3005,
                    "name": "Example",
                    "bikesAvailable": 2,
                    "docksAvailable": 8,
                    "totalDocks": 10,
                },
            }
        ]
    }
    result = normalize_geojson(payload, datetime(2026, 1, 1, tzinfo=UTC))
    assert result.loc[0, "station_id"] == 3005
    assert result.loc[0, "latitude"] == 34.0


def test_future_labels_require_persistent_zero() -> None:
    frame = pd.DataFrame(
        {
            "station_id": [1] * 6,
            "collected_at_utc": pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC"),
            "bikes_available": [3, 1, 0, 0, 2, 2],
            "docks_available": [7, 9, 10, 10, 8, 8],
        }
    )
    result = add_future_availability_labels(frame, snapshots_ahead=6, confirmations=2)
    assert result.iloc[0]["target_empty_30m"] == 1
    assert result.iloc[3]["target_empty_30m"] == 0
