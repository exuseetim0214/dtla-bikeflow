import pandas as pd
import pytest

from bikeflow.spatial import haversine_meters, nearest_station_map


def test_haversine_zero_distance() -> None:
    assert float(haversine_meters(34.0, -118.0, 34.0, -118.0)) == pytest.approx(0.0)


def test_nearest_station_map_excludes_self() -> None:
    stations = pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "latitude": [34.0, 34.001, 34.1],
            "longitude": [-118.0, -118.0, -118.0],
        }
    )
    result = nearest_station_map(stations, k=1)
    assert result[1][0]["station_id"] == 2
    assert result[1][0]["distance_m"] > 0
