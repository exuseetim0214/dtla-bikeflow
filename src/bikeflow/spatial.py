from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8


def haversine_meters(lat1: object, lon1: object, lat2: object, lon2: object) -> np.ndarray:
    lat1_r, lon1_r, lat2_r, lon2_r = map(
        np.radians,
        [
            np.asarray(lat1, dtype=float),
            np.asarray(lon1, dtype=float),
            np.asarray(lat2, dtype=float),
            np.asarray(lon2, dtype=float),
        ],
    )
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(value))


def nearest_station_map(stations: pd.DataFrame, *, k: int = 5) -> dict[int, list[dict[str, float]]]:
    required = {"station_id", "latitude", "longitude"}
    if not required.issubset(stations.columns):
        raise ValueError(f"Stations must contain {sorted(required)}")
    values = stations.dropna(subset=list(required)).reset_index(drop=True)
    result: dict[int, list[dict[str, float]]] = {}
    for _, station in values.iterrows():
        distances = haversine_meters(
            station["latitude"],
            station["longitude"],
            values["latitude"].to_numpy(),
            values["longitude"].to_numpy(),
        )
        order = np.argsort(distances)
        neighbors = []
        for index in order:
            candidate = int(values.iloc[index]["station_id"])
            if candidate == int(station["station_id"]):
                continue
            neighbors.append({"station_id": candidate, "distance_m": float(distances[index])})
            if len(neighbors) == k:
                break
        result[int(station["station_id"])] = neighbors
    return result
