from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from bikeflow.config import resolve_path

DEFAULT_STATUS_URL = "https://bikeshare.metro.net/stations/json/"


def normalize_geojson(payload: dict[str, Any], collected_at: datetime) -> pd.DataFrame:
    records = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        coordinates = feature.get("geometry", {}).get("coordinates", [None, None])
        records.append(
            {
                "collected_at_utc": collected_at.astimezone(UTC),
                "station_id": properties.get("kioskId"),
                "station_name": properties.get("name"),
                "latitude": properties.get("latitude", coordinates[1]),
                "longitude": properties.get("longitude", coordinates[0]),
                "bikes_available": properties.get("bikesAvailable"),
                "docks_available": properties.get("docksAvailable"),
                "total_docks": properties.get("totalDocks"),
                "station_status": properties.get("kioskStatus"),
                "public_status": properties.get("kioskPublicStatus"),
                "connection_status": properties.get("kioskConnectionStatus"),
                "is_virtual": properties.get("isVirtual"),
                "is_event_based": properties.get("isEventBased"),
            }
        )
    return pd.DataFrame.from_records(records)


def collect_status_snapshot(
    url: str = DEFAULT_STATUS_URL,
    output_dir: str | Path = "data/status",
) -> dict[str, Any]:
    collected_at = datetime.now(UTC)
    response = requests.get(url, timeout=(15, 60), headers={"User-Agent": "dtla-bikeflow/0.1"})
    response.raise_for_status()
    payload = response.json()
    frame = normalize_geojson(payload, collected_at)
    root = resolve_path(output_dir)
    date_dir = root / collected_at.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    stem = collected_at.strftime("%Y%m%dT%H%M%SZ")
    parquet_path = date_dir / f"station-status-{stem}.parquet"
    json_path = date_dir / f"station-status-{stem}.json"
    frame.to_parquet(parquet_path, index=False)
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "collected_at_utc": collected_at.isoformat(),
        "stations": len(frame),
        "parquet_path": str(parquet_path),
        "raw_json_path": str(json_path),
    }


def add_future_availability_labels(
    frame: pd.DataFrame,
    *,
    snapshots_ahead: int = 6,
    confirmations: int = 2,
) -> pd.DataFrame:
    required = {"station_id", "collected_at_utc", "bikes_available", "docks_available"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Status frame must contain {sorted(required)}")
    output = frame.sort_values(["station_id", "collected_at_utc"]).copy()

    def label_group(group: pd.DataFrame) -> pd.DataFrame:
        bikes = group["bikes_available"].eq(0)
        docks = group["docks_available"].eq(0)
        empty_label = pd.Series(False, index=group.index)
        full_label = pd.Series(False, index=group.index)
        for step in range(1, snapshots_ahead):
            empty_pair = bikes.shift(-step) & bikes.shift(-(step + confirmations - 1))
            full_pair = docks.shift(-step) & docks.shift(-(step + confirmations - 1))
            empty_label |= empty_pair.fillna(False)
            full_label |= full_pair.fillna(False)
        group["target_empty_30m"] = empty_label.astype("int8")
        group["target_full_30m"] = full_label.astype("int8")
        return group

    labeled = [label_group(group) for _, group in output.groupby("station_id", sort=False)]
    return pd.concat(labeled).sort_index()
