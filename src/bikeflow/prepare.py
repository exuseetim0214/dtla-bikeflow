from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bikeflow.config import load_yaml, resolve_path


def load_station_table(raw_root: Path) -> pd.DataFrame:
    paths = sorted((raw_root / "stations").glob("stations-*.csv"))
    if not paths:
        raise FileNotFoundError("Station table missing. Run `bikeflow download` first.")
    stations = pd.read_csv(paths[-1], encoding="utf-8-sig")
    stations.columns = [column.strip().lower().replace(" ", "_") for column in stations.columns]
    stations = stations.rename(columns={"kiosk_id": "station_id", "kiosk_name": "station_name"})
    stations["station_id"] = pd.to_numeric(stations["station_id"], errors="coerce").astype("Int64")
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    stations["region"] = stations["region"].astype("string").str.strip()
    stations["go_live_date"] = pd.to_datetime(stations["go_live_date"], errors="coerce")
    return stations


def _aggregate_trip_flows(
    trip_files: list[Path],
    dtla_ids: set[int],
    virtual_ids: set[int],
    frequency: str,
    chunksize: int = 150_000,
) -> tuple[pd.Series, pd.Series, dict[int, tuple[pd.Timestamp, pd.Timestamp]]]:
    departures_parts: list[pd.Series] = []
    arrivals_parts: list[pd.Series] = []
    bounds: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}

    def update_bounds(station: pd.Series, timestamp: pd.Series) -> None:
        values = pd.DataFrame({"station_id": station, "timestamp": timestamp}).dropna()
        if values.empty:
            return
        grouped = values.groupby("station_id")["timestamp"].agg(["min", "max"])
        for station_id, row in grouped.iterrows():
            key = int(station_id)
            if key in bounds:
                bounds[key] = (min(bounds[key][0], row["min"]), max(bounds[key][1], row["max"]))
            else:
                bounds[key] = (row["min"], row["max"])

    for path in trip_files:
        usecols = ["start_time", "end_time", "start_station", "end_station"]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            start_station = pd.to_numeric(chunk["start_station"], errors="coerce").astype("Int64")
            end_station = pd.to_numeric(chunk["end_station"], errors="coerce").astype("Int64")
            start_time = pd.to_datetime(chunk["start_time"], errors="coerce")
            end_time = pd.to_datetime(chunk["end_time"], errors="coerce")

            start_mask = start_station.isin(dtla_ids - virtual_ids) & start_time.notna()
            end_mask = end_station.isin(dtla_ids - virtual_ids) & end_time.notna()
            starts = pd.DataFrame(
                {
                    "station_id": start_station[start_mask].astype(int),
                    "bin_start": start_time[start_mask].dt.floor(frequency),
                }
            )
            ends = pd.DataFrame(
                {
                    "station_id": end_station[end_mask].astype(int),
                    "bin_start": end_time[end_mask].dt.floor(frequency),
                }
            )
            if not starts.empty:
                departures_parts.append(starts.groupby(["station_id", "bin_start"]).size())
                update_bounds(starts["station_id"], starts["bin_start"])
            if not ends.empty:
                arrivals_parts.append(ends.groupby(["station_id", "bin_start"]).size())
                update_bounds(ends["station_id"], ends["bin_start"])

    def collapse(parts: list[pd.Series], name: str) -> pd.Series:
        if not parts:
            return pd.Series(dtype="int32", name=name)
        values = pd.concat(parts).groupby(level=[0, 1]).sum().astype("int32")
        values.name = name
        return values

    return collapse(departures_parts, "departures"), collapse(arrivals_parts, "arrivals"), bounds


def build_station_panel(
    station_id: int,
    station: pd.Series,
    departures: pd.Series,
    arrivals: pd.Series,
    bounds: tuple[pd.Timestamp, pd.Timestamp],
    *,
    frequency: str,
    lags: list[int],
    rolling_windows: list[int],
    horizons: list[int],
) -> pd.DataFrame:
    first_bin, last_bin = bounds
    bins = pd.date_range(first_bin.floor(frequency), last_bin.ceil(frequency), freq=frequency)

    def station_counts(values: pd.Series) -> pd.Series:
        station_level = values.index.get_level_values(0)
        if station_id not in station_level:
            return pd.Series(0, index=bins, dtype="int32")
        return values.xs(station_id, level=0).reindex(bins, fill_value=0)

    station_departures = station_counts(departures)
    station_arrivals = station_counts(arrivals)
    station_departures = station_departures.astype("int16")
    station_arrivals = station_arrivals.astype("int16")

    offset = pd.tseries.frequencies.to_offset(frequency)
    forecast_time = bins + offset
    frame = pd.DataFrame(
        {
            "station_id": np.int32(station_id),
            "forecast_time": forecast_time,
            "latitude": np.float32(station["latitude"]),
            "longitude": np.float32(station["longitude"]),
            "station_age_days": (forecast_time - station["go_live_date"]).days.astype("int32"),
        }
    )
    dep = pd.Series(station_departures.to_numpy(), index=frame.index)
    arr = pd.Series(station_arrivals.to_numpy(), index=frame.index)

    for lag in sorted(set(lags)):
        shift = lag - 1
        frame[f"departures_lag_{lag}"] = dep.shift(shift).astype("float32")
        frame[f"arrivals_lag_{lag}"] = arr.shift(shift).astype("float32")
        frame[f"net_flow_lag_{lag}"] = (arr - dep).shift(shift).astype("float32")

    for window in sorted(set(rolling_windows)):
        frame[f"departures_roll_sum_{window}"] = (
            dep.rolling(window, min_periods=1).sum().astype("float32")
        )
        frame[f"arrivals_roll_sum_{window}"] = (
            arr.rolling(window, min_periods=1).sum().astype("float32")
        )
        frame[f"net_flow_roll_sum_{window}"] = (
            (arr - dep).rolling(window, min_periods=1).sum().astype("float32")
        )

    hour = frame["forecast_time"].dt.hour + frame["forecast_time"].dt.minute / 60
    weekday = frame["forecast_time"].dt.dayofweek
    frame["hour"] = frame["forecast_time"].dt.hour.astype("int8")
    frame["day_of_week"] = weekday.astype("int8")
    frame["is_weekend"] = weekday.ge(5).astype("int8")
    frame["month"] = frame["forecast_time"].dt.month.astype("int8")
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    frame["dow_sin"] = np.sin(2 * np.pi * weekday / 7).astype("float32")
    frame["dow_cos"] = np.cos(2 * np.pi * weekday / 7).astype("float32")

    for horizon in sorted(set(horizons)):
        future_dep = sum(dep.shift(-step) for step in range(1, horizon + 1))
        future_arr = sum(arr.shift(-step) for step in range(1, horizon + 1))
        minutes = int(offset.nanos // 60_000_000_000) * horizon
        frame[f"target_departures_{minutes}m"] = future_dep.astype("float32")
        frame[f"target_arrivals_{minutes}m"] = future_arr.astype("float32")
        frame[f"target_net_flow_{minutes}m"] = (future_arr - future_dep).astype("float32")

    return frame


def prepare_dataset(config_path: str | Path = "configs/pipeline.yaml") -> dict[str, Any]:
    config = load_yaml(config_path)
    raw_root = resolve_path(config["paths"]["raw_dir"])
    processed_root = resolve_path(config["paths"]["processed_dir"])
    artifact_root = resolve_path(config["paths"]["artifact_dir"])
    frequency = config["data"]["time_bin"]
    region = config["data"]["region"]
    virtual_ids = {int(value) for value in config["data"]["virtual_station_ids"]}
    lags = [int(value) for value in config["features"]["demand_lags"]]
    windows = [int(value) for value in config["features"]["rolling_windows"]]
    horizons = [int(value) for value in config["targets"]["demand_horizons_bins"]]

    stations = load_station_table(raw_root)
    selected = stations.loc[stations["region"].eq(region)].dropna(
        subset=["station_id", "latitude", "longitude", "go_live_date"]
    )
    station_lookup = selected.set_index("station_id")
    dtla_ids = set(station_lookup.index.astype(int))
    trip_files = sorted((raw_root / "trips").glob("metro-trips-*.csv"))
    if not trip_files:
        raise FileNotFoundError("Trip CSVs missing. Run `bikeflow download` first.")

    departures, arrivals, bounds = _aggregate_trip_flows(
        trip_files, dtla_ids, virtual_ids, frequency
    )
    panel_root = processed_root / "panel"
    panel_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for station_id in sorted(bounds):
        if station_id not in station_lookup.index:
            continue
        panel = build_station_panel(
            station_id,
            station_lookup.loc[station_id],
            departures,
            arrivals,
            bounds[station_id],
            frequency=frequency,
            lags=lags,
            rolling_windows=windows,
            horizons=horizons,
        )
        station_dir = panel_root / f"station_id={station_id}"
        station_dir.mkdir(parents=True, exist_ok=True)
        path = station_dir / "panel.parquet"
        panel.to_parquet(path, index=False)
        written.append(
            {
                "station_id": station_id,
                "rows": len(panel),
                "start": panel["forecast_time"].min().isoformat(),
                "end": panel["forecast_time"].max().isoformat(),
                "path": str(path),
            }
        )

    station_output = processed_root / "stations.parquet"
    selected.to_parquet(station_output, index=False)
    summary = {
        "frequency": frequency,
        "region": region,
        "source_trip_files": len(trip_files),
        "stations_written": len(written),
        "panel_rows": sum(item["rows"] for item in written),
        "partitions": written,
        "station_table": str(station_output),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
