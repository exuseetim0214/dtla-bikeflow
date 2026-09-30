from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bikeflow.config import resolve_path

EXPECTED_COLUMNS = {
    "trip_id",
    "duration",
    "start_time",
    "end_time",
    "start_station",
    "start_lat",
    "start_lon",
    "end_station",
    "end_lat",
    "end_lon",
    "bike_id",
    "plan_duration",
    "trip_route_category",
    "passholder_type",
    "bike_type",
}


def _read_stations(raw_root: Path) -> tuple[pd.DataFrame, Path]:
    files = sorted((raw_root / "stations").glob("stations-*.csv"))
    if not files:
        raise FileNotFoundError("Station table missing. Run `bikeflow download` first.")
    path = files[-1]
    stations = pd.read_csv(path, encoding="utf-8-sig")
    stations.columns = [column.strip().lower().replace(" ", "_") for column in stations.columns]
    stations = stations.rename(columns={"kiosk_id": "station_id", "kiosk_name": "station_name"})
    stations["station_id"] = pd.to_numeric(stations["station_id"], errors="coerce").astype("Int64")
    stations["region"] = stations["region"].astype("string").str.strip()
    return stations, path


def audit_dataset(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "reports",
    *,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    raw_root = resolve_path(raw_dir)
    output_root = resolve_path(output_dir)
    trip_files = sorted((raw_root / "trips").glob("metro-trips-*.csv"))
    if not trip_files:
        raise FileNotFoundError("Trip CSVs missing. Run `bikeflow download` first.")

    stations, station_path = _read_stations(raw_root)
    region_by_id = stations.set_index("station_id")["region"]
    summaries: list[dict[str, Any]] = []
    global_station_ids: set[int] = set()
    total_rows = 0
    total_start_dtla = 0
    total_end_dtla = 0
    total_cross_out = 0
    total_cross_in = 0
    global_min: pd.Timestamp | None = None
    global_max: pd.Timestamp | None = None

    for path in trip_files:
        quarter_rows = 0
        missing = {name: 0 for name in EXPECTED_COLUMNS}
        quarter_ids: set[int] = set()
        quarter_min: pd.Timestamp | None = None
        quarter_max: pd.Timestamp | None = None
        duplicate_ids = 0
        seen_trip_ids: set[str] = set()
        start_dtla = end_dtla = cross_out = cross_in = 0

        header = set(pd.read_csv(path, nrows=0).columns)
        absent = sorted(EXPECTED_COLUMNS - header)
        if absent:
            raise ValueError(f"{path.name} is missing columns: {absent}")

        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            quarter_rows += len(chunk)
            for column in EXPECTED_COLUMNS:
                missing[column] += int(chunk[column].isna().sum())
            ids = chunk["trip_id"].astype("string")
            duplicate_ids += int(ids.isin(seen_trip_ids).sum()) + int(ids.duplicated().sum())
            seen_trip_ids.update(ids.dropna().tolist())

            start_ids = pd.to_numeric(chunk["start_station"], errors="coerce").astype("Int64")
            end_ids = pd.to_numeric(chunk["end_station"], errors="coerce").astype("Int64")
            quarter_ids.update(start_ids.dropna().astype(int).tolist())
            quarter_ids.update(end_ids.dropna().astype(int).tolist())
            start_region = start_ids.map(region_by_id)
            end_region = end_ids.map(region_by_id)
            start_mask = start_region.eq("DTLA").fillna(False)
            end_mask = end_region.eq("DTLA").fillna(False)
            start_dtla += int(start_mask.sum())
            end_dtla += int(end_mask.sum())
            cross_out += int((start_mask & ~end_mask).sum())
            cross_in += int((~start_mask & end_mask).sum())

            times = pd.to_datetime(chunk["start_time"], errors="coerce")
            chunk_min, chunk_max = times.min(), times.max()
            if pd.notna(chunk_min):
                quarter_min = chunk_min if quarter_min is None else min(quarter_min, chunk_min)
            if pd.notna(chunk_max):
                quarter_max = chunk_max if quarter_max is None else max(quarter_max, chunk_max)

        quarter = path.stem.removeprefix("metro-trips-")
        summaries.append(
            {
                "quarter": quarter,
                "rows": quarter_rows,
                "start_min": quarter_min.isoformat() if quarter_min is not None else None,
                "start_max": quarter_max.isoformat() if quarter_max is not None else None,
                "unique_station_ids": len(quarter_ids),
                "duplicate_trip_ids_within_quarter": duplicate_ids,
                "start_dtla": start_dtla,
                "end_dtla": end_dtla,
                "dtla_to_outside": cross_out,
                "outside_to_dtla": cross_in,
                "missing": missing,
            }
        )
        total_rows += quarter_rows
        global_station_ids.update(quarter_ids)
        total_start_dtla += start_dtla
        total_end_dtla += end_dtla
        total_cross_out += cross_out
        total_cross_in += cross_in
        if quarter_min is not None:
            global_min = quarter_min if global_min is None else min(global_min, quarter_min)
        if quarter_max is not None:
            global_max = quarter_max if global_max is None else max(global_max, quarter_max)

    station_ids = set(stations["station_id"].dropna().astype(int))
    dtla = stations.loc[stations["region"].eq("DTLA")]
    repository_root = resolve_path(".")
    result = {
        "source_files": len(trip_files),
        "station_table": str(station_path.relative_to(repository_root)),
        "total_rows": total_rows,
        "start_min": global_min.isoformat() if global_min is not None else None,
        "start_max": global_max.isoformat() if global_max is not None else None,
        "unique_station_ids": len(global_station_ids),
        "unmatched_station_ids": sorted(global_station_ids - station_ids),
        "station_table_rows": len(stations),
        "dtla_station_rows": len(dtla),
        "dtla_active_rows": int(dtla["status"].eq("Active").sum()),
        "start_dtla": total_start_dtla,
        "end_dtla": total_end_dtla,
        "dtla_to_outside": total_cross_out,
        "outside_to_dtla": total_cross_in,
        "quarters": summaries,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "data_audit.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path = output_root / "data_audit.md"
    rows = "\n".join(
        f"| {q['quarter']} | {q['rows']:,} | {q['start_dtla']:,} | {q['end_dtla']:,} | "
        f"{q['unique_station_ids']:,} |"
        for q in summaries
    )
    md_path.write_text(
        "# Metro Bike data audit\n\n"
        f"- Coverage: `{result['start_min']}` through `{result['start_max']}`\n"
        f"- Trip rows: **{total_rows:,}**\n"
        f"- Source files: **{len(trip_files)}**\n"
        f"- Station IDs observed: **{len(global_station_ids)}**\n"
        f"- Unmatched station IDs: **{len(result['unmatched_station_ids'])}**\n"
        f"- DTLA starts / ends: **{total_start_dtla:,} / {total_end_dtla:,}**\n\n"
        "| Quarter | Trips | DTLA starts | DTLA ends | Stations |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{rows}\n",
        encoding="utf-8",
    )
    result["json_report"] = str(json_path)
    result["markdown_report"] = str(md_path)
    return result
