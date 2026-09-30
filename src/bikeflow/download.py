from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from bikeflow.config import load_json, resolve_path

USER_AGENT = "dtla-bikeflow/0.1 (+https://bikeshare.metro.net/about/data/)"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, force: bool = False) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return {
            "url": url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "status": "cached",
        }

    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp_handle:
        temp_path = Path(temp_handle.name)
        try:
            with requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=(15, 120),
                stream=True,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temp_handle.write(chunk)
            temp_path.replace(destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "status": "downloaded",
    }


def _extract_single_csv(archive: Path, destination: Path, *, force: bool = False) -> Path:
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {archive}, found {members}")
        member = members[0]
        if Path(member).is_absolute() or ".." in Path(member).parts:
            raise ValueError(f"Unsafe ZIP member: {member}")
        with (
            bundle.open(member) as source,
            tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp_handle,
        ):
            temp_path = Path(temp_handle.name)
            try:
                shutil.copyfileobj(source, temp_handle)
                temp_path.replace(destination)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
    return destination


def download_dataset(
    manifest_path: str | Path = "configs/data_manifest.json",
    raw_dir: str | Path = "data/raw",
    *,
    force: bool = False,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    raw_root = resolve_path(raw_dir)
    archives_dir = raw_root / "archives"
    trips_dir = raw_root / "trips"
    stations_dir = raw_root / "stations"
    manifest_dir = raw_root / "manifests"

    records: list[dict[str, Any]] = []
    for item in manifest["trip_archives"]:
        quarter = item["quarter"]
        archive_path = archives_dir / f"metro-trips-{quarter}.zip"
        record = _download(item["url"], archive_path, force=force)
        csv_path = trips_dir / f"metro-trips-{quarter}.csv"
        _extract_single_csv(archive_path, csv_path, force=force)
        record.update(
            {
                "quarter": quarter,
                "extracted_path": str(csv_path),
                "extracted_bytes": csv_path.stat().st_size,
                "extracted_sha256": sha256_file(csv_path),
            }
        )
        records.append(record)

    station_item = manifest["station_table"]
    station_path = stations_dir / f"stations-{station_item['as_of']}.csv"
    station_record = _download(station_item["url"], station_path, force=force)
    station_record["as_of"] = station_item["as_of"]

    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_page": manifest["source_page"],
        "trip_archives": records,
        "station_table": station_record,
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_path = manifest_dir / "download_manifest.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["manifest_path"] = str(output_path)
    return result
