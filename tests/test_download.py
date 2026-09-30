import zipfile
from pathlib import Path

from bikeflow.download import _extract_single_csv, sha256_file


def test_extract_single_csv(tmp_path: Path) -> None:
    archive = tmp_path / "quarter.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/trips.csv", "trip_id,start_time\n1,2026-01-01\n")
    destination = tmp_path / "trips.csv"
    _extract_single_csv(archive, destination)
    assert destination.read_text().startswith("trip_id")
    assert len(sha256_file(destination)) == 64
