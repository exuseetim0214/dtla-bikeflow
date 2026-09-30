from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_json(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value
