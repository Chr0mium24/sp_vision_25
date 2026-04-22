from __future__ import annotations

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

_yaml = YAML(typ="safe")
_yaml_round_trip = YAML()
_yaml_round_trip.preserve_quotes = True


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = _yaml.load(handle)
    return data or {}


def read_scalar(path: Path, key: str) -> str | None:
    data = load_yaml(path)
    value = data.get(key)
    if value is None:
        return None
    return str(value)


def update_scalar(path: Path, key: str, value: Any) -> None:
    with path.open("r", encoding="utf-8") as handle:
        data = _yaml_round_trip.load(handle) or {}
    data[key] = value
    with path.open("w", encoding="utf-8") as handle:
        _yaml_round_trip.dump(data, handle)
