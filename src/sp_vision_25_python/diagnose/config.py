from __future__ import annotations

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

_yaml = YAML(typ="safe")


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
