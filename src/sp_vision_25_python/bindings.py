from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_python_dir() -> Path:
    return repo_root() / "build" / "python"


@dataclass(frozen=True)
class BindingStatus:
    available: bool
    path: str | None


def ensure_binding_path() -> str | None:
    path = build_python_dir()
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        return path_str
    return None


def load_bindings():
    ensure_binding_path()
    return importlib.import_module("sp_vision_bindings")


def binding_status() -> BindingStatus:
    try:
        module = load_bindings()
    except Exception:
        return BindingStatus(False, None)
    return BindingStatus(True, getattr(module, "__file__", None))
