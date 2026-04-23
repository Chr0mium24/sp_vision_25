from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "CMakeLists.txt").exists() and (candidate / "cpp").is_dir() and (
            candidate / "python"
        ).is_dir():
            return candidate
    return current


def build_dir() -> Path:
    return repo_root() / "build"


def build_python_dir() -> Path:
    return build_dir() / "python"


def diagnostics_dir() -> Path:
    return repo_root() / "diagnostics"


def diagnose_script(*parts: str) -> Path:
    return diagnostics_dir().joinpath(*parts)
