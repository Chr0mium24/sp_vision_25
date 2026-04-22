from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    build_python = Path(__file__).resolve().parents[1] / "build" / "python"
    if build_python.exists():
        sys.path.insert(0, str(build_python))
