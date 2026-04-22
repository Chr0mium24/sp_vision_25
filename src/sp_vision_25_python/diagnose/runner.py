from __future__ import annotations

import subprocess
from pathlib import Path

from .paths import repo_root


def run_script(script: Path, args: list[str]) -> int:
    if not script.exists():
        raise FileNotFoundError(f"Missing diagnose script: {script}")

    completed = subprocess.run(
        [str(script), *args],
        cwd=repo_root(),
        check=False,
    )
    return completed.returncode

