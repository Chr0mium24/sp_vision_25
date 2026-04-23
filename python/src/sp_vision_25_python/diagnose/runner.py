from __future__ import annotations

import subprocess
from pathlib import Path

from .paths import repo_root


def run_command(args: list[str]) -> int:
    completed = subprocess.run(
        args,
        cwd=repo_root(),
        check=False,
    )
    return completed.returncode


def run_executable(path: Path, args: list[str]) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing diagnose binary: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Diagnose binary is not a file: {path}")
    return run_command([str(path), *args])


def run_script(script: Path, args: list[str]) -> int:
    if not script.exists():
        raise FileNotFoundError(f"Missing diagnose script: {script}")

    return run_command([str(script), *args])
