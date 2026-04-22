from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rich.console import Console

from .config import read_scalar
from .paths import repo_root


def video_devices() -> list[Path]:
    dev_dir = Path("/dev")
    return sorted(dev_dir.glob("video*"))


def serial_by_id_entries() -> list[Path]:
    return sorted(Path("/dev/serial/by-id").glob("*"))


def default_config_path() -> Path:
    return repo_root() / "configs" / "standard3.yaml"


def print_camera_info(console: Console | None = None) -> None:
    printer = console or Console()
    devices = video_devices()

    printer.print("[camera] /dev/video*", markup=False)
    if devices:
        for device in devices:
            printer.print(f"  {device}", markup=False)
    else:
        printer.print("  (no /dev/video*)", markup=False)

    if shutil.which("v4l2-ctl"):
        printer.print("[camera] v4l2-ctl --list-devices", markup=False)
        subprocess.run(["v4l2-ctl", "--list-devices"], check=False)
    else:
        printer.print("[camera] v4l2-ctl not found (sudo apt install v4l-utils)", markup=False)


def print_gimbal_port_info(config_path: Path | None = None, console: Console | None = None) -> None:
    printer = console or Console()
    cfg = config_path or default_config_path()

    printer.print("[diagnose] /dev/serial/by-id", markup=False)
    entries = serial_by_id_entries()
    if entries:
        for entry in entries:
            printer.print(f"  {entry}", markup=False)
    else:
        printer.print("  (no /dev/serial/by-id)", markup=False)

    com_port = read_scalar(cfg, "com_port")
    if not com_port:
        printer.print(f"[diagnose] com_port not found in {cfg}", markup=False)
        return

    printer.print(f"[diagnose] com_port from {cfg}: {com_port}", markup=False)
    if Path(com_port).exists():
        subprocess.run(
            ["udevadm", "info", "-a", "-n", com_port],
            check=False,
        )
    else:
        printer.print(f"[diagnose] {com_port} not present", markup=False)
