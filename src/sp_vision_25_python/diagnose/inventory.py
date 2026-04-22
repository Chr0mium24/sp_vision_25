from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import build_dir


@dataclass(frozen=True)
class BinarySpec:
    label: str
    path: Path


def _camera_specs() -> tuple[BinarySpec, ...]:
    base = build_dir() / "bin" / "tests" / "camera"
    return (
        BinarySpec("camera_test", base / "camera_test"),
        BinarySpec("camera_detect_test", base / "camera_detect_test"),
        BinarySpec("camera_window_test", base / "camera_window_test"),
        BinarySpec("camera_save_test", base / "camera_save_test"),
        BinarySpec("usbcamera_test", base / "usbcamera_test"),
        BinarySpec("usbcamera_detect_test", base / "usbcamera_detect_test"),
        BinarySpec("camera_thread_test", base / "camera_thread_test"),
        BinarySpec("handeye_test", base / "handeye_test"),
    )


def _gimbal_specs() -> tuple[BinarySpec, ...]:
    base = build_dir() / "bin" / "diag" / "gimbal"
    return (
        BinarySpec("gimbal_link_diag_test", base / "gimbal_link_diag_test"),
        BinarySpec("gimbal_serial_probe", base / "gimbal_serial_probe"),
    )


def _auto_aim_specs() -> tuple[BinarySpec, ...]:
    diag_dir = build_dir() / "bin" / "diag" / "auto_aim"
    buff_dir = build_dir() / "bin" / "diag" / "auto_buff"
    test_aim_dir = build_dir() / "bin" / "tests" / "auto_aim"
    test_buff_dir = build_dir() / "bin" / "tests" / "auto_buff"
    return (
        BinarySpec("auto_aim_ui_test", diag_dir / "auto_aim_ui_test"),
        BinarySpec("auto_aim_ui_tune", diag_dir / "auto_aim_ui_tune"),
        BinarySpec("auto_aim_test", test_aim_dir / "auto_aim_test"),
        BinarySpec("detector_video_test", test_aim_dir / "detector_video_test"),
        BinarySpec("auto_power_rune_test", test_buff_dir / "auto_power_rune_test"),
        BinarySpec("auto_buff_debug", buff_dir / "auto_buff_debug"),
        BinarySpec("auto_buff_debug_mpc", buff_dir / "auto_buff_debug_mpc"),
    )


def binary_specs(domain: str) -> tuple[BinarySpec, ...]:
    if domain == "camera":
        return _camera_specs()
    if domain == "gimbal":
        return _gimbal_specs()
    if domain == "auto_aim":
        return _auto_aim_specs()
    raise ValueError(f"Unknown binary domain: {domain}")


def binary_path(domain: str, label: str) -> Path:
    for spec in binary_specs(domain):
        if spec.label == label:
            return spec.path
    raise KeyError(f"Unknown binary label {label!r} for domain {domain!r}")


def print_binary_status(domain: str) -> None:
    for spec in binary_specs(domain):
        status = "ok" if spec.path.is_file() and os_access(spec.path) else "missing"
        print(f"[{status}] {spec.path}")


def os_access(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_mode & 0o111 != 0
