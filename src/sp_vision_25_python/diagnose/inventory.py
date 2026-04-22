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
    return ()


def _auto_aim_specs() -> tuple[BinarySpec, ...]:
    diag_dir = build_dir() / "bin" / "diag" / "auto_aim"
    buff_dir = build_dir() / "bin" / "diag" / "auto_buff"
    test_aim_dir = build_dir() / "bin" / "tests" / "auto_aim"
    test_buff_dir = build_dir() / "bin" / "tests" / "auto_buff"
    return (
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


def print_gimbal_status() -> None:
    print("[python] sp-vision-diagnose gimbal quick")
    print("[python] sp-vision-diagnose gimbal rxonly")
    print("[python] sp-vision-diagnose gimbal proto")
    print("[python] sp-vision-diagnose gimbal probe")
    print("[python] sp-vision-diagnose gimbal probe-raw")
    print("[python] sp-vision-diagnose gimbal scan")
    print("[python] sp-vision-diagnose gimbal snapshot")
    print("[python] sp-vision-diagnose gimbal watch")
    print("[python] sp-vision-diagnose gimbal control")
    print("[python] sp-vision-diagnose gimbal script-control")
    print("[python] sp-vision-diagnose gimbal axis")
    print("[python] sp-vision-diagnose gimbal manual-axis")
    specs = _gimbal_specs()
    if specs:
        print("[legacy]")
        for spec in specs:
            status = "ok" if spec.path.is_file() and os_access(spec.path) else "missing"
            print(f"[{status}] {spec.path}")
    else:
        print("[legacy] none")


def print_auto_aim_status() -> None:
    print("[python] sp-vision-diagnose auto-aim armor-box")
    print("[python] sp-vision-diagnose auto-aim armor-intent")
    print("[python] sp-vision-diagnose auto-aim armor-rec")
    print("[python] sp-vision-diagnose auto-aim armor-tune")
    print("[python] sp-vision-diagnose auto-aim armor-offline")
    print("[python] sp-vision-diagnose auto-aim rune-box")
    print("[python] sp-vision-diagnose auto-aim rune-rec")
    print("[python] sp-vision-diagnose auto-aim rune-tune")
    print("[python] sp-vision-diagnose auto-aim rune-online")
    print("[python] sp-vision-diagnose auto-aim rune-online-mpc")
    specs = _auto_aim_specs()
    if specs:
        print("[legacy]")
        for spec in specs:
            status = "ok" if spec.path.is_file() and os_access(spec.path) else "missing"
            print(f"[{status}] {spec.path}")
    else:
        print("[legacy] none")


def os_access(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_mode & 0o111 != 0
