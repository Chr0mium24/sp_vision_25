from __future__ import annotations

from pathlib import Path

from .inventory import binary_path
from .runner import run_executable
from .system import default_config_path, gimbal_scan_ports


CAMERA_HELP = """Usage:
  sp-vision-diagnose camera <action> [config.yaml] [extra args...]

Actions:
  info
  list
  release
  tune
  quick
  detect
  window
  save
  usb
  usb-detect
  thread
  handeye
  help
"""

GIMBAL_HELP = """Usage:
  sp-vision-diagnose gimbal <action> [config.yaml] [extra args...]

Actions:
  quick
  rxonly
  proto
  probe
  probe-raw
  scan
  snapshot
  watch
  control
  script-control
  axis
  manual-axis
  port-info
  help
"""

AUTO_AIM_HELP = """Usage:
  sp-vision-diagnose auto-aim <action> [config.yaml] [extra args...]

Actions:
  list
  armor-box
  armor-intent
  armor-rec
  armor-tune
  armor-offline
  rune-box
  rune-rec
  rune-tune
  rune-online
  rune-online-mpc
  help
"""


def _config_path(config: Path | None) -> Path:
    return config or default_config_path()


def _has_show_arg(extra_args: list[str]) -> bool:
    for arg in extra_args:
        if arg in {"--show", "-s"} or arg.startswith("--show="):
            return True
    return False


def _pop_input_arg(extra_args: list[str], default: str) -> tuple[str, list[str]]:
    if extra_args and not extra_args[0].startswith("--"):
        return extra_args[0], extra_args[1:]
    return default, extra_args


def handle_camera_action(action: str, config: Path | None, extra_args: list[str]) -> int | None:
    cfg = _config_path(config)

    if action == "help":
        print(CAMERA_HELP)
        return 0
    if action == "quick":
        return run_executable(binary_path("camera", "camera_test"), [f"--config-path={cfg}", *extra_args])
    if action == "detect":
        return run_executable(binary_path("camera", "camera_detect_test"), [str(cfg), *extra_args])
    if action == "window":
        return run_executable(binary_path("camera", "camera_window_test"), [str(cfg), *extra_args])
    if action == "save":
        return run_executable(binary_path("camera", "camera_save_test"), [str(cfg), *extra_args])
    if action == "usb":
        return run_executable(binary_path("camera", "usbcamera_test"), [str(cfg), *extra_args])
    if action == "usb-detect":
        return run_executable(binary_path("camera", "usbcamera_detect_test"), [str(cfg), *extra_args])
    if action == "thread":
        return run_executable(binary_path("camera", "camera_thread_test"), [str(cfg), *extra_args])
    if action == "handeye":
        return run_executable(binary_path("camera", "handeye_test"), [f"--config-path={cfg}", *extra_args])
    return None


def handle_gimbal_action(action: str, config: Path | None, extra_args: list[str]) -> int | None:
    cfg = _config_path(config)

    if action == "help":
        print(GIMBAL_HELP)
        return 0
    if action == "quick":
        return run_executable(
            binary_path("gimbal", "gimbal_link_diag_test"),
            [str(cfg), "--duration-ms=3000", "--summary-ms=1000", *extra_args],
        )
    if action == "rxonly":
        return run_executable(
            binary_path("gimbal", "gimbal_link_diag_test"),
            [str(cfg), "--no-send", "--duration-ms=3000", "--summary-ms=1000", *extra_args],
        )
    if action == "proto":
        return run_executable(
            binary_path("gimbal", "gimbal_link_diag_test"),
            [str(cfg), "--no-send", "--require-rx", "--duration-ms=2200", "--summary-ms=1000", *extra_args],
        )
    if action == "probe":
        return run_executable(
            binary_path("gimbal", "gimbal_serial_probe"),
            [str(cfg), "--duration-ms=3000", "--summary-ms=1000", *extra_args],
        )
    if action == "probe-raw":
        return run_executable(
            binary_path("gimbal", "gimbal_serial_probe"),
            [str(cfg), "--duration-ms=1200", "--summary-ms=1200", "--raw-log", "--hex-len=32", *extra_args],
        )
    if action == "scan":
        ports = ",".join(gimbal_scan_ports())
        return run_executable(
            binary_path("gimbal", "gimbal_link_diag_test"),
            [str(cfg), f"--ports={ports}", "--duration-ms=3000", "--summary-ms=1000", *extra_args],
        )
    if action == "snapshot":
        return run_executable(
            binary_path("gimbal", "gimbal_ui_test"),
            [str(cfg), "--mode=read", "--dump-once", "--wait-valid-ms=1500", "--nogui", *extra_args],
        )
    if action == "watch":
        return run_executable(
            binary_path("gimbal", "gimbal_ui_test"),
            [str(cfg), "--mode=read", "--nogui", *extra_args],
        )
    if action == "control":
        return run_executable(binary_path("gimbal", "gimbal_ui_test"), [str(cfg), "--mode=control", *extra_args])
    if action == "script-control":
        return run_executable(
            binary_path("gimbal", "gimbal_ui_test"),
            [
                str(cfg),
                "--mode=control",
                "--no-input",
                "--duration-ms=5000",
                "--yaw-deg=3",
                "--pitch-deg=-1",
                "--tracking=1",
                "--fric-on=1",
                "--fire-mode=1",
                *extra_args,
            ],
        )
    if action == "axis":
        return run_executable(binary_path("gimbal", "gimbal_axis_diag_test"), [str(cfg), *extra_args])
    if action == "manual-axis":
        return run_executable(binary_path("gimbal", "gimbal_manual_axis_diag_test"), [str(cfg), *extra_args])
    return None


def handle_auto_aim_action(action: str, config: Path | None, extra_args: list[str]) -> int | None:
    cfg = _config_path(config)

    if action == "help":
        print(AUTO_AIM_HELP)
        return 0
    if action == "armor-box":
        args = [str(cfg), *extra_args]
        if not _has_show_arg(extra_args):
            args.insert(1, "--show=true")
        return run_executable(binary_path("auto_aim", "auto_aim_ui_test"), args)
    if action == "armor-intent":
        args = [str(cfg), "--no-send=true", *extra_args]
        if not _has_show_arg(extra_args):
            args.insert(1, "--show=true")
        return run_executable(binary_path("auto_aim", "auto_aim_ui_test"), args)
    if action == "armor-rec":
        return run_executable(binary_path("auto_aim", "auto_aim_ui_test"), [str(cfg), *extra_args])
    if action == "armor-tune":
        args = [str(cfg), *extra_args]
        if not _has_show_arg(extra_args):
            args.insert(1, "--show=true")
        return run_executable(binary_path("auto_aim", "auto_aim_ui_tune"), args)
    if action == "armor-offline":
        input_prefix, remaining = _pop_input_arg(extra_args, "assets/demo/demo")
        return run_executable(
            binary_path("auto_aim", "auto_aim_test"),
            [f"--config-path={cfg}", input_prefix, *remaining],
        )
    if action in {"rune-box", "rune-rec"}:
        input_prefix, remaining = _pop_input_arg(extra_args, "assets/demo/power_rune_demo")
        return run_executable(
            binary_path("auto_aim", "auto_power_rune_test"),
            [f"--config-path={cfg}", input_prefix, *remaining],
        )
    if action == "rune-online":
        return run_executable(binary_path("auto_aim", "auto_buff_debug"), [str(cfg), *extra_args])
    if action == "rune-online-mpc":
        return run_executable(binary_path("auto_aim", "auto_buff_debug_mpc"), [str(cfg), *extra_args])
    return None
