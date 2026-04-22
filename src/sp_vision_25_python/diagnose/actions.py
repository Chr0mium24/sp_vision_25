from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rich.console import Console

from .config import read_scalar, update_scalar
from .inventory import binary_path
from .runner import run_executable
from .paths import repo_root
from .system import command_exists, default_config_path, gimbal_scan_ports, run_and_capture, run_silently


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

CAMERA_RELEASE_PATTERNS = [
    "component_container_mt",
    "ros2 launch rm_bringup",
    "ros2 launch foxglove_bridge",
    "rm_serial_driver_node",
    "armor_solver_node",
]


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


def _parse_flag(extra_args: list[str], name: str, default: str | bool) -> tuple[str | bool, list[str]]:
    remaining = []
    found = False
    value: str | bool = default
    for arg in extra_args:
        if arg.startswith(f"{name}="):
            value = arg.split("=", 1)[1]
            found = True
        elif arg == name and isinstance(default, bool):
            value = True
            found = True
        elif arg == f"--no-{name.lstrip('-')}":
            value = False
            found = True
        else:
            remaining.append(arg)
    return value if found else default, remaining


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _set_numeric_scalar(path: Path, key: str, value: float) -> None:
    if float(value).is_integer():
        update_scalar(path, key, int(value))
    else:
        update_scalar(path, key, float(value))


def _print_lines(lines: str) -> None:
    for line in lines.splitlines():
        print(line)


def handle_camera_action(action: str, config: Path | None, extra_args: list[str]) -> int | None:
    cfg = _config_path(config)

    if action == "help":
        print(CAMERA_HELP)
        return 0
    if action == "release":
        return _run_camera_release(extra_args)
    if action == "tune":
        return _run_camera_tune(cfg, extra_args)
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


def _run_camera_release(extra_args: list[str]) -> int:
    vidpid = "2bdf:0001"
    force = False
    for arg in extra_args:
        if arg.startswith("--vidpid="):
            vidpid = arg.split("=", 1)[1]
        elif arg == "--force":
            force = True

    if os.geteuid() != 0:
        print("[release] Please run with sudo.")
        print("  sudo diagnostics/camera/diagnose.sh release")
        return 1

    print(f"[release] target vid:pid={vidpid}")
    matches = []
    if command_exists("lsusb"):
        completed = run_and_capture(["lsusb"])
        for line in completed.stdout.splitlines():
            if vidpid.lower() in line.lower():
                matches.append(line)
    if matches:
        print("[release] matched devices:")
        for line in matches:
            print(line)
    else:
        print(f"[release] No USB device matched {vidpid} in lsusb.")

    usb_nodes: list[str] = []
    for line in matches:
        parts = line.split()
        if len(parts) < 4:
            continue
        bus = parts[1]
        dev = parts[3].rstrip(":")
        usb_nodes.append(f"/dev/bus/usb/{bus}/{dev}")

    if command_exists("docker"):
        completed = run_and_capture(["docker", "ps", "--format", "{{.ID}} {{.Names}} {{.Image}} {{.Command}}"])
        docker_rows = [
            line
            for line in completed.stdout.splitlines()
            if any(token in line for token in ("rm_bringup", "foxglove", "ros2", "camera_detector"))
        ]
        if docker_rows:
            print("[release] stopping docker containers:")
            for row in docker_rows:
                print(row)
            docker_ids = sorted({row.split()[0] for row in docker_rows if row.split()})
            for container_id in docker_ids:
                subprocess.run(["docker", "update", "--restart=no", container_id], check=False)
            subprocess.run(["docker", "stop", *docker_ids], check=False)
        else:
            print("[release] no matching running docker container found.")
    else:
        print("[release] docker not found, skip container stop.")

    for pattern in CAMERA_RELEASE_PATTERNS:
        if command_exists("pgrep"):
            completed = run_and_capture(["pgrep", "-f", pattern])
            pids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if pids:
                subprocess.run(["kill", *pids], check=False)

    for pattern in CAMERA_RELEASE_PATTERNS:
        if command_exists("pgrep"):
            completed = run_and_capture(["pgrep", "-f", pattern])
            pids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if pids:
                subprocess.run(["kill", "-9", *pids], check=False)

    for node in usb_nodes:
        if not Path(node).exists():
            continue
        print(f"[release] usb holder before: {node}")
        if command_exists("fuser"):
            subprocess.run(["fuser", "-v", node], check=False)
            if force:
                subprocess.run(["fuser", "-k", "-9", node], check=False)
            else:
                subprocess.run(["fuser", "-k", node], check=False)

    print("[release] remaining related processes:")
    if command_exists("pgrep"):
        completed = run_and_capture(
            [
                "pgrep",
                "-af",
                "component_container_mt|rm_bringup|ros2 launch|foxglove_bridge|rm_serial_driver_node|armor_solver_node|containerd-shim",
            ]
        )
        if completed.stdout.strip():
            _print_lines(completed.stdout.rstrip())
        else:
            print("  (none)")

    if not usb_nodes:
        print(f"[release] no usb node derived from lsusb {vidpid}.")
    else:
        for node in usb_nodes:
            if not Path(node).exists():
                continue
            print(f"[release] usb holder after: {node}")
            if command_exists("fuser"):
                completed = run_and_capture(["fuser", "-v", node])
                if completed.stdout.strip():
                    _print_lines(completed.stdout.rstrip())
                else:
                    print("  (free)")

    return 0


def _run_camera_tune(cfg: Path, extra_args: list[str]) -> int:
    scale = "0.5"
    quiet = True
    remaining = []
    for arg in extra_args:
        if arg.startswith("--scale="):
            scale = arg.split("=", 1)[1]
        elif arg == "--show-log":
            quiet = False
        else:
            remaining.append(arg)

    exposure = _to_float(read_scalar(cfg, "exposure_ms"), 2.5)
    gain = _to_float(read_scalar(cfg, "gain"), 16.9)

    def print_config() -> None:
        print(f"[camera] config: {cfg}")
        print(f"[camera] camera_name={read_scalar(cfg, 'camera_name') or '<missing>'} exposure_ms={read_scalar(cfg, 'exposure_ms') or '<missing>'} gain={read_scalar(cfg, 'gain') or '<missing>'} vid_pid={read_scalar(cfg, 'vid_pid') or '<missing>'}")

    window_proc: subprocess.Popen[str] | None = None

    def stop_window_process() -> None:
        nonlocal window_proc
        if window_proc is not None and window_proc.poll() is None:
            window_proc.terminate()
            try:
                window_proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                window_proc.kill()
                window_proc.wait(timeout=1)
        window_proc = None

    def start_window_process() -> None:
        nonlocal window_proc
        stop_window_process()
        print(f"[tune] reload window: exposure_ms={exposure} gain={gain} scale={scale}")
        stdout = None if not quiet else subprocess.DEVNULL
        stderr = None if not quiet else subprocess.DEVNULL
        window_proc = subprocess.Popen(
            [str(binary_path("camera", "camera_window_test")), str(cfg), f"--scale={scale}", *remaining],
            cwd=repo_root(),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

    def apply_config() -> None:
        _set_numeric_scalar(cfg, "exposure_ms", exposure)
        _set_numeric_scalar(cfg, "gain", gain)
        print_config()

    print_config()
    print("[tune] commands:")
    print("  e [num]: set exposure_ms (no num -> prompt)")
    print("  g [num]: set gain (no num -> prompt)")
    print("  r: reload window, p: print current values, q: quit")

    start_window_process()
    try:
        while True:
            try:
                line = input("tune> ")
            except EOFError:
                break

            if line == "q":
                break
            if line == "r":
                start_window_process()
                continue
            if line == "p":
                print_config()
                continue
            if line == "":
                continue
            if line == "e" or line.startswith("e "):
                val = line[2:].strip() if line.startswith("e ") else input("exposure_ms> ").strip()
                if val:
                    exposure = _to_float(val, exposure)
                    apply_config()
                    start_window_process()
                else:
                    print(f"[tune] invalid exposure value: {val}")
                continue
            if line == "g" or line.startswith("g "):
                val = line[2:].strip() if line.startswith("g ") else input("gain> ").strip()
                if val:
                    gain = _to_float(val, gain)
                    apply_config()
                    start_window_process()
                else:
                    print(f"[tune] invalid gain value: {val}")
                continue
            print(f"[tune] unknown command: {line} (supported: r p q e g)")
    finally:
        stop_window_process()
        print("[tune] exit.")

    return 0


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
    if action == "rune-tune":
        return _run_rune_tune(cfg, extra_args)
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


def _run_rune_tune(cfg: Path, extra_args: list[str]) -> int:
    input_prefix, rune_args = _pop_input_arg(extra_args, "assets/demo/power_rune_demo")
    yaw = _to_float(read_scalar(cfg, "yaw_offset"), 0.0)
    pitch = _to_float(read_scalar(cfg, "pitch_offset"), 0.0)
    fire_gap = _to_float(read_scalar(cfg, "fire_gap_time"), 0.7)
    predict_time = _to_float(read_scalar(cfg, "predict_time"), 0.12)

    def print_config() -> None:
        print(f"[rune-tune] config={cfg}")
        print(
            "[rune-tune] yaw_offset="
            f"{yaw} pitch_offset={pitch} fire_gap_time={fire_gap} predict_time={predict_time}"
        )

    def apply_config() -> None:
        _set_numeric_scalar(cfg, "yaw_offset", yaw)
        _set_numeric_scalar(cfg, "pitch_offset", pitch)
        _set_numeric_scalar(cfg, "fire_gap_time", fire_gap)
        _set_numeric_scalar(cfg, "predict_time", predict_time)

    def run_preview() -> int:
        print(f"[rune-tune] run: {binary_path('auto_aim', 'auto_power_rune_test')} --config-path={cfg} {input_prefix} {' '.join(rune_args)}")
        return run_executable(
            binary_path("auto_aim", "auto_power_rune_test"),
            [f"--config-path={cfg}", input_prefix, *rune_args],
        )

    print_config()
    print("[rune-tune] commands:")
    print("  y [num]: set yaw_offset(deg)")
    print("  i [num]: set pitch_offset(deg)")
    print("  f [num]: set fire_gap_time(s, >=0)")
    print("  t [num]: set predict_time(s, >=0)")
    print("  r: rerun power rune visualize")
    print("  p: print current params")
    print("  q: quit")

    while True:
        try:
            line = input("rune-tune> ")
        except EOFError:
            break

        if line == "q":
            break
        if line == "p":
            print_config()
            continue
        if line == "r":
            run_preview()
            continue
        if line == "":
            continue
        if line == "y" or line.startswith("y "):
            val = line[2:].strip() if line.startswith("y ") else input("yaw_offset(deg)> ").strip()
            if val:
                yaw = _to_float(val, yaw)
                apply_config()
                print_config()
            else:
                print(f"[rune-tune] invalid yaw_offset: {val}")
            continue
        if line == "i" or line.startswith("i "):
            val = line[2:].strip() if line.startswith("i ") else input("pitch_offset(deg)> ").strip()
            if val:
                pitch = _to_float(val, pitch)
                apply_config()
                print_config()
            else:
                print(f"[rune-tune] invalid pitch_offset: {val}")
            continue
        if line == "f" or line.startswith("f "):
            val = line[2:].strip() if line.startswith("f ") else input("fire_gap_time(s)> ").strip()
            if val:
                fire_gap = max(_to_float(val, fire_gap), 0.0)
                apply_config()
                print_config()
            else:
                print(f"[rune-tune] invalid fire_gap_time: {val}")
            continue
        if line == "t" or line.startswith("t "):
            val = line[2:].strip() if line.startswith("t ") else input("predict_time(s)> ").strip()
            if val:
                predict_time = max(_to_float(val, predict_time), 0.0)
                apply_config()
                print_config()
            else:
                print(f"[rune-tune] invalid predict_time: {val}")
            continue
        print(f"[rune-tune] unknown command: {line}")

    return 0
