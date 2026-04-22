from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .bindings import load_bindings
from .system import default_config_path


DEFAULT_DURATION_MS = 3000
DEFAULT_SUMMARY_MS = 1000


def _to_command(bindings: Any, command_tuple: tuple[Any, ...]):
    command = bindings.Command()
    command.control = bool(command_tuple[0])
    command.shoot = bool(command_tuple[1])
    command.yaw = float(command_tuple[2])
    command.pitch = float(command_tuple[3])
    command.horizon_distance = float(command_tuple[4])
    return command


def _maybe_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def run_auto_aim_live(
    config_path: Path | None,
    extra_args: list[str],
    *,
    show: bool,
    no_send: bool,
    use_enemy_color: bool = True,
    duration_ms: int = DEFAULT_DURATION_MS,
    summary_ms: int = DEFAULT_SUMMARY_MS,
    title: str = "auto-aim",
) -> int:
    cfg = config_path or default_config_path()
    bindings = load_bindings()
    cv2 = _maybe_cv2() if show else None

    if "--duration-ms=" in " ".join(extra_args):
        for arg in extra_args:
            if arg.startswith("--duration-ms="):
                duration_ms = max(1, int(float(arg.split("=", 1)[1])))
            elif arg.startswith("--summary-ms="):
                summary_ms = max(100, int(float(arg.split("=", 1)[1])))

    print(f"[{title}] config={cfg} show={'on' if show else 'off'} send={'off' if no_send else 'on'}")
    if cv2 is None and show:
        print(f"[{title}] cv2 not available, running without window")

    try:
        camera = bindings.Camera(str(cfg))
    except Exception as exc:
        print(f"[{title}] camera init failed: {exc}")
        return 1

    cboard = None
    if not no_send:
        try:
            cboard = bindings.CBoard(str(cfg))
        except Exception as exc:
            print(f"[{title}] cboard init failed, switching to no-send mode: {exc}")
            no_send = True

    runtime = bindings.Runtime(str(cfg), False)
    start = time.monotonic()
    last_summary = start
    frame_index = 0
    last_command = (False, False, 0.0, 0.0, 0.0)

    if cv2 is not None and show:
        try:
            cv2.namedWindow("Auto Aim Python", cv2.WINDOW_NORMAL)
        except Exception:
            cv2 = None

    while True:
        now = time.monotonic()
        if (now - start) * 1000.0 >= duration_ms:
            break

        try:
            image, timestamp_ns = camera.read()
        except Exception as exc:
            print(f"[{title}] camera read failed: {exc}")
            break

        if image is None or getattr(image, "size", 0) == 0:
            continue

        if cboard is not None:
            try:
                quat = cboard.imu_at(timestamp_ns)
                bullet_speed = float(cboard.bullet_speed)
            except Exception as exc:
                print(f"[{title}] cboard read failed: {exc}")
                quat = (1.0, 0.0, 0.0, 0.0)
                bullet_speed = 25.0
        else:
            quat = (1.0, 0.0, 0.0, 0.0)
            bullet_speed = 25.0

        try:
            result = runtime.step(image, quat, bullet_speed, frame_index, use_enemy_color, True)
        except Exception as exc:
            print(f"[{title}] runtime step failed: {exc}")
            break

        command = result["command"]
        last_command = command
        if cboard is not None and not no_send:
            try:
                cboard.send(_to_command(bindings, command))
            except Exception as exc:
                print(f"[{title}] send failed: {exc}")

        if show and cv2 is not None:
            try:
                display = image.copy()
                cv2.putText(
                    display,
                    f"state={result['tracker_state']} yaw={command[2]:+.3f} pitch={command[3]:+.3f}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Auto Aim Python", display)
                if cv2.waitKey(1) == ord("q"):
                    break
            except Exception:
                cv2 = None

        if now - last_summary >= summary_ms / 1000.0:
            print(
                f"[{title}] frame={frame_index} state={result['tracker_state']} "
                f"control={int(bool(command[0]))} shoot={int(bool(command[1]))} "
                f"yaw={command[2]:+.3f} pitch={command[3]:+.3f}"
            )
            last_summary = now

        frame_index += 1

    print(
        f"[{title}] done frames={frame_index} last_control={int(bool(last_command[0]))} "
        f"last_shoot={int(bool(last_command[1]))}"
    )
    return 0
