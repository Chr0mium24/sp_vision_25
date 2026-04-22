from __future__ import annotations

import fcntl
import os
import sys
import termios
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAD2DEG = 57.29577951308232
DEG2RAD = 1.0 / RAD2DEG


def _load_bindings() -> Any:
    try:
        import sp_vision_bindings as bindings
    except ImportError as exc:  # pragma: no cover - import path is exercised in integration tests
        raise RuntimeError(
            "sp_vision_bindings is not available. Run ./build.sh or uv sync first."
        ) from exc
    return bindings


def _mode_name(mode: Any) -> str:
    return {
        0: "IDLE",
        1: "AUTO_AIM",
        2: "SMALL_BUFF",
        3: "BIG_BUFF",
    }.get(int(mode), "INVALID")


def _state_dict(state: Any) -> dict[str, Any]:
    return {
        "yaw": float(state.yaw),
        "yaw_vel": float(state.yaw_vel),
        "pitch": float(state.pitch),
        "pitch_vel": float(state.pitch_vel),
        "roll": float(state.roll),
        "yaw_odom": float(state.yaw_odom),
        "pitch_odom": float(state.pitch_odom),
        "bullet_speed": float(state.bullet_speed),
        "bullet_count": int(state.bullet_count),
        "robot_id": int(state.robot_id),
    }


def _rx_dict(rx: Any) -> dict[str, Any]:
    age = getattr(rx, "last_good_frame_age_ms", None)
    return {
        "good_frames": int(rx.good_frames),
        "crc_fail": int(rx.crc_fail),
        "short_read": int(rx.short_read),
        "header_mismatch": int(rx.header_mismatch),
        "reconnect_count": int(rx.reconnect_count),
        "consecutive_crc_fail": int(rx.consecutive_crc_fail),
        "last_header": int(rx.last_header),
        "last_rx_crc": int(rx.last_rx_crc),
        "last_calc_crc": int(rx.last_calc_crc),
        "last_good_frame_age_ms": None if age is None else float(age),
    }


def _ypr_tuple(ypr: Any) -> tuple[float, float, float]:
    return float(ypr[0]), float(ypr[1]), float(ypr[2])


def _wait_valid(gimbal: Any, wait_valid_ms: int) -> bool:
    deadline = time.monotonic() + max(0, wait_valid_ms) / 1000.0
    while time.monotonic() < deadline:
        if gimbal.has_valid_q():
            return True
        time.sleep(0.01)
    return gimbal.has_valid_q()


def _take_flag(extra_args: list[str], name: str, default: bool) -> tuple[bool, list[str]]:
    remaining: list[str] = []
    value = default
    found = False
    negated = name if name.startswith("--no-") else f"--no-{name.lstrip('-')}"
    for arg in extra_args:
        if arg == name:
            value = True
            found = True
        elif arg == negated:
            value = False
            found = True
        else:
            remaining.append(arg)
    return value if found else default, remaining


def _take_int(extra_args: list[str], name: str, default: int) -> tuple[int, list[str]]:
    remaining: list[str] = []
    value = default
    found = False
    for arg in extra_args:
        if arg.startswith(f"{name}="):
            try:
                value = int(float(arg.split("=", 1)[1]))
                found = True
            except ValueError:
                pass
        else:
            remaining.append(arg)
    return value if found else default, remaining


def _take_float(extra_args: list[str], name: str, default: float) -> tuple[float, list[str]]:
    remaining: list[str] = []
    value = default
    found = False
    for arg in extra_args:
        if arg.startswith(f"{name}="):
            try:
                value = float(arg.split("=", 1)[1])
                found = True
            except ValueError:
                pass
        else:
            remaining.append(arg)
    return value if found else default, remaining


def _take_string(extra_args: list[str], name: str, default: str) -> tuple[str, list[str]]:
    remaining: list[str] = []
    value = default
    found = False
    for arg in extra_args:
        if arg.startswith(f"{name}="):
            value = arg.split("=", 1)[1]
            found = True
        else:
            remaining.append(arg)
    return value if found else default, remaining


def _parse_control_options(extra_args: list[str]) -> tuple[dict[str, Any], list[str]]:
    options: dict[str, Any] = {
        "wait_valid_ms": 1500,
        "duration_ms": 5000,
        "summary_ms": 1000,
        "loop_ms": 10,
        "mode": "control",
        "tracking": True,
        "fric_on": True,
        "fire_mode": 1,
        "yaw_deg": 3.0,
        "pitch_deg": -1.0,
        "no_input": False,
    }
    remaining = list(extra_args)
    for key in ("--wait-valid-ms", "--duration-ms", "--summary-ms", "--loop-ms", "--fire-mode"):
        if key in {"--fire-mode"}:
            value, remaining = _take_int(remaining, key, int(options[key.lstrip("-").replace("-", "_")]))
        else:
            value, remaining = _take_int(remaining, key, int(options[key.lstrip("-").replace("-", "_")]))
        options[key.lstrip("-").replace("-", "_")] = value
    value, remaining = _take_float(remaining, "--yaw-deg", float(options["yaw_deg"]))
    options["yaw_deg"] = value
    value, remaining = _take_float(remaining, "--pitch-deg", float(options["pitch_deg"]))
    options["pitch_deg"] = value
    value, remaining = _take_string(remaining, "--mode", str(options["mode"]))
    options["mode"] = value
    value, remaining = _take_flag(remaining, "--tracking", bool(options["tracking"]))
    options["tracking"] = value
    value, remaining = _take_flag(remaining, "--fric-on", bool(options["fric_on"]))
    options["fric_on"] = value
    value, remaining = _take_flag(remaining, "--no-input", bool(options["no_input"]))
    options["no_input"] = value
    return options, remaining


class TerminalRawMode:
    def __init__(self) -> None:
        self._orig: Any | None = None
        self._enabled = False

    def enable(self) -> bool:
        if not sys.stdin.isatty():
            return False
        try:
            self._orig = termios.tcgetattr(sys.stdin.fileno())
        except termios.error:
            return False

        raw = termios.tcgetattr(sys.stdin.fileno())
        raw[3] &= ~(termios.ICANON | termios.ECHO)
        raw[6][termios.VMIN] = 0
        raw[6][termios.VTIME] = 0
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, raw)
            flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except OSError:
            return False
        self._enabled = True
        return True

    def close(self) -> None:
        if self._enabled and self._orig is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._orig)
            self._enabled = False

    def __enter__(self) -> "TerminalRawMode":
        self.enable()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class GimbalSnapshot:
    mode: str
    state: dict[str, Any]
    rx_stats: dict[str, Any]
    ypr: tuple[float, float, float]


class GimbalSession:
    def __init__(self, config_path: Path, wait_for_first_q: bool = False) -> None:
        bindings = _load_bindings()
        self._bindings = bindings
        self._gimbal = bindings.Gimbal(str(config_path), wait_for_first_q)

    @property
    def backend(self) -> Any:
        return self._gimbal

    def snapshot(self) -> GimbalSnapshot:
        mode = _mode_name(self._gimbal.mode())
        state = _state_dict(self._gimbal.state())
        rx_stats = _rx_dict(self._gimbal.rx_stats())
        ypr = _ypr_tuple(self._gimbal.ypr_now())
        return GimbalSnapshot(mode=mode, state=state, rx_stats=rx_stats, ypr=ypr)

    def send(self, *, tracking: bool, fire: bool, yaw_deg: float, pitch_deg: float) -> None:
        self._gimbal.send(
            bool(tracking),
            bool(fire),
            float(yaw_deg) * DEG2RAD,
            0.0,
            0.0,
            float(pitch_deg) * DEG2RAD,
            0.0,
            0.0,
        )

    def stop(self) -> None:
        self.send(tracking=False, fire=False, yaw_deg=0.0, pitch_deg=0.0)


def _render_snapshot(snapshot: GimbalSnapshot) -> None:
    state = snapshot.state
    rx = snapshot.rx_stats
    ypr = snapshot.ypr
    print("\033[2J\033[H", end="")
    print("Gimbal Python Session")
    print(
        f"mode={snapshot.mode} valid_q={int(rx['good_frames'] > 0)} "
        f"ypr(deg)=({ypr[0] * RAD2DEG:+.2f}, {ypr[1] * RAD2DEG:+.2f}, {ypr[2] * RAD2DEG:+.2f})"
    )
    print(
        f"state(deg): yaw={state['yaw'] * RAD2DEG:+.2f} pitch={state['pitch'] * RAD2DEG:+.2f} "
        f"roll={state['roll'] * RAD2DEG:+.2f} yaw_odom={state['yaw_odom'] * RAD2DEG:+.2f} "
        f"pitch_odom={state['pitch_odom'] * RAD2DEG:+.2f}"
    )
    print(
        f"vel(rad/s): yaw={state['yaw_vel']:+.3f} pitch={state['pitch_vel']:+.3f} "
        f"bullet_speed={state['bullet_speed']:.2f} bullet_count={state['bullet_count']} "
        f"robot_id={state['robot_id']}"
    )
    print(
        f"rx: good={rx['good_frames']} crc_fail={rx['crc_fail']} short_read={rx['short_read']} "
        f"bad_header={rx['header_mismatch']} reconnect={rx['reconnect_count']} "
        f"consec_crc={rx['consecutive_crc_fail']} last_crc(rx/calc)=0x{rx['last_rx_crc']:04X}/0x{rx['last_calc_crc']:04X}"
    )
    if rx["last_good_frame_age_ms"] is not None:
        print(f"last good frame age: {rx['last_good_frame_age_ms']:.1f} ms")
    print("Keys: q quit | w/s pitch +/- | a/d yaw -/+ | [/] step | 0 reset | c tracking | r fric | 1 off 2 ready 3 single 4 fire | f toggle fire | space pulse")


def run_gimbal_snapshot(config_path: Path, extra_args: list[str]) -> int:
    options, remaining = _parse_control_options(extra_args)
    if remaining:
        print(f"[gimbal snapshot] ignored args: {' '.join(remaining)}")

    session = GimbalSession(config_path)
    if not _wait_valid(session.backend, int(options["wait_valid_ms"])):
        print(f"[gimbal snapshot] failed to receive valid gimbal feedback within {options['wait_valid_ms']} ms")
        return 2

    _render_snapshot(session.snapshot())
    return 0


def run_gimbal_watch(config_path: Path, extra_args: list[str]) -> int:
    options, remaining = _parse_control_options(extra_args)
    if remaining:
        print(f"[gimbal watch] ignored args: {' '.join(remaining)}")

    session = GimbalSession(config_path)
    if not _wait_valid(session.backend, int(options["wait_valid_ms"])):
        print(f"[gimbal watch] failed to receive valid gimbal feedback within {options['wait_valid_ms']} ms")
        return 2

    duration_ms = int(options["duration_ms"])
    loop_ms = max(1, int(options["loop_ms"]))
    summary_ms = max(loop_ms, int(options["summary_ms"]))
    deadline = time.monotonic() + duration_ms / 1000.0
    next_summary = 0.0
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_summary:
                _render_snapshot(session.snapshot())
                next_summary = now + summary_ms / 1000.0
            time.sleep(loop_ms / 1000.0)
    finally:
        session.stop()
    return 0


def _read_key() -> tuple[str | None, int | None]:
    try:
        data = os.read(sys.stdin.fileno(), 1)
    except BlockingIOError:
        return None, None
    except OSError:
        return None, None
    if not data:
        return None, None

    ch = data[0]
    if ch == ord("q"):
        return "quit", ch
    if ch == 27:
        try:
            seq = os.read(sys.stdin.fileno(), 2)
        except OSError:
            return None, None
        if len(seq) < 2 or seq[0] != ord("["):
            return None, None
        mapping = {ord("A"): "up", ord("B"): "down", ord("C"): "right", ord("D"): "left"}
        return mapping.get(seq[1]), seq[1]
    return "char", ch


def _run_gimbal_control_loop(config_path: Path, extra_args: list[str], scripted: bool) -> int:
    options, remaining = _parse_control_options(extra_args)
    if remaining:
        print(f"[gimbal control] ignored args: {' '.join(remaining)}")

    session = GimbalSession(config_path)
    if not _wait_valid(session.backend, int(options["wait_valid_ms"])):
        print(f"[gimbal control] failed to receive valid gimbal feedback within {options['wait_valid_ms']} ms")
        return 2

    if scripted:
        options["no_input"] = True

    if options["mode"] == "read":
        return run_gimbal_watch(config_path, extra_args)

    tracking = bool(options["tracking"])
    fric_on = bool(options["fric_on"])
    fire_mode = int(options["fire_mode"])
    fire_pulse_until = 0.0
    fire_pulse = False
    yaw_deg = 0.0
    pitch_deg = 0.0
    step_deg = 5.0
    loop_ms = max(1, int(options["loop_ms"]))
    summary_ms = max(loop_ms, int(options["summary_ms"]))
    duration_ms = int(options["duration_ms"])
    deadline = time.monotonic() + duration_ms / 1000.0 if options["no_input"] else None

    raw_mode = TerminalRawMode()
    if not options["no_input"] and not raw_mode.enable():
        print("[gimbal control] stdin is not a TTY; falling back to scripted control")
        options["no_input"] = True
        deadline = time.monotonic() + duration_ms / 1000.0

    try:
        next_summary = 0.0
        while True:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break

            key, ch = (None, None)
            if not options["no_input"]:
                key, ch = _read_key()

            if key == "quit":
                break
            if key == "up" or (key == "char" and ch in {ord("w"), ord("W")}):
                pitch_deg += step_deg
            elif key == "down" or (key == "char" and ch in {ord("s"), ord("S")}):
                pitch_deg -= step_deg
            elif key == "left" or (key == "char" and ch in {ord("a"), ord("A")}):
                yaw_deg -= step_deg
            elif key == "right" or (key == "char" and ch in {ord("d"), ord("D")}):
                yaw_deg += step_deg
            elif key == "char" and ch == ord("["):
                step_deg = max(0.1, step_deg / 2.0)
            elif key == "char" and ch == ord("]"):
                step_deg = min(20.0, step_deg * 2.0)
            elif key == "char" and ch == ord("0"):
                yaw_deg = 0.0
                pitch_deg = 0.0
            elif key == "char" and ch in {ord("c"), ord("C")}:
                tracking = not tracking
            elif key == "char" and ch in {ord("r"), ord("R")}:
                fric_on = not fric_on
            elif key == "char" and ch == ord("1"):
                fire_mode = 0
            elif key == "char" and ch == ord("2"):
                fire_mode = 1
            elif key == "char" and ch == ord("3"):
                fire_mode = 2
                fire_pulse = True
                fire_pulse_until = now + 0.2
            elif key == "char" and ch == ord("4"):
                fire_mode = 3
            elif key == "char" and ch in {ord("f"), ord("F")}:
                fire_mode = 3 if fire_mode == 0 else 0
            elif key == "char" and ch == ord(" "):
                fire_pulse = True
                fire_pulse_until = now + 0.2

            if fire_pulse and now >= fire_pulse_until:
                fire_pulse = False

            if not options["no_input"]:
                session.send(
                    tracking=tracking,
                    fire=bool(fire_mode == 3 or fire_pulse),
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                )
            else:
                session.send(
                    tracking=bool(options["tracking"]),
                    fire=bool(int(options["fire_mode"]) == 3 or fire_pulse),
                    yaw_deg=float(options["yaw_deg"]),
                    pitch_deg=float(options["pitch_deg"]),
                )

            if now >= next_summary:
                snapshot = session.snapshot()
                _render_snapshot(snapshot)
                print(
                    f"CMD(deg): yaw={yaw_deg:+.2f} pitch={pitch_deg:+.2f} step={step_deg:.2f} "
                    f"tracking={int(tracking)} fric={int(fric_on)} fire_mode={fire_mode} pulse={int(fire_pulse)}"
                )
                next_summary = now + summary_ms / 1000.0

            time.sleep(loop_ms / 1000.0)
    finally:
        raw_mode.close()
        session.stop()

    return 0


def run_gimbal_control(config_path: Path, extra_args: list[str]) -> int:
    return _run_gimbal_control_loop(config_path, extra_args, scripted=False)


def run_gimbal_script_control(config_path: Path, extra_args: list[str]) -> int:
    return _run_gimbal_control_loop(config_path, extra_args, scripted=True)
