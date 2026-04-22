from __future__ import annotations

import fcntl
import math
import os
import sys
import termios
import time
from itertools import permutations, product
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


def _rotation_matrix(ypr: tuple[float, float, float]) -> np.ndarray:
    yaw, pitch, roll = ypr
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _eulers_zyx(R: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    return yaw, pitch, roll


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


@dataclass
class AxisResult:
    command_name: str
    step_deg: float
    delta_deg: tuple[float, float, float]
    dominant_axis: int
    sign_match: bool


def _axis_name(axis: int) -> str:
    return {0: "yaw", 1: "pitch", 2: "roll"}.get(axis, "unknown")


def _delta_deg(after: tuple[float, float, float], before: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple((a - b) * RAD2DEG for a, b in zip(after, before))


def _dominant_axis(delta: tuple[float, float, float]) -> int:
    return max(range(3), key=lambda i: abs(delta[i]))


def _average_ypr(session: GimbalSession, duration_ms: int, loop_ms: int) -> tuple[float, float, float]:
    start = time.monotonic()
    total = np.zeros(3, dtype=float)
    count = 0
    while time.monotonic() - start < duration_ms / 1000.0:
        total += np.array(session.snapshot().ypr, dtype=float)
        count += 1
        time.sleep(loop_ms / 1000.0)
    if count == 0:
        return session.snapshot().ypr
    averaged = total / count
    return float(averaged[0]), float(averaged[1]), float(averaged[2])


def _send_plan(session: GimbalSession, yaw: float, pitch: float) -> None:
    session.send(tracking=True, fire=False, yaw_deg=yaw, pitch_deg=pitch)


def run_gimbal_manual_axis(config_path: Path, extra_args: list[str]) -> int:
    options, remaining = _take_int(list(extra_args), "--wait-valid-ms", 1500)
    wait_valid_ms = max(100, options)
    remaining_args = remaining
    options, remaining_args = _take_int(remaining_args, "--sample-ms", 700)
    sample_ms = max(100, options)
    options, remaining_args = _take_int(remaining_args, "--loop-ms", 5)
    loop_ms = max(1, options)
    if remaining_args:
        print(f"[manual-axis] ignored args: {' '.join(remaining_args)}")

    session = GimbalSession(config_path)
    if not _wait_valid(session.backend, wait_valid_ms):
        print(f"[manual-axis] failed to receive valid gimbal feedback within {wait_valid_ms} ms")
        return 2

    steps = [
        ("up", "请手动向上抬枪口，保持住后按回车"),
        ("down", "请手动向下压枪口，保持住后按回车"),
        ("right", "请手动向右转云台，保持住后按回车"),
        ("left", "请手动向左转云台，保持住后按回车"),
    ]

    print("[manual-axis] 纯读取模式，不会下发控制。每一步先回到自然位置。")
    print("[manual-axis] 采样前会先记录当前基准，再记录你保持动作时的反馈变化。")
    print("")

    try:
        for name, prompt in steps:
            print("------------------------------------------------------------")
            print(f"[manual-axis] step={name}")
            print("[manual-axis] 先回到自然位置，然后按回车记录 baseline")
            input()
            baseline = _average_ypr(session, sample_ms, loop_ms)
            print(
                f"[manual-axis] baseline(deg): yaw={baseline[0] * RAD2DEG:+.2f} "
                f"pitch={baseline[1] * RAD2DEG:+.2f} roll={baseline[2] * RAD2DEG:+.2f}"
            )
            print(prompt)
            input()
            moved = _average_ypr(session, sample_ms, loop_ms)
            delta = _delta_deg(moved, baseline)
            axis = _dominant_axis(delta)
            print(
                f"[manual-axis] step={name} delta(deg): yaw={delta[0]:+.2f} pitch={delta[1]:+.2f} "
                f"roll={delta[2]:+.2f} dominant={_axis_name(axis)}"
            )
    finally:
        session.stop()

    print("------------------------------------------------------------")
    print("[manual-axis] 解释：")
    print("  up/down 主要应该对应 pitch")
    print("  right/left 主要应该对应 yaw")
    print("  如果 up/down 主要对应了 roll，说明 C 板安装轴和枪口轴存在 90 度映射问题")
    print("  如果 up 主要对应 pitch 但符号和预期相反，更像是 pitch 符号反了")
    return 0


def run_gimbal_axis(config_path: Path, extra_args: list[str]) -> int:
    options = {
        "step_deg": 5.0,
        "settle_ms": 700,
        "sample_ms": 250,
        "wait_valid_ms": 1500,
        "loop_ms": 5,
    }
    remaining = list(extra_args)

    for key in ("--step-deg", "--settle-ms", "--sample-ms", "--wait-valid-ms", "--loop-ms"):
        if key in {"--step-deg"}:
            value, remaining = _take_float(remaining, key, float(options["step_deg"]))
            options["step_deg"] = float(value)
        else:
            value, remaining = _take_int(remaining, key, int(options[key.lstrip("-").replace("-", "_")]))
            options[key.lstrip("-").replace("-", "_")] = int(value)
    if remaining:
        print(f"[axis] ignored args: {' '.join(remaining)}")

    step_deg = max(0.5, min(15.0, float(options["step_deg"])))
    settle_ms = max(100, int(options["settle_ms"]))
    sample_ms = max(50, int(options["sample_ms"]))
    wait_valid_ms = max(100, int(options["wait_valid_ms"]))
    loop_ms = max(1, int(options["loop_ms"]))

    session = GimbalSession(config_path)
    if not _wait_valid(session.backend, wait_valid_ms):
        print(f"[axis] failed to receive valid gimbal feedback within {wait_valid_ms} ms")
        return 2

    base_snapshot = session.snapshot()
    base_yaw = base_snapshot.state["yaw"]
    base_pitch = base_snapshot.state["pitch"]

    print(f"[axis] baseline command yaw={base_yaw * RAD2DEG:+.2f}deg pitch={base_pitch * RAD2DEG:+.2f}deg")
    _send_plan(session, base_yaw * RAD2DEG, base_pitch * RAD2DEG)
    time.sleep(settle_ms / 1000.0)

    yaw_result = _axis_step(session, "yaw+", base_yaw, base_pitch, step_deg, True, settle_ms, sample_ms, loop_ms)
    pitch_result = _axis_step(session, "pitch+", base_yaw, base_pitch, step_deg, False, settle_ms, sample_ms, loop_ms)
    _print_axis_result(yaw_result)
    _print_axis_result(pitch_result)
    _print_axis_conclusion(yaw_result, pitch_result)

    print("")
    print("[sweep] ranking axis-aligned R_gimbal2imubody candidates:")
    candidates = _build_candidates()
    candidate_results = _evaluate_candidates(
        session, candidates, sample_ms, loop_ms, base_yaw, base_pitch, step_deg, settle_ms
    )
    for idx, result in enumerate(candidate_results[:5], start=1):
        _print_candidate(result, idx)

    if candidate_results:
        best = candidate_results[0]
        row = best["candidate"]["row_major"]
        print(
            "[recommend] try setting R_gimbal2imubody: "
            f"[{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}, {row[6]}, {row[7]}, {row[8]}]"
        )

    session.stop()
    return 0


def _axis_step(
    session: GimbalSession,
    command_name: str,
    base_yaw: float,
    base_pitch: float,
    step_deg: float,
    command_yaw: bool,
    settle_ms: int,
    sample_ms: int,
    loop_ms: int,
) -> AxisResult:
    _send_plan(session, base_yaw * RAD2DEG, base_pitch * RAD2DEG)
    time.sleep(settle_ms / 1000.0)
    base = _average_ypr(session, sample_ms, loop_ms)

    target_yaw = base_yaw
    target_pitch = base_pitch
    if command_yaw:
        target_yaw += step_deg * DEG2RAD
    else:
        target_pitch += step_deg * DEG2RAD

    _send_plan(session, target_yaw * RAD2DEG, target_pitch * RAD2DEG)
    time.sleep(settle_ms / 1000.0)
    moved = _average_ypr(session, sample_ms, loop_ms)

    _send_plan(session, base_yaw * RAD2DEG, base_pitch * RAD2DEG)
    time.sleep(settle_ms / 1000.0)

    delta = _delta_deg(moved, base)
    dominant_axis = _dominant_axis(delta)
    expected_axis = 0 if command_yaw else 1
    sign_match = dominant_axis == expected_axis and ((step_deg > 0) == (delta[expected_axis] > 0))
    return AxisResult(command_name, step_deg, delta, dominant_axis, sign_match)


def _print_axis_result(result: AxisResult) -> None:
    print(
        f"[axis] cmd={result.command_name} step={result.step_deg:+.2f}deg "
        f"delta(yaw={result.delta_deg[0]:+.2f}, pitch={result.delta_deg[1]:+.2f}, roll={result.delta_deg[2]:+.2f}) "
        f"dominant={_axis_name(result.dominant_axis)} sign_ok={int(result.sign_match)}"
    )


def _print_axis_conclusion(yaw_result: AxisResult, pitch_result: AxisResult) -> None:
    yaw_axis_ok = yaw_result.dominant_axis == 0
    pitch_axis_ok = pitch_result.dominant_axis == 1
    axis_mapping_ok = yaw_axis_ok and pitch_axis_ok
    sign_ok = yaw_result.sign_match and pitch_result.sign_match

    print("")
    if axis_mapping_ok and sign_ok:
        print("[conclusion] yaw/pitch commands map to the expected feedback axes with the expected sign.")
        print("[conclusion] This looks like a small zero-offset problem. Check yaw_offset/pitch_offset first.")
        return

    if not axis_mapping_ok:
        print("[conclusion] At least one commanded axis moves the wrong feedback axis.")
        print("[conclusion] This points to a C-board or IMU axis mapping problem. Check R_gimbal2imubody.")
    else:
        print("[conclusion] Axis mapping looks right, but at least one axis sign is opposite.")
        print("[conclusion] This points to an axis direction/sign problem in the IMU-to-gimbal mapping.")

    if yaw_result.dominant_axis == 2 or pitch_result.dominant_axis == 2:
        print("[hint] Strong roll response to yaw/pitch commands often means the C-board is mounted 90 degrees off.")


def _build_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    perm_axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for axes in perm_axes:
        for sx, sy, sz in product((-1, 1), repeat=3):
            R = np.zeros((3, 3), dtype=float)
            R[axes[0], 0] = sx
            R[axes[1], 1] = sy
            R[axes[2], 2] = sz
            if round(float(np.linalg.det(R))) != 1:
                continue
            row_major = [int(R[0, 0]), int(R[0, 1]), int(R[0, 2]), int(R[1, 0]), int(R[1, 1]), int(R[1, 2]), int(R[2, 0]), int(R[2, 1]), int(R[2, 2])]
            candidates.append({"R": R, "row_major": row_major, "label": f"[{row_major[0]},{row_major[1]},{row_major[2]};{row_major[3]},{row_major[4]},{row_major[5]};{row_major[6]},{row_major[7]},{row_major[8]}]"})
    return candidates


def _evaluate_candidates(
    session: GimbalSession,
    candidates: list[dict[str, Any]],
    sample_ms: int,
    loop_ms: int,
    base_yaw: float,
    base_pitch: float,
    step_deg: float,
    settle_ms: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    _send_plan(session, base_yaw * RAD2DEG, base_pitch * RAD2DEG)
    time.sleep(settle_ms / 1000.0)
    baseline_samples = [_average_transformed_state(session, c["R"], sample_ms, loop_ms) for c in candidates]

    _send_plan(session, (base_yaw + step_deg * DEG2RAD) * RAD2DEG, base_pitch * RAD2DEG)
    time.sleep(settle_ms / 1000.0)
    yaw_samples = [_average_transformed_state(session, c["R"], sample_ms, loop_ms) for c in candidates]

    _send_plan(session, base_yaw * RAD2DEG, base_pitch * RAD2DEG)
    time.sleep(settle_ms / 1000.0)
    _send_plan(session, base_yaw * RAD2DEG, (base_pitch + step_deg * DEG2RAD) * RAD2DEG)
    time.sleep(settle_ms / 1000.0)
    pitch_samples = [_average_transformed_state(session, c["R"], sample_ms, loop_ms) for c in candidates]

    _send_plan(session, base_yaw * RAD2DEG, base_pitch * RAD2DEG)
    time.sleep(settle_ms / 1000.0)

    for idx, candidate in enumerate(candidates):
        yaw_delta = _delta_deg(yaw_samples[idx], baseline_samples[idx])
        pitch_delta = _delta_deg(pitch_samples[idx], baseline_samples[idx])
        yaw_dom = _dominant_axis(yaw_delta)
        pitch_dom = _dominant_axis(pitch_delta)
        yaw_sign = yaw_dom == 0 and yaw_delta[0] > 0.0
        pitch_sign = pitch_dom == 1 and pitch_delta[1] > 0.0
        score = (
            abs(yaw_delta[0]) * 2.0 + abs(pitch_delta[1]) * 2.0 - abs(yaw_delta[1]) - abs(yaw_delta[2]) - abs(pitch_delta[0]) - abs(pitch_delta[2])
        )
        if not yaw_sign:
            score -= 6.0
        if not pitch_sign:
            score -= 6.0
        if yaw_dom != 0:
            score -= 4.0
        if pitch_dom != 1:
            score -= 4.0
        results.append(
            {
                "candidate": candidate,
                "yaw_delta": yaw_delta,
                "pitch_delta": pitch_delta,
                "yaw_dom": yaw_dom,
                "pitch_dom": pitch_dom,
                "yaw_sign": yaw_sign,
                "pitch_sign": pitch_sign,
                "score": score,
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _average_transformed_state(
    session: GimbalSession, R_gimbal2imubody: np.ndarray, duration_ms: int, loop_ms: int
) -> tuple[float, float, float]:
    start = time.monotonic()
    total = np.zeros(3, dtype=float)
    count = 0
    while time.monotonic() - start < duration_ms / 1000.0:
        q = session.snapshot().ypr
        R_imubody2imuabs = _rotation_matrix(q)
        R_gimbal2world = R_gimbal2imubody.T @ R_imubody2imuabs @ R_gimbal2imubody
        total += np.array(_eulers_zyx(R_gimbal2world), dtype=float)
        count += 1
        time.sleep(loop_ms / 1000.0)
    if count == 0:
        q = session.snapshot().ypr
        R_imubody2imuabs = _rotation_matrix(q)
        R_gimbal2world = R_gimbal2imubody.T @ R_imubody2imuabs @ R_gimbal2imubody
        return _eulers_zyx(R_gimbal2world)
    averaged = total / count
    return float(averaged[0]), float(averaged[1]), float(averaged[2])


def _print_candidate(result: dict[str, Any], rank: int) -> None:
    print(
        f"[candidate {rank}] score={result['score']:.2f} "
        f"R_gimbal2imubody={result['candidate']['label']} "
        f"yaw(delta={result['yaw_delta'][0]:+.2f},{result['yaw_delta'][1]:+.2f},{result['yaw_delta'][2]:+.2f} "
        f"dominant={_axis_name(result['yaw_dom'])} sign_ok={int(result['yaw_sign'])}) "
        f"pitch(delta={result['pitch_delta'][0]:+.2f},{result['pitch_delta'][1]:+.2f},{result['pitch_delta'][2]:+.2f} "
        f"dominant={_axis_name(result['pitch_dom'])} sign_ok={int(result['pitch_sign'])})"
    )
