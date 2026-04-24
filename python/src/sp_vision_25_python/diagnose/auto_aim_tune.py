from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from ..bindings import load_bindings
from .config import load_yaml
from .gimbal_session import Key, TerminalRawMode, read_key
from ..calibration.math_tools import RAD2DEG, eulers_from_quaternion

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.default_flow_style = False


@dataclass
class UiState:
    tracking: bool = True
    fric_on: bool = True
    fire_pulse: bool = False
    fire_pulse_until: float = 0.0
    fire_mode: int = 0

    bullet_speed: float = 25.0
    speed_step: float = 0.2
    yaw_offset_delta_deg: float = 0.0
    pitch_offset_delta_deg: float = 0.0
    offset_step_deg: float = 0.2


@dataclass
class ConfigState:
    min_detect_count: int = 5
    max_temp_lost_count: int = 15
    outpost_max_temp_lost_count: int = 75

    yaw_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    comming_angle_deg: float = 55.0
    leaving_angle_deg: float = 20.0
    decision_speed: float = 7.0
    high_speed_delay_time: float = 0.0
    low_speed_delay_time: float = 0.0

    first_tolerance_deg: float = 3.0
    second_tolerance_deg: float = 2.0
    judge_distance: float = 2.0
    auto_fire: bool = True


@dataclass
class TuneParam:
    key: str
    label: str
    kind: str
    step: float
    min_value: float
    max_value: float
    attr: str


def _timestamp_string() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _trim_float(value: str) -> str:
    if "." not in value:
        return value
    value = value.rstrip("0")
    if value.endswith("."):
        value = value[:-1]
    return value


def _format_double(value: float, precision: int = 3) -> str:
    return _trim_float(f"{value:.{precision}f}")


def _load_config_state(config_path: Path) -> ConfigState:
    yaml = load_yaml(config_path)
    return ConfigState(
        min_detect_count=int(yaml["min_detect_count"]),
        max_temp_lost_count=int(yaml["max_temp_lost_count"]),
        outpost_max_temp_lost_count=int(yaml["outpost_max_temp_lost_count"]),
        yaw_offset_deg=float(yaml["yaw_offset"]),
        pitch_offset_deg=float(yaml["pitch_offset"]),
        comming_angle_deg=float(yaml["comming_angle"]),
        leaving_angle_deg=float(yaml["leaving_angle"]),
        decision_speed=float(yaml["decision_speed"]),
        high_speed_delay_time=float(yaml["high_speed_delay_time"]),
        low_speed_delay_time=float(yaml["low_speed_delay_time"]),
        first_tolerance_deg=float(yaml["first_tolerance"]),
        second_tolerance_deg=float(yaml["second_tolerance"]),
        judge_distance=float(yaml["judge_distance"]),
        auto_fire=bool(yaml["auto_fire"]),
    )


def _build_tune_params(state: ConfigState) -> list[TuneParam]:
    return [
        TuneParam("min_detect_count", "min_detect_count", "int", 1, 1, 100, "min_detect_count"),
        TuneParam("max_temp_lost_count", "max_temp_lost_count", "int", 1, 1, 200, "max_temp_lost_count"),
        TuneParam(
            "outpost_max_temp_lost_count",
            "outpost_max_temp_lost_count",
            "int",
            1,
            1,
            500,
            "outpost_max_temp_lost_count",
        ),
        TuneParam("yaw_offset", "yaw_offset(deg)", "double", 0.1, -20, 20, "yaw_offset_deg"),
        TuneParam("pitch_offset", "pitch_offset(deg)", "double", 0.1, -20, 20, "pitch_offset_deg"),
        TuneParam("comming_angle", "comming_angle(deg)", "double", 1.0, 0, 180, "comming_angle_deg"),
        TuneParam("leaving_angle", "leaving_angle(deg)", "double", 1.0, 0, 180, "leaving_angle_deg"),
        TuneParam("decision_speed", "decision_speed", "double", 0.1, 0, 50, "decision_speed"),
        TuneParam(
            "high_speed_delay_time", "high_speed_delay_time", "double", 0.005, 0, 1, "high_speed_delay_time"
        ),
        TuneParam(
            "low_speed_delay_time", "low_speed_delay_time", "double", 0.005, 0, 1, "low_speed_delay_time"
        ),
        TuneParam("first_tolerance", "first_tolerance(deg)", "double", 0.1, 0, 10, "first_tolerance_deg"),
        TuneParam("second_tolerance", "second_tolerance(deg)", "double", 0.1, 0, 10, "second_tolerance_deg"),
        TuneParam("judge_distance", "judge_distance", "double", 0.1, 0, 10, "judge_distance"),
        TuneParam("auto_fire", "auto_fire", "bool", 1, 0, 1, "auto_fire"),
    ]


def _param_value(state: ConfigState, param: TuneParam) -> str:
    value = getattr(state, param.attr)
    if param.kind == "int":
        return str(int(value))
    if param.kind == "bool":
        return "true" if bool(value) else "false"
    return _format_double(float(value), 3)


def _adjust_param(state: ConfigState, param: TuneParam, direction: float) -> None:
    if param.kind == "int":
        value = int(getattr(state, param.attr))
        delta = int(param.step * direction)
        if delta == 0:
            delta = 1 if direction > 0 else -1
        value += delta
        value = max(int(param.min_value), value)
        if param.max_value > param.min_value:
            value = min(int(param.max_value), value)
        setattr(state, param.attr, value)
        return
    if param.kind == "double":
        value = float(getattr(state, param.attr)) + param.step * direction
        value = max(param.min_value, value)
        if param.max_value > param.min_value:
            value = min(param.max_value, value)
        setattr(state, param.attr, value)
        return
    if param.kind == "bool":
        setattr(state, param.attr, not bool(getattr(state, param.attr)))


def _export_config(config_path: Path, state: ConfigState, ui: UiState) -> tuple[bool, str, str]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = _yaml.load(handle) or CommentedMap()

    replacements = {
        "min_detect_count": state.min_detect_count,
        "max_temp_lost_count": state.max_temp_lost_count,
        "outpost_max_temp_lost_count": state.outpost_max_temp_lost_count,
        "yaw_offset": float(state.yaw_offset_deg + ui.yaw_offset_delta_deg),
        "pitch_offset": float(state.pitch_offset_deg + ui.pitch_offset_delta_deg),
        "comming_angle": float(state.comming_angle_deg),
        "leaving_angle": float(state.leaving_angle_deg),
        "decision_speed": float(state.decision_speed),
        "high_speed_delay_time": float(state.high_speed_delay_time),
        "low_speed_delay_time": float(state.low_speed_delay_time),
        "first_tolerance": float(state.first_tolerance_deg),
        "second_tolerance": float(state.second_tolerance_deg),
        "judge_distance": float(state.judge_distance),
        "auto_fire": bool(state.auto_fire),
    }
    missing = [key for key in replacements if key not in data]
    for key, value in replacements.items():
        data[key] = value

    out_path = config_path.with_name(f"{config_path.stem}_{_timestamp_string()}{config_path.suffix or '.yaml'}")
    with out_path.open("w", encoding="utf-8") as handle:
        _yaml.dump(data, handle)

    if missing:
        return False, str(out_path), f"missing keys: {', '.join(missing)}"
    return True, str(out_path), ""


def _print_tui(
    ui: UiState,
    state: ConfigState,
    gimbal_state: Any,
    ypr_deg: np.ndarray,
    command: tuple[bool, bool, float, float, float],
    target_count: int,
    tracker_state: str,
    tune_params: list[TuneParam],
    selected_param: int,
    save_status: str,
    log_on: bool,
    dt: float,
) -> None:
    print("\033[2J\033[H", end="")
    print(
        "Auto Aim UI Tune\n"
        f"dt: {dt * 1e3:.1f}ms  tracking:{1 if ui.tracking else 0}  fric:{1 if ui.fric_on else 0}  "
        f"fire_mode:{ui.fire_mode}  pulse:{1 if ui.fire_pulse else 0}  log:{1 if log_on else 0}\n"
        f"bullet_speed: {ui.bullet_speed:.2f} (step {ui.speed_step:.2f})  offset_step: {ui.offset_step_deg:.2f}deg\n"
        f"offset_delta (deg): yaw:{ui.yaw_offset_delta_deg:+.2f}  pitch:{ui.pitch_offset_delta_deg:+.2f}\n"
        f"offset_yaml  (deg): yaw:{state.yaw_offset_deg:.2f}  pitch:{state.pitch_offset_deg:.2f}  "
        f"export: yaw:{state.yaw_offset_deg + ui.yaw_offset_delta_deg:.2f} pitch:{state.pitch_offset_deg + ui.pitch_offset_delta_deg:.2f}\n"
        f"cmd   (deg): yaw:{command[2] * RAD2DEG:+.2f}  pitch:{command[3] * RAD2DEG:+.2f}  "
        f"control:{1 if command[0] else 0}  targets:{target_count}  state:{tracker_state}\n"
        f"fb    (deg): yaw:{ypr_deg[0]:+.2f}  pitch:{ypr_deg[1]:+.2f}  roll:{ypr_deg[2]:+.2f}\n"
        f"fb    (rad): yaw:{gimbal_state.yaw:+.3f}  pitch:{gimbal_state.pitch:+.3f}  roll:{gimbal_state.roll:+.3f}  "
        f"yaw_vel:{gimbal_state.yaw_vel:+.3f}  pitch_vel:{gimbal_state.pitch_vel:+.3f}\n"
        "Realtime: bullet_speed, offset_delta, tracking/fric/fire\n"
        "Restart: tune params (j/k select, -/= adjust, u toggle), R export\n"
    )

    for i, param in enumerate(tune_params):
        prefix = ">" if i == selected_param else " "
        print(f"{prefix} {param.label:<26} : {_param_value(state, param)}")

    if save_status:
        print(f"Save: {save_status}")

    print(
        "Keys: q quit | w/s or Up/Down pitch_delta +/- | a/d or Left/Right yaw_delta -/+ | [/] step\n"
        "      z/x bullet_speed -/+ | ,/. speed_step | 0 reset_offset | p reset_speed | c tracking\n"
        "      r fric | 1 off 2 ready 3 single 4 fire | f toggle fire | space single pulse\n"
        "      j/k select | -/= adjust | u toggle | R save | L log"
    )


def _parse_show(extra_args: list[str], default: bool = True) -> tuple[bool, list[str]]:
    show = default
    remaining: list[str] = []
    for arg in extra_args:
        if arg in {"--show", "-s"}:
            show = True
        elif arg.startswith("--show="):
            show = arg.split("=", 1)[1].lower() not in {"0", "false", "no", "off"}
        else:
            remaining.append(arg)
    return show, remaining


def run_auto_aim_tune(config_path: Path, extra_args: list[str]) -> int:
    show, extra_args = _parse_show(extra_args, True)
    if extra_args:
        # keep compatibility with existing shell-style invocation; ignore leftovers gracefully
        pass

    bindings = load_bindings()
    gimbal = bindings.Gimbal(str(config_path))
    camera = bindings.Camera(str(config_path))
    runtime = bindings.Runtime(str(config_path), False)
    solver = bindings.Solver(str(config_path))

    ui = UiState()
    state = _load_config_state(config_path)
    tune_params = _build_tune_params(state)
    selected_param = 0
    save_status = ""
    log_enabled = False
    log_file = None
    log_path = ""

    terminal = TerminalRawMode()
    terminal.enable()

    use_gui = show
    if use_gui:
        try:
            cv2.namedWindow("Auto Aim UI Tune", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Auto Aim UI Tune", 1280, 720)
        except cv2.error:
            use_gui = False

    t0 = time.monotonic()
    last_loop = t0

    try:
        while True:
            img, timestamp_ns = camera.read()
            if img is None or getattr(img, "size", 0) == 0:
                continue

            now = time.monotonic()
            dt = now - last_loop
            last_loop = now

            gs = gimbal.state()
            q = gimbal.q_at_ns(int(timestamp_ns))
            solver.set_R_gimbal2world_quat(q)
            output = runtime.step(np.asarray(img), q, ui.bullet_speed)
            armors = output["armors"]
            targets = output["targets"]
            command = output["command"]
            tracker_state = output["tracker_state"]

            if ui.fire_pulse and now >= ui.fire_pulse_until:
                ui.fire_pulse = False

            send_yaw = float(command[2]) + ui.yaw_offset_delta_deg / RAD2DEG
            send_pitch = float(command[3]) + ui.pitch_offset_delta_deg / RAD2DEG

            vision = bindings.VisionToGimbal()
            vision.tracking = 1 if (ui.tracking and command[0]) else 0
            vision.pitch = float(send_pitch)
            vision.yaw = float(send_yaw)
            fire_cmd = ui.fire_mode
            if ui.fire_pulse:
                fire_cmd = 2
            if not vision.tracking:
                fire_cmd = 0
            vision.fire = int(fire_cmd)
            vision.fric_on = 1 if ui.fric_on else 0
            gimbal.send_vision(vision)

            ypr_deg = np.asarray(eulers_from_quaternion(q, 2, 1, 0) * RAD2DEG)
            _print_tui(
                ui, state, gs, ypr_deg, command, len(targets), tracker_state, tune_params,
                selected_param, save_status, log_enabled, dt,
            )

            if log_enabled and log_file is not None:
                data: dict[str, Any] = {
                    "t": now - t0,
                    "dt": dt,
                    "targets": len(targets),
                    "tracker_state": tracker_state,
                    "command_yaw": float(command[2]),
                    "command_pitch": float(command[3]),
                    "send_yaw": float(send_yaw),
                    "send_pitch": float(send_pitch),
                    "gimbal_yaw": float(gs.yaw),
                    "gimbal_pitch": float(gs.pitch),
                    "gimbal_yaw_vel": float(gs.yaw_vel),
                    "gimbal_pitch_vel": float(gs.pitch_vel),
                    "yaw_err": float(send_yaw - gs.yaw),
                    "pitch_err": float(send_pitch - gs.pitch),
                    "bullet_speed": ui.bullet_speed,
                    "yaw_offset_base_deg": state.yaw_offset_deg,
                    "pitch_offset_base_deg": state.pitch_offset_deg,
                    "yaw_offset_delta_deg": ui.yaw_offset_delta_deg,
                    "pitch_offset_delta_deg": ui.pitch_offset_delta_deg,
                    "fire_mode": ui.fire_mode,
                    "tracking": ui.tracking,
                    "fric_on": ui.fric_on,
                    "auto_fire": state.auto_fire,
                }
                if armors:
                    armor = armors[0]
                    data["armor_center_norm_x"] = float(armor.center_norm[0])
                    data["armor_center_norm_y"] = float(armor.center_norm[1])
                    data["armor_confidence"] = float(armor.confidence)
                log_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                log_file.flush()

            if use_gui:
                draw = np.asarray(img).copy()
                if targets:
                    target = targets[0]
                    draw_text = f"[{tracker_state}]"
                    cv2.putText(draw, draw_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    for xyza in target.armor_xyza_list:
                        xyz = np.asarray(xyza[:3], dtype=float)
                        yaw = float(xyza[3])
                        image_points = solver.reproject_armor(xyz, yaw, target.armor_type, target.name)
                        for pt in image_points:
                            cv2.circle(draw, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 255, 0), -1)

                if tune_params:
                    param = tune_params[selected_param]
                    cv2.putText(
                        draw, f"sel:{param.label}={_param_value(state, param)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                    )

                cv2.putText(
                    draw,
                    f"spd:{ui.bullet_speed:.2f} off_y:{ui.yaw_offset_delta_deg:+.2f} "
                    f"off_p:{ui.pitch_offset_delta_deg:+.2f} fire:{ui.fire_mode} log:{1 if log_enabled else 0}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                )
                cv2.imshow("Auto Aim UI Tune", cv2.resize(draw, None, fx=0.5, fy=0.5))
                gui_key = cv2.waitKey(1)
            else:
                gui_key = -1

            key = gui_key
            if key == -1:
                ev = read_key()
                if ev is None:
                    continue
                if ev.key == Key.Quit:
                    break
                if ev.key == Key.Left:
                    key = 81
                elif ev.key == Key.Right:
                    key = 83
                elif ev.key == Key.Up:
                    key = 82
                elif ev.key == Key.Down:
                    key = 84
                elif ev.key == Key.Char:
                    key = ev.ch

            if key == ord("q"):
                break
            if key == ord("c"):
                ui.tracking = not ui.tracking
            if key == ord("r"):
                ui.fric_on = not ui.fric_on
            if key == ord("1"):
                ui.fire_mode = 0
            if key == ord("2"):
                ui.fire_mode = 1
            if key == ord("3"):
                ui.fire_mode = 2
            if key == ord("4"):
                ui.fire_mode = 3
            if key == ord("f"):
                ui.fire_mode = 0 if ui.fire_mode == 3 else 3
            if key == ord(" "):
                ui.fire_pulse = True
                ui.fire_pulse_until = now + 0.12
            if key == ord("0"):
                ui.yaw_offset_delta_deg = 0.0
                ui.pitch_offset_delta_deg = 0.0
            if key == ord("p"):
                ui.bullet_speed = 25.0
            if key == ord("["):
                ui.offset_step_deg = max(0.01, ui.offset_step_deg - 0.05)
            if key == ord("]"):
                ui.offset_step_deg = min(5.0, ui.offset_step_deg + 0.05)
            if key == ord(","):
                ui.speed_step = max(0.1, ui.speed_step - 0.1)
            if key == ord("."):
                ui.speed_step = min(10.0, ui.speed_step + 0.1)
            if key == ord("z"):
                ui.bullet_speed = max(0.0, ui.bullet_speed - ui.speed_step)
            if key == ord("x"):
                ui.bullet_speed += ui.speed_step
            if key in {ord("a"), 81}:
                ui.yaw_offset_delta_deg -= ui.offset_step_deg
            if key in {ord("d"), 83}:
                ui.yaw_offset_delta_deg += ui.offset_step_deg
            if key in {ord("w"), 82}:
                ui.pitch_offset_delta_deg += ui.offset_step_deg
            if key in {ord("s"), 84}:
                ui.pitch_offset_delta_deg -= ui.offset_step_deg
            if key == ord("j") and tune_params:
                selected_param = (selected_param + 1) % len(tune_params)
            if key == ord("k") and tune_params:
                selected_param = len(tune_params) - 1 if selected_param == 0 else selected_param - 1
            if key in {ord("-"), ord("_")} and tune_params:
                _adjust_param(state, tune_params[selected_param], -1.0)
            if key in {ord("="), ord("+")} and tune_params:
                _adjust_param(state, tune_params[selected_param], 1.0)
            if key == ord("u") and tune_params:
                if tune_params[selected_param].kind == "bool":
                    _adjust_param(state, tune_params[selected_param], 1.0)
            if key == ord("R"):
                ok, out_path, err = _export_config(config_path, state, ui)
                save_status = f"saved: {out_path}" if ok else f"save failed: {err or 'unknown'}"
            if key == ord("L"):
                if log_enabled:
                    if log_file is not None:
                        log_file.close()
                    log_file = None
                    log_enabled = False
                    save_status = "log: off"
                else:
                    Path("logs").mkdir(parents=True, exist_ok=True)
                    log_path = f"logs/auto_aim_ui_{_timestamp_string()}.jsonl"
                    log_file = Path(log_path).open("w", encoding="utf-8")
                    log_enabled = True
                    save_status = f"log: {log_path}"

            time.sleep(0.005)
    finally:
        if log_file is not None:
            log_file.close()
        stop = bindings.VisionToGimbal()
        stop.tracking = 0
        stop.pitch = 0.0
        stop.yaw = 0.0
        stop.fire = 0
        stop.fric_on = 0
        gimbal.send_vision(stop)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    return 0
