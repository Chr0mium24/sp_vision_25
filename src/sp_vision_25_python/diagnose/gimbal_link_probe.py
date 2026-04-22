from __future__ import annotations

import struct
import time
import sys
from pathlib import Path
from typing import Any

import serial

from .bindings import load_bindings
from .config import read_scalar
from .system import default_config_path

K_RX_HEADER = 0x5A
K_TX_HEADER = 0xA5
K_EXTENDED_FRAME_SIZE = 49
K_RAD2DEG = 57.29577951308232
K_DEFAULT_PORTS = ["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyS0"]


def _crc16(data: bytes) -> int:
    bindings = load_bindings()
    return int(bindings.crc16(data))


def _check_crc16(data: bytes) -> bool:
    bindings = load_bindings()
    return bool(bindings.check_crc16(data))


def _build_ports(config_path: str | None, ports_arg: str) -> list[str]:
    ports: list[str] = []
    if ports_arg:
        ports = [token.strip() for token in ports_arg.split(",") if token.strip()]
    if not ports and config_path:
        try:
            com_port = read_scalar(Path(config_path), "com_port")
            if com_port:
                ports.append(com_port)
        except Exception:
            pass
    if not ports:
        ports = list(K_DEFAULT_PORTS)
    return ports


def _open_first_available(ports: list[str], baud: int) -> tuple[serial.Serial | None, str, str]:
    last_reason = ""
    for port in ports:
        try:
            ser = serial.Serial(port=port, baudrate=baud, timeout=0, write_timeout=0.1)
            return ser, port, ""
        except Exception as exc:
            last_reason = str(exc)
    return None, "", last_reason


def _parse_frame(frame: bytes) -> dict[str, Any]:
    flags = frame[1]
    return {
        "valid": True,
        "detect_color": flags & 0x01,
        "reset_tracker": (flags >> 1) & 0x01,
        "yaw": struct.unpack_from("<f", frame, 2)[0],
        "pitch": struct.unpack_from("<f", frame, 6)[0],
        "roll": struct.unpack_from("<f", frame, 10)[0],
        "yaw_odom": struct.unpack_from("<f", frame, 14)[0],
        "pitch_odom": struct.unpack_from("<f", frame, 18)[0],
        "yaw_vel": struct.unpack_from("<f", frame, 22)[0],
        "pitch_vel": struct.unpack_from("<f", frame, 26)[0],
        "robot_id": frame[46],
        "t": time.monotonic(),
    }


def _parse_stream(buffer: bytearray, fail_hex_len: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    stats = {
        "drop_bytes": 0,
        "headers_seen": 0,
        "extended_ok": 0,
        "extended_crc_fail": 0,
        "has_fail_sample": False,
        "last_extended_crc_rx": 0,
        "last_extended_crc_calc": 0,
        "last_fail_prefix": "",
    }
    frame: dict[str, Any] | None = None

    while True:
        try:
            idx = buffer.index(K_RX_HEADER)
        except ValueError:
            if len(buffer) > K_EXTENDED_FRAME_SIZE - 1:
                drop = len(buffer) - (K_EXTENDED_FRAME_SIZE - 1)
                stats["drop_bytes"] += drop
                del buffer[:drop]
            break

        if idx > 0:
            stats["drop_bytes"] += idx
            del buffer[:idx]

        if len(buffer) < K_EXTENDED_FRAME_SIZE:
            break
        stats["headers_seen"] += 1
        candidate = bytes(buffer[:K_EXTENDED_FRAME_SIZE])
        if _check_crc16(candidate):
            stats["extended_ok"] += 1
            frame = _parse_frame(candidate)
            del buffer[:K_EXTENDED_FRAME_SIZE]
            continue

        stats["extended_crc_fail"] += 1
        stats["has_fail_sample"] = True
        stats["last_extended_crc_calc"] = _crc16(candidate[:-2])
        stats["last_extended_crc_rx"] = int.from_bytes(candidate[-2:], "little")
        stats["last_fail_prefix"] = " ".join(f"{b:02X}" for b in candidate[:fail_hex_len])
        if len(candidate) > fail_hex_len:
            stats["last_fail_prefix"] += " ..."
        del buffer[0]
    return frame, stats


def _age_ms(t: float | None) -> float:
    if t is None:
        return -1.0
    return (time.monotonic() - t) * 1000.0


def _build_tx(
    tracking: int, fric_on: int, fire_mode: int, yaw_deg: float, pitch_deg: float
) -> bytes:
    fire = max(0, min(3, int(fire_mode)))
    yaw = yaw_deg / K_RAD2DEG
    pitch = pitch_deg / K_RAD2DEG
    payload = struct.pack("<BBffBB", K_TX_HEADER, int(tracking), pitch, yaw, fire, int(fric_on))
    checksum = _crc16(payload)
    return payload + struct.pack("<H", checksum)


def run_gimbal_link_diag(config_path: Path, extra_args: list[str]) -> int:
    ports_arg = ""
    baud = 115200
    duration_ms = 3000
    summary_ms = 1000
    loop_ms = 2
    no_send = False
    require_rx = False
    tracking = 1
    fric_on = 1
    fire_mode = 0
    yaw_deg = 0.0
    pitch_deg = 0.0
    remaining = list(extra_args)
    for arg in list(remaining):
        if arg.startswith("--ports="):
            ports_arg = arg.split("=", 1)[1]
            remaining.remove(arg)
        elif arg.startswith("--baud="):
            baud = max(1, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg.startswith("--duration-ms="):
            duration_ms = max(1, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg.startswith("--summary-ms="):
            summary_ms = max(100, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg.startswith("--loop-ms="):
            loop_ms = max(1, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg == "--no-send":
            no_send = True
            remaining.remove(arg)
        elif arg == "--require-rx":
            require_rx = True
            remaining.remove(arg)
        elif arg.startswith("--tracking="):
            tracking = 1 if int(float(arg.split("=", 1)[1])) != 0 else 0
            remaining.remove(arg)
        elif arg.startswith("--fric-on="):
            fric_on = 1 if int(float(arg.split("=", 1)[1])) != 0 else 0
            remaining.remove(arg)
        elif arg.startswith("--fire-mode="):
            fire_mode = int(float(arg.split("=", 1)[1]))
            remaining.remove(arg)
        elif arg.startswith("--yaw-deg="):
            yaw_deg = float(arg.split("=", 1)[1])
            remaining.remove(arg)
        elif arg.startswith("--pitch-deg="):
            pitch_deg = float(arg.split("=", 1)[1])
            remaining.remove(arg)
    if remaining:
        print(f"[gimbal link] ignored args: {' '.join(remaining)}")

    cfg = str(config_path or default_config_path())
    ports = _build_ports(cfg, ports_arg)
    if not ports:
        print("no serial ports available.")
        return 1

    tx = _build_tx(tracking, fric_on, fire_mode, yaw_deg, pitch_deg)
    print(
        f"gimbal_link_diag_test: baud={baud} duration={duration_ms}ms summary={summary_ms}ms "
        f"loop={loop_ms}ms send={'off' if no_send else 'on'} cmd(track={tracking} fric={fric_on} "
        f"fire={max(0, min(3, int(fire_mode)))} yaw={yaw_deg / K_RAD2DEG:.3f} pitch={pitch_deg / K_RAD2DEG:.3f}) "
        f"ports={','.join(ports)}"
    )
    ser, opened_port, reason = _open_first_available(ports, baud)
    if ser is None:
        print(f"failed to open serial ports: {reason}")
        return 1
    print(f"[diag] opened {opened_port}")

    stats = {
        "tx_ok": 0,
        "tx_exc": 0,
        "read_calls": 0,
        "read_zero": 0,
        "read_exception": 0,
        "bytes": 0,
        "extended_ok": 0,
        "extended_crc_fail": 0,
    }
    last_stats = dict(stats)
    frame: dict[str, Any] | None = None
    buffer = bytearray()
    start_t = time.monotonic()
    last_summary_t = start_t

    try:
        while time.monotonic() - start_t < duration_ms / 1000.0:
            if not no_send:
                try:
                    ser.write(tx)
                    stats["tx_ok"] += 1
                except Exception:
                    stats["tx_exc"] += 1

            try:
                avail = int(getattr(ser, "in_waiting", 0))
                stats["read_calls"] += 1
                if avail == 0:
                    stats["read_zero"] += 1
                else:
                    chunk = ser.read(max(1, min(avail, 4096)))
                    if chunk:
                        stats["bytes"] += len(chunk)
                        buffer.extend(chunk)
                        frame, parsed = _parse_stream(buffer, 24)
                        stats["extended_ok"] += parsed["extended_ok"]
                        stats["extended_crc_fail"] += parsed["extended_crc_fail"]
                    else:
                        stats["read_zero"] += 1
            except Exception:
                stats["read_exception"] += 1

            now = time.monotonic()
            if now - last_summary_t >= summary_ms / 1000.0:
                d_tx = stats["tx_ok"] - last_stats["tx_ok"]
                d_bytes = stats["bytes"] - last_stats["bytes"]
                d_ok49 = stats["extended_ok"] - last_stats["extended_ok"]
                d_crc49 = stats["extended_crc_fail"] - last_stats["extended_crc_fail"]
                frame_age = -1.0 if frame is None else (time.monotonic() - frame["t"]) * 1000.0
                print(
                    f"[diag][{summary_ms}ms] port={opened_port} tx={stats['tx_ok']}(+{d_tx}) "
                    f"bytes={stats['bytes']}(+{d_bytes}) ok49={stats['extended_ok']}(+{d_ok49}) "
                    f"crc49={stats['extended_crc_fail']}(+{d_crc49}) age={frame_age:.0f}ms"
                )
                if frame:
                    frame_age = -1.0 if frame is None else (time.monotonic() - frame["t"]) * 1000.0
                    print(
                        "[frame] proto=49B yaw={yaw:.2f}deg pitch={pitch:.2f}deg roll={roll:.2f}deg "
                        "yaw_odom={yaw_odom:.3f} pitch_odom={pitch_odom:.3f} yaw_vel={yaw_vel:.3f} "
                        "pitch_vel={pitch_vel:.3f} color={detect_color} reset={reset_tracker} robot_id={robot_id}".format(
                            yaw=frame["yaw"] * K_RAD2DEG,
                            pitch=frame["pitch"] * K_RAD2DEG,
                            roll=frame["roll"] * K_RAD2DEG,
                            **frame,
                        )
                    )
                last_stats = dict(stats)
                last_summary_t = now
            time.sleep(loop_ms / 1000.0)
    finally:
        try:
            ser.close()
        except Exception:
            pass

    print(
        "done: tx_ok={tx_ok} tx_exc={tx_exc} bytes={bytes} ok49={ok49} crc49={crc49} read_exc={read_exc}".format(
            tx_ok=stats["tx_ok"],
            tx_exc=stats["tx_exc"],
            bytes=stats["bytes"],
            ok49=stats["extended_ok"],
            crc49=stats["extended_crc_fail"],
            read_exc=stats["read_exception"],
        )
    )
    if require_rx and stats["extended_ok"] == 0:
        print("require-rx enabled but no valid rx frame observed.", file=sys.stderr)
        return 2
    return 0


def run_gimbal_serial_probe(config_path: Path, extra_args: list[str]) -> int:
    ports_arg = ""
    baud = 115200
    duration_ms = 0
    summary_ms = 1000
    sleep_ms = 2
    read_max = 256
    hex_len = 24
    raw_log = False
    reopen_ms = 1000
    remaining = list(extra_args)
    for arg in list(remaining):
        if arg.startswith("--ports="):
            ports_arg = arg.split("=", 1)[1]
            remaining.remove(arg)
        elif arg.startswith("--baud="):
            baud = max(1, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg.startswith("--duration-ms="):
            duration_ms = max(0, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg.startswith("--summary-ms="):
            summary_ms = max(50, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg.startswith("--sleep-ms="):
            sleep_ms = max(0, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg.startswith("--read-max="):
            read_max = max(1, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg.startswith("--hex-len="):
            hex_len = max(4, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
        elif arg == "--raw-log":
            raw_log = True
            remaining.remove(arg)
        elif arg.startswith("--reopen-ms="):
            reopen_ms = max(10, int(float(arg.split("=", 1)[1])))
            remaining.remove(arg)
    if remaining:
        print(f"[gimbal probe] ignored args: {' '.join(remaining)}")

    cfg = str(config_path or default_config_path())
    ports = _build_ports(cfg, ports_arg)
    print(f"gimbal_serial_probe: baud={baud} ports={','.join(ports)}")

    ser: serial.Serial | None = None
    opened_port = ""
    fail_reason = ""
    stats = {
        "open_ok": 0,
        "open_fail": 0,
        "read_calls": 0,
        "read_zero": 0,
        "read_exception": 0,
        "bytes": 0,
        "drop_bytes": 0,
        "headers_seen": 0,
        "extended_ok": 0,
        "extended_crc_fail": 0,
        "has_fail_sample": False,
        "last_extended_crc_rx": 0,
        "last_extended_crc_calc": 0,
        "last_fail_prefix": "",
    }
    last_stats = dict(stats)
    stream_buffer = bytearray()
    frame: dict[str, Any] | None = None
    t_start = time.monotonic()
    t_last_summary = t_start
    t_last_open_try = t_start - 10.0
    t_last_data: float | None = None
    last_chunk_hex = ""

    try:
        while True:
            now = time.monotonic()
            if duration_ms > 0 and (now - t_start) * 1000.0 >= duration_ms:
                break

            if ser is None:
                if (now - t_last_open_try) * 1000.0 >= reopen_ms:
                    t_last_open_try = now
                    ser, opened_port, fail_reason = _open_first_available(ports, baud)
                    if ser is not None:
                        stats["open_ok"] += 1
                        print(f"[probe] opened {opened_port}")
                    else:
                        stats["open_fail"] += 1
                        print(f"[probe] open failed ({fail_reason})")
            else:
                try:
                    avail = int(getattr(ser, "in_waiting", 0))
                    to_read = max(1, min(read_max, avail))
                    chunk = ser.read(to_read)
                    stats["read_calls"] += 1
                    if not chunk:
                        stats["read_zero"] += 1
                    else:
                        stats["bytes"] += len(chunk)
                        t_last_data = now
                        stream_buffer.extend(chunk)
                        last_chunk_hex = " ".join(f"{b:02X}" for b in chunk[:hex_len]) + (
                            " ..." if len(chunk) > hex_len else ""
                        )
                        if raw_log:
                            print(f"[raw] n={len(chunk)} hex={last_chunk_hex}")
                    frame, parsed = _parse_stream(stream_buffer, hex_len)
                    if parsed["extended_ok"]:
                        stats["extended_ok"] += parsed["extended_ok"]
                    if parsed["extended_crc_fail"]:
                        stats["extended_crc_fail"] += parsed["extended_crc_fail"]
                    if parsed["drop_bytes"]:
                        stats["drop_bytes"] += parsed["drop_bytes"]
                    if parsed["headers_seen"]:
                        stats["headers_seen"] += parsed["headers_seen"]
                    if parsed["has_fail_sample"]:
                        stats["has_fail_sample"] = True
                        stats["last_extended_crc_rx"] = parsed["last_extended_crc_rx"]
                        stats["last_extended_crc_calc"] = parsed["last_extended_crc_calc"]
                        stats["last_fail_prefix"] = parsed["last_fail_prefix"]
                except Exception as exc:
                    stats["read_exception"] += 1
                    print(f"[probe] read exception on {opened_port}: {exc}")
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None

            now = time.monotonic()
            if (now - t_last_summary) * 1000.0 >= summary_ms:
                d_bytes = stats["bytes"] - last_stats["bytes"]
                d_reads = stats["read_calls"] - last_stats["read_calls"]
                d_zero = stats["read_zero"] - last_stats["read_zero"]
                d_drop = stats["drop_bytes"] - last_stats["drop_bytes"]
                d_headers = stats["headers_seen"] - last_stats["headers_seen"]
                d_ok49 = stats["extended_ok"] - last_stats["extended_ok"]
                d_crc49 = stats["extended_crc_fail"] - last_stats["extended_crc_fail"]
                frame_age = -1.0 if frame is None else (time.monotonic() - frame["t"]) * 1000.0
                print(
                    f"[probe][{summary_ms}ms] port={opened_port if ser and ser.is_open else '<closed>'} "
                    f"bytes={d_bytes} reads={d_reads} zero={d_zero} drop={d_drop} hdr={d_headers} "
                    f"ok49={d_ok49} crc49={d_crc49} frame_age={frame_age:.0f}ms "
                    f"data_age={_age_ms(t_last_data):.0f}ms"
                )
                if frame:
                    frame_age = -1.0 if frame is None else (time.monotonic() - frame["t"]) * 1000.0
                    print(
                        "[frame] proto=49B yaw={yaw:.2f}deg pitch={pitch:.2f}deg roll={roll:.2f}deg "
                        "yaw_odom={yaw_odom:.3f} pitch_odom={pitch_odom:.3f} yaw_vel={yaw_vel:.3f} "
                        "pitch_vel={pitch_vel:.3f} color={detect_color} reset={reset_tracker} robot_id={robot_id}".format(
                            yaw=frame["yaw"] * K_RAD2DEG,
                            pitch=frame["pitch"] * K_RAD2DEG,
                            roll=frame["roll"] * K_RAD2DEG,
                            **frame,
                        )
                    )
                elif last_chunk_hex:
                    print(f"[frame] no valid frame yet, last_chunk={last_chunk_hex}")
                if stats["has_fail_sample"]:
                    print(
                        f"[fail] crc49(rx/calc)=0x{stats['last_extended_crc_rx']:04X}/0x{stats['last_extended_crc_calc']:04X} "
                        f"prefix={stats['last_fail_prefix']}"
                    )
                last_stats = dict(stats)
                t_last_summary = now
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    print(
        "done total: open_ok={open_ok} open_fail={open_fail} bytes={bytes} reads={reads} zero={zero} "
        "drop={drop} ok49={ok49} crc49={crc49} read_exc={read_exc}".format(
            open_ok=stats["open_ok"],
            open_fail=stats["open_fail"],
            bytes=stats["bytes"],
            reads=stats["read_calls"],
            zero=stats["read_zero"],
            drop=stats["drop_bytes"],
            ok49=stats["extended_ok"],
            crc49=stats["extended_crc_fail"],
            read_exc=stats["read_exception"],
        )
    )
    return 0
