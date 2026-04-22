from pathlib import Path

import builtins

import sp_vision_25_python.diagnose.actions as actions


def test_camera_quick_uses_config_flag(monkeypatch):
    recorded = []

    def fake_run(path, args):
        recorded.append((path, args))
        return 0

    monkeypatch.setattr(actions, "run_executable", fake_run)
    result = actions.handle_camera_action("quick", Path("configs/demo.yaml"), ["--fps=60"])
    assert result == 0
    assert recorded == [
        (
            actions.binary_path("camera", "camera_test"),
            ["--config-path=configs/demo.yaml", "--fps=60"],
        )
    ]


def test_gimbal_scan_uses_detected_ports(monkeypatch):
    recorded = []

    monkeypatch.setattr(
        actions,
        "run_gimbal_link_diag",
        lambda config, args: recorded.append((config, args)) or 0,
    )
    monkeypatch.setattr(actions, "gimbal_scan_ports", lambda: ["/dev/ttyUSB0", "/dev/ttyUSB1"])
    result = actions.handle_gimbal_action("scan", Path("configs/demo.yaml"), ["--loop-ms=50"])
    assert result == 0
    assert recorded == [
        (
            Path("configs/demo.yaml"),
            ["--ports=/dev/ttyUSB0,/dev/ttyUSB1", "--duration-ms=3000", "--summary-ms=1000", "--loop-ms=50"],
        )
    ]


def test_auto_aim_armor_box_adds_show(monkeypatch):
    recorded = []

    def fake_run(path, args):
        recorded.append((path, args))
        return 0

    monkeypatch.setattr(actions, "run_executable", fake_run)
    result = actions.handle_auto_aim_action("armor-box", Path("configs/demo.yaml"), ["--scale=0.7"])
    assert result == 0
    assert recorded == [
        (
            actions.binary_path("auto_aim", "auto_aim_ui_test"),
            ["configs/demo.yaml", "--show=true", "--scale=0.7"],
        )
    ]


def test_auto_aim_armor_box_keeps_show(monkeypatch):
    recorded = []

    def fake_run(path, args):
        recorded.append((path, args))
        return 0

    monkeypatch.setattr(actions, "run_executable", fake_run)
    result = actions.handle_auto_aim_action("armor-box", Path("configs/demo.yaml"), ["--show=false"])
    assert result == 0
    assert recorded == [
        (
            actions.binary_path("auto_aim", "auto_aim_ui_test"),
            ["configs/demo.yaml", "--show=false"],
        )
    ]


def test_auto_aim_rune_box_uses_default_input(monkeypatch):
    recorded = []

    def fake_run(path, args):
        recorded.append((path, args))
        return 0

    monkeypatch.setattr(actions, "run_executable", fake_run)
    result = actions.handle_auto_aim_action("rune-box", Path("configs/demo.yaml"), ["--start-index=0"])
    assert result == 0
    assert recorded == [
        (
            actions.binary_path("auto_aim", "auto_power_rune_test"),
            ["--config-path=configs/demo.yaml", "assets/demo/power_rune_demo", "--start-index=0"],
        )
    ]


def test_camera_release_requires_sudo(monkeypatch):
    monkeypatch.setattr(actions.os, "geteuid", lambda: 1000)
    result = actions._run_camera_release([])
    assert result == 1


def test_camera_release_builds_commands(monkeypatch):
    captured = []

    monkeypatch.setattr(actions.os, "geteuid", lambda: 0)
    monkeypatch.setattr(actions, "command_exists", lambda name: True)
    def fake_capture(args):
        if args[0] == "lsusb":
            return type("R", (), {"stdout": "Bus 001 Device 002: ID 2bdf:0001 foo\n"})()
        if args[0] == "docker":
            return type("R", (), {"stdout": "abc123 rm_bringup foo bar\n"})()
        if args[0] == "pgrep":
            return type("R", (), {"stdout": "1234\n"})()
        return type("R", (), {"stdout": ""})()

    monkeypatch.setattr(actions, "run_and_capture", fake_capture)
    monkeypatch.setattr(actions.subprocess, "run", lambda args, check=False: captured.append(args))
    result = actions._run_camera_release(["--force"])
    assert result == 0
    assert any(item[:2] == ["docker", "update"] for item in captured)


def test_rune_tune_updates_config_and_replays(monkeypatch, tmp_path):
    cfg = tmp_path / "demo.yaml"
    cfg.write_text(
        "yaw_offset: 0\npitch_offset: 0\nfire_gap_time: 0.7\npredict_time: 0.12\n",
        encoding="utf-8",
    )
    recorded = []
    inputs = iter(["y 1.5", "f 0.2", "r", "q"])

    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    monkeypatch.setattr(actions, "run_executable", lambda path, args: recorded.append((path, args)) or 0)
    result = actions._run_rune_tune(cfg, [])
    assert result == 0
    assert recorded
    updated = cfg.read_text(encoding="utf-8")
    assert "yaw_offset: 1.5" in updated
    assert "fire_gap_time: 0.2" in updated
