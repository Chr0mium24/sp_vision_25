from pathlib import Path

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

    def fake_run(path, args):
        recorded.append((path, args))
        return 0

    monkeypatch.setattr(actions, "run_executable", fake_run)
    monkeypatch.setattr(actions, "gimbal_scan_ports", lambda: ["/dev/ttyUSB0", "/dev/ttyUSB1"])
    result = actions.handle_gimbal_action("scan", Path("configs/demo.yaml"), ["--loop-ms=50"])
    assert result == 0
    assert recorded == [
        (
            actions.binary_path("gimbal", "gimbal_link_diag_test"),
            [
                "configs/demo.yaml",
                "--ports=/dev/ttyUSB0,/dev/ttyUSB1",
                "--duration-ms=3000",
                "--summary-ms=1000",
                "--loop-ms=50",
            ],
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
