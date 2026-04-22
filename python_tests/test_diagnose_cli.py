from typer.testing import CliRunner

import sp_vision_25_python.diagnose.main as main
from sp_vision_25_python.diagnose.main import app


runner = CliRunner()


def test_status_command_smokes():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "sp_vision_25 diagnose status" in result.stdout


def test_bindings_command_smokes():
    result = runner.invoke(app, ["bindings"])
    assert result.exit_code == 0
    assert "sp_vision_bindings" in result.stdout
    assert "Gimbal" in result.stdout


def test_camera_list_is_handled_in_python():
    result = runner.invoke(app, ["camera", "list"])
    assert result.exit_code == 0
    assert "camera_test" in result.stdout
    assert "camera_window_test" in result.stdout


def test_gimbal_list_is_handled_in_python():
    result = runner.invoke(app, ["gimbal", "list"])
    assert result.exit_code == 0
    assert "gimbal_link_diag_test" in result.stdout


def test_auto_aim_list_is_handled_in_python():
    result = runner.invoke(app, ["auto-aim", "list"])
    assert result.exit_code == 0
    assert "auto_aim_ui_test" in result.stdout


def test_camera_info_is_handled_in_python(monkeypatch):
    called = []

    def fake_print_camera_info(console=None):
        called.append(console is not None)

    monkeypatch.setattr(main, "print_camera_info", fake_print_camera_info)
    result = runner.invoke(app, ["camera", "info"])
    assert result.exit_code == 0
    assert called == [True]


def test_gimbal_port_info_is_handled_in_python(monkeypatch):
    called = []

    def fake_print_gimbal_port_info(config_path=None, console=None):
        called.append((config_path, console is not None))

    monkeypatch.setattr(main, "print_gimbal_port_info", fake_print_gimbal_port_info)
    result = runner.invoke(app, ["gimbal", "port-info"])
    assert result.exit_code == 0
    assert called == [(None, True)]


def test_camera_quick_is_handled_in_python(monkeypatch):
    called = []

    def fake_handle(action, config, extra_args):
        called.append((action, config, extra_args))
        return 0

    monkeypatch.setattr(main, "handle_camera_action", fake_handle)
    result = runner.invoke(app, ["camera", "quick", "--fps=60"])
    assert result.exit_code == 0
    assert called == [("quick", None, ["--fps=60"])]


def test_camera_release_is_handled_in_python(monkeypatch):
    called = []

    def fake_handle(action, config, extra_args):
        called.append((action, config, extra_args))
        return 0

    monkeypatch.setattr(main, "handle_camera_action", fake_handle)
    result = runner.invoke(app, ["camera", "release", "--force"])
    assert result.exit_code == 0
    assert called == [("release", None, ["--force"])]


def test_camera_tune_is_handled_in_python(monkeypatch):
    called = []

    def fake_handle(action, config, extra_args):
        called.append((action, config, extra_args))
        return 0

    monkeypatch.setattr(main, "handle_camera_action", fake_handle)
    result = runner.invoke(app, ["camera", "tune", "--show-log"])
    assert result.exit_code == 0
    assert called == [("tune", None, ["--show-log"])]


def test_gimbal_quick_is_handled_in_python(monkeypatch):
    called = []

    def fake_handle(action, config, extra_args):
        called.append((action, config, extra_args))
        return 0

    monkeypatch.setattr(main, "handle_gimbal_action", fake_handle)
    result = runner.invoke(app, ["gimbal", "quick"])
    assert result.exit_code == 0
    assert called == [("quick", None, [])]


def test_gimbal_axis_is_handled_in_python(monkeypatch):
    called = []

    import sp_vision_25_python.diagnose.actions as actions
    from sp_vision_25_python.diagnose.system import default_config_path

    monkeypatch.setattr(
        actions, "run_gimbal_axis", lambda config, extra_args: called.append((config, extra_args)) or 0
    )
    result = runner.invoke(app, ["gimbal", "axis", "--step-deg=6"])
    assert result.exit_code == 0
    assert called == [(default_config_path(), ["--step-deg=6"])]


def test_gimbal_manual_axis_is_handled_in_python(monkeypatch):
    called = []

    import sp_vision_25_python.diagnose.actions as actions
    from sp_vision_25_python.diagnose.system import default_config_path

    monkeypatch.setattr(
        actions,
        "run_gimbal_manual_axis",
        lambda config, extra_args: called.append((config, extra_args)) or 0,
    )
    result = runner.invoke(app, ["gimbal", "manual-axis", "--wait-valid-ms=1000"])
    assert result.exit_code == 0
    assert called == [(default_config_path(), ["--wait-valid-ms=1000"])]


def test_gimbal_snapshot_is_handled_in_python(monkeypatch):
    called = []

    import sp_vision_25_python.diagnose.actions as actions
    from sp_vision_25_python.diagnose.system import default_config_path

    monkeypatch.setattr(
        actions,
        "run_gimbal_snapshot",
        lambda config, extra_args: called.append((config, extra_args)) or 0,
    )
    result = runner.invoke(app, ["gimbal", "snapshot", "--wait-valid-ms=100"])
    assert result.exit_code == 0
    assert called == [(default_config_path(), ["--wait-valid-ms=100"])]


def test_gimbal_watch_is_handled_in_python(monkeypatch):
    called = []

    import sp_vision_25_python.diagnose.actions as actions
    from sp_vision_25_python.diagnose.system import default_config_path

    monkeypatch.setattr(
        actions, "run_gimbal_watch", lambda config, extra_args: called.append((config, extra_args)) or 0
    )
    result = runner.invoke(app, ["gimbal", "watch", "--duration-ms=1"])
    assert result.exit_code == 0
    assert called == [(default_config_path(), ["--duration-ms=1"])]


def test_gimbal_control_is_handled_in_python(monkeypatch):
    called = []

    import sp_vision_25_python.diagnose.actions as actions
    from sp_vision_25_python.diagnose.system import default_config_path

    monkeypatch.setattr(
        actions, "run_gimbal_control", lambda config, extra_args: called.append((config, extra_args)) or 0
    )
    result = runner.invoke(app, ["gimbal", "control", "--mode=control"])
    assert result.exit_code == 0
    assert called == [(default_config_path(), ["--mode=control"])]


def test_gimbal_script_control_is_handled_in_python(monkeypatch):
    called = []

    import sp_vision_25_python.diagnose.actions as actions
    from sp_vision_25_python.diagnose.system import default_config_path

    monkeypatch.setattr(
        actions,
        "run_gimbal_script_control",
        lambda config, extra_args: called.append((config, extra_args)) or 0,
    )
    result = runner.invoke(app, ["gimbal", "script-control", "--no-input"])
    assert result.exit_code == 0
    assert called == [(default_config_path(), ["--no-input"])]


def test_auto_aim_armor_box_is_handled_in_python(monkeypatch):
    called = []

    def fake_handle(action, config, extra_args):
        called.append((action, config, extra_args))
        return 0

    monkeypatch.setattr(main, "handle_auto_aim_action", fake_handle)
    result = runner.invoke(app, ["auto-aim", "armor-box", "--show=true"])
    assert result.exit_code == 0
    assert called == [("armor-box", None, ["--show=true"])]


def test_auto_aim_rune_tune_is_handled_in_python(monkeypatch):
    called = []

    def fake_handle(action, config, extra_args):
        called.append((action, config, extra_args))
        return 0

    monkeypatch.setattr(main, "handle_auto_aim_action", fake_handle)
    result = runner.invoke(app, ["auto-aim", "rune-tune", "assets/demo/power_rune_demo"])
    assert result.exit_code == 0
    assert called == [("rune-tune", None, ["assets/demo/power_rune_demo"])]


def test_tui_command_launches_dashboard(monkeypatch):
    called = []

    monkeypatch.setattr(main, "launch_tui", lambda: called.append(True))
    result = runner.invoke(app, ["tui"])
    assert result.exit_code == 0
    assert called == [True]
