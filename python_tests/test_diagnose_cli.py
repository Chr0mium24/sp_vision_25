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


def _invoke_with_guard(monkeypatch, args):
    def boom(*_args, **_kwargs):
        raise AssertionError("diagnose script bridge should not run for list actions")

    monkeypatch.setattr(main, "_run_diagnose_script", boom)
    result = runner.invoke(app, args)
    return result


def test_camera_list_is_handled_in_python(monkeypatch):
    result = _invoke_with_guard(monkeypatch, ["camera", "list"])
    assert result.exit_code == 0
    assert "camera_test" in result.stdout
    assert "camera_window_test" in result.stdout


def test_gimbal_list_is_handled_in_python(monkeypatch):
    result = _invoke_with_guard(monkeypatch, ["gimbal", "list"])
    assert result.exit_code == 0
    assert "gimbal_ui_test" in result.stdout


def test_auto_aim_list_is_handled_in_python(monkeypatch):
    result = _invoke_with_guard(monkeypatch, ["auto-aim", "list"])
    assert result.exit_code == 0
    assert "auto_aim_ui_test" in result.stdout


def test_camera_info_is_handled_in_python(monkeypatch):
    called = []

    def fake_print_camera_info(console=None):
        called.append(console is not None)

    def boom(*_args, **_kwargs):
        raise AssertionError("bridge should not run")

    monkeypatch.setattr(main, "_run_diagnose_script", boom)
    monkeypatch.setattr(main, "print_camera_info", fake_print_camera_info)
    result = runner.invoke(app, ["camera", "info"])
    assert result.exit_code == 0
    assert called == [True]


def test_gimbal_port_info_is_handled_in_python(monkeypatch):
    called = []

    def fake_print_gimbal_port_info(config_path=None, console=None):
        called.append((config_path, console is not None))

    def boom(*_args, **_kwargs):
        raise AssertionError("bridge should not run")

    monkeypatch.setattr(main, "_run_diagnose_script", boom)
    monkeypatch.setattr(main, "print_gimbal_port_info", fake_print_gimbal_port_info)
    result = runner.invoke(app, ["gimbal", "port-info"])
    assert result.exit_code == 0
    assert called == [(None, True)]


def test_camera_quick_is_handled_in_python(monkeypatch):
    called = []

    def boom(*_args, **_kwargs):
        raise AssertionError("bridge should not run")

    def fake_handle(action, config, extra_args):
        called.append((action, config, extra_args))
        return 0

    monkeypatch.setattr(main, "_run_diagnose_script", boom)
    monkeypatch.setattr(main, "handle_camera_action", fake_handle)
    result = runner.invoke(app, ["camera", "quick", "--fps=60"])
    assert result.exit_code == 0
    assert called == [("quick", None, ["--fps=60"])]


def test_gimbal_quick_is_handled_in_python(monkeypatch):
    called = []

    def boom(*_args, **_kwargs):
        raise AssertionError("bridge should not run")

    def fake_handle(action, config, extra_args):
        called.append((action, config, extra_args))
        return 0

    monkeypatch.setattr(main, "_run_diagnose_script", boom)
    monkeypatch.setattr(main, "handle_gimbal_action", fake_handle)
    result = runner.invoke(app, ["gimbal", "quick"])
    assert result.exit_code == 0
    assert called == [("quick", None, [])]


def test_auto_aim_armor_box_is_handled_in_python(monkeypatch):
    called = []

    def boom(*_args, **_kwargs):
        raise AssertionError("bridge should not run")

    def fake_handle(action, config, extra_args):
        called.append((action, config, extra_args))
        return 0

    monkeypatch.setattr(main, "_run_diagnose_script", boom)
    monkeypatch.setattr(main, "handle_auto_aim_action", fake_handle)
    result = runner.invoke(app, ["auto-aim", "armor-box", "--show=true"])
    assert result.exit_code == 0
    assert called == [("armor-box", None, ["--show=true"])]
