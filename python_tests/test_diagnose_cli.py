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
