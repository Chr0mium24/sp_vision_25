from typer.testing import CliRunner

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
