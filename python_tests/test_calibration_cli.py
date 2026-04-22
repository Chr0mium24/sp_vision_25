from typer.testing import CliRunner

from sp_vision_25_python.calibration.main import app

runner = CliRunner()


def test_calibration_help_smokes():
    result = runner.invoke(app, ["help"])
    assert result.exit_code == 0
    assert "sp-vision-calibration" in result.stdout
    assert "calibrate-camera" in result.stdout
    assert "split-video" in result.stdout
