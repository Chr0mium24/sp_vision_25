from __future__ import annotations

from pathlib import Path

import typer

from .workflow import calibrate_camera, calibrate_handeye, calibrate_robotworld_handeye, capture, split_video

app = typer.Typer(help="Python calibration tools for sp_vision_25.", add_completion=False)


@app.command("capture")
def capture_cmd(
    config_path: Path = typer.Argument(Path("configs/calibration.yaml"), help="YAML config path"),
    output_folder: Path = typer.Argument(Path("assets/img_with_q"), help="Output folder"),
    imu: bool = typer.Option(False, "--imu", help="Enable IMU capture"),
    no_show: bool = typer.Option(False, "--no-show", help="Disable preview windows"),
) -> None:
    raise typer.Exit(code=capture(config_path, output_folder, imu, show=not no_show))


@app.command("calibrate-camera")
def calibrate_camera_cmd(
    input_folder: Path = typer.Argument(Path("assets/img_with_q"), help="Input folder"),
    config_path: Path = typer.Option(Path("configs/calibration.yaml"), "--config-path", "-c", help="YAML config path"),
    no_show: bool = typer.Option(False, "--no-show", help="Disable preview windows"),
) -> None:
    raise typer.Exit(code=calibrate_camera(config_path, input_folder, show=not no_show))


@app.command("calibrate-handeye")
def calibrate_handeye_cmd(
    input_folder: Path = typer.Argument(Path("assets/img_with_q"), help="Input folder"),
    config_path: Path = typer.Option(Path("configs/calibration.yaml"), "--config-path", "-c", help="YAML config path"),
    no_show: bool = typer.Option(False, "--no-show", help="Disable preview windows"),
) -> None:
    raise typer.Exit(code=calibrate_handeye(config_path, input_folder, show=not no_show))


@app.command("calibrate-robotworld-handeye")
def calibrate_robotworld_handeye_cmd(
    input_folder: Path = typer.Argument(Path("assets/img_with_q"), help="Input folder"),
    config_path: Path = typer.Option(Path("configs/calibration.yaml"), "--config-path", "-c", help="YAML config path"),
    no_show: bool = typer.Option(False, "--no-show", help="Disable preview windows"),
) -> None:
    raise typer.Exit(code=calibrate_robotworld_handeye(config_path, input_folder, show=not no_show))


@app.command("split-video")
def split_video_cmd(
    input_path: Path = typer.Argument(..., help="Base input path without extension"),
    output_path: Path = typer.Option(Path("records/Big/2024-05-14_11-6-26"), "--output-path", "-p", help="Base output path without extension"),
    start_index: int = typer.Option(0, "--start-index", "-s", help="Start frame index"),
    end_index: int = typer.Option(0, "--end-index", "-e", help="End frame index"),
    no_show: bool = typer.Option(False, "--no-show", help="Disable preview windows"),
) -> None:
    raise typer.Exit(code=split_video(input_path, output_path, start_index, end_index, show=not no_show))


@app.command("help")
def help_cmd() -> None:
    print(
        """
Usage:
  sp-vision-calibration <command> [args...]

Commands:
  capture
  calibrate-camera
  calibrate-handeye
  calibrate-robotworld-handeye
  split-video
  help
""".strip()
    )


def main() -> None:
    app()
