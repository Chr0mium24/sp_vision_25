from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .actions import handle_auto_aim_action, handle_camera_action, handle_gimbal_action
from .bindings import binding_status
from .inventory import print_binary_status
from .system import print_camera_info, print_gimbal_port_info
from .paths import build_dir, build_python_dir, repo_root
from .tui import launch_tui

app = typer.Typer(
    help="Python diagnose control plane for sp_vision_25.",
    add_completion=False,
)

camera_app = typer.Typer(
    help="Camera diagnose actions.",
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)

gimbal_app = typer.Typer(
    help="Gimbal diagnose actions.",
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)

auto_aim_app = typer.Typer(
    help="Auto-aim diagnose actions.",
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)

console = Console()
EXTRA_CONTEXT = {"allow_extra_args": True, "ignore_unknown_options": True}


def _print_status_row(table: Table, label: str, value: str) -> None:
    table.add_row(label, value)


def _echo_help(text: str) -> None:
    print(text.strip())


def _exit_from_handler(result: int | None) -> None:
    raise typer.Exit(code=0 if result is None else result)


def _extra_command(app: typer.Typer, name: str):
    return app.command(name, context_settings=EXTRA_CONTEXT)


def _split_config_arg(extra_args: list[str], default: Path | None = None) -> tuple[Path | None, list[str]]:
    if extra_args and extra_args[0].endswith(".yaml"):
        return Path(extra_args[0]), extra_args[1:]
    return default, extra_args


@app.command()
def status() -> None:
    """Show workspace, build, and binding status."""

    table = Table(title="sp_vision_25 diagnose status", show_lines=False)
    table.add_column("Item", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    _print_status_row(table, "repo_root", str(repo_root()))
    _print_status_row(table, "build_dir", "exists" if build_dir().exists() else "missing")
    _print_status_row(
        table, "build_python", "exists" if build_python_dir().exists() else "missing"
    )
    binding = binding_status()
    _print_status_row(table, "pybind11", "available" if binding.available else "missing")
    _print_status_row(table, "binding_path", binding.path or "-")
    console.print(table)


@app.command()
def bindings() -> None:
    """Show the current Python binding surface."""

    module = None
    error: str | None = None
    try:
        from .bindings import load_bindings

        module = load_bindings()
    except Exception as exc:  # pragma: no cover - defensive display path
        error = str(exc)

    table = Table(title="sp_vision_bindings", show_lines=False)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")

    if module is None:
        _print_status_row(table, "import", f"failed: {error}")
    else:
        _print_status_row(table, "import", "ok")
        for symbol in [
            "Armor",
            "Aimer",
            "Gimbal",
            "GimbalMode",
            "GimbalRxStats",
            "GimbalState",
            "Command",
            "ExtendedKalmanFilter",
            "Runtime",
            "Solver",
            "Target",
            "Tracker",
            "VisionToGimbal",
        ]:
            _print_status_row(table, symbol, "yes" if hasattr(module, symbol) else "no")

    console.print(table)


@app.command()
def tui() -> None:
    """Launch the Textual diagnose dashboard."""

    launch_tui()


@camera_app.command("help")
def camera_help() -> None:
    _echo_help(
        """
        Usage:
          sp-vision-diagnose camera <action> [config.yaml] [extra args...]

        Actions:
          info
          list
          release
          tune
          quick
          detect
          window
          save
          usb
          usb-detect
          thread
          handeye
          help
        """
    )


@camera_app.command("list")
def camera_list() -> None:
    print_binary_status("camera")


@camera_app.command("info")
def camera_info() -> None:
    print_camera_info(console)


@_extra_command(camera_app, "release")
def camera_release(ctx: typer.Context) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("release", config, extra_args))


@_extra_command(camera_app, "tune")
def camera_tune(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("tune", config, extra_args))


@_extra_command(camera_app, "quick")
def camera_quick(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("quick", config, extra_args))


@_extra_command(camera_app, "detect")
def camera_detect(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("detect", config, extra_args))


@_extra_command(camera_app, "window")
def camera_window(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("window", config, extra_args))


@_extra_command(camera_app, "save")
def camera_save(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("save", config, extra_args))


@_extra_command(camera_app, "usb")
def camera_usb(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("usb", config, extra_args))


@_extra_command(camera_app, "usb-detect")
def camera_usb_detect(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("usb-detect", config, extra_args))


@_extra_command(camera_app, "thread")
def camera_thread(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("thread", config, extra_args))


@_extra_command(camera_app, "handeye")
def camera_handeye(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_camera_action("handeye", config, extra_args))


@gimbal_app.command("help")
def gimbal_help() -> None:
    _echo_help(
        """
        Usage:
          sp-vision-diagnose gimbal <action> [config.yaml] [extra args...]

        Actions:
          quick
          rxonly
          proto
          probe
          probe-raw
          scan
          snapshot
          watch
          control
          script-control
          axis
          manual-axis
          port-info
          help
        """
    )


@gimbal_app.command("list")
def gimbal_list() -> None:
    print_binary_status("gimbal")


@_extra_command(gimbal_app, "quick")
def gimbal_quick(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("quick", config, extra_args))


@_extra_command(gimbal_app, "rxonly")
def gimbal_rxonly(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("rxonly", config, extra_args))


@_extra_command(gimbal_app, "proto")
def gimbal_proto(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("proto", config, extra_args))


@_extra_command(gimbal_app, "probe")
def gimbal_probe(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("probe", config, extra_args))


@_extra_command(gimbal_app, "probe-raw")
def gimbal_probe_raw(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("probe-raw", config, extra_args))


@_extra_command(gimbal_app, "scan")
def gimbal_scan(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("scan", config, extra_args))


@_extra_command(gimbal_app, "snapshot")
def gimbal_snapshot(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("snapshot", config, extra_args))


@_extra_command(gimbal_app, "watch")
def gimbal_watch(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("watch", config, extra_args))


@_extra_command(gimbal_app, "control")
def gimbal_control(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("control", config, extra_args))


@_extra_command(gimbal_app, "script-control")
def gimbal_script_control(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("script-control", config, extra_args))


@_extra_command(gimbal_app, "axis")
def gimbal_axis(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("axis", config, extra_args))


@_extra_command(gimbal_app, "manual-axis")
def gimbal_manual_axis(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_gimbal_action("manual-axis", config, extra_args))


@gimbal_app.command("port-info")
def gimbal_port_info(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    print_gimbal_port_info(config, console)


@auto_aim_app.command("help")
def auto_aim_help() -> None:
    _echo_help(
        """
        Usage:
          sp-vision-diagnose auto-aim <action> [config.yaml] [extra args...]

        Actions:
          list
          armor-box
          armor-intent
          armor-rec
          armor-tune
          armor-offline
          rune-box
          rune-rec
          rune-tune
          rune-online
          rune-online-mpc
          help
        """
    )


@auto_aim_app.command("list")
def auto_aim_list() -> None:
    print_binary_status("auto_aim")


@_extra_command(auto_aim_app, "armor-box")
def auto_aim_armor_box(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("armor-box", config, extra_args))


@_extra_command(auto_aim_app, "armor-intent")
def auto_aim_armor_intent(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("armor-intent", config, extra_args))


@_extra_command(auto_aim_app, "armor-rec")
def auto_aim_armor_rec(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("armor-rec", config, extra_args))


@_extra_command(auto_aim_app, "armor-tune")
def auto_aim_armor_tune(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("armor-tune", config, extra_args))


@_extra_command(auto_aim_app, "armor-offline")
def auto_aim_armor_offline(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("armor-offline", config, extra_args))


@_extra_command(auto_aim_app, "rune-box")
def auto_aim_rune_box(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("rune-box", config, extra_args))


@_extra_command(auto_aim_app, "rune-rec")
def auto_aim_rune_rec(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("rune-rec", config, extra_args))


@_extra_command(auto_aim_app, "rune-tune")
def auto_aim_rune_tune(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("rune-tune", config, extra_args))


@_extra_command(auto_aim_app, "rune-online")
def auto_aim_rune_online(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("rune-online", config, extra_args))


@_extra_command(auto_aim_app, "rune-online-mpc")
def auto_aim_rune_online_mpc(
    ctx: typer.Context,
) -> None:
    config, extra_args = _split_config_arg(list(ctx.args))
    _exit_from_handler(handle_auto_aim_action("rune-online-mpc", config, extra_args))


app.add_typer(camera_app, name="camera")
app.add_typer(gimbal_app, name="gimbal")
app.add_typer(auto_aim_app, name="auto-aim")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
