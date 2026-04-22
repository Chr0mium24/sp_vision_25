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


def _print_status_row(table: Table, label: str, value: str) -> None:
    table.add_row(label, value)


def _echo_help(text: str) -> None:
    print(text.strip())


def _exit_from_handler(result: int | None) -> None:
    raise typer.Exit(code=0 if result is None else result)


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
            "Command",
            "ExtendedKalmanFilter",
            "Runtime",
            "Solver",
            "Target",
            "Tracker",
        ]:
            _print_status_row(table, symbol, "yes" if hasattr(module, symbol) else "no")

    console.print(table)


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


@camera_app.command("release")
def camera_release(ctx: typer.Context) -> None:
    _exit_from_handler(handle_camera_action("release", None, list(ctx.args)))


@camera_app.command("tune")
def camera_tune(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_camera_action("tune", config, list(ctx.args)))


@camera_app.command("quick")
def camera_quick(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_camera_action("quick", config, list(ctx.args)))


@camera_app.command("detect")
def camera_detect(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_camera_action("detect", config, list(ctx.args)))


@camera_app.command("window")
def camera_window(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_camera_action("window", config, list(ctx.args)))


@camera_app.command("save")
def camera_save(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_camera_action("save", config, list(ctx.args)))


@camera_app.command("usb")
def camera_usb(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_camera_action("usb", config, list(ctx.args)))


@camera_app.command("usb-detect")
def camera_usb_detect(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_camera_action("usb-detect", config, list(ctx.args)))


@camera_app.command("thread")
def camera_thread(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_camera_action("thread", config, list(ctx.args)))


@camera_app.command("handeye")
def camera_handeye(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_camera_action("handeye", config, list(ctx.args)))


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


@gimbal_app.command("quick")
def gimbal_quick(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("quick", config, list(ctx.args)))


@gimbal_app.command("rxonly")
def gimbal_rxonly(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("rxonly", config, list(ctx.args)))


@gimbal_app.command("proto")
def gimbal_proto(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("proto", config, list(ctx.args)))


@gimbal_app.command("probe")
def gimbal_probe(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("probe", config, list(ctx.args)))


@gimbal_app.command("probe-raw")
def gimbal_probe_raw(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("probe-raw", config, list(ctx.args)))


@gimbal_app.command("scan")
def gimbal_scan(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("scan", config, list(ctx.args)))


@gimbal_app.command("snapshot")
def gimbal_snapshot(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("snapshot", config, list(ctx.args)))


@gimbal_app.command("watch")
def gimbal_watch(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("watch", config, list(ctx.args)))


@gimbal_app.command("control")
def gimbal_control(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("control", config, list(ctx.args)))


@gimbal_app.command("script-control")
def gimbal_script_control(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("script-control", config, list(ctx.args)))


@gimbal_app.command("axis")
def gimbal_axis(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("axis", config, list(ctx.args)))


@gimbal_app.command("manual-axis")
def gimbal_manual_axis(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_gimbal_action("manual-axis", config, list(ctx.args)))


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


@auto_aim_app.command("armor-box")
def auto_aim_armor_box(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("armor-box", config, list(ctx.args)))


@auto_aim_app.command("armor-intent")
def auto_aim_armor_intent(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("armor-intent", config, list(ctx.args)))


@auto_aim_app.command("armor-rec")
def auto_aim_armor_rec(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("armor-rec", config, list(ctx.args)))


@auto_aim_app.command("armor-tune")
def auto_aim_armor_tune(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("armor-tune", config, list(ctx.args)))


@auto_aim_app.command("armor-offline")
def auto_aim_armor_offline(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("armor-offline", config, list(ctx.args)))


@auto_aim_app.command("rune-box")
def auto_aim_rune_box(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("rune-box", config, list(ctx.args)))


@auto_aim_app.command("rune-rec")
def auto_aim_rune_rec(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("rune-rec", config, list(ctx.args)))


@auto_aim_app.command("rune-tune")
def auto_aim_rune_tune(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("rune-tune", config, list(ctx.args)))


@auto_aim_app.command("rune-online")
def auto_aim_rune_online(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("rune-online", config, list(ctx.args)))


@auto_aim_app.command("rune-online-mpc")
def auto_aim_rune_online_mpc(
    ctx: typer.Context,
    config: Path | None = typer.Argument(None),
) -> None:
    _exit_from_handler(handle_auto_aim_action("rune-online-mpc", config, list(ctx.args)))


app.add_typer(camera_app, name="camera")
app.add_typer(gimbal_app, name="gimbal")
app.add_typer(auto_aim_app, name="auto-aim")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
