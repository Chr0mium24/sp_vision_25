from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .bindings import binding_status
from .inventory import print_binary_status
from .system import print_camera_info, print_gimbal_port_info
from .paths import build_dir, build_python_dir, diagnose_script, repo_root
from .runner import run_script

app = typer.Typer(
    help="Python diagnose control plane for sp_vision_25.",
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)

console = Console()


def _print_status_row(table: Table, label: str, value: str) -> None:
    table.add_row(label, value)


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


def _run_diagnose_script(domain: str, action: str, config: Path | None, extra_args: list[str]) -> int:
    script = diagnose_script(domain, "diagnose.sh")
    args = [action]
    if config is not None:
        args.append(str(config))
    args.extend(extra_args)
    return run_script(script, args)


def _run_list(domain: str) -> None:
    print_binary_status(domain)


@app.command()
def camera(
    ctx: typer.Context,
    action: str,
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Bridge to diagnostics/camera/diagnose.sh while we migrate the workflow."""

    if action == "list":
        _run_list("camera")
        return
    if action == "info":
        print_camera_info(console)
        return
    raise typer.Exit(code=_run_diagnose_script("camera", action, config, list(ctx.args)))


@app.command()
def gimbal(
    ctx: typer.Context,
    action: str,
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Bridge to diagnostics/gimbal/diagnose.sh while we migrate the workflow."""

    if action == "list":
        _run_list("gimbal")
        return
    if action == "port-info":
        print_gimbal_port_info(config, console)
        return
    raise typer.Exit(code=_run_diagnose_script("gimbal", action, config, list(ctx.args)))


@app.command(name="auto-aim")
def auto_aim(
    ctx: typer.Context,
    action: str,
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Bridge to diagnostics/auto_aim/diagnose.sh while we migrate the workflow."""

    if action == "list":
        _run_list("auto_aim")
        return
    raise typer.Exit(code=_run_diagnose_script("auto_aim", action, config, list(ctx.args)))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
