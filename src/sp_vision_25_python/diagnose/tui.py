from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static, TabPane, TabbedContent
from rich.panel import Panel
from rich.table import Table

from .bindings import binding_status
from .paths import build_dir, build_python_dir, repo_root


def _status_table() -> Table:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Item", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("repo_root", str(repo_root()))
    table.add_row("build_dir", "exists" if build_dir().exists() else "missing")
    table.add_row("build_python", "exists" if build_python_dir().exists() else "missing")
    binding = binding_status()
    table.add_row("pybind11", "available" if binding.available else "missing")
    table.add_row("binding_path", binding.path or "-")
    return table


class DiagnoseTUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    TabbedContent {
        height: 1fr;
    }

    .panel {
        padding: 1 2;
    }
    """

    BINDINGS = [("r", "refresh_data", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            yield TabPane("Overview", Static(_status_table(), classes="panel"))
            yield TabPane(
                "Camera",
                Static(
                    Panel(
                        "camera list, info, release, tune, and high-frequency command entry point",
                        title="Camera",
                    ),
                    classes="panel",
                ),
            )
            yield TabPane(
                "Gimbal",
                Static(
                    Panel(
                        "gimbal quick/rxonly/probe/scan/control and port-info entry point",
                        title="Gimbal",
                    ),
                    classes="panel",
                ),
            )
            yield TabPane(
                "Auto Aim",
                Static(
                    Panel(
                        "auto-aim armor/rune commands plus future white-box diagnostics",
                        title="Auto Aim",
                    ),
                    classes="panel",
                ),
            )
        yield Footer()

    def action_refresh_data(self) -> None:
        self.refresh()


def launch_tui() -> None:
    DiagnoseTUI().run()
