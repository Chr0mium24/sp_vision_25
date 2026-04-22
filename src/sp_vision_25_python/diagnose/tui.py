from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static, TabPane, TabbedContent
from rich.panel import Panel
from rich.table import Table

from .bindings import binding_status
from .inventory import binary_specs
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


def _domain_table(domain: str) -> Table:
    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("Binary", style="cyan", no_wrap=True)
    table.add_column("Status", style="white", no_wrap=True)
    table.add_column("Path", style="white")
    available = 0
    specs = binary_specs(domain)
    for spec in specs:
        exists = spec.path.is_file() and spec.path.stat().st_mode & 0o111 != 0
        status = "ok" if exists else "missing"
        if exists:
            available += 1
        table.add_row(spec.label, status, str(spec.path))
    table.caption = f"{available}/{len(specs)} binaries available"
    return table


def _domain_panel(domain: str, title: str, intro: str, shortcuts: list[str]) -> Panel:
    shortcut_table = Table(show_header=False, box=None, pad_edge=False)
    shortcut_table.add_column("Key", style="cyan", no_wrap=True)
    shortcut_table.add_column("Command", style="white")
    for shortcut in shortcuts:
        shortcut_table.add_row("•", shortcut)

    inner = Table.grid(expand=True)
    inner.add_row(intro)
    inner.add_row("")
    inner.add_row(shortcut_table)
    inner.add_row("")
    inner.add_row(_domain_table(domain))
    return Panel(inner, title=title)


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

    BINDINGS = [
        ("r", "refresh_data", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            yield TabPane("Overview", Static(id="overview", classes="panel"))
            yield TabPane("Camera", Static(id="camera", classes="panel"))
            yield TabPane("Gimbal", Static(id="gimbal", classes="panel"))
            yield TabPane("Auto Aim", Static(id="auto_aim", classes="panel"))
        yield Footer()

    def on_mount(self) -> None:
        self._sync_panels()

    def _sync_panels(self) -> None:
        self.query_one("#overview", Static).update(
            Panel(
                _status_table(),
                title="Workspace",
                subtitle="Press r to refresh",
            )
        )
        self.query_one("#camera", Static).update(
            _domain_panel(
                "camera",
                "Camera",
                "Camera diagnose currently covers list/info/release/tune and the C++ test entry points.",
                [
                    "sp-vision-diagnose camera list",
                    "sp-vision-diagnose camera info",
                    "sp-vision-diagnose camera release --force",
                    "sp-vision-diagnose camera tune configs/standard3.yaml",
                ],
            )
        )
        self.query_one("#gimbal", Static).update(
            _domain_panel(
                "gimbal",
                "Gimbal",
                "Gimbal diagnose now exposes link checks, serial probes, scan helpers, and port info.",
                [
                    "sp-vision-diagnose gimbal list",
                    "sp-vision-diagnose gimbal quick configs/standard3.yaml",
                    "sp-vision-diagnose gimbal port-info",
                ],
            )
        )
        self.query_one("#auto_aim", Static).update(
            _domain_panel(
                "auto_aim",
                "Auto Aim",
                "Auto-aim diagnose is already routing through Python and can drive the real C++ binaries.",
                [
                    "sp-vision-diagnose auto-aim list",
                    "sp-vision-diagnose auto-aim armor-box configs/standard3.yaml",
                    "sp-vision-diagnose auto-aim rune-tune configs/sentry.yaml assets/demo/power_rune_demo",
                ],
            )
        )

    def action_refresh_data(self) -> None:
        self._sync_panels()


def launch_tui() -> None:
    DiagnoseTUI().run()
