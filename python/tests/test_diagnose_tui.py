from rich.panel import Panel

from sp_vision_25_python.diagnose.tui import DiagnoseTUI, _domain_panel, _status_table


def test_tui_class_can_be_instantiated():
    app = DiagnoseTUI()
    assert app is not None
    assert app.BINDINGS


def test_tui_panels_are_renderables():
    assert _status_table().row_count >= 5
    assert isinstance(
        _domain_panel("camera", "Camera", "intro", ["sp-vision-diagnose camera list"]),
        Panel,
    )
