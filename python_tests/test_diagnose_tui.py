from sp_vision_25_python.diagnose.tui import DiagnoseTUI


def test_tui_class_can_be_instantiated():
    app = DiagnoseTUI()
    assert app is not None
    assert app.BINDINGS
