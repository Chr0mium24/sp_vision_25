from pathlib import Path

import sp_vision_bindings as svb


def test_tracker_can_step_and_report_debug():
    config = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"
    solver = svb.Solver(str(config))
    tracker = svb.Tracker(str(config), solver)

    assert tracker.state == "lost"

    result = tracker.track([])

    assert result["state"] in {"lost", "detecting", "tracking", "temp_lost", "switching"}
    assert isinstance(result["debug"], dict)
    assert "valid" in result["debug"]
