from pathlib import Path

import numpy as np

import sp_vision_bindings as svb


def test_runtime_constructs_and_steps_on_blank_image():
    config = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"
    runtime = svb.Runtime(str(config), False)

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = runtime.step(image, (1.0, 0.0, 0.0, 0.0))

    assert isinstance(result, dict)
    assert "command" in result
    assert "tracker_state" in result
    assert "debug" in result
