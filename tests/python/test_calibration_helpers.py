from __future__ import annotations

import numpy as np

from sp_vision_25_python.calibration.math_tools import eulers_from_matrix, rotation_matrix
from sp_vision_25_python.calibration.workflow import _centers_3d_board, _centers_3d_planar


def test_planar_centers_shape_and_order():
    points = _centers_3d_planar((3, 2), 10.0)
    assert points.shape == (6, 3)
    assert np.allclose(points[0], [0.0, 0.0, 0.0])
    assert np.allclose(points[1], [10.0, 0.0, 0.0])
    assert np.allclose(points[3], [0.0, 10.0, 0.0])


def test_board_centers_shape_and_order():
    points = _centers_3d_board((3, 2), 10.0)
    assert points.shape == (6, 3)
    assert np.allclose(points[0], [0.0, 15.0, 10.0])
    assert np.allclose(points[1], [0.0, 5.0, 10.0])
    assert np.allclose(points[3], [0.0, 15.0, 0.0])


def test_rotation_matrix_roundtrip_for_zyx():
    ypr = (0.3, -0.2, 0.1)
    R = rotation_matrix(ypr)
    recovered = eulers_from_matrix(R, 2, 1, 0)
    assert np.allclose(recovered, ypr, atol=1e-6)
