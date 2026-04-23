import numpy as np
import pytest

sp_vision_bindings = pytest.importorskip("sp_vision_bindings")


def test_ekf_linear_predict_update() -> None:
    ekf = sp_vision_bindings.ExtendedKalmanFilter(
        np.array([1.0, 2.0], dtype=np.float64),
        np.eye(2, dtype=np.float64),
    )

    x_pred = ekf.predict(np.eye(2, dtype=np.float64), np.zeros((2, 2), dtype=np.float64))
    np.testing.assert_allclose(x_pred, [1.0, 2.0])
    np.testing.assert_allclose(ekf.x, [1.0, 2.0])

    x_upd = ekf.update(
        np.array([1.5, 1.5], dtype=np.float64),
        np.eye(2, dtype=np.float64),
        0.1 * np.eye(2, dtype=np.float64),
    )

    np.testing.assert_allclose(x_upd, ekf.x)
    assert "nis" in ekf.data
    assert "nees" in ekf.data
    assert ekf.last_nis >= 0.0


def test_ekf_custom_callbacks() -> None:
    ekf = sp_vision_bindings.ExtendedKalmanFilter(
        np.array([0.0, 0.0], dtype=np.float64),
        np.eye(2, dtype=np.float64),
    )

    ekf.predict_custom(
        np.eye(2, dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        lambda x: x + np.array([1.0, -1.0], dtype=np.float64),
    )

    np.testing.assert_allclose(ekf.x, [1.0, -1.0])

    ekf.update_custom(
        np.array([1.0, -1.0], dtype=np.float64),
        np.eye(2, dtype=np.float64),
        0.1 * np.eye(2, dtype=np.float64),
        lambda x: x,
    )

    np.testing.assert_allclose(ekf.x, [1.0, -1.0], atol=1e-8)

