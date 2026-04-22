from __future__ import annotations

import math

import numpy as np

RAD2DEG = 57.29577951308232
DEG2RAD = 1.0 / RAD2DEG


def limit_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle


def rotation_matrix(ypr: tuple[float, float, float]) -> np.ndarray:
    yaw, pitch, roll = ypr
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    return np.array(
        [
            [cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
            [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
            [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll],
        ],
        dtype=float,
    )


def _quaternion_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def eulers_from_quaternion(
    q: tuple[float, float, float, float], axis0: int, axis1: int, axis2: int, extrinsic: bool = False
) -> np.ndarray:
    return eulers_from_matrix(_quaternion_to_matrix(q), axis0, axis1, axis2, extrinsic)


def eulers_from_matrix(
    R: np.ndarray, axis0: int, axis1: int, axis2: int, extrinsic: bool = False
) -> np.ndarray:
    if not extrinsic:
        axis0, axis2 = axis2, axis0

    i, j, k = axis0, axis1, axis2
    is_proper = i == k
    if is_proper:
        k = 3 - i - j
    sign = (i - j) * (j - k) * (k - i) // 2

    coeffs = np.array(R, dtype=float)
    # Convert to quaternion-like parameters, matching tools::eulers()
    # q = [w, x, y, z]
    # Eigen::Quaterniond coeffs() order is [x, y, z, w]
    # Build the same a,b,c,d terms as the C++ implementation.
    # Reconstruct quaternion from matrix using standard formulas.
    trace = float(coeffs[0, 0] + coeffs[1, 1] + coeffs[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (coeffs[2, 1] - coeffs[1, 2]) / s
        y = (coeffs[0, 2] - coeffs[2, 0]) / s
        z = (coeffs[1, 0] - coeffs[0, 1]) / s
    elif coeffs[0, 0] > coeffs[1, 1] and coeffs[0, 0] > coeffs[2, 2]:
        s = math.sqrt(1.0 + coeffs[0, 0] - coeffs[1, 1] - coeffs[2, 2]) * 2.0
        w = (coeffs[2, 1] - coeffs[1, 2]) / s
        x = 0.25 * s
        y = (coeffs[0, 1] + coeffs[1, 0]) / s
        z = (coeffs[0, 2] + coeffs[2, 0]) / s
    elif coeffs[1, 1] > coeffs[2, 2]:
        s = math.sqrt(1.0 + coeffs[1, 1] - coeffs[0, 0] - coeffs[2, 2]) * 2.0
        w = (coeffs[0, 2] - coeffs[2, 0]) / s
        x = (coeffs[0, 1] + coeffs[1, 0]) / s
        y = 0.25 * s
        z = (coeffs[1, 2] + coeffs[2, 1]) / s
    else:
        s = math.sqrt(1.0 + coeffs[2, 2] - coeffs[0, 0] - coeffs[1, 1]) * 2.0
        w = (coeffs[1, 0] - coeffs[0, 1]) / s
        x = (coeffs[0, 2] + coeffs[2, 0]) / s
        y = (coeffs[1, 2] + coeffs[2, 1]) / s
        z = 0.25 * s
    xyzw = np.array([x, y, z, w], dtype=float)

    if is_proper:
        a = xyzw[3]
        b = xyzw[i]
        c = xyzw[j]
        d = xyzw[k] * sign
    else:
        a = xyzw[3] - xyzw[j]
        b = xyzw[i] + xyzw[k] * sign
        c = xyzw[j] + xyzw[3]
        d = xyzw[k] * sign - xyzw[i]

    result = np.zeros(3, dtype=float)
    n2 = a * a + b * b + c * c + d * d
    result[1] = math.acos(max(-1.0, min(1.0, 2 * (a * a + b * b) / n2 - 1)))

    half_sum = math.atan2(b, a)
    half_diff = math.atan2(-d, c)

    eps = 1e-7
    safe1 = abs(result[1]) >= eps
    safe2 = abs(result[1] - math.pi) >= eps
    safe = safe1 and safe2
    if safe:
        result[0] = half_sum + half_diff
        result[2] = half_sum - half_diff
    else:
        if not extrinsic:
            result[0] = 0.0
            if not safe1:
                result[2] = 2 * half_sum
            if not safe2:
                result[2] = -2 * half_diff
        else:
            result[2] = 0.0
            if not safe1:
                result[0] = 2 * half_sum
            if not safe2:
                result[0] = 2 * half_diff

    result = np.array([limit_rad(v) for v in result], dtype=float)

    if not is_proper:
        result[2] *= sign
        result[1] -= math.pi / 2.0

    if not extrinsic:
        result[0], result[2] = result[2], result[0]

    return result
