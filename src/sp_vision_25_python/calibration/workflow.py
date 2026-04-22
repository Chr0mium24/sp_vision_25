from __future__ import annotations

import math
import signal
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from ..bindings import load_bindings
from ..diagnose.config import load_yaml
from .math_tools import RAD2DEG, eulers_from_matrix, eulers_from_quaternion

_yaml = YAML()
_yaml.default_flow_style = False


def _flow_list(values: list[float] | np.ndarray) -> CommentedSeq:
    seq = CommentedSeq([float(value) for value in np.asarray(values, dtype=float).reshape(-1)])
    seq.fa.set_flow_style()
    return seq


def _dump_map(data: dict[str, Any], comments: list[str] | None = None) -> None:
    if comments:
        for comment in comments:
            print(f"# {comment}")
    cm = CommentedMap()
    for key, value in data.items():
        cm[key] = _flow_list(value) if isinstance(value, (list, tuple, np.ndarray)) else value
    _yaml.dump(cm, sys.stdout)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pattern_mode(pattern_type: str) -> str:
    return "chessboard" if pattern_type.lower() in {"chessboard", "checkerboard"} else "circles"


def _detect_pattern(img: np.ndarray, pattern_size: tuple[int, int], pattern_type: str) -> tuple[bool, list[list[float]]]:
    mode = _pattern_mode(pattern_type)
    if mode == "chessboard":
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        success, corners = cv2.findChessboardCorners(img, pattern_size, flags)
        if success:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1),
            )
        return success, [] if not success else corners.reshape(-1, 2).tolist()

    success, corners = cv2.findCirclesGrid(img, pattern_size, cv2.CALIB_CB_SYMMETRIC_GRID)
    return success, [] if not success else corners.reshape(-1, 2).tolist()


def _centers_3d_planar(pattern_size: tuple[int, int], center_distance: float) -> np.ndarray:
    cols, rows = pattern_size
    points = [[j * center_distance, i * center_distance, 0.0] for i in range(rows) for j in range(cols)]
    return np.asarray(points, dtype=np.float32)


def _centers_3d_board(pattern_size: tuple[int, int], center_distance: float) -> np.ndarray:
    cols, rows = pattern_size
    points = [
        [0.0, (-j + 0.5 * cols) * center_distance, (-i + 0.5 * rows) * center_distance]
        for i in range(rows)
        for j in range(cols)
    ]
    return np.asarray(points, dtype=np.float32)


def _show_image(title: str, image: np.ndarray, scale: float = 0.5) -> bool:
    try:
        display = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
        cv2.imshow(title, display)
        return True
    except cv2.error as exc:
        print(f"[warn] GUI unavailable for '{title}': {exc}")
        return False


def _wait_key(delay: int = 0) -> int:
    try:
        return int(cv2.waitKey(delay))
    except cv2.error:
        return -1


def _camera_pattern_info(config_path: Path) -> tuple[tuple[int, int], float, str]:
    yaml = load_yaml(config_path)
    return (
        (int(yaml["pattern_cols"]), int(yaml["pattern_rows"])),
        float(yaml["center_distance_mm"]),
        str(yaml.get("pattern_type", "circle_grid")),
    )


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return np.empty((0, 0, 3), dtype=np.uint8) if img is None else img


def _load_quaternion(path: Path) -> tuple[float, float, float, float]:
    values = path.read_text(encoding="utf-8").split()
    if len(values) < 4:
        raise ValueError(f"Invalid quaternion file: {path}")
    return tuple(map(float, values[:4]))  # type: ignore[return-value]


def _rotation_from_quaternion(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def capture(config_path: Path, output_folder: Path, imu: bool, show: bool = True) -> int:
    bindings = load_bindings()
    yaml = load_yaml(config_path)
    pattern_size, _, pattern_type = _camera_pattern_info(config_path)

    img_folder = output_folder if imu else output_folder.parent / "img"
    _ensure_dir(img_folder)

    if imu:
        com_port = str(yaml.get("com_port", ""))
        if not com_port or not Path(com_port).exists():
            print(f"[imu] com_port missing or not found: {com_port}")
            return 1
        gimbal = bindings.CBoard(str(config_path))
    else:
        gimbal = None

    camera = bindings.Camera(str(config_path))
    print("默认标定板尺寸为10列7行")
    count = 0
    window_ok = True
    while True:
        img, timestamp_ns = camera.read()
        if img is None or getattr(img, "size", 0) == 0:
            print("[capture] empty frame, stopping")
            break

        q = (1.0, 0.0, 0.0, 0.0)
        if gimbal is not None:
            q = gimbal.imu_at(int(timestamp_ns))

        frame = np.asarray(img).copy()
        if gimbal is not None:
            ypr = eulers_from_quaternion(q, 2, 1, 0) * RAD2DEG
            cv2.putText(frame, f"Z {ypr[0]:.2f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, f"Y {ypr[1]:.2f}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, f"X {ypr[2]:.2f}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        success, centers_2d = _detect_pattern(frame, pattern_size, pattern_type)
        if success:
            cv2.drawChessboardCorners(
                frame,
                pattern_size,
                np.asarray(centers_2d, dtype=np.float32).reshape(-1, 1, 2),
                success,
            )
        if show and window_ok:
            window_ok = _show_image("Press s to save, q to quit", frame, 0.5)
        key = _wait_key(1) if (show and window_ok) else -1
        if key == ord("q"):
            break
        if key != ord("s"):
            continue

        count += 1
        cv2.imwrite(str(img_folder / f"{count}.jpg"), np.asarray(img))
        if gimbal is not None:
            (img_folder / f"{count}.txt").write_text(f"{q[0]} {q[1]} {q[2]} {q[3]}", encoding="utf-8")
        print(f"[{count}] Saved in {img_folder}")

    if show:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print("注意四元数输出顺序为wxyz")
    return 0


def calibrate_camera(config_path: Path, input_folder: Path, show: bool = True) -> int:
    pattern_size, center_distance_mm, pattern_type = _camera_pattern_info(config_path)
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None

    for index in range(1, 10_000):
        img_path = input_folder / f"{index}.jpg"
        img = _load_image(img_path)
        if img.size == 0:
            break
        img_size = (int(img.shape[1]), int(img.shape[0]))
        success, centers_2d = _detect_pattern(img, pattern_size, pattern_type)
        drawing = img.copy()
        if success:
            cv2.drawChessboardCorners(
                drawing,
                pattern_size,
                np.asarray(centers_2d, dtype=np.float32).reshape(-1, 1, 2),
                success,
            )
        if show:
            _show_image("Press any to continue", drawing, 0.5)
            _wait_key(0)
        print(f"[{'success' if success else 'failure'}] {img_path}")
        if not success:
            continue
        obj_points.append(_centers_3d_planar(pattern_size, center_distance_mm))
        img_points.append(np.asarray(centers_2d, dtype=np.float32))

    if not obj_points or img_size is None:
        print("[error] No valid calibration images found.")
        return 1

    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    camera_matrix[0, 0] = 1.0
    camera_matrix[1, 1] = 1.0
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-16)
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        img_size,
        camera_matrix,
        dist_coeffs,
        flags=cv2.CALIB_FIX_K3,
        criteria=criteria,
    )

    error_sum = 0.0
    total_points = 0
    for obj, img_pt, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs, strict=False):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        total_points += len(projected)
        error_sum += float(np.linalg.norm(img_pt - projected, axis=1).sum())
    error = error_sum / total_points if total_points else 0.0

    print(f"# 重投影误差: {error:.4f}px")
    _dump_map(
        {
            "camera_matrix": _to_list(camera_matrix),
            "distort_coeffs": _to_list(dist_coeffs),
        }
    )
    return 0


def _to_list(array: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(array, dtype=float).reshape(-1)]


def calibrate_handeye(config_path: Path, input_folder: Path, show: bool = True) -> int:
    yaml = load_yaml(config_path)
    pattern_size, center_distance_mm, pattern_type = _camera_pattern_info(config_path)
    r_gimbal2imubody = np.asarray(yaml["R_gimbal2imubody"], dtype=float).reshape(3, 3)
    camera_matrix = np.asarray(yaml["camera_matrix"], dtype=float).reshape(3, 3)
    dist_coeffs = np.asarray(yaml["distort_coeffs"], dtype=float).reshape(-1, 1)

    R_gimbal2world_list: list[np.ndarray] = []
    t_gimbal2world_list: list[np.ndarray] = []
    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []

    for index in range(1, 10_000):
        img_path = input_folder / f"{index}.jpg"
        q_path = input_folder / f"{index}.txt"
        img = _load_image(img_path)
        if img.size == 0 or not q_path.exists():
            break
        q = _load_quaternion(q_path)
        R_imubody2imuabs = _rotation_from_quaternion(q)
        R_gimbal2world = r_gimbal2imubody.T @ R_imubody2imuabs @ r_gimbal2imubody
        ypr = eulers_from_matrix(R_gimbal2world, 2, 1, 0) * RAD2DEG

        drawing = img.copy()
        cv2.putText(drawing, f"yaw   {ypr[0]:.2f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(drawing, f"pitch {ypr[1]:.2f}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(drawing, f"roll  {ypr[2]:.2f}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        success, centers_2d = _detect_pattern(img, pattern_size, pattern_type)
        if success:
            cv2.drawChessboardCorners(
                drawing,
                pattern_size,
                np.asarray(centers_2d, dtype=np.float32).reshape(-1, 1, 2),
                success,
            )
        if show:
            _show_image("Press any to continue", drawing, 0.5)
            _wait_key(0)
        print(f"[{'success' if success else 'failure'}] {img_path}")
        if not success:
            continue

        center_points = _centers_3d_planar(pattern_size, center_distance_mm)
        solved, rvec, tvec = cv2.solvePnP(
            center_points,
            np.asarray(centers_2d, dtype=np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not solved:
            continue
        R_gimbal2world_list.append(R_gimbal2world)
        t_gimbal2world_list.append(np.zeros((3, 1), dtype=float))
        rvecs.append(rvec)
        tvecs.append(tvec)

    if not rvecs:
        print("[error] No valid calibration images found.")
        return 1

    R_camera2gimbal, t_camera2gimbal = cv2.calibrateHandEye(
        R_gimbal2world_list,
        t_gimbal2world_list,
        rvecs,
        tvecs,
    )
    t_camera2gimbal = np.asarray(t_camera2gimbal, dtype=float) / 1e3

    R_gimbal2ideal = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=float)
    R_camera2ideal = R_gimbal2ideal @ np.asarray(R_camera2gimbal, dtype=float)
    camera_ypr = eulers_from_matrix(R_camera2ideal, 1, 0, 2) * RAD2DEG

    print(f"# 相机同理想情况的偏角: yaw{camera_ypr[0]:.2f} pitch{camera_ypr[1]:.2f} roll{camera_ypr[2]:.2f} degree")
    _dump_map(
        {
            "R_gimbal2imubody": _to_list(r_gimbal2imubody),
            "R_camera2gimbal": _to_list(R_camera2gimbal),
            "t_camera2gimbal": _to_list(t_camera2gimbal),
        }
    )
    return 0


def calibrate_robotworld_handeye(config_path: Path, input_folder: Path, show: bool = True) -> int:
    yaml = load_yaml(config_path)
    pattern_size, center_distance_mm, pattern_type = _camera_pattern_info(config_path)
    r_gimbal2imubody = np.asarray(yaml["R_gimbal2imubody"], dtype=float).reshape(3, 3)
    camera_matrix = np.asarray(yaml["camera_matrix"], dtype=float).reshape(3, 3)
    dist_coeffs = np.asarray(yaml["distort_coeffs"], dtype=float).reshape(-1, 1)

    R_world2gimbal_list: list[np.ndarray] = []
    t_world2gimbal_list: list[np.ndarray] = []
    R_world2cam_list: list[np.ndarray] = []
    t_world2cam_list: list[np.ndarray] = []

    for index in range(1, 10_000):
        img_path = input_folder / f"{index}.jpg"
        q_path = input_folder / f"{index}.txt"
        img = _load_image(img_path)
        if img.size == 0 or not q_path.exists():
            break
        q = _load_quaternion(q_path)
        R_imubody2imuabs = _rotation_from_quaternion(q)
        R_gimbal2world = r_gimbal2imubody.T @ R_imubody2imuabs @ r_gimbal2imubody
        ypr = eulers_from_matrix(R_gimbal2world, 2, 1, 0) * RAD2DEG

        drawing = img.copy()
        cv2.putText(drawing, f"yaw   {ypr[0]:.2f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(drawing, f"pitch {ypr[1]:.2f}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(drawing, f"roll  {ypr[2]:.2f}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        success, centers_2d = _detect_pattern(img, pattern_size, pattern_type)
        if success:
            cv2.drawChessboardCorners(
                drawing,
                pattern_size,
                np.asarray(centers_2d, dtype=np.float32).reshape(-1, 1, 2),
                success,
            )
        if show:
            _show_image("Press any to continue", drawing, 0.5)
            _wait_key(0)
        print(f"[{'success' if success else 'failure'}] {img_path}")
        if not success:
            continue

        center_points = _centers_3d_board(pattern_size, center_distance_mm)
        solved, rvec, tvec = cv2.solvePnP(
            center_points,
            np.asarray(centers_2d, dtype=np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not solved:
            continue
        R_world2cam, _ = cv2.Rodrigues(rvec)
        R_world2gimbal_list.append(R_gimbal2world.T)
        t_world2gimbal_list.append(np.zeros((3, 1), dtype=float))
        R_world2cam_list.append(R_world2cam)
        t_world2cam_list.append(tvec)

    if not R_world2cam_list:
        print("[error] No valid calibration images found.")
        return 1

    R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
        R_world2cam_list,
        t_world2cam_list,
        R_world2gimbal_list,
        t_world2gimbal_list,
    )
    t_gripper2cam = np.asarray(t_gripper2cam, dtype=float) / 1e3
    t_base2world = np.asarray(t_base2world, dtype=float) / 1e3

    R_camera2gimbal = np.asarray(R_gripper2cam, dtype=float).T
    R_board2world = np.asarray(R_base2world, dtype=float).T
    t_camera2gimbal = -R_camera2gimbal @ t_gripper2cam
    t_board2world = -R_board2world @ t_base2world

    R_camera2ideal = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=float) @ R_camera2gimbal
    camera_ypr = eulers_from_matrix(R_camera2ideal, 1, 0, 2) * RAD2DEG
    distance = math.hypot(float(t_board2world[0]), float(t_board2world[1]))
    board_ypr = eulers_from_matrix(R_board2world, 2, 1, 0) * RAD2DEG

    print(f"# 相机同理想情况的偏角: yaw{camera_ypr[0]:.2f} pitch{camera_ypr[1]:.2f} roll{camera_ypr[2]:.2f} degree")
    print(f"# 标定板到世界坐标系原点的水平距离: {distance:.2f} m")
    print(f"# 标定板同竖直摆放时的偏角: yaw{board_ypr[0]:.2f} pitch{board_ypr[1]:.2f} roll{board_ypr[2]:.2f} degree")
    _dump_map(
        {
            "R_gimbal2imubody": _to_list(r_gimbal2imubody),
            "R_camera2gimbal": _to_list(R_camera2gimbal),
            "t_camera2gimbal": _to_list(t_camera2gimbal),
        }
    )
    return 0


def split_video(input_path: Path, output_path: Path, start_index: int = 0, end_index: int = 0, show: bool = True) -> int:
    stop_requested = False

    def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)
    video_path = Path(f"{input_path}.avi")
    text_path = Path(f"{input_path}.txt")
    video = cv2.VideoCapture(str(video_path))
    text_lines = text_path.read_text(encoding="utf-8").splitlines()

    video.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))

    outvideo = cv2.VideoWriter(f"{output_path}.avi", fourcc, fps, (frame_width, frame_height))
    outtext = Path(f"{output_path}.txt").open("w", encoding="utf-8")

    try:
        for frame_count in range(start_index, len(text_lines)):
            if stop_requested:
                break
            if end_index > 0 and frame_count > end_index:
                break
            ok, img = video.read()
            if not ok or img is None or img.size == 0:
                break
            outvideo.write(img)
            outtext.write(text_lines[frame_count] + "\n")
            if show:
                cv2.imshow("result", cv2.resize(img, None, fx=0.8, fy=0.8))
                if cv2.waitKey(1) == ord("q"):
                    break
    finally:
        signal.signal(signal.SIGINT, old_handler)
        outtext.close()
        outvideo.release()
        video.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    return 0
