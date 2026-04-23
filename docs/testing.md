# 测试与联调

本文档说明当前测试链路、常用命令和 `tests/` 的职责。

## 1. 先构建，再测试

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Python 测试：

```bash
uv --project python run pytest tests/python
```

如果你希望显式使用独立缓存目录：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv --project python run pytest tests/python
```

## 2. `tests/` 的角色

`tests/` 现在是统一测试入口，但内部仍然分两类职责：

- `tests/cpp/`
  - 构建真实 C++ 测试程序
  - 覆盖设备联调、离线回放和 smoke test
- `tests/python/`
  - 覆盖 Python CLI
  - 覆盖 pybind11 暴露出来的真实 C++ 类型

它不是一个“只有断言的单元测试目录”。这里既有自动化测试，也有联调程序。

## 3. C++ 测试输出位置

大多数 C++ 测试输出到：

```text
build/bin/tests/<group>/<executable>
```

当前主要分组：

- `auto_aim`
- `auto_buff`
- `camera`
- `gimbal`
- `planner`
- `system`
- `ros2`，仅在 ROS2 依赖满足时编译

## 4. 常用 C++ 测试命令

### 4.1 相机

```bash
build/bin/tests/camera/camera_test --config-path=configs/standard3.yaml
build/bin/tests/camera/camera_window_test configs/standard3.yaml
build/bin/tests/camera/camera_detect_test configs/standard3.yaml
```

### 4.2 云台

```bash
build/bin/tests/gimbal/gimbal_test configs/standard3.yaml
build/bin/tests/gimbal/gimbal_response_test configs/standard3.yaml
```

### 4.3 自瞄

```bash
build/bin/tests/auto_aim/auto_aim_test
build/bin/tests/auto_aim/detector_video_test
build/bin/tests/auto_buff/auto_power_rune_test
```

### 4.4 系统链路

```bash
build/bin/tests/system/cboard_test
build/bin/tests/system/dm_test
```

## 5. Python diagnose 命令

日常联调建议优先使用 `sp-vision-diagnose`，它负责统一命令入口和会话组织：

```bash
uv --project python run sp-vision-diagnose status
uv --project python run sp-vision-diagnose camera list
uv --project python run sp-vision-diagnose gimbal list
uv --project python run sp-vision-diagnose auto-aim list
```

更细的使用方法见：

- [相机 Diagnose](./diagnose/camera.md)
- [云台 Diagnose](./diagnose/gimbal.md)
- [自瞄 Diagnose](./diagnose/auto_aim.md)

## 6. Python 标定命令

标定入口统一为：

```bash
uv --project python run sp-vision-calibration help
```

常用命令：

```bash
uv --project python run sp-vision-calibration capture configs/calibration.yaml assets/img_with_q --imu
uv --project python run sp-vision-calibration calibrate-camera assets/img_with_q -c configs/calibration.yaml
uv --project python run sp-vision-calibration calibrate-handeye assets/img_with_q -c configs/calibration.yaml
uv --project python run sp-vision-calibration calibrate-robotworld-handeye assets/img_with_q -c configs/calibration.yaml
```

## 7. 推荐联调顺序

1. `sp-vision-diagnose status`
2. `sp-vision-diagnose camera info`
3. `sp-vision-diagnose gimbal port-info`
4. `sp-vision-diagnose camera quick configs/standard3.yaml`
5. `sp-vision-diagnose gimbal quick configs/standard3.yaml`
6. `sp-vision-diagnose auto-aim armor-box configs/standard3.yaml`

## 8. 常见问题

- 相机被占用：先执行 `sudo uv --project python run sp-vision-diagnose camera release --force`
- pybind11 模块缺失：确认已经执行过 C++ 构建，再运行 `uv --project python run sp-vision-diagnose bindings`
- 没有图形界面：优先使用无窗口动作，例如 `camera quick`、`auto-aim armor-rec`
