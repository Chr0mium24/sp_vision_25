# 60. `tests/` 与 `diagnostics/`

## 60.1 `diagnostics/` 目录职责

`diagnostics/` 面向“在线定位问题、调参数、做串口链路排查”。它们通常不是比赛程序，而是工程保障工具。

## 60.2 `diagnostics/auto_aim/`

| 文件 | 作用 |
| --- | --- |
| `diagnostics/auto_aim/auto_aim_ui_test.cpp` | 在线自瞄 UI/TUI，支持不下发控制、实时看命令与回授差。 |
| `diagnostics/auto_aim/auto_aim_ui_tune.cpp` | 在线自瞄调参器，可改 YAML 参数并导出。 |
| `diagnostics/auto_aim/diagnose.sh` | 自瞄/打符诊断入口脚本，封装多种模式与示例命令。 |

## 60.3 `diagnostics/camera/`

| 文件 | 作用 |
| --- | --- |
| `diagnostics/camera/diagnose.sh` | 相机诊断脚本，封装占用释放、曝光调参、预览、保存、线程测试等流程。 |

## 60.4 `diagnostics/gimbal/`

| 文件 | 作用 |
| --- | --- |
| `diagnostics/gimbal/gimbal_ui_test.cpp` | 云台读写 UI/交互诊断程序。 |
| `diagnostics/gimbal/gimbal_axis_diag_test.cpp` | 主动小角度运动，判断 yaw/pitch 轴映射是否正确。 |
| `diagnostics/gimbal/gimbal_manual_axis_diag_test.cpp` | 人手拨动云台时读取姿态，判断轴向与符号。 |
| `diagnostics/gimbal/gimbal_serial_probe.cpp` | 原始串口字节流探针，检查是否能收到合法帧。 |
| `diagnostics/gimbal/gimbal_link_diag_test.cpp` | 协议级链路诊断，验证收发与 CRC。 |
| `diagnostics/gimbal/diagnose.sh` | 云台诊断脚本，统一封装 quick/probe/axis/watch 等动作。 |

## 60.5 `tests/auto_aim/`

| 文件 | 作用 |
| --- | --- |
| `tests/auto_aim/auto_aim_test.cpp` | 离线自瞄回放测试。 |
| `tests/auto_aim/detector_video_test.cpp` | 离线检测视频测试。 |
| `tests/auto_aim/fire_test.cpp` | 开火逻辑或发射相关测试。 |

## 60.6 `tests/auto_buff/`

| 文件 | 作用 |
| --- | --- |
| `tests/auto_buff/auto_power_rune_test.cpp` | 离线打符测试/可视化测试。 |

## 60.7 `tests/camera/`

| 文件 | 作用 |
| --- | --- |
| `tests/camera/camera_test.cpp` | 相机基础读流测试。 |
| `tests/camera/camera_detect_test.cpp` | 相机+检测联调测试。 |
| `tests/camera/camera_save_test.cpp` | 手动保存采样图像。 |
| `tests/camera/camera_thread_test.cpp` | 多线程相机/检测链路测试。 |
| `tests/camera/camera_window_test.cpp` | 相机图像窗口预览。 |
| `tests/camera/usbcamera_test.cpp` | USB 相机基础测试。 |
| `tests/camera/usbcamera_detect_test.cpp` | USB 相机+检测联调。 |
| `tests/camera/multi_usbcamera_test.cpp` | 多 USB 相机并行读流测试。 |
| `tests/camera/handeye_test.cpp` | 手眼结果投影验证。 |

## 60.8 `tests/gimbal/`

| 文件 | 作用 |
| --- | --- |
| `tests/gimbal/gimbal_test.cpp` | 云台基础通信/姿态测试。 |
| `tests/gimbal/gimbal_response_test.cpp` | 云台响应与跟踪效果测试。 |

## 60.9 `tests/planner/`

| 文件 | 作用 |
| --- | --- |
| `tests/planner/planner_test.cpp` | 在线/联动规划器测试。 |
| `tests/planner/planner_test_offline.cpp` | 离线轨迹规划测试。 |

## 60.10 `tests/system/`

| 文件 | 作用 |
| --- | --- |
| `tests/system/cboard_test.cpp` | C 板通信测试。 |
| `tests/system/dm_test.cpp` | 达妙 IMU 测试。 |

## 60.11 `tests/ros2/`

| 文件 | 作用 |
| --- | --- |
| `tests/ros2/publish_test.cpp` | ROS2 发布测试。 |
| `tests/ros2/subscribe_test.cpp` | ROS2 订阅测试。 |
| `tests/ros2/topic_loop_test.cpp` | ROS2 话题闭环测试。 |

## 60.12 为什么这些目录重要

- `src/` 负责“跑比赛”。
- `diagnostics/` 负责“找到为什么跑不好”。
- `tests/` 负责“在更小的闭环里复现问题”。

