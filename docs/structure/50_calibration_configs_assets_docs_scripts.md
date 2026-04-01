# 50. 标定、配置、素材、脚本与顶层文件

## 50.1 顶层文件

| 文件 | 作用 |
| --- | --- |
| `CMakeLists.txt` | 整个工程的总构建入口。 |
| `.clang-format` | 代码格式化规则。 |
| `.gitignore` | Git 忽略规则。 |
| `.gitmodules` | Git 子模块配置。 |
| `LICENSE` | 开源协议。 |
| `readme.md` | 项目说明、运行环境、设计理念、整体工作流。 |
| `build.sh` | 一键 `cmake + build` 脚本，兼容 ROS2 `RMW_IMPLEMENTATION` 配置。 |
| `autostart.sh` | 通过 `screen` 启动看门狗/自启动流程。 |
| `buff_layout.xml` | 打符/PlotJuggler 等布局配置。 |
| `mpc_layout.xml` | MPC 调试布局配置。 |

## 50.2 `calibration/`

| 文件 | 作用 |
| --- | --- |
| `calibration/capture.cpp` | 采集标定图片，支持同时保存图像和姿态四元数。 |
| `calibration/calibrate_camera.cpp` | 相机内参标定。 |
| `calibration/calibrate_handeye.cpp` | 经典手眼标定。 |
| `calibration/calibrate_robotworld_handeye.cpp` | robot-world-handeye 联合标定。 |
| `calibration/split_video.cpp` | 从录制的 `avi+txt` 中截取指定帧段。 |

## 50.3 `configs/`

| 文件 | 作用 |
| --- | --- |
| `configs/standard3.yaml` | 常用步兵配置。 |
| `configs/standard4.yaml` | 另一套步兵配置。 |
| `configs/sentry.yaml` | 哨兵配置。 |
| `configs/uav.yaml` | 无人机配置。 |
| `configs/ascento.yaml` | 特定平台/实验配置。 |
| `configs/demo.yaml` | 离线演示/示例配置。 |
| `configs/example.yaml` | 参数示例配置。 |
| `configs/handeye.yaml` | `handeye_test` 模板配置，补充网格和延时参数。 |
| `configs/mvs.yaml` | 相机或 SDK 相关配置。 |
| `configs/camera.yaml` | 后相机/辅助相机配置。 |
| `configs/calibration.yaml` | 标定流程参数。 |
| `configs/backup.standard3.yaml` | `standard3` 的备份版本。 |

## 50.4 `assets/`

| 文件 | 作用 |
| --- | --- |
| `assets/yolov5.bin` | YOLOv5 OpenVINO 权重。 |
| `assets/yolov5.xml` | YOLOv5 OpenVINO 网络结构。 |
| `assets/yolov8.bin` | YOLOv8 OpenVINO 权重。 |
| `assets/yolov8.xml` | YOLOv8 OpenVINO 网络结构。 |
| `assets/yolo11.bin` | YOLO11 OpenVINO 权重。 |
| `assets/yolo11.xml` | YOLO11 OpenVINO 网络结构。 |
| `assets/yolo11_buff_int8.bin` | 打符 YOLO11 INT8 权重。 |
| `assets/yolo11_buff_int8.xml` | 打符 YOLO11 INT8 网络结构。 |
| `assets/tiny_resnet.onnx` | 装甲图案分类 ONNX 模型。 |
| `assets/best2-sim.onnx` | 其他实验/仿真模型。 |
| `assets/standard_fanblade.jpg` | 打符相关静态图片素材。 |
| `assets/demo/demo.avi` | 离线演示视频。 |
| `assets/demo/demo.txt` | 与 `demo.avi` 配套的姿态时间戳文本。 |

## 50.5 `docs/`

| 文件 | 作用 |
| --- | --- |
| `docs/calibration_workflow.md` | 标定工作流说明。 |
| `docs/test_chain_and_usage.md` | 测试链路与使用说明。 |
| `docs/gimbal_ros2_transport.md` | 云台 ROS2 传输设计与现状说明。 |
| `docs/diagnose/auto_aim_diagnose.md` | 自瞄诊断说明。 |
| `docs/diagnose/camera_diagnose.md` | 相机诊断说明。 |
| `docs/diagnose/gimbal_diagnose.md` | 云台诊断说明。 |

## 50.6 `scripts/`

| 文件 | 作用 |
| --- | --- |
| `scripts/prechange_backup.sh` | 改动前做快照备份、状态导出和烟雾检查。 |
| `scripts/run_gimbal_ros2_bridge.sh` | 启动 `gimbal_ros2_bridge` 的辅助脚本。 |
