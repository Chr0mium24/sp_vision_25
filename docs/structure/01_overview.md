# 1. 项目总览

## 1.1 一句话理解

`sp_vision_25` 是一个 RoboMaster 视觉框架，核心目标是把“相机取流、姿态获取、目标检测、位姿解算、状态估计、瞄准/规划、发射控制、诊断调参、离线复现”放进同一套 C++ 工程里。

## 1.2 架构分层

- `tools/`：工具层，提供数学、日志、轨迹、EKF、绘图、录制、并发容器。
- `io/`：设备抽象层，负责相机、串口云台、C 板 CAN、USB 相机、ROS2 通信。
- `tasks/`：算法层，分为 `auto_aim`、`auto_buff`、`omniperception`。
- `src/`：主程序入口，不同兵种/模式对上面三层做组合。
- `diagnostics/`：在线诊断与调参程序。
- `tests/`：模块级联调、离线回放、通信验证程序。
- `calibration/`：相机内参、手眼、robot-world-handeye 标定工具。
- `configs/`：运行参数、外参、阈值、模型路径、设备信息。
- `assets/`：模型权重、演示素材、示例图片。
- `docs/`：已有工作流/诊断文档；`docs/structure/` 是本次新增的结构文档。

## 1.3 构建关系

`CMakeLists.txt` 体现了最重要的依赖方向：

- `tools` 编译成对象库。
- `io` 编译成静态库。
- `tasks/auto_aim`、`tasks/auto_buff`、`tasks/omniperception` 编译成对象库。
- `src/`、`tests/`、`diagnostics/`、`calibration/` 下的可执行程序把这些库按需链接起来。
- ROS2 相关目标只有在找到 `ament_cmake/rclcpp/sp_msgs` 时才编译。

## 1.4 运行主链路

项目里有两套典型主链路：

- 标准自瞄链路：`Camera/Gimbal or CBoard -> YOLO/Detector -> Solver -> Tracker -> Aimer/Planner -> Shooter -> Gimbal/CBoard`
- 打符链路：`Camera/Gimbal or CBoard -> Buff_Detector -> Buff Solver -> Buff Target -> Buff Aimer -> Gimbal/CBoard`

## 1.5 常见入口

- 步兵/无人机/C 板链路：`src/standard.cpp`、`src/mt_standard.cpp`、`src/uav.cpp`
- MPC 自瞄：`src/standard_mpc.cpp`、`src/auto_aim_debug_mpc.cpp`
- 打符调试：`src/auto_buff_debug.cpp`、`src/auto_buff_debug_mpc.cpp`
- 哨兵/全向感知：`src/sentry*.cpp`

