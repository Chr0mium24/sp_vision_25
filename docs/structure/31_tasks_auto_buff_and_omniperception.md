# 31. `tasks/auto_buff/` 与 `tasks/omniperception/`

## 31.1 `tasks/auto_buff/` 目录职责

`auto_buff` 负责打符。它和 `auto_aim` 很像，也走“检测 -> 解算 -> 状态估计 -> 决策”四步，但对象从装甲板换成了符环/扇叶。

## 31.2 `tasks/auto_buff/` 文件职责

| 文件 | 作用 |
| --- | --- |
| `tasks/auto_buff/CMakeLists.txt` | 定义 `auto_buff` 对象库。 |
| `tasks/auto_buff/buff_type.hpp` | 扇叶、符、跟踪状态等基础类型定义。 |
| `tasks/auto_buff/buff_type.cpp` | `FanBlade/PowerRune` 的构造与几何整理。 |
| `tasks/auto_buff/buff_detector.hpp` | 打符检测器声明。 |
| `tasks/auto_buff/buff_detector.cpp` | 使用 `YOLO11_BUFF` 和图像后处理找扇叶与圆心。 |
| `tasks/auto_buff/buff_solver.hpp` | 打符位姿解算器声明。 |
| `tasks/auto_buff/buff_solver.cpp` | 对目标扇叶做 PnP、坐标变换和重投影。 |
| `tasks/auto_buff/buff_target.hpp` | 小符/大符目标估计器声明。 |
| `tasks/auto_buff/buff_target.cpp` | 小符/大符 EKF、方向判断、正弦速度拟合。 |
| `tasks/auto_buff/buff_aimer.hpp` | 打符瞄准器声明。 |
| `tasks/auto_buff/buff_aimer.cpp` | 时间补偿、弹道迭代、MPC 风格导数估计。 |
| `tasks/auto_buff/yolo11_buff.hpp` | 打符专用 YOLO11 推理器声明。 |
| `tasks/auto_buff/yolo11_buff.cpp` | 打符 YOLO11 OpenVINO 推理与框/关键点解析。 |
| `tasks/auto_buff/buff_predict.hpp` | 旧版/实验性预测器集合，未在当前 `CMakeLists` 里编译。 |

## 31.3 `tasks/auto_buff/` 核心函数

- `Buff_Detector::detect`：当前主用的在线打符检测入口。
- `Buff_Detector::get_r_center`：根据扇叶几何反推圆心。
- `Solver::solve`：将 `PowerRune` 从图像系解到世界系。
- `SmallTarget::get_target/update/predict`：小符 EKF 主链路。
- `BigTarget::get_target/update/predict`：大符状态估计与角速度拟合主链路。
- `Aimer::aim`：输出传统 `io::Command`。
- `Aimer::mpc_aim`：输出带速度/加速度的 `auto_aim::Plan`，用于串口云台或 MPC 链路。
- `YOLO11_BUFF::get_onecandidatebox/get_multicandidateboxes`：打符推理主入口。

## 31.4 `tasks/omniperception/` 目录职责

`omniperception` 负责“多相机辅助感知/切目标”。它不是替代主相机自瞄，而是在主相机丢目标或需要高优先级切换时提供额外视角。

## 31.5 `tasks/omniperception/` 文件职责

| 文件 | 作用 |
| --- | --- |
| `tasks/omniperception/CMakeLists.txt` | 定义 `omniperception` 对象库。 |
| `tasks/omniperception/detection.hpp` | 多相机识别结果结构 `DetectionResult`。 |
| `tasks/omniperception/decider.hpp` | 全向感知决策器声明。 |
| `tasks/omniperception/decider.cpp` | 多相机目标过滤、优先级排序、角度转换、辅助命令生成。 |
| `tasks/omniperception/perceptron.hpp` | 多相机并行感知器声明。 |
| `tasks/omniperception/perceptron.cpp` | 4 路 USB 相机并行推理，并输出 `DetectionResult` 队列。 |

## 31.6 `tasks/omniperception/` 核心函数

- `Perceptron::parallel_infer`：每个 USB 相机一条线程，不断抓图并做 YOLO 推理。
- `Perceptron::get_detection_queue`：把当前缓存的多路检测结果取出给主线程。
- `Decider::armor_filter`：按敌我颜色、无敌状态、禁打目标过滤。
- `Decider::set_priority`：根据战术模式赋优先级。
- `Decider::sort`：对多路 `DetectionResult` 统一排序。
- `Decider::decide(...)`：在主相机丢失目标时，给出辅助转向命令。
- `Decider::get_target_info`：提取当前目标信息发给 ROS2/导航。

## 31.7 关键数据结构

- `PowerRune`：打符主观测结构，包含圆心、扇叶列表、世界坐标结果。
- `FanBlade`：单个扇叶或关键点集合。
- `auto_buff::Target`：打符目标估计基类。
- `DetectionResult`：多相机检测结果快照，包含 `armors/timestamp/delta_yaw/delta_pitch`。

