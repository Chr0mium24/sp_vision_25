# 70. 核心函数总表

这份文档只列“真正把主链路串起来”的函数，不追求列出所有辅助函数。

## 70.1 主程序装配层

| 位置 | 核心函数/逻辑 | 作用 |
| --- | --- | --- |
| `src/standard.cpp` | `main` | 最短自瞄链路，适合看整体调用顺序。 |
| `src/mt_standard.cpp` | `main` + `detect_thread` | 典型 C 板版并行链路。 |
| `src/standard_mpc.cpp` | `main` + `plan_thread` | 把目标队列和 MPC 规划线程解耦。 |
| `src/sentry*.cpp` | `main` | 把 ROS2、主相机、全向感知拼在一起。 |

## 70.2 IO 层

### 相机

- `io::Camera::Camera`：按 YAML 选择具体相机 SDK。
- `io::Camera::read`：统一读帧接口。
- `io::USBCamera::open/read/try_open`：USB 相机初始化、读帧、重连。
- `io::MindVision::open/read/try_open`：迈德威视工业相机读流。
- `io::HikRobot::capture_start/read/reset_usb`：海康工业相机读流和恢复。

### 姿态/控制

- `io::CBoard::imu_at`：对 CAN 回传四元数做时间插值。
- `io::CBoard::callback`：解析 IMU/bullet speed/mode CAN 帧。
- `io::CBoard::send`：把 `io::Command` 编码成 CAN 包。
- `io::Gimbal::q`：按时间戳球面插值云台姿态。
- `io::Gimbal::read_thread`：串口协议解析、统计与状态更新核心。
- `io::Gimbal::send`：下发 yaw/pitch/fire/fric_on。
- `io::Gimbal::reconnect`：串口异常恢复。

### ROS2

- `io::ROS2::publish`：把当前目标信息发布给导航。
- `Subscribe2Nav::subscribe_enemy_status`：获取敌方无敌状态。
- `Subscribe2Nav::subscribe_autoaim_target`：获取导航指定的优先打击目标。

## 70.3 自瞄算法层

### 总运行时

- `auto_aim::Runtime::step`
  - 输入：`image/timestamp/q_gimbal2world/bullet_speed`
  - 调用链：`solver.set_R_gimbal2world -> yolo.detect -> tracker.track -> aimer.aim`
  - 输出：`armors/targets/command/tracker_state`

### 检测

- `auto_aim::YOLO::detect`
  - 对外屏蔽 `YOLOV5/8/11` 差异。
- `auto_aim::YOLOV5::detect/postprocess`
  - 旧版框架，偏框检测+后处理。
- `auto_aim::YOLOV8::detect/postprocess`
  - 关键点型检测器，后处理中还会调用分类/类型判断。
- `auto_aim::YOLO11::detect/postprocess`
  - 当前较新的关键点型检测器。
- `auto_aim::Detector::detect`
  - 传统图像处理检测器，适合回退方案和局部测试。
- `auto_aim::Classifier::classify/ovclassify`
  - 给传统检测补编号。

### 位姿解算

- `auto_aim::Solver::set_R_gimbal2world`
  - 用 IMU 四元数刷新世界坐标系旋转。
- `auto_aim::Solver::solve`
  - `Armor.points -> xyz_in_gimbal/world + ypr/ypd`
- `auto_aim::Solver::optimize_yaw`
  - 通过重投影误差搜索更合理的 world yaw。
- `auto_aim::Solver::reproject_armor`
  - 调试重投影的核心接口。

### 跟踪与估计

- `auto_aim::Tracker::track`
  - 自瞄状态机总入口。
- `auto_aim::Tracker::state_machine`
  - 管理 `lost/detecting/tracking/temp_lost/switching`。
- `auto_aim::Tracker::set_target`
  - 按目标类型选择半径、装甲数、初值协方差。
- `auto_aim::Tracker::update_target`
  - 先预测，再融合同名同类型装甲观测。
- `auto_aim::Target::predict`
  - 11 维目标状态前向预测。
- `auto_aim::Target::update`
  - 在多块装甲板里选最匹配的观测进行更新。
- `auto_aim::Target::armor_xyza_list`
  - 从整车状态反推出每块装甲板的世界坐标与朝向。
- `auto_aim::Target::h_jacobian`
  - EKF 观测模型雅可比。

### 决策

- `auto_aim::Aimer::aim`
  - 做处理延迟补偿、飞行时间迭代、输出最终 `io::Command`。
- `auto_aim::Aimer::choose_aim_point`
  - 在多块装甲板中选当前可射击的那块。
- `auto_aim::Shooter::shoot`
  - 判断“现在该不该真开火”。
- `auto_aim::Planner::plan`
  - MPC 决策器，输出 `Plan`。
- `auto_aim::Planner::get_trajectory`
  - 构造参考轨迹给 TinyMPC。

### 多线程

- `auto_aim::multithread::MultiThreadDetector::push/pop`
  - 异步推理输入输出。
- `auto_aim::multithread::CommandGener::generate_command`
  - 后台消费最新目标，生成控制并发到 `CBoard`。

## 70.4 打符算法层

- `auto_buff::Buff_Detector::detect`
  - 打符在线检测主入口。
- `auto_buff::Solver::solve`
  - 目标扇叶与圆心位姿解算。
- `auto_buff::SmallTarget::get_target/update/predict`
  - 小符估计主链路。
- `auto_buff::BigTarget::get_target/update/predict`
  - 大符估计主链路，外加正弦速度拟合。
- `auto_buff::Aimer::aim`
  - 输出传统打符命令。
- `auto_buff::Aimer::mpc_aim`
  - 输出含速度和加速度的打符计划。

## 70.5 全向感知层

- `omniperception::Perceptron::parallel_infer`
  - 每路 USB 相机一条线程，独立做 YOLO。
- `omniperception::Perceptron::get_detection_queue`
  - 把结果批量交给主线程。
- `omniperception::Decider::armor_filter`
  - 颜色、禁打、无敌状态过滤。
- `omniperception::Decider::set_priority`
  - 战术优先级赋值。
- `omniperception::Decider::sort`
  - 多相机结果统一排序。
- `omniperception::Decider::decide`
  - 丢失主目标时给出辅助朝向命令。

## 70.6 工具层

- `tools::ExtendedKalmanFilter::predict/update`
- `tools::Trajectory::Trajectory`
- `tools::xyz2ypd / ypd2xyz / *_jacobian`
- `tools::Recorder::record`
- `tools::Plotter::plot`
- `tools::RansacSineFitter::fit`

