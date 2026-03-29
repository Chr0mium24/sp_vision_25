# 72. 数据流

## 72.1 标准自瞄数据流

1. `io::Camera::read` 产出图像和时间戳。
2. `io::Gimbal::q` 或 `io::CBoard::imu_at` 产出与图像对齐的姿态四元数。
3. `auto_aim::Solver::set_R_gimbal2world` 刷新坐标系。
4. `auto_aim::YOLO::detect` 或 `auto_aim::Detector::detect` 产出 `std::list<Armor>`。
5. `auto_aim::Solver::solve` 将每个 `Armor` 补全 3D 位姿。
6. `auto_aim::Tracker::track` 维护单目标 `Target` 状态机与 EKF。
7. `auto_aim::Aimer::aim` 或 `auto_aim::Planner::plan` 产出控制命令。
8. `auto_aim::Shooter::shoot` 决定是否真的开火。
9. `io::Gimbal::send` 或 `io::CBoard::send` 下发命令。

## 72.2 多线程自瞄数据流

1. 采图线程不断读图。
2. `MultiThreadDetector::push` 把图像和时间戳压入异步推理队列。
3. `MultiThreadDetector::pop/debug_pop` 取出推理完成的装甲板结果。
4. 主线程完成 `Solver -> Tracker`。
5. `CommandGener::push` 把最新目标交给后台命令线程。
6. `CommandGener::generate_command` 在后台调用 `Aimer/Shooter` 并发给 C 板。

## 72.3 MPC 自瞄数据流

1. 主线程完成 `Camera -> YOLO -> Tracker`。
2. 最新 `Target` 压入 `target_queue`。
3. 规划线程读取云台状态 `GimbalState`。
4. `Planner::plan` 生成 `Plan`。
5. `Gimbal::send` 发送位置、速度、加速度前馈。

## 72.4 打符数据流

1. `Camera::read` 取图。
2. `Buff_Detector::detect` 识别扇叶和圆心。
3. `auto_buff::Solver::solve` 求出 `PowerRune` 在世界系下的位置与姿态。
4. `SmallTarget/BigTarget::get_target` 维护符运动状态。
5. `auto_buff::Aimer::aim` 或 `mpc_aim` 计算控制量。
6. `CBoard::send` 或 `Gimbal::send` 下发。

## 72.5 哨兵/全向感知数据流

1. 主相机链路正常时，走标准自瞄链。
2. 若 `Tracker` 进入 `lost` 或 `switching`：
3. `Perceptron` 从多 USB 相机并行产出 `DetectionResult` 队列。
4. `Decider::sort` 对多路结果统一过滤和排序。
5. `Decider::decide` 生成辅助转向命令。
6. 目标重新进入主相机视野后，`Tracker` 切回正常 `tracking`。

## 72.6 标定数据流

1. `capture.cpp` 采图并可选记录 IMU 四元数。
2. `calibrate_camera.cpp` 用标定板图像求 `camera_matrix/distort_coeffs`。
3. `calibrate_handeye.cpp` 或 `calibrate_robotworld_handeye.cpp` 联合图像和姿态求外参。
4. 结果回填到 `configs/*.yaml`，被 `Solver` 等模块加载。

