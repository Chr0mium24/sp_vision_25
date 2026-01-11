# 标定流程（实操版）

本文档给出当前系统的可执行标定流程：串口云台、暂无弹速反馈，且发射硬件尚未可用。

## 发射未恢复（无需实弹）

### 1) 相机内参 + 畸变
- 工具：`calibration/calibrate_camera.cpp`
- 输入：同一文件夹内的圆点标定图，命名为 `1.jpg`、`2.jpg` ...
- 配置：`configs/calibration.yaml`
  - `pattern_cols`、`pattern_rows`、`center_distance_mm`
- 运行：
  - `./build/calibrate_camera -c configs/calibration.yaml assets/img_with_q`
- 输出：
  - `camera_matrix`、`distort_coeffs`，并在 stdout 打印重投影误差
- 应用：
  - 粘贴到当前使用的配置文件（如 `configs/standard3.yaml`）
- 验证：
  - 用任意重投影可视化工具确认点位落在标定板上

### 2) 相机 -> 云台外参
- 目标：填充配置中的 `R_camera2gimbal`、`t_camera2gimbal`
- 步骤：
  - 将标定板固定在云台正前方（姿态稳定且可测）
  - 云台在多个 yaw/pitch 姿态下采集图像和云台姿态
  - 使用手眼标定解算相机到云台的外参
- 验证：
  - 多姿态下的重投影点应一致对齐

### 3) 云台 -> IMU 外参
- 目标：填充 `R_gimbal2imubody`
- 步骤：
  - 分别做纯 yaw 与纯 pitch 旋转
  - 检查 IMU 报告的轴是否与云台轴对齐（避免交叉轴耦合）
  - 调整 `R_gimbal2imubody` 使轴对齐
- 验证：
  - 纯 yaw 不应引入明显的 pitch/roll 偏差

### 4) Yaw/Pitch 零偏
- 目标：调整 `yaw_offset`、`pitch_offset`
- 步骤：
  - 放置静态目标在画面中心
  - 在不发射的情况下观察瞄准误差
  - 调整零偏直到自动瞄准能稳定居中

### 5) 检测稳定性
- 调整检测/跟踪相关参数：
  - `enemy_color`、`min_confidence`、`roi`、传统阈值
- 验证：
  - 目标连续跟踪、误检低

### 6) 固定弹速（临时）
- 在无反馈条件下先设定固定弹速：
  - 从粗略值开始（如 22 或 25）
  - 等发射功能恢复后再精调

## 发射恢复（可实弹）

### 1) 协议 + 方向检查
- 核对 yaw/pitch 方向与符号：
  - 正 yaw 向右、正 pitch 向上（或与项目约定一致）
- 先修正协议或符号再进入调参

### 2) 弹速标定
- 实测枪口速度：
  - 有测速门则直接测量，否则用固定距离实射反推
- 更新固定弹速或接入反馈

### 3) 弹道模型 + 零偏微调
- 近/中/远距离实射
- 调整：
  - `pitch_offset`
  - 弹速
  - 延时参数

### 4) 开火判定阈值
- 调整：
  - `first_tolerance`、`second_tolerance`、`judge_distance`
- 目标：
  - 稳定目标 -> 开火
  - 不稳定目标 -> 不开火

### 5) 高速目标延时
- 调整：
  - `high_speed_delay_time`、`low_speed_delay_time`
- 目标：
  - 不超调、不滞后

## 关键配置字段（参考）

相机：
- `camera_matrix`
- `distort_coeffs`
- `camera_name`
- `exposure_ms`
- `gain`
- `vid_pid`

外参：
- `R_camera2gimbal`
- `t_camera2gimbal`
- `R_gimbal2imubody`

零偏与瞄准：
- `yaw_offset`
- `pitch_offset`
- `decision_speed`
- `comming_angle`
- `leaving_angle`
- `high_speed_delay_time`
- `low_speed_delay_time`

检测相关：
- `enemy_color`
- `min_confidence`
- `roi`
- 传统检测器阈值
