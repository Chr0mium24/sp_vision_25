# 05 `standard3` 的其它参数是什么

## 结论

`configs/standard3.yaml` 现在可以分成 9 组参数。

下面按用途整理，而不是简单把 YAML 原样抄一遍。

## 1. 检测主配置

- `enemy_color: blue`
- `yolo_name: yolov5`
- `device: CPU`
- `min_confidence: 0.8`
- `use_traditional: true`

含义：

- 主检测器是 YOLOv5
- 运行在 CPU
- 识别后还会走一次传统角点修正

## 2. 模型路径

- `classify_model: assets/tiny_resnet.onnx`
- `yolo11_model_path: assets/yolo11.xml`
- `yolov8_model_path: assets/yolov8.xml`
- `yolov5_model_path: assets/yolov5.xml`

这里虽然配了多套模型路径，但当前 `yolo_name` 决定实际用哪一个。

## 3. ROI 和传统视觉参数

ROI：

- `roi.x = 420`
- `roi.y = 50`
- `roi.width = 600`
- `roi.height = 600`
- `use_roi = false`

传统视觉：

- `threshold = 150`
- `max_angle_error = 45`
- `min_lightbar_ratio = 1.5`
- `max_lightbar_ratio = 20`
- `min_lightbar_length = 8`
- `min_armor_ratio = 1`
- `max_armor_ratio = 5`
- `max_side_ratio = 1.5`
- `max_rectangular_error = 25`

## 4. Tracker 参数

- `min_detect_count = 5`
- `max_temp_lost_count = 15`
- `outpost_max_temp_lost_count = 75`

作用是控制初始化、临时丢失和前哨站容错。

## 5. Aimer 与 Shooter 参数

Aimer：

- `yaw_offset = 2`
- `pitch_offset = 13`
- `comming_angle = 55`
- `leaving_angle = 20`
- `decision_speed = 7`
- `high_speed_delay_time = 0.0`
- `low_speed_delay_time = 0.0`

Shooter：

- `first_tolerance = 3`
- `second_tolerance = 2`
- `judge_distance = 2`
- `auto_fire = true`

## 6. 相机参数

- `camera_name = hikrobot`
- `exposure_ms = 2.0`
- `gain = 16.9`
- `vid_pid = 2bdf:0001`

## 7. 外参和内参

IMU 相关：

- `R_gimbal2imubody = [0,-1,0;1,0,0;0,0,1]`

相机内参：

- `camera_matrix`
- `distort_coeffs`

手眼外参：

- `R_camera2gimbal`
- `t_camera2gimbal`

## 8. CBoard / Gimbal / ROS2 Transport

CBoard：

- `quaternion_canid = 0x100`
- `bullet_speed_canid = 0x101`
- `send_canid = 0xff`
- `can_interface = can0`

Gimbal：

- `com_port = /dev/ttyACM0`
- `yaw_kp = 0`
- `yaw_kd = 0`
- `pitch_kp = 0`
- `pitch_kd = 0`

ROS2 transport：

- `gimbal_transport = serial`
- `gimbal_to_vision_topic = /gimbal_to_vision`
- `vision_to_gimbal_topic = /visionToGimbal`
- `gimbal_ros2_node_name = sp_vision_gimbal_transport`

## 9. Planner / Buff 参数

Planner：

- `fire_thresh = 0.0035`
- `max_yaw_acc = 50`
- `Q_yaw = [9e6, 0]`
- `R_yaw = [1]`
- `max_pitch_acc = 100`
- `Q_pitch = [9e6, 0]`
- `R_pitch = [1]`

Buff：

- `model = assets/yolo11_buff_int8.xml`
- `fire_gap_time = 0.700`
- `predict_time = 0.120`

## 最值得先看的参数

如果你是为了排故，最先看这几项：

1. `enemy_color`
2. `use_traditional`
3. `pitch_offset`
4. `R_gimbal2imubody`
5. `camera_matrix / distort_coeffs`
6. `R_camera2gimbal / t_camera2gimbal`
7. `gimbal_transport`

## 配置位置

完整原文件见：

- `configs/standard3.yaml`
