# 标定流程

当前标定链路由 Python 入口 `sp-vision-calibration` 统一承载，底层采集和部分能力仍复用真实 C++ 绑定。

## 1. 命令总览

```bash
uv --project python run sp-vision-calibration help
```

当前支持：

- `capture`
- `calibrate-camera`
- `calibrate-handeye`
- `calibrate-robotworld-handeye`
- `split-video`

## 2. 采集数据

采集图像与姿态对：

```bash
uv --project python run sp-vision-calibration capture \
    configs/calibration.yaml \
    assets/img_with_q \
    --imu
```

常用参数：

- `configs/calibration.yaml`：标定配置
- `assets/img_with_q`：输出目录
- `--imu`：同时采集 IMU / 姿态信息
- `--no-show`：无图形环境时关闭预览

## 3. 相机内参标定

```bash
uv --project python run sp-vision-calibration calibrate-camera \
    assets/img_with_q \
    -c configs/calibration.yaml
```

输入要求：

- 标定板图像放在同一目录
- 配置中正确设置 `pattern_cols`、`pattern_rows`、`center_distance_mm`

输出结果：

- `camera_matrix`
- `distort_coeffs`
- 终端中的重投影误差

## 4. 手眼标定

```bash
uv --project python run sp-vision-calibration calibrate-handeye \
    assets/img_with_q \
    -c configs/calibration.yaml
```

目标：

- 求解 `R_camera2gimbal`
- 求解 `t_camera2gimbal`

建议：

- 采集多个稳定姿态
- 同一批数据里图像和姿态文件保持一一对应

## 5. Robot-World Hand-Eye

```bash
uv --project python run sp-vision-calibration calibrate-robotworld-handeye \
    assets/img_with_q \
    -c configs/calibration.yaml
```

适用场景：

- 需要同时求相机外参与世界系关系
- 希望把板位姿和相机位姿一起纳入解算

## 6. 视频切分

```bash
uv --project python run sp-vision-calibration split-video \
    records/Big/2024-05-14_11-6-26 \
    -p records/Big/2024-05-14_11-6-26_cut \
    --start-index=0 \
    --end-index=0
```

`input_path` 和 `output_path` 都使用不带扩展名的前缀路径。

## 7. 建议流程

1. 采集标定图像和姿态
2. 运行 `calibrate-camera`
3. 运行 `calibrate-handeye`
4. 必要时运行 `calibrate-robotworld-handeye`
5. 将结果写回实际运行配置
6. 用 `camera window` 或 `auto-aim armor-box` 做联调确认

## 8. 关键配置字段

- `camera_matrix`
- `distort_coeffs`
- `R_camera2gimbal`
- `t_camera2gimbal`
- `R_gimbal2imubody`
- `yaw_offset`
- `pitch_offset`
