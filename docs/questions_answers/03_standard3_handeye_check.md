# 03 `standard3` 的手眼外参是不是被调坏了

## 结论

从当前仓库内容看，`standard3` 被改动了，但被改的不是你贴出来的这组 `R_camera2gimbal / t_camera2gimbal`。

这组手眼外参目前在以下文件里是一致的：

- `configs/standard3.yaml`
- `configs/backup.standard3.yaml`
- `configs/handeye.yaml`

所以仅从仓库内容判断，不能说是这组手眼外参被“调坏了”。

## 对比结果

`configs/backup.standard3.yaml` 和 `configs/standard3.yaml` 的 diff 里，真正变化较大的项是：

- `enemy_color`: `red -> blue`
- `pitch_offset`: `6.5 -> 13`
- `exposure_ms`: `2.5 -> 2.0`
- `R_gimbal2imubody`: 单位阵 -> `[0,-1,0;1,0,0;0,0,1]`
- `camera_matrix / distort_coeffs`: 换成了另一套内参
- 新增了 ROS2 transport 配置

但 `R_camera2gimbal` 和 `t_camera2gimbal` 没变。

## 代码和配置依据

当前 `standard3`：

- `configs/standard3.yaml:61`
- `configs/standard3.yaml:71-72`

备份配置：

- `configs/backup.standard3.yaml:62`
- `configs/backup.standard3.yaml:71-72`

## 更值得怀疑的参数

如果现在 `standard3` 表现异常，更应该优先怀疑这些项：

1. `R_gimbal2imubody`
2. `pitch_offset`
3. 新相机内参 `camera_matrix / distort_coeffs`
4. `enemy_color`

原因很直接：

- `R_gimbal2imubody` 会直接影响世界系姿态解算
- `pitch_offset` 从 `6.5` 跳到 `13`，变化很大
- 内参换了会直接影响 PnP 解算
- `enemy_color` 错了会直接打不到正确目标

## 推荐排查顺序

1. 先确认敌我颜色
2. 用 `gimbal_axis_diag_test` 复核 `R_gimbal2imubody`
3. 做重投影可视化，确认新内参是否正确
4. 再调 `pitch_offset`
5. 最后才怀疑 `R_camera2gimbal / t_camera2gimbal`

## 结论复述

你贴出来的那组 `R_camera2gimbal / t_camera2gimbal` 本身没有在 `standard3` 里被单独改坏；当前更像是“IMU 外参、内参和 pitch 零偏一起被换了”。
