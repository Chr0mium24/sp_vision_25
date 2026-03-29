# 71. 核心数据结构

## 71.1 控制与通信

### `io::Command`

```cpp
struct Command {
  bool control;
  bool shoot;
  double yaw;
  double pitch;
  double horizon_distance;
};
```

作用：

- 是算法层统一输出。
- `control=false` 代表本帧不接管。
- `shoot=true` 代表本帧允许开火。
- `yaw/pitch` 是发送给云台或 C 板的目标角。

### `io::GimbalToVision`

串口下位机到视觉的扩展包，核心字段有：

- `yaw/pitch/roll`
- `yaw_odom/pitch_odom`
- `yaw_vel/pitch_vel`
- `aim_x/aim_y/aim_z`
- `robot_id`

### `io::VisionToGimbal`

视觉到下位机的控制包，核心字段有：

- `tracking`
- `pitch`
- `yaw`
- `fire`
- `fric_on`

### `io::GimbalState`

是 `Gimbal` 对外暴露的“云台状态快照”，主程序和调试程序都直接读它。

## 71.2 自瞄观测与状态

### `auto_aim::Lightbar`

表示单个灯条，包含：

- 几何中心、上下端点、方向向量
- 长宽比、角度误差
- 原始 `RotatedRect`

### `auto_aim::Armor`

贯穿检测、解算、跟踪全流程的核心结构，包含：

- 图像层字段：`points/center/box/pattern/confidence`
- 类别字段：`color/name/type/priority/class_id`
- 3D 字段：`xyz_in_gimbal/world`、`ypr_in_gimbal/world`、`ypd_in_world`
- 调试字段：`yaw_raw/duplicated`

### `auto_aim::Target`

内部最关键的是 11 维 EKF 状态向量：

```text
x = [center_x, vx, center_y, vy, center_z, vz, angle, angular_vel, r, l, h]
```

字段含义：

- `center_x/center_y/center_z`：整车旋转中心
- `vx/vy/vz`：中心速度
- `angle`：当前参考装甲板角度
- `angular_vel`：整车 yaw 角速度
- `r`：主半径
- `l`：长短半径差
- `h`：上下装甲高度差

它还维护：

- `name/armor_type/priority`
- `jumped`：是否发生过装甲板切换
- `last_id`：上一次匹配到的装甲板编号

### `auto_aim::RuntimeInput / RuntimeOutput`

`RuntimeInput`：

- 输入图像
- 时间戳
- 当前 `q_gimbal2world`
- 弹速
- 是否使用敌方颜色过滤

`RuntimeOutput`：

- 观测装甲板列表
- 跟踪目标列表
- 最终控制命令
- 跟踪器状态字符串

### `auto_aim::AimPoint`

- `valid`
- `xyza`

其中 `xyza` 表示选中的待打装甲板位置和角度。

### `auto_aim::Plan`

MPC 输出结构：

```text
control, fire,
target_yaw, target_pitch,
yaw, yaw_vel, yaw_acc,
pitch, pitch_vel, pitch_acc
```

它不仅给目标角，还给速度和加速度前馈。

## 71.3 打符观测与状态

### `auto_buff::FanBlade`

表示单个扇叶：

- `center`
- `points`
- `angle/width/height`
- `type`

### `auto_buff::PowerRune`

表示当前符：

- `r_center`
- `fanblades`
- `light_num`
- `xyz_in_world / ypr_in_world / ypd_in_world`
- `blade_xyz_in_world / blade_ypd_in_world`

### `auto_buff::Target`

是小符/大符估计器的基类，包含：

- EKF `x/P/A/Q/H/R`
- 方向投票器 `Voter`
- `first_in_/unsolvable_`

#### 小符状态向量

`SmallTarget` 使用 7 维状态，可概括为：

```text
[R_yaw, v_R_yaw, R_pitch, R_dis, yaw, angle(roll), spd]
```

#### 大符状态向量

`BigTarget` 使用 10 维状态，可概括为：

```text
[R_yaw, v_R_yaw, R_pitch, R_dis, yaw, angle(roll), spd, a, w, fi]
```

其中：

- `spd`：角速度
- `a/w/fi`：大符正弦速度模型参数

## 71.4 全向感知数据

### `omniperception::DetectionResult`

```text
armors + timestamp + delta_yaw + delta_pitch
```

它不是最终目标状态，而是“某一路辅助相机此刻看到了什么，以及从当前主云台转过去需要补多少角”。

## 71.5 工具层共享结构

### `tools::Trajectory`

- `unsolvable`
- `fly_time`
- `pitch`

### `tools::Frame`

多线程检测链里的统一帧容器：

- `id`
- `img`
- `t`
- `q`
- `armors`

### `tools::ExtendedKalmanFilter`

- `x`：状态向量
- `P`：状态协方差
- `data`：残差、NIS、NEES 等调试指标
- `recent_nis_failures`：最近窗口内检验失败记录

