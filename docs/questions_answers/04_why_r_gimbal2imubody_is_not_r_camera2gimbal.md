# 04 为什么 `R_gimbal2imubody` 不应该和 `R_camera2gimbal` 一样

## 结论

不应该默认一样。

这两个矩阵描述的是完全不同的坐标变换：

- `R_camera2gimbal`：相机坐标系到云台坐标系
- `R_gimbal2imubody`：云台坐标系到 IMU 本体坐标系

它们只有在极特殊的机械安装条件下才可能恰好相同，通常不会相同。

## 代码里的作用

`Solver` 启动时会分别读这两个矩阵：

```cpp
R_gimbal2imubody_ = ...
R_camera2gimbal_ = ...
```

见 `tasks/auto_aim/solver.cpp:31-36`。

随后这两个矩阵分别参与两条不同链路：

### 1. 姿态链路

```cpp
R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
```

见 `tasks/auto_aim/solver.cpp:48-52`。

这里 `R_gimbal2imubody` 的任务是把 IMU 测得的姿态换成“云台坐标语义”。

### 2. 位置链路

```cpp
armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
```

见 `tasks/auto_aim/solver.cpp:65-68`。

这里 `R_camera2gimbal` 的任务是把 PnP 解出来的相机系目标坐标，变成云台系目标坐标。

## 直观理解

这两个矩阵分别回答两个不同问题：

- 相机是怎么装在云台上的？
- IMU 是怎么装在云台上的？

只要相机和 IMU 不在同一位置、同一朝向安装，它们的外参就不会一样。

## 对 `standard3` 的判断

`standard3` 当前配置是：

```yaml
R_gimbal2imubody: [0, -1, 0, 1, 0, 0, 0, 0, 1]
R_camera2gimbal:  [-0.027..., -0.126..., 0.991..., ...]
```

这很正常。

前者更像“轴交换 + 符号约定”，后者更像“真实手眼标定结果”。

## 什么时候该怀疑它们

应该怀疑的是“某一个矩阵是否和对应硬件关系匹配”，而不是“两个矩阵为什么不相等”。

排查标准：

- 纯 yaw 转动时，`R_gimbal2imubody` 是否让 pitch/roll 基本不串轴
- 重投影是否说明 `R_camera2gimbal / t_camera2gimbal` 对得上画面

## 结论复述

`R_gimbal2imubody` 和 `R_camera2gimbal` 描述的不是同一件事，所以不应该拿“是否一样”当判断标准。
