# 07 当前 ROS2 传输链路到底是什么

## 结论

你描述的方向基本对，但要改一个词：当前实现不是 ROS2 service，而是 ROS2 topic。

当前仓库已经实现的是：

1. `sp_vision` 内部的 `io::Gimbal` 支持 `serial` 和 `ros2` 两种传输后端
2. 外部提供一个 `gimbal_ros2_bridge` 程序，把真实串口桥接成两个 topic

## `sp_vision` 侧的行为

当配置里：

```yaml
gimbal_transport: "ros2"
```

且编译时带了 ROS2 support，`io::Gimbal` 会：

- 订阅 `/gimbal_to_vision`
- 发布 `/vision_to_gimbal`

见：

- `io/gimbal/gimbal.cpp:65-116`

更具体地说：

- `ros2_tx_publisher_` 发布 `vision_to_gimbal`：`io/gimbal/gimbal.cpp:79-81`
- `ros2_rx_subscription_` 订阅 `gimbal_to_vision`：`io/gimbal/gimbal.cpp:82-105`

## Bridge 侧的行为

`diagnostics/gimbal/gimbal_ros2_bridge.cpp` 当前实现的是：

- 串口读 `GimbalToVision`，发布到 `/gimbal_to_vision`
- 订阅 `/vision_to_gimbal`，收到后写串口

见：

- 建 publisher/subscriber：`diagnostics/gimbal/gimbal_ros2_bridge.cpp:149-152`
- 收到 `vision_to_gimbal` 后写串口：`diagnostics/gimbal/gimbal_ros2_bridge.cpp:181-218`
- 从串口读 `GimbalToVision` 并发布：`diagnostics/gimbal/gimbal_ros2_bridge.cpp:221-257`

## 包格式

当前 topic 载荷不是自定义 msg，而是原始字节数组：

- topic 类型：`std_msgs/msg/UInt8MultiArray`
- `/gimbal_to_vision` 载荷大小：`sizeof(GimbalToVision)`
- `/vision_to_gimbal` 载荷大小：`sizeof(VisionToGimbal)`

文档说明见：

- `docs/gimbal_ros2_transport.md:48-63`

## 这和你描述的链路对齐情况

你的理解基本可以改写成：

### 当前 `sp_vision` 主程序

- 不直接占串口
- 通过 `io::Gimbal` 订阅 `gimbal_to_vision`
- 通过 `io::Gimbal` 发布 `vision_to_gimbal`

### 当前 bridge 程序

- 独占串口
- 读串口后发布 `gimbal_to_vision`
- 收到 `vision_to_gimbal` 后写串口

这和你想的方向是一致的。

## 一个容易混淆的点

包头不是都 `5A`。

当前协议里：

- `GimbalToVision` 头是 `0x5A`
- `VisionToGimbal` 头是 `0xA5`

见：

- `io/gimbal/gimbal.hpp:24-56`

所以如果你说“5A 开头的自瞄包”，更准确地说那是“下位机发给视觉的包”。

## 结论复述

当前代码已经实现了你想要的 topic 化传输主干，但它是“topic bridge”，不是“ROS2 service 接口”。
