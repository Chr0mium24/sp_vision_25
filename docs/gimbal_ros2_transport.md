# Gimbal ROS2 Transport

## Status

当前仓库已经实现 `io::Gimbal` 的双后端传输：

- keep the current serial transport as the default path
- optionally compile a ROS2 transport when ROS2 dependencies are available
- allow the same upper-layer auto-aim code to work against either
  - a real serial port
  - ROS2 topics `/gimbal_to_vision` and `/vision_to_gimbal`

上层 API 保持不变：

- `io::Gimbal::q(...)`
- `io::Gimbal::mode()`
- `io::Gimbal::state()`
- `io::Gimbal::send(...)`

## Constraints

- `sp_vision` is not a standard ROS2 package today
- we should not require a ROS2 workspace to keep the current serial workflow working
- message definitions should avoid introducing a custom ROS2 msg package in this repository for now

## Transport Model

### Serial backend

The existing backend remains unchanged in behavior:

- open `com_port`
- read `GimbalToVision`
- write `VisionToGimbal`
- maintain CRC / reconnect / interpolation

### ROS2 backend

当 ROS2 可用且运行时配置选择 `ros2` 时，`io::Gimbal` 会：

- subscribe to `/gimbal_to_vision`
- publish to `/vision_to_gimbal`
- parse the incoming topic payload into the same `GimbalToVision` packet layout
- serialize outgoing `VisionToGimbal` using the same packet layout

The ROS2 backend emulates the serial stream semantics, so the rest of the code can continue to treat it as a transport only.

## Topic Types

For phase 1, use `std_msgs/msg/UInt8MultiArray` for both topics.

Reason:

- available from standard ROS2 packages
- preserves exact packet layout
- lets bridge tools reuse the current packed structs and CRC helpers
- avoids introducing a custom msg package before the auto-aim project itself is ROS2-native

Topic contract:

- `/gimbal_to_vision`: payload size must be `sizeof(io::GimbalToVision)`
- `/vision_to_gimbal`: payload size must be `sizeof(io::VisionToGimbal)`

## Runtime Config

相关 YAML 键如下：

```yaml
gimbal_transport: "serial"          # serial | ros2
gimbal_to_vision_topic: "/gimbal_to_vision"
vision_to_gimbal_topic: "/vision_to_gimbal"
gimbal_ros2_node_name: "sp_vision_gimbal_transport"
```

Behavior:

- no key present: default to `serial`
- `ros2` selected without ROS2 support compiled in: fail fast with a clear log

## Bridge Program

仓库同时提供了一个 ROS2 bridge 程序，把真实串口桥接到这两个 topic：

- read serial `GimbalToVision` packets and publish `/gimbal_to_vision`
- subscribe to `/vision_to_gimbal` and write serial `VisionToGimbal`

This lets a ROS2 navigation stack own the physical serial device while `sp_vision` consumes the mirrored topic transport.

可执行文件：

- `gimbal_ros2_bridge`

辅助脚本：

- `scripts/run_gimbal_ros2_bridge.sh`

Example:

```bash
./build.sh
./scripts/run_gimbal_ros2_bridge.sh configs/standard3.yaml --ports=/dev/ttyACM0
```

## Build Strategy

### Without ROS2

- compile serial backend only
- keep current binaries working

### With ROS2 core packages

Require:

- `ament_cmake`
- `rclcpp`
- `std_msgs`

Compile:

- ROS2 gimbal transport backend
- gimbal ROS2 bridge executable

### With full navigation ROS2 packages

If `sp_msgs` is also available, continue compiling the existing navigation ROS2 glue:

- `io/ros2/publish2nav.cpp`
- `io/ros2/subscribe2nav.cpp`
- `io/ros2/ros2.cpp`

This keeps old and new ROS2 paths independent.

## Implemented Scope

当前已实现：

1. `io::Gimbal` 运行时切换 `serial/ros2`
2. `gimbal_ros2_bridge`
3. `scripts/run_gimbal_ros2_bridge.sh`
4. 示例配置字段：`gimbal_transport`、topic 名称、node 名称

## Future Work

- replace raw byte topics with a dedicated ROS2 msg package after the auto-aim stack is more ROS2-native
- add optional transport diagnostics topic
- add timeout / stale-data guards for the ROS2 backend
