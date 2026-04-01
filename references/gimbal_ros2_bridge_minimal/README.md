# gimbal_ros2_bridge_minimal

最小可运行的 Linux 串口 <-> ROS2 topic 桥接参考。

目录内容：

- `src/gimbal_ros2_bridge_minimal.cpp`：桥接主程序
- `src/bridge_packets.hpp`：协议结构体和 CRC
- `bridge.yaml`：最小配置
- `build.sh`：独立编译
- `run_gimbal_ros2_bridge.sh`：独立运行

依赖：

- ROS2 `rclcpp`
- `std_msgs`
- `yaml-cpp`
- 目录内自带的 `third_party/serial`

这个目录现在已经把主仓库里的 `io/serial` 直接拷进来了，可以整目录移走单独用。
在 Linux 下，这份 `serial` 库底层还是 `termios`。

编译：

```bash
source /opt/ros/humble/setup.bash
cd references/gimbal_ros2_bridge_minimal
bash build.sh
```

运行：

```bash
source /opt/ros/humble/setup.bash
cd references/gimbal_ros2_bridge_minimal
bash run_gimbal_ros2_bridge.sh bridge.yaml
```

如果要改 topic 或串口，直接改 `bridge.yaml`。
如果要临时覆盖串口，也可以：

```bash
bash run_gimbal_ros2_bridge.sh bridge.yaml --ports=/dev/ttyACM0
```
