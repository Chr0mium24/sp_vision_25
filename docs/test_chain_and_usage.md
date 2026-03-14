# 测试链路与调用说明

本文档用于统一说明 `sp_vision_25` 当前测试链路、可执行文件分类和调用方式。

## 1. 编译与输出目录

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

测试可执行文件统一输出到：

```bash
build/bin/tests/<group>/<executable>
```

例如：

```bash
build/bin/tests/gimbal/gimbal_ui_test
build/bin/tests/auto_aim/auto_aim_ui_test
build/bin/tests/auto_buff/auto_power_rune_test
```

## 2. 核心测试链路（推荐）

### 2.1 云台通信收发（在线）

目标：验证视觉到下位机发送、下位机姿态/状态回读是否正常。

主程序：

- `gimbal_ui_test`（推荐，支持 `read/control` 两模式）
- `gimbal_test`（基础连通性与发射节奏）
- `gimbal_response_test`（响应分析）

调用示例：

```bash
# 只读模式：只看回读，不发控制
./build/bin/tests/gimbal/gimbal_ui_test configs/standard3.yaml --mode=read --nogui

# 控制模式：发送 yaw/pitch/tracking/fire/fric（键盘可交互）
./build/bin/tests/gimbal/gimbal_ui_test configs/standard3.yaml --mode=control

# 脚本化反复调试（无键盘输入，跑5秒自动退出）
./build/bin/tests/gimbal/gimbal_ui_test configs/standard3.yaml --mode=control --no-input --duration-ms=5000 --yaw-deg=3 --pitch-deg=-1 --tracking=1 --fric-on=1 --fire-mode=1
```

### 2.2 车的自瞄闭环（在线）

目标：跑通相机 + 云台 + 自瞄主流程闭环。

主程序：

- `auto_aim_ui_test`（在线联调）
- `standard`（比赛主链路）

调用示例：

```bash
# TUI模式（默认）
./build/bin/tests/auto_aim/auto_aim_ui_test configs/standard3.yaml

# 带图像窗口
./build/bin/tests/auto_aim/auto_aim_ui_test configs/standard3.yaml --show=true
```

说明：`auto_aim_ui_test` 与 `standard` 均复用 `tasks/auto_aim/auto_aim_runtime.*`，测试链路和主链路核心计算保持一致。

### 2.3 自瞄调参（在线 + 导出参数）

目标：在线调参并导出新 YAML 配置。

主程序：

- `auto_aim_ui_tune`

调用示例：

```bash
./build/bin/tests/auto_aim/auto_aim_ui_tune configs/standard3.yaml --show=true
```

关键交互：

- `j/k` 选参数，`-/=` 调整，`u` 切换布尔参数
- `R` 导出新配置（写到原配置目录，文件名带时间戳）
- `L` 开关日志（`logs/auto_aim_ui_*.jsonl`）

### 2.4 Power Rune（离线回放）

目标：离线验证打符识别/解算/瞄准逻辑。

主程序：

- `auto_power_rune_test`（由 `auto_buff_test` 重命名）

调用示例：

```bash
# input-path 传“前缀”，程序会读取 <prefix>.avi 和 <prefix>.txt
./build/bin/tests/auto_buff/auto_power_rune_test --config-path=configs/sentry.yaml assets/demo/power_rune_demo --start-index=0 --end-index=0
```

说明：该测试是离线数据回放，不依赖真实云台在线回传。

## 3. 其他测试分组

可执行文件分组如下：

- `auto_aim`: `auto_aim_test`, `auto_aim_ui_test`, `auto_aim_ui_tune`, `detector_video_test`, `fire_test`
- `auto_buff`: `auto_power_rune_test`
- `gimbal`: `gimbal_ui_test`, `hero_gimbal_ui_test`, `gimbal_test`, `gimbal_response_test`, `usb_aim_rx_test`
- `camera`: `camera_test`, `camera_detect_test`, `camera_save_test`, `camera_thread_test`, `camera_window_test`, `usbcamera_test`, `usbcamera_detect_test`, `multi_usbcamera_test`, `handeye_test`
- `planner`: `planner_test`, `planner_test_offline`
- `system`: `cboard_test`, `dm_test`
- `ros2`（仅 ROS2 依赖满足时编译）：`publish_test`, `subscribe_test`, `topic_loop_test`

## 4. 常见问题

- 查看参数：所有测试都支持 `--help`。
- 串口权限：若访问 `/dev/tty*` 失败，先检查 `dialout` 权限与 udev 规则。
- 无显示环境：加 `--nogui` 或不传 `--show=true`，使用 TUI 调试。
- 只编译单个测试：

```bash
cmake --build build -j"$(nproc)" --target gimbal_ui_test
cmake --build build -j"$(nproc)" --target auto_aim_ui_test
cmake --build build -j"$(nproc)" --target auto_aim_ui_tune
cmake --build build -j"$(nproc)" --target auto_power_rune_test
```
