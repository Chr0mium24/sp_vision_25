# 测试链路与调用说明

本文档用于统一说明 `sp_vision_25` 当前测试链路、可执行文件分类和调用方式。

## 1. 编译与输出目录

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

自动化测试可执行文件输出到：

```bash
build/bin/tests/<group>/<executable>
```

联调诊断工具输出到：

```bash
build/bin/diag/<group>/<executable>
```

例如：

```bash
build/bin/tests/auto_buff/auto_power_rune_test
build/bin/diag/gimbal/gimbal_ui_test
build/bin/diag/auto_aim/auto_aim_ui_test
```

## 2. 核心测试链路（推荐）

### 2.1 云台通信收发（在线）

目标：验证视觉到下位机发送、下位机姿态/状态回读是否正常。

主程序：

- `gimbal_link_diag_test`（非英雄协议快速诊断：可选发包+持续统计收包）
- `gimbal_ui_test`（推荐，支持 `read/control` 两模式）
- `gimbal_test`（基础连通性与发射节奏）
- `gimbal_response_test`（响应分析）

调用示例：

```bash
# 一键封装脚本（推荐，完整说明见 [docs/diagnose/gimbal_diagnose.md](diagnose/gimbal_diagnose.md)）
./diagnostics/gimbal/diagnose.sh quick
./diagnostics/gimbal/diagnose.sh rxonly
./diagnostics/gimbal/diagnose.sh proto

# 非英雄协议快速诊断：先看串口是否有字节、是否有有效帧（推荐先跑）
./build/bin/diag/gimbal/gimbal_link_diag_test configs/standard3.yaml --duration-ms=3000 --summary-ms=1000

# 仅收包诊断（不发控制）
./build/bin/diag/gimbal/gimbal_link_diag_test configs/standard3.yaml --no-send --duration-ms=3000 --summary-ms=1000

# 只读模式：只看回读，不发控制
./build/bin/diag/gimbal/gimbal_ui_test configs/standard3.yaml --mode=read --nogui

# 控制模式：发送 yaw/pitch/tracking/fire/fric（键盘可交互）
./build/bin/diag/gimbal/gimbal_ui_test configs/standard3.yaml --mode=control

# 脚本化反复调试（无键盘输入，跑5秒自动退出）
./build/bin/diag/gimbal/gimbal_ui_test configs/standard3.yaml --mode=control --no-input --duration-ms=5000 --yaw-deg=3 --pitch-deg=-1 --tracking=1 --fric-on=1 --fire-mode=1
```

说明：

- 该链路默认使用非英雄协议（`0xA5` 下发 + `0x5A` 回传），并兼容 28B/49B 回传帧。
- `hero_gimbal_ui_test`、`usb_aim_rx_test` 属于英雄链路，非英雄车调试时不建议混用。

### 2.2 车的自瞄闭环（在线）

目标：跑通相机 + 云台 + 自瞄主流程闭环。

推荐入口：

- `diagnostics/auto_aim/diagnose.sh`（完整说明见 [docs/diagnose/auto_aim_diagnose.md](diagnose/auto_aim_diagnose.md)）

调用示例：

```bash
./diagnostics/auto_aim/diagnose.sh armor-rec configs/standard3.yaml
./diagnostics/auto_aim/diagnose.sh armor-box configs/standard3.yaml
```

说明：`auto_aim_ui_test` 与 `standard` 均复用 `tasks/auto_aim/auto_aim_runtime.*`，联调链路与主链路核心计算保持一致。

### 2.3 自瞄调参（在线 + 导出参数）

目标：在线调参并导出新 YAML 配置。

调用示例：

```bash
./diagnostics/auto_aim/diagnose.sh armor-tune configs/standard3.yaml
```

关键交互：

- `j/k` 选参数，`-/=` 调整，`u` 切换布尔参数
- `R` 导出新配置（写到原配置目录，文件名带时间戳）
- `L` 开关日志（`logs/auto_aim_ui_*.jsonl`）

### 2.4 Power Rune（离线回放）

目标：离线验证打符识别/解算/瞄准逻辑。

调用示例：

```bash
./diagnostics/auto_aim/diagnose.sh rune-box configs/sentry.yaml assets/demo/power_rune_demo --start-index=0 --end-index=0
./diagnostics/auto_aim/diagnose.sh rune-tune configs/sentry.yaml assets/demo/power_rune_demo --start-index=0 --end-index=0
```

说明：该测试是离线数据回放，不依赖真实云台在线回传。

### 2.5 相机联通与预览（在线）

目标：快速确认相机设备存在、取流正常、可选检测链路可跑通。

推荐入口：

- `diagnostics/camera/diagnose.sh`（完整说明见 [docs/diagnose/camera_diagnose.md](diagnose/camera_diagnose.md)）

调用示例：

```bash
./diagnostics/camera/diagnose.sh info
sudo ./diagnostics/camera/diagnose.sh release
./diagnostics/camera/diagnose.sh tune configs/standard3.yaml --scale=0.7
./diagnostics/camera/diagnose.sh quick configs/standard3.yaml
./diagnostics/camera/diagnose.sh window configs/standard3.yaml --scale=0.7
./diagnostics/camera/diagnose.sh detect configs/standard3.yaml
./diagnostics/camera/diagnose.sh usb configs/sentry.yaml --name=video0 -d
```

## 3. 可执行文件分组

可执行文件分组如下：

- 自动化测试（`build/bin/tests`）：
- `auto_aim`: `auto_aim_test`, `detector_video_test`, `fire_test`
- `auto_buff`: `auto_power_rune_test`
- `gimbal`: `gimbal_test`, `gimbal_response_test`
- `camera`: `camera_test`, `camera_detect_test`, `camera_save_test`, `camera_thread_test`, `camera_window_test`, `usbcamera_test`, `usbcamera_detect_test`, `multi_usbcamera_test`, `handeye_test`
- `planner`: `planner_test`, `planner_test_offline`
- `system`: `cboard_test`, `dm_test`
- `ros2`（仅 ROS2 依赖满足时编译）：`publish_test`, `subscribe_test`, `topic_loop_test`
- 联调诊断（`build/bin/diag`）：
- `auto_aim`: `auto_aim_ui_test`, `auto_aim_ui_tune`
- `gimbal`: `gimbal_ui_test`, `gimbal_link_diag_test`, `gimbal_serial_probe`, `hero_gimbal_ui_test`, `usb_aim_rx_test`

## 4. 常见问题

- 查看参数：所有测试都支持 `--help`。
- 串口权限：若访问 `/dev/tty*` 失败，先检查 `dialout` 权限与 udev 规则。
- 无显示环境：加 `--nogui` 或不传 `--show=true`，使用 TUI 调试。
- 电控侧日志（`references` 工程）：
  - USB 链路统计日志在 `SEML/App/Robo/Robo_USB.c`，默认每 `500ms` 通过 `USART1` 输出一行：
  - `rx_poll/rx_empty/rx_total_bytes/rx_auto_aim_ok/rx_crc_fail/tx_busy` 等关键计数。
  - 控制状态日志在 `SEML/App/Robo/Robo_Control.c`，默认状态变化或 `1s` 周期输出：
  - `s1/s2/src/remote_off/aa/shoot`，可直接判断“中档是否仍为遥控控制”。
  - 两处可通过宏关闭或调频：`USB_LINK_DIAG_LOG_ENABLE`、`USB_LINK_DIAG_LOG_PERIOD_MS`、`CONTROL_LINK_DIAG_LOG_ENABLE`、`CONTROL_LINK_DIAG_LOG_PERIOD_MS`。
- 只编译单个测试：

```bash
cmake --build build -j"$(nproc)" --target gimbal_ui_test
cmake --build build -j"$(nproc)" --target auto_aim_ui_test
cmake --build build -j"$(nproc)" --target auto_aim_ui_tune
cmake --build build -j"$(nproc)" --target auto_power_rune_test
```
