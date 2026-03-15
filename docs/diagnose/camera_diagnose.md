# Camera Diagnose 使用说明

本文档说明 `diagnostics/camera/diagnose.sh` 的常用用法。

## 1. 前置条件

- 已完成编译，且存在 `build/bin/tests/camera/*` 可执行文件。
- 相机已连接，工业相机 SDK / USB 驱动已就绪。

如缺少二进制，可先执行：

```bash
bash build.sh
```

## 2. 基本用法

```bash
diagnostics/camera/diagnose.sh <action> [config.yaml] [extra args...]
```

不带参数时会显示帮助说明。

## 3. 常用动作

### 3.1 查看设备

```bash
./diagnostics/camera/diagnose.sh info
```

作用：

- 打印 `/dev/video*`
- 若安装了 `v4l2-ctl`，打印 `v4l2-ctl --list-devices`

### 3.2 工业相机快速联通

```bash
./diagnostics/camera/diagnose.sh quick configs/standard3.yaml
```

作用：跑 `camera_test`，持续输出 FPS。

### 3.3 一键释放相机占用（ROS2/容器）

当工业相机被 `component_container_mt` 或 `rm_bringup` 占用时，先执行：

```bash
sudo ./diagnostics/camera/diagnose.sh release
```

默认目标设备为 `2bdf:0001`（HikRobot MV-CS016-10UC）。可选参数：

```bash
sudo ./diagnostics/camera/diagnose.sh release --vidpid=2bdf:0001 --force
```

说明：

- `release` 会尝试停止匹配容器（`rm_bringup/foxglove/ros2/camera_detector`）。
- 尝试结束相关 ROS2 进程并释放 `/dev/bus/usb/...` 句柄。
- `--force` 会对占用句柄执行更强的 kill（`-9`）。

### 3.4 交互调参并重载窗口（推荐）

```bash
./diagnostics/camera/diagnose.sh tune configs/standard3.yaml --scale=0.7
```

作用：

- 在终端交互调 `exposure_ms/gain`
- 每次改参后自动重载 `camera_window_test`
- 默认静默窗口日志，不在终端持续刷 FPS
- `--scale=<ratio>` 只影响窗口显示缩放，不改变相机采集参数

交互命令：

- `e` 或 `e <num>`: 设置 `exposure_ms`（不带数值会继续提示输入）
- `g` 或 `g <num>`: 设置 `gain`（不带数值会继续提示输入）
- `r`: 手动重载窗口
- `p`: 打印当前参数
- `q`: 退出

可选：

```bash
./diagnostics/camera/diagnose.sh tune configs/standard3.yaml --scale=0.7 --show-log
```

`--show-log` 会显示窗口进程日志（包含 FPS 输出）。

### 3.5 工业相机窗口预览

```bash
./diagnostics/camera/diagnose.sh window configs/standard3.yaml --scale=0.7
```

作用：打开预览窗口，按 `q` 退出。

### 3.6 工业相机 + 检测链路

```bash
./diagnostics/camera/diagnose.sh detect configs/standard3.yaml
```

可选传统方法：

```bash
./diagnostics/camera/diagnose.sh detect configs/standard3.yaml --tradition=true
```

### 3.7 采图保存

```bash
./diagnostics/camera/diagnose.sh save configs/standard3.yaml --output-folder=assets/camera_captures
```

在窗口内按键：

- `s`: 保存当前帧
- `q`: 退出

### 3.8 USB 相机联通 / 检测

```bash
./diagnostics/camera/diagnose.sh usb configs/sentry.yaml --name=video0 -d
./diagnostics/camera/diagnose.sh usb-detect configs/sentry.yaml --name=video0 -d
```

### 3.9 线程与手眼测试

```bash
./diagnostics/camera/diagnose.sh thread configs/ascento.yaml
./diagnostics/camera/diagnose.sh handeye configs/handeye.yaml -d
```

## 4. 排查建议

- 工业相机无图像：先确认 SDK 安装、配置里的 `camera_name` 与设备匹配。
- 工业相机被占用：先跑 `sudo ./diagnostics/camera/diagnose.sh release` 再重试 `quick/window/detect`。
- USB 相机无图像：先跑 `info`，确认 `/dev/video*` 是否存在、`--name` 是否匹配。
- 无图形界面环境：优先使用 `quick`（无窗口）或去掉 `-d`。
