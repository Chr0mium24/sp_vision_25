# Auto Aim Diagnose 使用说明

本文档说明 `sp-vision-diagnose auto-aim` 的用法，覆盖装甲板与 Power Rune 的识别、画框和调参。

## 1. 前置条件

- 已完成编译，且 Python 入口可用：
  - `sp-vision-diagnose`
  - `sp-vision-calibration`（如果你还在跑标定流程）
- 相机和云台在线联调时，串口/相机链路已正常（可先用 gimbal/camera diagnose 确认）。

如缺少入口，可先执行：

```bash
bash build.sh
```

## 2. 基本用法

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim <action> [config.yaml] [extra args...]
```

不带参数时会显示帮助说明。

## 3. 常用动作

### 3.1 查看可执行文件状态

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim list
```

### 3.2 装甲板在线识别 + 画框（推荐）

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim armor-box configs/standard3.yaml
```

说明：

- 调用 Python diagnose 的 `armor-box` 入口
- 图像窗口里会显示目标框和重投影框
- TUI 同时显示 `targets/state/cmd` 等关键状态
- 按 `S` 可将当前帧完整快照保存到 `logs/auto_aim_snapshots/<timestamp>_<index>/`
- 快照目录包含 `raw.png`、`annotated.png`、`frame.json`，其中 `frame.json` 会展开 `solver -> tracker/target -> aimer -> final command` 的中间数据流

### 3.3 装甲板在线识别（无 GUI）

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim armor-rec configs/standard3.yaml
```

说明：适合远程终端，仅看识别状态和控制输出。

### 3.4 装甲板在线调参（可导出 YAML）

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim armor-tune configs/standard3.yaml
```

说明：

- 当前 `armor-tune` 仍然通过过渡性的 C++ `auto_aim_ui_tune.cpp` 承担交互编辑，但 Python 入口已经接管命令路由和文档入口。
- 后续若继续精简，会优先把 YAML 回写和交互面板再往 Python TUI 收口。

关键交互：

- `j/k` 选参数，`-/=` 调整，`u` 切换布尔参数
- `R` 导出新配置（文件名带时间戳）
- `L` 开关日志（`logs/auto_aim_ui_*.jsonl`）

### 3.5 装甲板离线回放

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim armor-offline configs/demo.yaml assets/demo/demo --start-index=0 --end-index=0
```

说明：`input-prefix` 传前缀，程序读取 `<prefix>.avi` 和 `<prefix>.txt`。

### 3.6 Power Rune 识别 + 画框（离线）

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim rune-box configs/sentry.yaml assets/demo/power_rune_demo --start-index=0 --end-index=0
```

说明：

- `rune-rec` 是 `rune-box` 的别名。
- 程序会在窗口显示打符目标框、重投影结果和调试曲线。

### 3.7 Power Rune 调参（脚本交互）

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim rune-tune configs/sentry.yaml assets/demo/power_rune_demo --start-index=0 --end-index=0
```

调参命令：

- `y [num]` 设置 `yaw_offset`
- `i [num]` 设置 `pitch_offset`
- `f [num]` 设置 `fire_gap_time`
- `t [num]` 设置 `predict_time`
- `r` 按当前参数重跑一次 `rune-box`
- `p` 打印当前参数
- `q` 退出

说明：`rune-tune` 会直接写回当前 `config.yaml`。

### 3.8 Power Rune 在线调试（可选）

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim rune-online configs/standard3.yaml
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose auto-aim rune-online-mpc configs/standard3.yaml
```

说明：依赖 `build/bin/diag/auto_buff/auto_buff_debug` 与 `build/bin/diag/auto_buff/auto_buff_debug_mpc`。

## 4. 排查建议

- `armor-box` 无目标：先确认相机取流正常、模型文件路径存在、云台姿态回传正常。
- `rune-box` 无图像：确认输入前缀对应的 `.avi/.txt` 同时存在。
- `rune-tune` 后效果变差：先备份配置，再按 `p` 核对参数并回退。
