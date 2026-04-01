# Auto Aim Diagnose 使用说明

本文档说明 `diagnostics/auto_aim/diagnose.sh` 的用法，覆盖装甲板与 Power Rune 的识别、画框和调参。

## 1. 前置条件

- 已完成编译，且存在下列可执行文件：
  - `build/bin/diag/auto_aim/auto_aim_ui_test`
  - `build/bin/diag/auto_aim/auto_aim_ui_tune`
  - `build/bin/tests/auto_buff/auto_power_rune_test`
- 相机和云台在线联调时，串口/相机链路已正常（可先用 gimbal/camera diagnose 确认）。

如缺少二进制，可先执行：

```bash
bash build.sh
```

## 2. 基本用法

```bash
diagnostics/auto_aim/diagnose.sh <action> [config.yaml] [extra args...]
```

不带参数时会显示帮助说明。

## 3. 常用动作

### 3.1 查看可执行文件状态

```bash
./diagnostics/auto_aim/diagnose.sh list
```

### 3.2 装甲板在线识别 + 画框（推荐）

```bash
./diagnostics/auto_aim/diagnose.sh armor-box configs/standard3.yaml
```

说明：

- 调用 `auto_aim_ui_test --show=true`
- 图像窗口里会显示目标框和重投影框
- TUI 同时显示 `targets/state/cmd` 等关键状态

### 3.3 装甲板在线识别 + 意图输出（不下发控制）

```bash
./diagnostics/auto_aim/diagnose.sh armor-intent configs/standard3.yaml
```

说明：

- 调用 `auto_aim_ui_test --show=true --no-send=true`
- 适合联调前先看目标框、重投影和期望 yaw/pitch
- 不会向下位机实际发送控制命令

### 3.4 装甲板在线识别（无 GUI）

```bash
./diagnostics/auto_aim/diagnose.sh armor-rec configs/standard3.yaml
```

说明：适合远程终端，仅看识别状态和控制输出。

### 3.5 装甲板在线调参（可导出 YAML）

```bash
./diagnostics/auto_aim/diagnose.sh armor-tune configs/standard3.yaml
```

关键交互：

- `j/k` 选参数，`-/=` 调整，`u` 切换布尔参数
- `R` 导出新配置（文件名带时间戳）
- `L` 开关日志（`logs/auto_aim_ui_*.jsonl`）

### 3.6 装甲板离线回放

```bash
./diagnostics/auto_aim/diagnose.sh armor-offline configs/demo.yaml assets/demo/demo --start-index=0 --end-index=0
```

说明：`input-prefix` 传前缀，程序读取 `<prefix>.avi` 和 `<prefix>.txt`。

### 3.7 Power Rune 识别 + 画框（离线）

```bash
./diagnostics/auto_aim/diagnose.sh rune-box configs/sentry.yaml path/to/power_rune_demo --start-index=0 --end-index=0
```

说明：

- `rune-rec` 是 `rune-box` 的别名。
- `input-prefix` 同样传前缀，程序读取 `<prefix>.avi` 和 `<prefix>.txt`。
- 程序会在窗口显示打符目标框、重投影结果和调试曲线。
- 仓库当前未附带 `power_rune_demo` 示例数据，需要自行准备离线回放素材。

### 3.8 Power Rune 调参（脚本交互）

```bash
./diagnostics/auto_aim/diagnose.sh rune-tune configs/sentry.yaml path/to/power_rune_demo --start-index=0 --end-index=0
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

### 3.9 Power Rune 在线调试（可选）

```bash
./diagnostics/auto_aim/diagnose.sh rune-online configs/standard3.yaml
./diagnostics/auto_aim/diagnose.sh rune-online-mpc configs/standard3.yaml
```

说明：依赖 `build/bin/diag/auto_buff/auto_buff_debug` 与 `build/bin/diag/auto_buff/auto_buff_debug_mpc`。

## 4. 排查建议

- `armor-box` 无目标：先确认相机取流正常、模型文件路径存在、云台姿态回传正常。
- `rune-box` 无图像：确认输入前缀对应的 `.avi/.txt` 同时存在。
- `rune-tune` 后效果变差：先备份配置，再按 `p` 核对参数并回退。
