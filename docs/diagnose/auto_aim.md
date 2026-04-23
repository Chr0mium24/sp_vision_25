# Auto Aim Diagnose

本文档说明 `sp-vision-diagnose auto-aim` 的当前用法。

## 1. 基本命令

```bash
uv --project python run sp-vision-diagnose auto-aim help
```

常用动作：

- `list`
- `armor-box`
- `armor-intent`
- `armor-rec`
- `armor-tune`
- `armor-offline`
- `rune-box`
- `rune-rec`
- `rune-tune`
- `rune-online`
- `rune-online-mpc`

## 2. 查看后端状态

```bash
uv --project python run sp-vision-diagnose auto-aim list
```

它会列出 Python 入口和相关 C++ 后端是否存在。

## 3. 装甲板联调

### 画框联调

```bash
uv --project python run sp-vision-diagnose auto-aim armor-box configs/standard3.yaml
```

适合在线查看：

- 检测框
- 重投影结果
- 目标状态与控制输出

当前调试图默认输出到：

- `logs/auto_aim/patterns`
- `logs/auto_aim/imgs`

### 无窗口识别

```bash
uv --project python run sp-vision-diagnose auto-aim armor-rec configs/standard3.yaml
```

适合远程终端或无图形环境。

### 意图/状态观察

```bash
uv --project python run sp-vision-diagnose auto-aim armor-intent configs/standard3.yaml
```

### 在线调参

```bash
uv --project python run sp-vision-diagnose auto-aim armor-tune configs/standard3.yaml
```

常见交互：

- `j/k`：切换参数
- `-/=`：调整数值
- `u`：切换布尔参数
- `R`：导出新配置
- `L`：记录日志

## 4. 离线回放

```bash
uv --project python run sp-vision-diagnose auto-aim armor-offline \
    configs/demo.yaml \
    assets/demo/demo \
    --start-index=0 \
    --end-index=0
```

输入前缀会映射到同名前缀的媒体与姿态文件。

## 5. Power Rune

### 离线画框

```bash
uv --project python run sp-vision-diagnose auto-aim rune-box \
    configs/sentry.yaml \
    assets/demo/power_rune_demo \
    --start-index=0 \
    --end-index=0
```

`rune-rec` 是它的别名入口。

### 离线调参

```bash
uv --project python run sp-vision-diagnose auto-aim rune-tune \
    configs/sentry.yaml \
    assets/demo/power_rune_demo \
    --start-index=0 \
    --end-index=0
```

常用交互：

- `y [num]`：设置 `yaw_offset`
- `i [num]`：设置 `pitch_offset`
- `f [num]`：设置 `fire_gap_time`
- `t [num]`：设置 `predict_time`
- `r`：按当前参数重跑
- `p`：打印当前参数
- `q`：退出

### 在线模式

```bash
uv --project python run sp-vision-diagnose auto-aim rune-online configs/standard3.yaml
uv --project python run sp-vision-diagnose auto-aim rune-online-mpc configs/standard3.yaml
```

这两条命令依赖保留中的 `build/bin/diag/auto_buff/*` 后端。

## 6. 常见问题

- `armor-box` 无目标：先检查相机链路、模型路径和云台回传
- `rune-box` 无图像：确认输入前缀对应的数据文件完整
- 调参后效果更差：先导出新配置，再和原始 YAML 比较回退
