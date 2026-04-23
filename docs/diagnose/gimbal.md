# Gimbal Diagnose

本文档说明 `sp-vision-diagnose gimbal` 的当前用法和常见判读方式。

## 1. 基本命令

```bash
uv --project python run sp-vision-diagnose gimbal help
```

常用动作：

- `list`
- `quick`
- `rxonly`
- `proto`
- `probe`
- `probe-raw`
- `scan`
- `snapshot`
- `watch`
- `control`
- `script-control`
- `axis`
- `manual-axis`
- `port-info`

## 2. 先看端口信息

```bash
uv --project python run sp-vision-diagnose gimbal port-info
uv --project python run sp-vision-diagnose gimbal port-info configs/standard3.yaml
```

作用：

- 打印配置里的串口路径
- 辅助确认 udev 规则是否落到预期设备

## 3. 推荐排查顺序

当你怀疑串口不通、协议不匹配或数据异常时，建议按顺序执行：

```bash
uv --project python run sp-vision-diagnose gimbal port-info
uv --project python run sp-vision-diagnose gimbal probe-raw configs/standard3.yaml
uv --project python run sp-vision-diagnose gimbal proto configs/standard3.yaml
uv --project python run sp-vision-diagnose gimbal quick configs/standard3.yaml
```

如果需要手动指定多个端口：

```bash
uv --project python run sp-vision-diagnose gimbal probe-raw configs/standard3.yaml --ports=/dev/ttyACM0,/dev/ttyACM1
```

## 4. 动作说明

### `quick`

```bash
uv --project python run sp-vision-diagnose gimbal quick configs/standard3.yaml
```

快速发包并读回传，适合做 3 秒联通检查。

### `rxonly`

```bash
uv --project python run sp-vision-diagnose gimbal rxonly configs/standard3.yaml
```

只读回传，不发控制。

### `proto`

```bash
uv --project python run sp-vision-diagnose gimbal proto configs/standard3.yaml
```

严格按协议判定是否收到了有效帧。

### `snapshot` / `watch`

```bash
uv --project python run sp-vision-diagnose gimbal snapshot configs/standard3.yaml
uv --project python run sp-vision-diagnose gimbal watch configs/standard3.yaml --duration-ms=3000
```

用于读取当前姿态和统计信息。

### `control` / `script-control`

```bash
uv --project python run sp-vision-diagnose gimbal control configs/standard3.yaml
uv --project python run sp-vision-diagnose gimbal script-control configs/standard3.yaml --no-input
```

用于交互控制或脚本化控制。

### `axis` / `manual-axis`

```bash
uv --project python run sp-vision-diagnose gimbal axis configs/standard3.yaml
uv --project python run sp-vision-diagnose gimbal manual-axis configs/standard3.yaml
```

用于轴向检查和人工确认方向。

## 5. 判读建议

- `bytes > 0` 且 `ok49 = 0`
  - 有字节流，但不是目标协议帧
- `ok49 > 0`
  - 协议已对上，链路基本可用
- `last_chunk` 长期全 `00`
  - 端口可能对了，但发送侧没有形成有效数据
- `proto` 失败
  - 当前端口、当前固件或当前协议设置下没有有效回传

## 6. 约定

- 当前非英雄协议默认使用 `0xA5` 下发、`0x5A` 回传
- 应用侧约定 `pitch` 抬头为正
