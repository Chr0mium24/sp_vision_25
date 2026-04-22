# Gimbal Diagnose 脚本说明

本文档说明 `sp-vision-diagnose gimbal` 的使用方法和判读规则，用于云台串口链路联调。

## 1. 基本用法

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal <action> [config.yaml] [extra args...]
```

- 默认配置文件：`configs/standard3.yaml`
- `extra args` 会透传给底层测试程序（例如 `--ports=...`、`--duration-ms=...`）。

## 2. Action 对照

| action | 作用 | 关键判定 | 示例 |
| --- | --- | --- | --- |
| `quick` | 发包+收包快速检查（3s） | `bytes>0` 且 `ok49>0` | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal quick` |
| `rxonly` | 只收包检查（不发控制） | `ok49>0` | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal rxonly` |
| `proto` | 严格协议检查（要求必须收到有效帧） | 结束码为 0 且有有效帧，否则失败 | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal proto` |
| `probe` | 原始字节流统计（不依赖协议） | 看 `bytes/drop/hdr` 变化 | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal probe` |
| `probe-raw` | 原始十六进制抽样 | 观察是否出现 `5A` 帧头 | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal probe-raw` |
| `scan` | 多端口扫描 | 找到哪个口有有效帧 | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal scan` |
| `snapshot` | 单次读姿态快照 | `valid=1` | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal snapshot` |
| `watch` | 持续读姿态（只读） | `RX stats good` 持续增长 | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal watch` |
| `control` | 交互控制模式 | 键控后反馈变化 | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal control` |
| `script-control` | 无输入脚本控制（5s） | 回传姿态/统计有变化 | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal script-control` |
| `port-info` | 打印配置端口和 udev 信息 | 端口存在且与设备 ID 对应 | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal port-info` |
| `help` | 显示帮助 | - | `UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal help` |

## 3. 推荐排查顺序

当你遇到“有字节但没有有效帧”或“数据疑似全 0”时，建议按下面顺序执行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal port-info
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal probe-raw --ports=/dev/ttyACM0,/dev/ttyACM1
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal proto --ports=/dev/ttyACM0,/dev/ttyACM1
UV_CACHE_DIR=/tmp/uv-cache uv run sp-vision-diagnose gimbal quick --ports=/dev/ttyACM0,/dev/ttyACM1
```

## 4. 输出判读

- `bytes>0` 且 `ok49=0`：有字节流，但不是目标协议帧。
- `last_chunk` 长期全 `00`：通常是端口对了但数据源异常（电控发送侧未形成有效帧，或读到非协议通道）。
- `ok49>0`：协议已对上，链路可用。
- `proto` 失败：当前端口/固件状态下没有有效回传帧。

## 5. 相关说明

- 非英雄协议默认为：`0xA5` 下发，`0x5A` 回传（49B 回传帧）。
- 应用侧与串口协议侧统一约定为“pitch 抬头为正”。
