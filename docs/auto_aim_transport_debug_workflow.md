# 自瞄传输链路排障说明

本文档面向“不改视觉算法，只排查视觉到电控之间的传输 / 轴向 / 协议差异”的场景。

适用问题：

- `diagnostics/gimbal/diagnose.sh control` 能连上，但车不按预期动
- 视觉算法本身跑通，但上车后自瞄不正常
- 一辆车正常，另一辆车不正常，想用工程方式做对比

不关注的内容：

- YOLO、跟踪器、解算器内部算法细节
- 电控控制器参数整定细节

## 1. 先说结论

当前仓库已经有两类可直接导出的日志：

1. 云台纯传输日志
   文件：`diagnostics/gimbal/gimbal_ui_test.cpp`
   导出格式：CSV
   作用：看“发了什么命令、回了什么姿态、CRC/短包/坏帧统计如何”

2. 自瞄端到端日志
   文件：`diagnostics/auto_aim/auto_aim_ui_test.cpp`
   导出格式：JSONL
   作用：看“视觉输出命令、加上 offset 后真正发送了什么、回授姿态是多少、目标状态是否稳定”

所以你完全可以把正常车和异常车各导出一份，再让我做对比分析。

## 2. 视觉侧数据流经过哪些文件

这里按最短主链路说明。

### 2.1 入口层

主程序入口：

- `src/standard.cpp`
- 调试入口：`diagnostics/auto_aim/auto_aim_ui_test.cpp`

这两者的职责都一样：

1. 从相机读图
2. 从云台读姿态回授
3. 调用 `auto_aim::Runtime::step(...)`
4. 把输出的 yaw/pitch/fire 打包发给下位机

### 2.2 运行时总入口

文件：

- `tasks/auto_aim/auto_aim_runtime.cpp`

关键函数：

- `auto_aim::Runtime::step`

调用顺序是：

1. `solver_.set_R_gimbal2world(input.q_gimbal2world)`
2. `yolo_.detect(...)`
3. `tracker_.track(...)`
4. `aimer_.aim(...)`

虽然这里经过了视觉算法，但对于你的问题，真正重要的是它最后产出的 `io::Command`。

### 2.3 视觉命令结构

文件：

- `io/command.hpp`

核心结构：

- `io::Command`

关键字段：

- `control`
- `shoot`
- `yaw`
- `pitch`

这就是视觉层最终准备交给电控的“抽象命令”。

### 2.4 从抽象命令变成串口包

文件：

- `diagnostics/auto_aim/auto_aim_ui_test.cpp`
- `src/standard.cpp`
- `io/gimbal/gimbal.hpp`
- `io/gimbal/gimbal.cpp`
- `io/gimbal/gimbal_protocol.hpp`

关键函数：

- `io::Gimbal::send(...)`
- `io::Gimbal::send(io::VisionToGimbal)`
- `io::Gimbal::send_packet(...)`
- `refresh_crc16(...)`

实际链路是：

1. `auto_aim_ui_test.cpp` 或 `standard.cpp` 拿到 `io::Command`
2. 组装成 `io::VisionToGimbal plan`
3. `gimbal.send(plan)`
4. 自动补 `0xA5` 包头和 CRC16
5. 串口写出

协议上：

- 下发包头：`0xA5`
- 回传包头：`0x5A`

## 3. 电控侧数据流经过哪些文件

这里按参考工程的 USB 自瞄链路说明。

### 3.1 视觉命令进入电控

文件：

- `references/.../SEML/App/Robo/Robo_USB.c`

关键函数：

- `receive()`
- `get_received_packet()`

逻辑：

1. USB 收到 `0xA5` 开头的数据
2. `memcpy` 到全局 `received_packet`
3. 某些车会在这里做 `pitch` 符号修正
4. 后续 `AA_Task` 通过 `get_received_packet()` 取走

### 3.2 自瞄任务读取视觉命令

文件：

- `references/.../SEML/App/Robo/Robo_AA.c`

关键函数：

- `AA_Task()`

逻辑：

1. `auto_aiming_control.receive = get_received_packet();`
2. 取出 `yaw / pitch / fire / fric_on`
3. 写入电控内部控制状态
4. 计算后再生成回传包 `auto_aiming_control.send`
5. `set_send_packet(...)`

### 3.3 电控闭环控制云台

文件：

- `references/.../SEML/App/Robo/Robo_gimbal.c`

关键函数：

- `Gimbal_Task()`

逻辑：

1. 读取 `Set_Gimbal_Yaw_Angle`
2. 读取 `Set_Gimbal_Pitch_Angle`
3. 读取 IMU 姿态
4. 做 yaw/pitch 闭环
5. 发电机控制量

### 3.4 电控回传姿态给视觉

文件：

- `references/.../SEML/App/Robo/Robo_AA.c`
- `references/.../SEML/App/Robo/Robo_USB.c`
- `io/gimbal/gimbal.cpp`

逻辑：

1. 电控把当前姿态写到 `auto_aiming_control.send`
2. `set_send_packet(...)`
3. `Robo_USB.c::send(...)` 打成 `0x5A` 包
4. 视觉侧 `io::Gimbal::read_thread()` 读到
5. `handle_rx_packet(...)` 更新：
   - `gimbal.state()`
   - `gimbal.q(...)`
   - `rx_stats()`

这就是完整闭环。

## 4. 你现在最该关注的文件

如果你的目标是排查“正常车 vs 异常车”的中间传输差异，优先看这些文件：

- 视觉侧发包：`diagnostics/auto_aim/auto_aim_ui_test.cpp`
- 视觉侧串口协议：`io/gimbal/gimbal.cpp`
- 电控侧收包：`references/.../Robo_USB.c`
- 电控侧消费视觉命令：`references/.../Robo_AA.c`
- 电控侧姿态闭环：`references/.../Robo_gimbal.c`

## 5. 当前已经存在的日志/导出能力

### 5.1 纯串口链路日志

命令：

```bash
./diagnostics/gimbal/diagnose.sh control configs/standard3.yaml --duration-ms=10000 --log-csv=logs/gimbal_good.csv
```

对应程序：

- `diagnostics/gimbal/gimbal_ui_test.cpp`

CSV 里主要有：

- `cmd_tracking/cmd_fric/cmd_fire`
- `cmd_yaw_rad/cmd_pitch_rad`
- `fb_yaw_rad/fb_pitch_rad/fb_roll_rad`
- `good/crc_fail/short_read/bad_header/reconnect`

这个日志回答的问题是：

- 视觉到底有没有发命令
- 电控到底有没有回包
- 回包 CRC/长度/包头是否正常
- 发出的 yaw/pitch 和回来的 yaw/pitch 是不是方向相反

### 5.2 自瞄端到端日志

命令：

```bash
./diagnostics/auto_aim/diagnose.sh armor-log configs/standard3.yaml --duration-ms=10000 --log-jsonl=logs/autoaim_good.jsonl
```

对应程序：

- `diagnostics/auto_aim/auto_aim_ui_test.cpp`

JSONL 里主要有：

- `tracker_state`
- `target_count`
- `command_yaw/command_pitch`
- `send_yaw/send_pitch`
- `feedback_yaw/feedback_pitch`
- `delta_yaw/delta_pitch`
- `armor_confidence`
- `aim_point_valid`

这个日志回答的问题是：

- 视觉是否持续稳定地产生命令
- 视觉发出的命令与云台反馈差多少
- 偏差是常量、抖动、还是符号反了
- 是“算法输出没问题但发出去不对”，还是“发出去对但车端执行不对”

### 5.3 通用文本日志

所有程序还会写：

- `logs/*.log`

来源：

- `tools/logger.cpp`

里面常见信息：

- `Gimbal read_thread started`
- CRC fail
- short_read
- reconnect
- tracker warning

### 5.4 单次快照

命令：

```bash
./diagnostics/gimbal/diagnose.sh snapshot configs/standard3.yaml
```

作用：

- 快速看当前有没有有效回传
- 快速确认 `yaw/pitch/roll` 是否有值
- 快速确认 `good/crc_fail/short_read/bad_header`

## 6. 正常车和异常车最值得先对比的地方

你给的两个参考工程，已经能看出几个很强的差异点。

### 6.1 `Robo_USB.c` 对 `pitch` 的处理不同

正常工程：

- `references/WeiyiHuang25-25_3v3_final_exam_infantary-settings/SEML/App/Robo/Robo_USB.c`

异常工程：

- `references/25-season-Hero-main/SEML/App/Robo/Robo_USB.c`

关键差异：

- 正常工程在 `receive()` 里有：
  - `received_packet.pitch = -received_packet.pitch;`
- 异常工程没有这一步

这意味着两辆车对“视觉协议里的 pitch 正方向”理解不一致。

### 6.2 `Robo_gimbal.c` 的闭环轴不同

正常工程：

- pitch 闭环使用 `gimbal.imu->pitch`

异常工程：

- pitch 闭环实际使用 `gimbal.imu->roll`

这说明异常车的枪管仰角轴是按 `roll` 在控制，不是标准的 `pitch`。

### 6.3 `Robo_AA.c` 的回传也不同

正常工程：

- 回传 `send.pitch = AHRS.euler_angle.pitch`
- 回传真实 `yaw_vel/pitch_vel`
- 回传 `yaw_odom/pitch_odom`
- 回传 `robot_id`

异常工程：

- 回传 `send.pitch = AHRS.euler_angle.roll`
- `send.roll = 0`
- `yaw_vel/pitch_vel/odom` 基本都置零

这会导致视觉侧看到的反馈语义和标准步兵车不一致。

## 7. 推荐的工程化排查顺序

不要一上来跑整套自瞄。按下面顺序做。

### 第一步：先验证纯传输

正常车、异常车都跑：

```bash
./diagnostics/gimbal/diagnose.sh snapshot configs/standard3.yaml
./diagnostics/gimbal/diagnose.sh proto configs/standard3.yaml
./diagnostics/gimbal/diagnose.sh control configs/standard3.yaml --duration-ms=10000 --log-csv=logs/gimbal_xxx.csv
```

你把两辆车的 `gimbal_*.csv` 给我，我先看：

- 包有没有进来
- CRC 是否稳定
- 命令发出后反馈是否同向
- pitch 到底是走 pitch 还是 roll 语义

### 第二步：再验证整套自瞄

正常车、异常车都跑：

```bash
./diagnostics/auto_aim/diagnose.sh armor-log configs/standard3.yaml --duration-ms=10000 --log-jsonl=logs/autoaim_xxx.jsonl
```

如果怕真实发包影响安全，可以先跑：

```bash
./diagnostics/auto_aim/diagnose.sh armor-intent configs/standard3.yaml
```

或者直接：

```bash
./build/bin/diag/auto_aim/auto_aim_ui_test configs/standard3.yaml --no-send=true --duration-ms=10000 --log-jsonl=logs/intent_xxx.jsonl
```

这样可以先看视觉输出和反馈差，不真的控制云台。

### 第三步：把四份文件给我

至少给我：

1. 正常车 `logs/gimbal_good.csv`
2. 异常车 `logs/gimbal_bad.csv`
3. 正常车 `logs/autoaim_good.jsonl`
4. 异常车 `logs/autoaim_bad.jsonl`

最好再加：

5. 对应时段的 `logs/*.log`
6. 两车使用的 YAML 配置

## 8. 我会怎么对比你给我的日志

拿到日志后，我会按下面几类问题判断：

1. 视觉命令是否稳定
   看 `command_yaw/command_pitch`

2. 真正下发是否稳定
   看 `send_yaw/send_pitch/send_tracking`

3. 云台回授是否跟着走
   看 `feedback_yaw/feedback_pitch`

4. 方向是否反了
   看 `delta_yaw/delta_pitch` 的符号

5. 协议是否脏
   看 `crc_fail/short_read/bad_header/reconnect`

6. 是否存在“视觉没问题，但电控轴语义不同”
   结合你两份 `Robo_USB.c / Robo_AA.c / Robo_gimbal.c` 的实现差异判断

## 9. 现在最像什么问题

从当前两份参考工程代码看，异常车最像是“电控侧轴语义和标准步兵车不一致”，不是视觉算法问题。

强信号有三个：

1. `Robo_USB.c` 对 `pitch` 是否取反不同
2. `Robo_gimbal.c` 一个用 `imu->pitch`，一个用 `imu->roll`
3. `Robo_AA.c` 一个按标准 `pitch` 回传，一个把枪管仰角塞在 `pitch<-roll`

这类问题最适合用本文档里的“同一命令、两车导出、做差分分析”的方法，不要靠肉眼猜。
