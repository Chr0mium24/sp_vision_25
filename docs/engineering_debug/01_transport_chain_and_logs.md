# 自瞄传输链路与日志导出

本文只讲工程链路，不讲视觉算法细节。

## 1. 你现在要排查的到底是什么

你描述的问题是：

- 视觉算法本身能跑
- `diagnostics/gimbal/diagnose.sh control` 上车后不能正常完成自瞄
- 一辆车正常，另一辆车不正常

这类问题通常不在 `YOLO / 跟踪 / 解算器` 本身，而在下面几类环节：

1. 视觉是否真的持续发出了正确的 `yaw / pitch / fire`
2. 电控是否真的收到了这份命令
3. 电控是否把 `pitch` 当成了同一个轴
4. 电控回传给视觉的姿态是否和视觉假设一致
5. 协议字段、包头、CRC、符号约定是否一致

## 2. 自瞄主链路经过哪些文件

最短链路可以按“视觉侧”和“电控侧”分开看。

### 2.1 视觉侧

#### 入口文件

- `src/standard.cpp`
- `diagnostics/auto_aim/auto_aim_ui_test.cpp`

这两个入口都在做同一件事：

1. 从相机读图
2. 从云台拿当前姿态反馈
3. 调 `auto_aim::Runtime::step(...)`
4. 把结果打成协议包发给车

#### 运行时总入口

文件：

- `tasks/auto_aim/auto_aim_runtime.cpp`

核心函数：

- `auto_aim::Runtime::step`

调用顺序是：

1. `solver_.set_R_gimbal2world(...)`
2. `yolo_.detect(...)`
3. `tracker_.track(...)`
4. `aimer_.aim(...)`

对工程排障来说，最重要的不是中间算法细节，而是最后产出的 `io::Command`。

#### “不同远近，pitch 不一样”是不是已经实现了

是，已经实现了。

负责这件事的不是 `Robo_USB.c` 或 `gimbal.send(...)`，而是视觉侧的：

- `tasks/auto_aim/solver.cpp`
- `tasks/auto_aim/aimer.cpp`
- `tools/trajectory.cpp`

它们分工如下：

1. `Solver::solve(...)`
   把图像里的装甲板解成三维位置，也就是目标相对云台/世界坐标系的 `xyz`
2. `Aimer::aim(...)`
   先选当前要打的那块装甲板，再根据目标三维位置、子弹速度、延迟时间做预测
3. `tools::Trajectory`
   根据
   - 子弹初速 `v0`
   - 水平距离 `d`
   - 高差 `h`
   求出
   - 飞行时间 `fly_time`
   - 对应的弹道 `pitch`

所以“目标越远，pitch 往往越大”这件事，在工程里不是靠经验写死，而是靠弹道解算实时算出来的。

现在这份实现的特点是：

- 已经考虑重力下坠
- 已经考虑子弹飞行时间
- 已经用飞行时间反过来迭代预测目标未来位置
- 当前 `tools::Trajectory` 默认是不考虑空气阻力的简化弹道

换句话说，现在有“空间解算 + 基础弹道补偿”，不是单纯拿 `atan2(z, d)` 直接当 pitch 发出去。

#### 抽象命令结构

文件：

- `io/command.hpp`

关键字段：

- `control`
- `shoot`
- `yaw`
- `pitch`

可以把它理解成“视觉已经算好的标准命令”。

#### 从命令变成串口包

文件：

- `io/gimbal/gimbal.cpp`
- `io/gimbal/gimbal_protocol.hpp`

关键函数：

- `io::Gimbal::send(...)`
- `io::Gimbal::send(io::VisionToGimbal)`
- `io::Gimbal::send_packet(...)`

这里会做三件事：

1. 把 `yaw / pitch / fire / tracking` 写进 `VisionToGimbal`
2. 自动补上下行包头 `0xA5`
3. 自动补 `CRC16`

所以你平时不用自己手写协议包，真正的发包出口就是这里。

#### 从车上收反馈回来

文件：

- `io/gimbal/gimbal.cpp`

关键函数：

- `io::Gimbal::read_thread()`
- `io::Gimbal::handle_rx_packet(...)`
- `io::Gimbal::state()`
- `io::Gimbal::q(...)`
- `io::Gimbal::rx_stats()`

它们负责：

1. 后台读取上位机回包
2. 检查包头 `0x5A`
3. 检查 `CRC16`
4. 更新当前姿态反馈
5. 统计好包、坏包、短包、重连次数

### 2.2 电控侧

#### USB/串口接收入口

参考工程文件：

- `references/25-season-Hero-main/SEML/App/Robo/Robo_USB.c`
- `references/WeiyiHuang25-25_3v3_final_exam_infantary-settings/SEML/App/Robo/Robo_USB.c`

关键函数：

- `receive()`
- `get_received_packet()`

作用：

1. 收到视觉发来的 `0xA5` 包
2. 拷贝到 `received_packet`
3. 某些工程会在这里对 `pitch` 做符号翻转
4. 给 `AA_Task()` 使用

#### 自瞄任务消费视觉命令

文件：

- `references/.../Robo_AA.c`

关键函数：

- `AA_Task()`

作用：

1. `auto_aiming_control.receive = get_received_packet();`
2. 取出 `yaw / pitch / fire / fric_on`
3. 写到 `Set_Gimbal_Yaw_Angle / Set_Gimbal_Pitch_Angle`
4. 组织回传给视觉的状态包

#### 云台闭环执行

文件：

- `references/.../Robo_gimbal.c`

关键函数：

- `Gimbal_Task()`

作用：

1. 读取 `Set_Gimbal_Yaw_Angle`
2. 读取 `Set_Gimbal_Pitch_Angle`
3. 结合 IMU 做闭环
4. 输出给云台电机

#### 电控回传状态

文件：

- `references/.../Robo_AA.c`
- `references/.../Robo_USB.c`
- `io/gimbal/gimbal.cpp`

路径：

1. `AA_Task()` 组 `send_packet`
2. `Robo_USB.c::send(...)` 发出 `0x5A`
3. `io::Gimbal::read_thread()` 收到
4. `handle_rx_packet(...)` 更新上位机反馈状态

## 3. 现成的日志能力有哪些

当前仓库已经有三层日志。

### 3.1 通用程序日志

文件：

- `tools/logger.cpp`

作用：

- 记录程序启动、异常、串口线程启动、错误信息

你可以在 `logs/*.log` 里看到这类信息，例如：

- 串口是否打开成功
- `read_thread` 是否启动
- 是否出现异常或重连

### 3.2 云台纯传输日志

入口：

- `diagnostics/gimbal/gimbal_ui_test.cpp`
- `diagnostics/gimbal/diagnose.sh control`

导出格式：

- `CSV`

典型字段：

- `cmd_tracking`
- `cmd_fric`
- `cmd_fire`
- `cmd_yaw_rad`
- `cmd_pitch_rad`
- `fb_yaw_rad`
- `fb_pitch_rad`
- `fb_roll_rad`
- `good`
- `crc_fail`
- `short_read`
- `bad_header`
- `reconnect`

这个日志回答的问题是：

1. 到底有没有发命令
2. 到底有没有回包
3. `pitch / yaw` 方向有没有反
4. 协议包有没有被破坏

### 3.3 自瞄端到端日志

入口：

- `diagnostics/auto_aim/auto_aim_ui_test.cpp`
- `diagnostics/auto_aim/diagnose.sh armor-log`

导出格式：

- `JSONL`

典型字段：

- `tracker_state`
- `target_count`
- `command_yaw`
- `command_pitch`
- `send_yaw`
- `send_pitch`
- `feedback_yaw`
- `feedback_pitch`
- `feedback_roll`
- `delta_yaw`
- `delta_pitch`
- `armor_confidence`
- `aim_point_valid`

这个日志回答的问题是：

1. 视觉是否稳定地产生命令
2. offset 之后真正发给车的值是什么
3. 车回来的姿态是不是沿着预期方向变化
4. 目标是否稳定

如果你专门想看“同一个目标远近变化时，pitch 有没有跟着变”，最关键的是这几列：

- `command_pitch`
- `send_pitch`
- `feedback_pitch`
- `aim_x`
- `aim_y`
- `aim_z`

看法很简单：

- 当 `aim_x/aim_y` 对应的目标距离变远时
- `command_pitch` 不应该一直恒定
- 它应当随着距离和高差变化

如果 `aim_x/aim_y/aim_z` 明显在变，但 `command_pitch` 基本不变，那才说明弹道解算链路可能有问题。

## 4. 你应该怎么导出“正常车 vs 异常车”日志

### 4.1 先导出纯传输日志

正常车：

```bash
./diagnostics/gimbal/diagnose.sh control configs/standard3.yaml --duration-ms=10000 --log-csv=logs/gimbal_good.csv
```

异常车：

```bash
./diagnostics/gimbal/diagnose.sh control configs/standard3.yaml --duration-ms=10000 --log-csv=logs/gimbal_bad.csv
```

### 4.2 再导出端到端自瞄日志

正常车：

```bash
./diagnostics/auto_aim/diagnose.sh armor-log configs/standard3.yaml --duration-ms=10000 --log-jsonl=logs/autoaim_good.jsonl
```

异常车：

```bash
./diagnostics/auto_aim/diagnose.sh armor-log configs/standard3.yaml --duration-ms=10000 --log-jsonl=logs/autoaim_bad.jsonl
```

### 4.3 如果你担心真的发到车上

可以先导出“视觉意图日志”：

```bash
./build/bin/diag/auto_aim/auto_aim_ui_test configs/standard3.yaml --no-send=true --duration-ms=10000 --log-jsonl=logs/intent_only.jsonl
```

这个模式下：

- 视觉照样算
- 但不真正下发给车

它可以先证明“视觉输出是否正常”，把问题和电控链路分开。

## 5. 你拿到日志后优先看什么

### 5.1 先看云台 CSV

先看这几列：

- `cmd_yaw_rad`
- `cmd_pitch_rad`
- `fb_yaw_rad`
- `fb_pitch_rad`
- `fb_roll_rad`
- `good`
- `crc_fail`

工程上最常见的现象是：

1. `cmd_pitch` 在变，但 `fb_pitch` 不变，`fb_roll` 在变
2. `cmd_pitch` 增大，`fb_pitch` 反向减小
3. `good` 很低、`crc_fail` 很高

这三种分别通常对应：

1. 轴映射错了
2. 符号约定反了
3. 协议链路本身不稳定

### 5.2 再看自瞄 JSONL

先看：

- `command_yaw`
- `command_pitch`
- `send_yaw`
- `send_pitch`
- `feedback_yaw`
- `feedback_pitch`
- `feedback_roll`
- `delta_yaw`
- `delta_pitch`

如果：

- `command_pitch` 是正常的
- `send_pitch` 也是正常的
- 但 `feedback_pitch` 不跟着变

那问题就不在视觉，而在发包后的链路。

## 6. 你后面给我哪些文件最有价值

优先给这四个：

- `logs/gimbal_good.csv`
- `logs/gimbal_bad.csv`
- `logs/autoaim_good.jsonl`
- `logs/autoaim_bad.jsonl`

如果你还想再进一步缩小范围，再补一个：

- `logs/intent_only.jsonl`

这样我就能分三层做判断：

1. 视觉有没有算对
2. 视觉有没有发对
3. 电控有没有按同一套坐标系去执行和回传
