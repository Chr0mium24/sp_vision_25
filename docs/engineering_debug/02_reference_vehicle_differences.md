# 正常车与异常车参考工程差异

本文只比较你给出的两套参考电控工程。

- 异常车：`references/25-season-Hero-main`
- 正常车：`references/WeiyiHuang25-25_3v3_final_exam_infantary-settings`

结论先说：

这两套工程不是“参数略有不同”，而是 `pitch / roll / 符号约定` 已经发生了工程分叉。

## 1. 最关键的三个差异

## 1.1 `Robo_USB.c` 收包时对 `pitch` 的处理不同

文件：

- 异常车：`references/25-season-Hero-main/SEML/App/Robo/Robo_USB.c`
- 正常车：`references/WeiyiHuang25-25_3v3_final_exam_infantary-settings/SEML/App/Robo/Robo_USB.c`

差异：

- 正常车在 `receive()` 里会做 `received_packet.pitch = -received_packet.pitch;`
- 异常车不会

这意味着：

- 同样一份视觉命令，下位机读到的 `pitch` 方向可能已经不同

工程含义：

- 如果视觉认为“抬头是正”
- 但下位机内部控制刚好约定“低头是正”
- 那么缺一次翻转或多一次翻转，都会导致自瞄方向反掉

## 1.2 `Robo_AA.c` 使用的“当前俯仰轴”不同

文件：

- 异常车：`references/25-season-Hero-main/SEML/App/Robo/Robo_AA.c`
- 正常车：`references/WeiyiHuang25-25_3v3_final_exam_infantary-settings/SEML/App/Robo/Robo_AA.c`

差异：

- 异常车：`st.current_pitch = AHRS.euler_angle.roll;`
- 正常车：`st.current_pitch = AHRS.euler_angle.pitch;`

这意味着：

- 异常车把“枪管仰角”映射到了 IMU 的 `roll`
- 正常车使用标准 `pitch`

工程含义：

- 两辆车的 C 板安装方向或内部坐标约定不一样
- 如果视觉仍然统一只发 `pitch` 字段，那么异常车需要在板端自己做“pitch 字段 -> 实际 roll 轴”的适配

## 1.3 `Robo_gimbal.c` 闭环控制读的 IMU 轴不同

文件：

- 异常车：`references/25-season-Hero-main/SEML/App/Robo/Robo_gimbal.c`
- 正常车：`references/WeiyiHuang25-25_3v3_final_exam_infantary-settings/SEML/App/Robo/Robo_gimbal.c`

差异：

- 异常车 pitch 闭环对比的是 `gimbal.imu->roll`
- 正常车 pitch 闭环对比的是 `gimbal.imu->pitch`

这意味着：

- 异常车整条“俯仰控制链”从反馈到闭环都走的是 `roll`
- 正常车整条链路走的是 `pitch`

所以你的问题非常像：

- 视觉输出没问题
- 但车端把 `pitch` 解释成了另一条轴

## 2. 反馈回视觉的字段也不同

异常车 `Robo_AA.c` 的回传方式大致是：

- `send.yaw = AHRS.euler_angle.yaw`
- `send.pitch = AHRS.euler_angle.roll`
- `send.roll = 0`

正常车则更接近标准语义：

- `send.yaw = AHRS.euler_angle.yaw`
- `send.pitch = AHRS.euler_angle.pitch`
- `send.roll = AHRS.euler_angle.roll`

工程含义：

- 即使视觉侧的协议结构没变
- 两辆车回来的 `feedback_pitch` 含义也未必相同

所以你导出的日志里，如果出现：

- 正常车 `feedback_pitch` 跟着命令走
- 异常车 `feedback_roll` 跟着命令走

这并不是视觉算法错误，而是板端轴定义不同。

## 3. 为什么 `diagnose.sh control` 会“看起来能连上，但自瞄不正常”

`diagnostics/gimbal/diagnose.sh control` 本质上只做这些事：

1. 构造 `VisionToGimbal`
2. 发 `yaw / pitch / fire / fric_on`
3. 读取车回传的状态

它不负责判断：

1. 下位机到底把 `pitch` 映射到哪个物理轴
2. 板端有没有再翻一次符号
3. 板端回传的 `feedback_pitch` 是不是真正的“枪管仰角”

所以会出现一种很典型的现象：

- 协议通了
- 包也能收发
- 但自瞄就是不对

这说明问题不在“连通性”，而在“同一字段是否被双方解释成了同一个物理量”。

## 4. 你应该如何用日志验证这个判断

### 4.1 验证轴映射

你只改 `pitch`，不改 `yaw`，看：

- `fb_pitch`
- `fb_roll`

如果异常车上：

- `fb_pitch` 几乎不变
- `fb_roll` 跟着变化

就说明异常车的仰角链路走的是 `roll`。

### 4.2 验证符号约定

你给一个小的正向 `pitch` 命令，看反馈是：

- 同方向增加
- 还是反方向减少

如果反向，就说明某一层多翻或少翻了一次。

### 4.3 验证视觉是否无罪

先用：

```bash
./build/bin/diag/auto_aim/auto_aim_ui_test configs/standard3.yaml --no-send=true --duration-ms=10000 --log-jsonl=logs/intent_only.jsonl
```

如果这份日志里的：

- `command_pitch`
- `send_pitch`

都稳定合理，那视觉侧已经足够清楚了。

## 5. 工程上最稳的处理方式

不要先改视觉算法。

优先原则是：

1. 保持视觉协议字段含义统一
2. 在板端做本车专属的轴向适配
3. 用日志证明适配前后差异

更具体地说：

- 如果异常车物理上确实是“仰角走 roll”
- 那就让异常车板端明确承担 `pitch 字段 -> roll 控制轴` 的转换责任
- 视觉仍然坚持发标准 `yaw / pitch`

这样后续：

- 正常车和异常车可以共用视觉程序
- 差异只留在各自电控工程里

## 6. 这两套参考工程里哪些文件最值得先对比

优先级从高到低：

1. `Robo_USB.c`
2. `Robo_AA.c`
3. `Robo_gimbal.c`
4. `Robo_Control.c`
5. `Robo_Shoot.c`

前 3 个决定了“视觉命令 -> 物理云台 -> 回传姿态”的主链路。

后 2 个主要影响：

- 发射模式是否真的生效
- 其他控制逻辑是否覆盖了 `AA_Task()` 的输出

## 7. 一句话判断

你当前最像的问题不是“视觉不会瞄”，而是：

- 正常车把 `pitch` 当 `pitch`
- 异常车把 `pitch` 当“对外叫 pitch、对内其实走 roll 的枪管仰角”

只要日志对出来这一点，后面就可以按工程适配处理，而不是继续怀疑视觉算法。
