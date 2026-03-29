# 40. `src/` 主程序入口

## 40.1 目录职责

`src/` 不实现底层算法，而是负责“把相机、姿态源、模式切换、任务模块和发送接口编排成一个完整程序”。

## 40.2 每个文件的作用

| 文件 | 作用 |
| --- | --- |
| `src/standard.cpp` | 最简串口云台自瞄主程序，走 `Runtime` 一站式链路。 |
| `src/mt_standard.cpp` | C 板版本的多线程主程序，自瞄与打符共存，检测和命令生成解耦。 |
| `src/standard_mpc.cpp` | 串口云台版本主程序，自瞄使用 `Planner` 输出带导数控制，打符也支持 `mpc_aim`。 |
| `src/auto_aim_debug_mpc.cpp` | MPC 自瞄调试程序，显示重投影结果并向 PlotJuggler 打点。 |
| `src/mt_auto_aim_debug.cpp` | 多线程自瞄调试程序，重点观察 EKF、重投影、命令与回授。 |
| `src/auto_buff_debug.cpp` | C 板版本打符调试程序，观察符观测、预测和命令。 |
| `src/auto_buff_debug_mpc.cpp` | 串口云台版本打符调试程序，输出打符 MPC 风格计划。 |
| `src/uav.cpp` | 无人机配置主程序，偏向传统检测器与 C 板通信。 |
| `src/uav_debug.cpp` | 无人机调试程序，输出传统自瞄调试信息。 |
| `src/sentry.cpp` | 哨兵主程序：主相机自瞄，丢失目标时切到多 USB + 后相机的全向感知。 |
| `src/sentry_bp.cpp` | 哨兵后备版本，主相机丢失时只用后相机辅助，不启用双 USB 全向感知。 |
| `src/sentry_debug.cpp` | 哨兵调试程序，额外显示重投影、状态估计与全向感知行为。 |
| `src/sentry_multithread.cpp` | 哨兵多线程全向感知版本，4 路 USB 相机并行推理。 |

## 40.3 这些入口大致怎么选

- 想看最短主链路：`standard.cpp`
- 想看 C 板通信和模式切换：`mt_standard.cpp`、`uav.cpp`
- 想看 MPC 决策：`standard_mpc.cpp`
- 想看哨兵/全向感知：`sentry.cpp` 或 `sentry_multithread.cpp`
- 想做在线调试：`*_debug.cpp`

