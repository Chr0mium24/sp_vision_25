# 10. `tools/` 目录

## 10.1 目录职责

`tools/` 是整个工程的基础设施层。它不关心兵种和任务，只提供“算法、线程、日志、可视化、录制、数学”这些共用能力。

## 10.2 文件职责

| 文件 | 作用 |
| --- | --- |
| `tools/CMakeLists.txt` | 定义 `tools` 对象库，决定哪些 `.cpp` 参与构建。 |
| `tools/crc.hpp` | CRC8/CRC16 接口声明。 |
| `tools/crc.cpp` | CRC 实现，给串口/CAN 协议包校验用。 |
| `tools/exiter.hpp` | 退出控制类声明，用于捕获退出信号。 |
| `tools/exiter.cpp` | 退出控制实现，供主循环安全停机。 |
| `tools/extended_kalman_filter.hpp` | 通用 EKF 类声明。 |
| `tools/extended_kalman_filter.cpp` | EKF 的预测/更新实现，并记录 NIS/NEES 调试数据。 |
| `tools/img_tools.hpp` | 画点、画线、画框、画字等图像调试接口。 |
| `tools/img_tools.cpp` | 图像调试绘制实现。 |
| `tools/logger.hpp` | 全局 logger 接口声明。 |
| `tools/logger.cpp` | spdlog 单例初始化与获取。 |
| `tools/math_tools.hpp` | 角度限制、欧拉角/四元数转换、坐标变换与雅可比接口。 |
| `tools/math_tools.cpp` | 上述数学函数实现，是解算器和 EKF 的底层工具。 |
| `tools/pid.hpp` | PID 控制器声明。 |
| `tools/pid.cpp` | PID 控制器实现，支持角度型误差。 |
| `tools/plotter.hpp` | PlotJuggler/远程绘图发送器声明。 |
| `tools/plotter.cpp` | 通过 UDP 发送 JSON 数据到 PlotJuggler。 |
| `tools/ransac_sine_fitter.hpp` | RANSAC 正弦拟合器声明。 |
| `tools/ransac_sine_fitter.cpp` | 大符角速度拟合用的正弦拟合实现。 |
| `tools/recorder.hpp` | 视频+姿态录制器声明。 |
| `tools/recorder.cpp` | 异步保存视频帧和四元数时间戳。 |
| `tools/thread_pool.hpp` | 线程池、按帧序恢复结果的 `OrderedQueue`、多线程帧结构。 |
| `tools/thread_safe_queue.hpp` | 有界线程安全队列模板。 |
| `tools/trajectory.hpp` | 无空气阻力弹道求解结构声明。 |
| `tools/trajectory.cpp` | 根据初速、水平距离、高差求飞行时间和 pitch。 |
| `tools/yaml.hpp` | 轻量 YAML 读取封装，减少重复判空与类型转换。 |

## 10.3 核心函数

### `tools::ExtendedKalmanFilter`

- `predict(F, Q)`：线性预测。
- `predict(F, Q, f)`：带非线性状态转移函数的预测。
- `update(z, H, R)`：线性观测更新。
- `update(z, H, R, h, z_subtract)`：带非线性观测和角度差处理的更新。

它是 `auto_aim::Target`、`auto_buff::Target`、`buff_predict.hpp` 的公共滤波核心。

### `tools::math_tools`

- `limit_rad`：把角度限制到 `(-pi, pi]`。
- `eulers`：四元数/旋转矩阵转欧拉角。
- `rotation_matrix`：`ypr -> R`。
- `xyz2ypd` 与 `ypd2xyz`：世界/云台空间坐标与球坐标切换。
- `xyz2ypd_jacobian` 与 `ypd2xyz_jacobian`：EKF 观测方程需要的雅可比。
- `delta_time`：统一的 steady_clock 时间差。

### 其他关键工具

- `Trajectory::Trajectory(v0, d, h)`：弹道求解，给 `Aimer` 和 `Planner` 用。
- `Recorder::record`：将图像、四元数、时间戳压入异步保存线程。
- `Plotter::plot`：把 JSON 调试数据发到外部绘图工具。
- `PID::calc`：闭环控制器基础实现。
- `RansacSineFitter::fit`：拟合大符转速曲线。

## 10.4 这里最关键的数据类型

- `tools::Trajectory`：`unsolvable/fly_time/pitch`，是最小弹道结果。
- `tools::Frame`：多线程检测链里的“帧载体”，包含 `img/t/q/armors`。
- `tools::ThreadSafeQueue<T>`：模块间异步解耦的基础队列。
- `tools::ExtendedKalmanFilter::x/P`：所有状态估计器共享的状态向量与协方差。

