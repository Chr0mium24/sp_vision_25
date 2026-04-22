# `tests/` 目录职责图

这份文档用于回答一个常见问题：`tests/` 是否和 `diagnose/` 重复。

结论先写在前面：

- `diagnose/` 是控制面，负责命令路由、TUI、参数编排、设备探测和日志展示
- `tests/` 是执行面，负责把真实 C++ 逻辑编成可执行程序，供 `diagnose/`、回归检查和硬件联调直接调用
- 现在这两个目录**有交集，但不重复**

## 1. 当前目录状态

`tests/` 里目前主要是这些组：

- `auto_aim`
- `auto_buff`
- `camera`
- `gimbal`
- `planner`
- `system`
- `ros2`

其中多数目标仍然被以下路径直接引用：

- [CMakeLists.txt](/home/cr/Codes/sp_vision_25/CMakeLists.txt)
- [src/sp_vision_25_python/diagnose/inventory.py](/home/cr/Codes/sp_vision_25/src/sp_vision_25_python/diagnose/inventory.py)
- [docs/test_chain_and_usage.md](/home/cr/Codes/sp_vision_25/docs/test_chain_and_usage.md)

## 2. 角色划分

### 2.1 现役保留

这类目标现在仍然有明确用途，建议保留为 C++ 可执行文件：

- `camera_test`
- `camera_detect_test`
- `camera_save_test`
- `camera_thread_test`
- `camera_window_test`
- `usbcamera_test`
- `usbcamera_detect_test`
- `multi_usbcamera_test`
- `handeye_test`
- `gimbal_test`
- `gimbal_response_test`
- `planner_test`
- `planner_test_offline`
- `auto_aim_test`
- `detector_video_test`
- `auto_power_rune_test`
- `cboard_test`
- `dm_test`

理由：

- 它们仍是 Python diagnose 的底层执行体，或者仍是硬件联调最直接的烟雾测试
- 其中不少测试直接映射到真实设备、相机、串口、云台和标定流程
- Python 侧如果完全重写，容易和线上 C++ 行为漂移

### 2.2 可迁 Python

这类目标未来可以考虑逐步转成 Python `pytest`、`pybind11` 调用，或被 Python diagnose 直接封装，但前提是对应的底层接口已经足够稳定：

- `planner_test`
- `planner_test_offline`
- `gimbal_test`
- `gimbal_response_test`
- `camera_window_test`
- `camera_thread_test`
- `detector_video_test`
- `auto_aim_test`

这类目标之所以“可迁”，不是因为它们没用，而是因为它们更接近：

- 算法回归
- 数据流回归
- 结果断言

而这些是 Python `pytest` 也能很好承接的。

### 2.3 暂不建议迁

以下目标更偏设备/系统层，通常更适合继续保持为独立 C++ smoke test：

- `camera_test`
- `camera_detect_test`
- `camera_save_test`
- `usbcamera_test`
- `usbcamera_detect_test`
- `handeye_test`
- `cboard_test`
- `dm_test`
- `auto_power_rune_test`

原因：

- 它们和硬件、SDK、采集链路、设备权限耦合更紧
- Python 迁移不一定带来明显收益，反而更容易把依赖层弄厚

### 2.4 可单独审视

`ros2` 目录下这些目标属于条件编译项：

- `publish_test`
- `subscribe_test`
- `topic_loop_test`

它们是否继续保留，取决于项目是否还要长期支持 ROS2 环境。

当前仓库已经把 ROS2 作为可选依赖处理，所以这部分更适合单独判断：

- 如果继续支持 ROS2：保留
- 如果明确放弃 ROS2：可以整体收缩或移除

## 3. 和 diagnose 的关系

`diagnose/` 和 `tests/` 的关系可以理解成：

- `diagnose/` = “怎么用”
- `tests/` = “用什么跑”

例如：

- `sp-vision-diagnose camera quick` 最终仍会调用 `camera_test`
- `sp-vision-diagnose auto-aim armor-box` 直接复用真实 C++ runtime，但它的很多回归样本仍来自 `tests/auto_aim`
- `sp-vision-diagnose gimbal quick` 现在已经走 Python 会话，但底层仍依赖真实 C++ 绑定和测试目标做验证

所以现在不能把 `tests/` 理解成“重复 diagnose”。

## 4. 什么时候可以删

一个 `tests/*.cpp` 只有在满足下面条件后，才值得考虑删除：

1. Python diagnose 或 `pytest` 已经完全接手这个测试的职责
2. `CMakeLists.txt` 不再编译这个目标
3. `src/sp_vision_25_python/diagnose/inventory.py` 不再列它为后端可执行文件
4. `docs/test_chain_and_usage.md` 不再把它写成当前链路
5. 没有任何诊断命令再调用它

## 5. 建议

当前最稳妥的策略不是删除 `tests/`，而是：

- 保留现役 smoke test
- 把更偏算法回归的部分逐步迁到 Python `pytest`
- 继续用 `inventory.py` 把 `diagnose` 入口和 `tests` 执行体区分开

