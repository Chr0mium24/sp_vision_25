# Diagnose Python 化迁移方案

本文档用于规划 `sp_vision_25` 的 diagnose/调参体系重构，目标是：

- 用 Python 统一承接 diagnose 命令入口、TUI、配置编辑、设备探测与日志展示
- 保留 C++ 作为实时算法与硬件交互基座
- 将“运行时可调参数”从“改 YAML 文件 + 重新加载”逐步迁移到“Python 控制面 -> C++ 运行时参数接口”
- 明确 Python 不重写核心算法，白盒 diagnose/test 必须复用真实 C++ 实现
- 将 `pybind11` 视为核心路线，而不是可选增强

本文档回答四个问题：

1. Python diagnose 应该如何组织目录和命令树
2. C++ 基座现在能否直接承接 Python 诊断层
3. 哪些配置应该由 Python 读，哪些应该由 C++ 自己持有
4. 迁移顺序如何安排，具体有哪些文件要改、每个文件应该怎么改

## 1. 当前问题总结

当前 diagnose 体系的主要问题不是“脚本语言是 Bash”，而是职责混杂。迁移已经进入收尾阶段，当前 Python diagnose 入口和 TUI 已经可用，三个旧 `diagnose.sh` 包装也已经删除：

- 旧的 `diagnostics/*/diagnose.sh` 过去同时负责命令路由、设备释放、参数编辑、二进制调用
- 多个 C++ diagnose 程序同时负责：
  - 业务逻辑
  - 命令行解析
  - TUI/GUI 展示
  - 参数调节与落盘
- 各 C++ 模块普遍直接读取 YAML 文件，配置入口分散，不利于 Python 统一控制

典型现状：

- `diagnostics/camera/diagnose.sh`：历史上集成设备释放、信息查看、配置编辑、命令分发
- `diagnostics/auto_aim/diagnose.sh`：历史上集成在线/离线调试、rune 调参、YAML 原位修改
- `diagnostics/gimbal/diagnose.sh`：历史上集成串口诊断、控制模式切换、端口信息查看
- `diagnostics/auto_aim/auto_aim_ui_tune.cpp`：既负责 runtime，又负责 TUI，又负责 YAML 回写
- `io/camera.cpp`、`io/cboard.cpp`、`tasks/auto_aim/detector.cpp`、`tasks/auto_aim/planner/planner.cpp` 等：构造时直接从 `config_path` 读取 YAML

这会导致三个长期问题：

- diagnose 行为难以统一复用
- 动态调参只能通过“写配置文件”间接实现
- Python 无法平滑接管控制面

除此之外，还有一个更关键的问题：

- 如果 Python diagnose/test 自己重写 CRC、卡尔曼滤波、planner、tracker 等逻辑，那么测试结果会逐渐偏离真实线上 C++ 行为，最终失去 diagnose 的可信度

因此本文档的前提约束是：

- Python 只负责控制面、配置、界面和编排
- 核心算法和核心协议逻辑必须复用现有 C++ 实现
- 对于白盒测试、精细调参与高可信 diagnose，必须优先通过 `pybind11` 复用 C++ 核心模块

## 2. 目标架构

建议迁移到三层结构：

### 2.1 C++ Core

职责：

- 硬件访问：相机、串口/CAN、IMU、OpenVINO runtime
- 核心算法：detector、solver、tracker、planner、aimer、runtime
- 运行时参数更新接口
- 状态快照/debug 数据输出接口

不负责：

- TUI/CLI 菜单
- diagnose 命令树
- YAML 编辑器

### 2.2 Python Diagnose App

职责：

- 统一命令入口
- Textual TUI
- YAML 配置文件读写与校验
- 外部设备信息采集
- 运行时调参面板
- 日志查看/会话记录
- 调用 C++ 二进制或 pybind11 模块

不负责：

- 重写 CRC
- 重写卡尔曼滤波
- 重写 planner/tracker/solver 等核心算法

推荐技术选型：

- CLI：`Typer`
- TUI：`Textual`
- 输出/日志：`Rich`
- YAML：`ruamel.yaml` 或 `PyYAML`

推荐优先使用 `Textual`，原因：

- 适合做参数表、状态面板、日志面板、快捷键菜单
- 比 C++ 手写 raw terminal 模式维护成本低
- 后续扩展到多页 TUI 更自然

### 2.3 Adapter 层

短期：

- Python 通过 `subprocess` 启动现有 C++ 可执行文件
- 以命令行参数传递配置路径、模式、运行时 patch 文件或 patch 参数

中期与长期：

- 通过 `pybind11` 暴露核心模块和运行时接口
- Python 直接持有 runtime/controller/diagnose session
- Python diagnose/test 通过绑定层复用真实 C++ 实现

这里需要明确：

- `subprocess` 只适合黑盒 diagnose 和命令入口统一
- 只要涉及白盒诊断、动态调参、精确状态观测、单元测试一致性，就必须接入 `pybind11`

## 2.4 黑盒 diagnose 与白盒 diagnose

建议将 Python diagnose 明确拆成两类：

### 黑盒 diagnose

特点：

- 通过现有 C++ 可执行文件工作
- Python 负责命令路由、日志、环境检查、TUI 编排
- 不直接进入算法内部

适用场景：

- 相机联通性检查
- 串口/端口扫描
- 启停现有 diagnose 程序
- 离线回放任务编排

实现方式：

- `subprocess` + 统一 CLI/TUI

### 白盒 diagnose

特点：

- 直接访问真实 C++ 模块内部状态和接口
- 需要运行时更新参数
- 需要精确复用线上实现

适用场景：

- CRC 测试
- 卡尔曼滤波单测与中间状态检查
- planner/tracker/solver 调参与快照
- runtime 内部状态观测

实现方式：

- `pybind11`

因此迁移策略不是“是否使用 pybind11”，而是：

- 黑盒能力短期可由 `subprocess` 承接
- 白盒能力必须由 `pybind11` 承接

## 3. Python Diagnose 目录结构

建议在仓库内增加如下目录：

```text
python/
├── diagnose/
│   ├── __init__.py
│   ├── main.py
│   ├── cli.py
│   ├── app.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── loader.py
│   │   ├── writer.py
│   │   └── patch.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── process_runner.py
│   │   ├── binary_registry.py
│   │   ├── session.py
│   │   └── event_stream.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── camera_service.py
│   │   ├── gimbal_service.py
│   │   ├── auto_aim_service.py
│   │   ├── runtime_patch_service.py
│   │   └── system_service.py
│   ├── tui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── screens/
│   │   │   ├── home.py
│   │   │   ├── camera.py
│   │   │   ├── gimbal.py
│   │   │   ├── auto_aim.py
│   │   │   └── tune.py
│   │   └── widgets/
│   │       ├── status_panel.py
│   │       ├── param_table.py
│   │       ├── log_view.py
│   │       └── binary_status.py
│   └── tools/
│       ├── __init__.py
│       ├── usb.py
│       ├── serial.py
│       ├── v4l2.py
│       └── env.py
└── bindings/
    └── README.md
```

各路径职责：

- 这个结构是逻辑分层图，不强制要求和仓库路径一一对应
- 当前实际落地路径放在 `src/sp_vision_25_python/diagnose/`，这样更适合 `uv` 打包和命令入口安装
- 绑定模块仍然建议继续放在 `bindings/` 下，与 diagnose 控制面解耦

说明：

- `python/diagnose/cli.py`：Typer 命令树入口
- `python/diagnose/config/*`：统一管理 YAML 读取、落盘、patch、schema
- `python/diagnose/core/*`：管理二进制路径、进程启动、日志流
- `python/diagnose/services/*`：按业务域封装 diagnose 行为
- `python/diagnose/tui/*`：Textual UI

绑定模块建议单独组织，不与 diagnose app 混在一起。推荐目录：

```text
bindings/
├── CMakeLists.txt
├── module.cpp
├── tools/
│   ├── crc_bindings.cpp
│   └── ekf_bindings.cpp
└── auto_aim/
    ├── planner_bindings.cpp
    ├── solver_bindings.cpp
    ├── tracker_bindings.cpp
    └── runtime_bindings.cpp
```

原因：

- diagnose app 迭代速度快
- 绑定层与 C++ ABI、头文件、生命周期管理强相关
- 单独组织更利于分批推进与测试

## 4. Python Diagnose 命令树

建议主入口命令统一为：

```bash
uv run sp-vision-diagnose
```

### 4.1 顶层命令树

```text
sp-vision-diagnose
├── build
├── doctor
├── config
├── camera
├── gimbal
├── auto-aim
├── tune
└── tui
```

### 4.2 详细命令树

```text
sp-vision-diagnose build
├── all
├── target <name>
└── status

sp-vision-diagnose doctor
├── env
├── binaries
├── camera
├── serial
└── openvino

sp-vision-diagnose config
├── show <config.yaml>
├── validate <config.yaml>
├── diff <old.yaml> <new.yaml>
├── patch <config.yaml> --set key=value
└── export-runtime <config.yaml>

sp-vision-diagnose camera
├── info
├── list
├── release
├── quick [config]
├── window [config]
├── detect [config]
├── usb [config]
├── thread [config]
└── handeye [config]

sp-vision-diagnose gimbal
├── quick [config]
├── rxonly [config]
├── proto [config]
├── probe [config]
├── scan [config]
├── snapshot [config]
├── watch [config]
├── control [config]
├── axis [config]
├── manual-axis [config]
└── port-info [config]

sp-vision-diagnose auto-aim
├── list
├── armor-box [config]
├── armor-intent [config]
├── armor-offline [config] <input-prefix>
├── rune-box [config] <input-prefix>
├── rune-online [config]
├── rune-online-mpc [config]
└── session [config]

sp-vision-diagnose tune
├── auto-aim [config]
├── camera [config]
├── planner [config]
└── export [config]

sp-vision-diagnose tui
├── home
├── camera [config]
├── gimbal [config]
├── auto-aim [config]
└── tune [config]
```

设计原则：

- 与历史 `diagnostics/*/diagnose.sh` 的动作尽量一一对应，降低迁移摩擦
- `doctor` 负责环境检查，避免把“环境问题”塞进业务命令
- `config` 与 `tune` 分离：
  - `config` 负责静态文件
  - `tune` 负责运行时调参

## 5. 配置模型建议

建议将配置分成三类：

### 5.1 静态配置

示例：

- `camera_name`
- `vid_pid`
- `com_port`
- `can_interface`
- `*_model_path`
- `device`

特点：

- 运行中通常不改
- 改了通常需要重建对象或重启模块

建议：

- 由 Python 负责 YAML 读写与校验
- C++ 在初始化时接收“静态配置对象”

### 5.2 运行时可调配置

示例：

- `yaw_offset`
- `pitch_offset`
- `decision_speed`
- `first_tolerance`
- `second_tolerance`
- `fire_thresh`
- `auto_fire`
- detector/planner 阈值参数

特点：

- diagnose/tune 中频繁改动
- 不应依赖“改 YAML 再重新读取”

建议：

- Python TUI 持有当前 runtime patch
- 优先通过 `pybind11` 发送到 C++
- 只有临时过渡阶段才考虑进程参数或 patch 文件
- 需要持久化时，再由 Python 写回 YAML

### 5.3 调试会话配置

示例：

- `show=true`
- `nogui`
- `duration-ms`
- `summary-ms`
- `input-prefix`
- `log-path`

特点：

- diagnose 会话级参数
- 不应该写入机器人主配置 YAML

建议：

- 完全放在 Python CLI/TUI 层

## 6. C++ 基座现状判断

### 6.1 现在能不能用

能用。

理由：

- 已完成基础编译，C++ 主体可正常构建
- `io`、`tasks`、`tools` 已具备较明确模块边界
- diagnose 相关可执行文件可以作为 Python 的第一阶段后端

但这里的“能用”仅表示：

- 可以作为黑盒 diagnose 的后端
- 可以作为 `pybind11` 绑定的源代码基础

不表示：

- Python 可以靠重写算法替代它
- 也不表示只靠 `subprocess` 就足够支撑长期 diagnose/test 体系

### 6.2 现在为什么还不够好

不够好主要体现在三个方面：

1. 配置读取分散
2. 运行时可调参数缺少显式接口
3. diagnose 程序把 UI/TUI/CLI 和核心逻辑绑在一起

因此结论是：

- C++ 基座足够作为迁移起点
- 但要支持长期的 Python diagnose，必须做接口整理
- 并且必须规划 `pybind11` 绑定层

## 7. C++ 配置接口改造总原则

总原则：

- “类不直接依赖配置文件路径”，而依赖“配置对象”
- “运行时参数可更新”与“静态初始化参数”分离
- “配置读取失败”不直接 `exit(1)`，而通过异常/错误返回上传

推荐引入以下对象：

```text
config/
├── static_config.hpp
├── runtime_config.hpp
├── config_loader.hpp
├── config_loader.cpp
└── runtime_patch.hpp
```

推荐数据结构：

- `StaticConfig`
- `CameraConfig`
- `GimbalConfig`
- `CBoardConfig`
- `AutoAimRuntimeConfig`
- `AutoAimTuningConfig`
- `PlannerConfig`
- `DetectorConfig`
- `RuntimePatch`

同时建议新增一组绑定目标：

- `sp_vision_bindings_tools`
- `sp_vision_bindings_auto_aim`
- `sp_vision_bindings_runtime`

## 8. 文件级改造清单

下面列出第一阶段应重点修改的文件。

### 8.1 配置基础设施

#### [tools/yaml.hpp](/home/cr/Codes/sp_vision_25/tools/yaml.hpp)

当前问题：

- `load` 和 `read` 出错时直接 `exit(1)`
- 无法被 Python diagnose 或 pybind11 优雅接住

建议修改：

- `load` 改为抛出 `std::runtime_error`
- `read` 改为：
  - 缺字段时抛出异常
  - 增加 `read_optional<T>()`
- 新增路径、字段名、上下文信息

目标：

- 让上层决定如何处理错误
- Python 层可以把错误显示在 TUI，而不是进程直接退出

这一步对 `pybind11` 也很关键：

- 绑定层最怕底层直接 `exit(1)`
- 改为异常后，Python 端才能拿到可展示、可断言的错误

#### 新增 `config/` 目录

建议新增文件：

- `config/config_types.hpp`
- `config/config_loader.hpp`
- `config/config_loader.cpp`
- `config/runtime_patch.hpp`

建议内容：

- 定义统一配置结构体
- 从 YAML 构造结构体
- 提供 patch merge 逻辑

### 8.2 IO 层

#### [io/camera.cpp](/home/cr/Codes/sp_vision_25/io/camera.cpp)

当前问题：

- 构造函数直接收 `config_path`
- 内部自己读取 YAML

建议修改：

- 保留旧接口：`Camera(const std::string& config_path)` 作为兼容层
- 新增主接口：`Camera(const CameraConfig& config)`
- 将 YAML 解析挪到 `config_loader`

目标：

- Python 或 C++ 上层都能直接传结构化配置

#### [io/cboard.cpp](/home/cr/Codes/sp_vision_25/io/cboard.cpp)

当前问题：

- `read_yaml` 和对象初始化耦合
- `send_transform` 读取逻辑散在内部

建议修改：

- 新增 `CBoardConfig`
- 构造函数改为优先接 `CBoardConfig`
- 将 CAN 接口、CAN ID、发送变换参数统一归档
- 提供 `update_send_transform(...)`

目标：

- 让动态调整 `send_yaw_scale` / `send_pitch_scale` 这类参数成为可能

#### [io/gimbal/gimbal.hpp](/home/cr/Codes/sp_vision_25/io/gimbal/gimbal.hpp)
#### [io/gimbal/gimbal.cpp](/home/cr/Codes/sp_vision_25/io/gimbal/gimbal.cpp)

当前问题：

- 构造时直接依赖 `config_path`
- 发送变换参数只在初始化时读入

建议修改：

- 新增 `GimbalConfig`
- `Gimbal(const GimbalConfig&, bool wait_for_first_q = true)`
- 新增：
  - `void update_send_transform(const SendTransform&)`
  - `SendTransform send_transform() const`

目标：

- 允许 Python diagnose 在控制会话中更新发送偏置，而不是写回 YAML 后重启

### 8.3 Auto Aim 层

#### [tasks/auto_aim/detector.cpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/detector.cpp)
#### [tasks/auto_aim/classifier.cpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/classifier.cpp)
#### [tasks/auto_aim/tracker.cpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/tracker.cpp)
#### [tasks/auto_aim/yolos/yolov5.cpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/yolos/yolov5.cpp)
#### [tasks/auto_aim/yolos/yolov8.cpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/yolos/yolov8.cpp)
#### [tasks/auto_aim/yolos/yolo11.cpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/yolos/yolo11.cpp)

当前问题：

- 多个模块各自 `YAML::LoadFile(config_path)`
- 同一组参数被分散读取

建议修改：

- 拆分出：
  - `DetectorConfig`
  - `ClassifierConfig`
  - `TrackerConfig`
  - `YoloConfig`
- 保留旧的 `config_path` 构造函数做兼容
- 新增 `update_config(...)` 或 `update_thresholds(...)`

目标：

- 让 Python 调参可以只更新 detector/tracker 阈值，不需要重建全链路

#### [tasks/auto_aim/planner/planner.hpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/planner/planner.hpp)
#### [tasks/auto_aim/planner/planner.cpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/planner/planner.cpp)

当前问题：

- planner 内部多次读 YAML
- solver setup 与运行逻辑耦合

建议修改：

- 新增 `PlannerConfig`
- 将以下参数收口：
  - `yaw_offset`
  - `pitch_offset`
  - `fire_thresh`
  - `decision_speed`
  - `high_speed_delay_time`
  - `low_speed_delay_time`
  - `max_yaw_acc`
  - `max_pitch_acc`
  - `Q_yaw/Q_pitch`
  - `R_yaw/R_pitch`
- 提供：
  - `update_runtime_config(...)`
  - `PlannerConfig config() const`

目标：

- 让 planner 成为最先支持热更新的模块
- 让 planner 成为第一批 `pybind11` 绑定对象

#### [tasks/auto_aim/auto_aim_runtime.cpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/auto_aim_runtime.cpp)
#### [tasks/auto_aim/auto_aim_runtime.hpp](/home/cr/Codes/sp_vision_25/tasks/auto_aim/auto_aim_runtime.hpp)

当前问题：

- Runtime 聚合了多个模块，但没有统一对外配置入口

建议修改：

- 新增 `AutoAimRuntimeConfig`
- 新增：
  - `void update_tuning(const AutoAimTuningConfig&)`
  - `AutoAimDebugSnapshot snapshot() const`
- 将 diagnose 程序需要的调试字段正式组织成 snapshot 接口

目标：

- 成为 Python diagnose 的优先绑定目标

建议在这里新增的不是文件路径接口，而是绑定友好的对象接口：

- `AutoAimDebugSnapshot snapshot() const`
- `void update_tuning(const AutoAimTuningConfig&)`
- `void reset()`
- `StepResult step(...)`

这样 Python 端才能真正把它当 diagnose session 使用

### 8.4 Tools 层优先绑定对象

这部分虽然不是 diagnose 可执行文件，但对“Python diagnose/test 是否可信”至关重要。

#### [tools/crc.hpp](/home/cr/Codes/sp_vision_25/tools/crc.hpp)
#### [tools/crc.cpp](/home/cr/Codes/sp_vision_25/tools/crc.cpp)

建议修改：

- 保持实现不变
- 增加清晰的绑定入口函数

建议绑定内容：

- `get_crc8`
- `get_crc16`
- 如有必要，增加 buffer/list/bytes 适配

目标：

- Python 侧协议测试直接复用真实 C++ CRC 实现

#### [tools/extended_kalman_filter.hpp](/home/cr/Codes/sp_vision_25/tools/extended_kalman_filter.hpp)
#### [tools/extended_kalman_filter.cpp](/home/cr/Codes/sp_vision_25/tools/extended_kalman_filter.cpp)

建议修改：

- 保持算法实现不变
- 整理构造、predict、update、state 获取接口
- 确保无 `exit(1)` 式错误处理

建议绑定内容：

- 构造
- `predict`
- `update`
- `state`
- 协方差或内部矩阵快照

目标：

- Python 单测和 diagnose 面板直接查看真实 EKF 行为

### 8.5 Diagnose 程序本身

#### [diagnostics/gimbal/gimbal_ui_test.cpp](/home/cr/Codes/sp_vision_25/diagnostics/gimbal/gimbal_ui_test.cpp)

当前状态：

- 已迁移到 Python 侧的 `GimbalSession`
- `sp-vision-diagnose gimbal snapshot/watch/control/script-control` 现在直接走 `pybind11`
- 对应 C++ 源文件和 CMake 目标已删除

建议后续：

- 继续收口 `gimbal_link_diag_test.cpp`、`gimbal_serial_probe.cpp`、`gimbal_axis_diag_test.cpp`、`gimbal_manual_axis_diag_test.cpp`

#### [diagnostics/auto_aim/auto_aim_ui_test.cpp](/home/cr/Codes/sp_vision_25/diagnostics/auto_aim/auto_aim_ui_test.cpp)

当前问题：

- 既有 UI，又有 snapshot 生成，又有 runtime 驱动

建议修改：

- 提取出无 UI 的 `AutoAimDiagnoseSession`
- 将 JSON snapshot 构造逻辑下沉为可复用接口

目标：

- Python 可以直接消费 snapshot

#### [diagnostics/auto_aim/auto_aim_ui_tune.cpp](/home/cr/Codes/sp_vision_25/diagnostics/auto_aim/auto_aim_ui_tune.cpp)

当前问题：

- 在 C++ 中直接加载/修改/导出 YAML
- 运行时调参与持久化写回耦合

建议修改：

- 停止继续扩展该文件
- 迁移目标：
  - 运行时 patch 交给 C++ runtime 接口
  - YAML 读写交给 Python

目标：

- 最终由 Python `tune auto-aim` 替代

### 8.6 Bash diagnose 脚本（历史）

#### [diagnostics/gimbal/diagnose.sh](/home/cr/Codes/sp_vision_25/diagnostics/gimbal/diagnose.sh)
#### [diagnostics/camera/diagnose.sh](/home/cr/Codes/sp_vision_25/diagnostics/camera/diagnose.sh)
#### [diagnostics/auto_aim/diagnose.sh](/home/cr/Codes/sp_vision_25/diagnostics/auto_aim/diagnose.sh)

建议修改：

- 不再继续扩展 Bash 能力
- 该阶段曾短暂改为薄 shim，现已删除：

```bash
#!/usr/bin/env bash
uv run sp-vision-diagnose gimbal "$@"
```

目标：

- 保持旧入口兼容的阶段已经结束
- 实际逻辑已经全部转移到 Python

## 9. 迁移顺序

建议分四个阶段。

### 第一阶段：Python 包装层落地（已完成）

目标：

- 不改 C++ 核心逻辑，只替换 Bash diagnose
- 明确这一步只解决黑盒 diagnose 的统一入口

内容：

- 建立 `python/diagnose/` 目录
- 引入 `Typer + Rich`
- 用 Python 复刻：
  - `diagnostics/gimbal/diagnose.sh`
  - `diagnostics/camera/diagnose.sh`
  - `diagnostics/auto_aim/diagnose.sh`
- 仍通过 `subprocess` 调现有二进制

优先级：

- `gimbal`
- `camera`
- `auto_aim`

原因：

- `gimbal` 命令边界最清晰
- `camera` 以系统信息与设备释放为主，Python 很适合
- `auto_aim` 最复杂，先保持外层包装

### 第二阶段：pybind11 第一批绑定 + 配置接口收口

目标：

- 建立白盒 diagnose/test 的最小可用路径
- 让主要核心类从“读文件”改成“接配置对象”

内容：

- 建立 `bindings/` 目录
- 接入 `pybind11`
- 第一批绑定：
  - `tools/crc`
  - `tools/extended_kalman_filter`
  - `tasks/auto_aim/planner`
  - `tasks/auto_aim/solver`
- 新增 `config/` 层
- 改造 `tools/yaml.hpp`
- 改造：
  - `io/camera.*`
  - `io/cboard.*`
  - `io/gimbal/*`
  - `tasks/auto_aim/*`
  - `tasks/auto_buff/*`

原因：

- 这批模块最能直接决定 diagnose/test 的可信度
- 如果 Python 自己重写它们，最终会和线上 C++ 行为漂移

### 第三阶段：运行时参数热更新与高层绑定

目标：

- 建立 Python diagnose -> C++ runtime 的调参通道

内容：

- 给 planner/detector/runtime 增加 `update_*` 接口
- 给 diagnose 会话增加 `snapshot()` 接口
- 绑定：
  - `tasks/auto_aim/tracker`
  - `tasks/auto_aim/auto_aim_runtime`
- 如有必要，补充进程模式作为兼容后备方案

建议：

- 先让 `planner` 和 `auto_aim_runtime` 支持热更新
- 再扩展到 detector/tracker

### 第四阶段：Python TUI 接管

目标：

- 用 `Textual` 接管主要 TUI

内容：

- `gimbal` 页面
- `camera` 页面
- `auto_aim` 页面
- `tune` 页面

最终结果：

- C++ 仅输出状态与接收控制
- Python 成为 diagnose 控制面
- Python 白盒测试与 diagnose 面板复用真实 C++ 实现

## 10. 动态调参链路建议

推荐链路如下：

```text
Python TUI
  -> 读取 YAML 形成 ConfigModel
  -> 启动 pybind11 runtime/session
  -> 发送 RuntimePatch
  -> 拉取 DebugSnapshot
  -> 用户确认后选择 Save
  -> Python 将 patch 合并写回 YAML
```

不推荐链路：

```text
Python 改 YAML
  -> C++ 每次重新 LoadFile
  -> 用文件当实时控制通道
```

原因：

- 文件 I/O 不适合作为运行时控制协议
- 难以区分“暂存值”和“持久化值”
- 会让不同模块读到不一致配置
- 更重要的是，这条链路无法让 Python 直接复用真实 C++ 内部状态

## 11. 第一批建议落地的改动

如果只做一轮最有价值的改造，建议目标如下：

1. 新建 Python diagnose CLI 骨架
2. 先把三个 `diagnose.sh` 迁成 Python 命令
3. 接入 `pybind11` 构建链路
4. 绑定 `tools/crc` 与 `tools/extended_kalman_filter`
5. 改 `tools/yaml.hpp`，去掉 `exit(1)`
6. 为 `Planner` 新增 `PlannerConfig` 与 `update_runtime_config`
7. 绑定 `Planner`
8. 为 `AutoAimRuntime` 增加 `snapshot()` 接口
9. 将 `auto_aim_ui_tune.cpp` 标记为过渡实现，不再扩展

这批改动完成后：

- diagnose 外层体验会立刻统一
- C++ 基座仍保持稳定
- Python 已经能开始复用一部分真实 C++ 实现
- 后续继续扩充绑定时不会推倒重来

## 12. 文档与代码同步建议

迁移开始后建议同步维护以下文档：

- 在 [docs/test_chain_and_usage.md](/home/cr/Codes/sp_vision_25/docs/test_chain_and_usage.md) 追加 Python diagnose 入口说明
- 在 `readme.md` 中保留 `sp-vision-diagnose` 用法，并在历史备注里说明旧 `diagnostics/*.sh` 已删除
- 为每个迁移完成的模块单独补一页：
  - `docs/diagnose/gimbal_python_diagnose.md`
  - `docs/diagnose/camera_python_diagnose.md`
  - `docs/diagnose/auto_aim_python_diagnose.md`

## 13. 最终建议

结论可以概括为三句话：

- Python diagnose 完全值得做，尤其 TUI 建议引入 `Textual`
- Python 不应该重写核心算法，`pybind11` 是白盒 diagnose/test 的核心路线
- C++ 基座现在能用，但必须把“配置读取”和“运行时参数”从文件路径里解耦出来
- 最稳的迁移顺序是：先统一黑盒入口，再接入第一批 `pybind11` 绑定，再收口配置接口与热更新，最后接管 TUI

建议下一步直接进入实现阶段时，先从 `gimbal diagnose` 开始。

原因：

- 命令边界最清晰
- 风险最低
- 最能验证 Python CLI/TUI 架构是否顺手

但在 `gimbal diagnose` 骨架并行推进的同时，建议立刻启动第一批绑定：

- `tools/crc`
- `tools/extended_kalman_filter`
- `tasks/auto_aim/planner`

因为这三类一旦不复用真实 C++ 实现，Python diagnose/test 的可信度就会快速下降。
