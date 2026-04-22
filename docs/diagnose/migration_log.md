# Diagnose Migration Log

这个文件是 diagnose Python 化与 `pybind11` 重构的迭代日志，按时间记录每次实际做了什么。

## 2026-04-22

### 1. 基线准备

- 安装了 Python 工作区基础依赖
- 建立了 `uv` 管理的 Python 环境
- 新增了 `pyproject.toml`、`.python-version` 和最小 `pytest` 基线

### 2. 文档收口

- 新建并完善了 [docs/diagnose_python_migration.md](/home/cr/Codes/sp_vision_25/docs/diagnose_python_migration.md)
- 明确了 Python diagnose 只做控制面
- 明确了白盒测试和高可信 diagnose 必须复用真实 C++ 实现
- 明确了 `pybind11` 是核心路线

### 3. 第一批绑定

- 新增了 `sp_vision_bindings` 共享模块
- 完成了 `tools/crc` 的 Python 绑定
- Python 侧已经可以直接调用真实 C++ CRC 实现

### 4. 当前进行中

- 开始接入 `tools/extended_kalman_filter`
- 为后续 `planner`、`solver`、`tracker`、`auto_aim_runtime` 的绑定做准备

### 5. 本轮验证

- 完成了 `tools/extended_kalman_filter` 的 Python 绑定
- 为 EKF 暴露了 `x`、`P`、`data`、`window_size`、`last_nis` 和预测/更新接口
- 新增了 `python_tests/test_ekf_binding.py`
- 通过了 `uv run pytest`
- 通过了完整的 `./build.sh`

### 6. 白盒核心继续推进

- 新增了 `auto_aim::Armor` 的 Python 绑定
- 新增了 `auto_aim::Solver` 的 Python 绑定
- Python 侧可以直接调用真实 C++ 的重投影与 solvePnP 逻辑
- 新增了 `python_tests/test_solver_binding.py`
- 当前 Python 测试已覆盖 CRC、EKF、import、Solver roundtrip

### 7. Target 绑定

- 新增了 `auto_aim::Target` 的 Python 绑定
- Python 侧可以直接基于真实 `Armor` 构造 `Target`
- Python 侧可以直接调用 `predict` / `update`
- 新增了 `python_tests/test_target_binding.py`
- 当前 Python 测试已覆盖 CRC、EKF、Armor、Solver、Target

### 8. Tracker 绑定

- 新增了 `auto_aim::Tracker` 的 Python 绑定
- Python 侧可以直接基于真实 `Solver` 和 `Armor` 列表驱动 Tracker 状态机
- Python 侧可以读取 `state` 与调试信息
- 新增了 `python_tests/test_tracker_binding.py`
- 当前 Python 测试已覆盖 CRC、EKF、Armor、Solver、Target、Tracker

### 9. Aimer 绑定

- 新增了 `auto_aim::Aimer` 的 Python 绑定
- Python 侧可以直接基于真实 `Target` 计算瞄准命令
- Python 侧可以读取 `AimerDebug`
- 新增了 `python_tests/test_aimer_binding.py`
- 当前 Python 测试已覆盖 CRC、EKF、Armor、Solver、Target、Tracker、Aimer

### 10. Runtime 绑定

- 新增了 `auto_aim::Runtime` 的 Python 绑定
- Python 侧可以直接输入 numpy 图像与四元数，调用真实 runtime step
- 新增了 `python_tests/test_runtime_binding.py`
- 当前 Python 测试已覆盖 CRC、EKF、Armor、Solver、Target、Tracker、Aimer、Runtime

### 11. Python diagnose 控制面骨架

- 新增了 `sp_vision_25_python.diagnose` 包
- 新增了统一 CLI 入口 `sp-vision-diagnose`
- 新增了 `status`、`bindings`、`camera`、`gimbal`、`auto-aim` 命令
- 现有 Bash diagnose 脚本已被 Python 命令统一桥接
- 将 Python 构建后端改为仓库内本地 backend，避免 `uv` 依赖外网拉取 build backend
- 统一入口已经可以直接查看构建状态、绑定状态，并转发到现有 camera/gimbal/auto_aim 诊断脚本

### 12. Python 先接管 list 类命令

- 新增了 `diagnose/inventory.py`
- 将 `camera list`、`gimbal list`、`auto-aim list` 改为 Python 直接输出
- 这三类列表命令现在只检查构建产物存在性，不再依赖 Bash 脚本
- 新增了对应 CLI 测试，验证这三条路径都已经从 Python 侧接管

### 13. Python 继续接管只读信息命令

- 新增了 `diagnose/config.py` 和 `diagnose/system.py`
- 将 `camera info` 改为 Python 直接输出 `/dev/video*` 和 `v4l2-ctl` 信息
- 将 `gimbal port-info` 改为 Python 直接读取 YAML 配置并输出串口信息
- 新增了对应 CLI 测试，确保这两条路径也不再回落到 Bash

### 14. Python 接管高频启动命令

- 新增了 `diagnose/actions.py`
- 将 `camera release` 和 `camera tune` 改为 Python 直接实现
- 将 `camera quick`、`camera detect`、`camera window`、`camera save`、`camera usb`、`camera usb-detect`、`camera thread`、`camera handeye` 改为 Python 直接启动 C++ 二进制
- 将 `gimbal quick`、`gimbal rxonly`、`gimbal proto`、`gimbal probe`、`gimbal probe-raw`、`gimbal scan`、`gimbal snapshot`、`gimbal watch`、`gimbal control`、`gimbal script-control`、`gimbal axis`、`gimbal manual-axis` 改为 Python 直接启动 C++ 二进制
- 将 `auto-aim armor-box`、`armor-intent`、`armor-rec`、`armor-tune`、`armor-offline`、`rune-box`、`rune-rec`、`rune-online`、`rune-online-mpc`、`rune-tune` 改为 Python 直接启动 C++ 二进制
- 新增了动作级测试和 CLI 分流测试，确保额外参数能正确传递且不会回落到 Bash

### 15. Bash 入口降级为 Python 薄壳

- 将 `diagnostics/camera/diagnose.sh`、`diagnostics/gimbal/diagnose.sh`、`diagnostics/auto_aim/diagnose.sh` 降级为 `uv run sp-vision-diagnose ...` 的薄包装
- 旧脚本入口仍可用，但真正的业务逻辑已经统一到 Python diagnose 控制面

### 16. Python diagnose 子应用整理

- 将 `sp-vision-diagnose` 拆成 `camera`、`gimbal`、`auto-aim` 三个 Typer 子应用
- 每个子应用按动作拆分为独立命令，减少了 `action` 分发的集中度
- 保留了 `help` 子命令，便于和旧脚本调用习惯兼容
- 这一步让后续接入 TUI/状态面板更容易继续演进

### 17. TUI 骨架

- 新增了 `diagnose/tui.py`
- 新增了 `sp-vision-diagnose tui` 命令
- TUI 当前以状态总览和三个业务域分栏为主，作为后续参数面板和实时状态展示的起点
- 新增了 TUI 冒烟测试，确保入口可以被正常启动

### 18. CLI 参数收口

- 将 `camera`、`gimbal`、`auto-aim` 子应用里需要透传额外参数的命令统一改成 `ctx.args` 解析
- 补齐了 `gimbal list`，让三个业务域的列表命令都由 Python 直接接管
- 修正了 `camera quick`、`camera release`、`camera tune`、`auto-aim armor-box`、`auto-aim rune-tune` 等命令的参数分流
- 强制重装了本地 `uv` 环境里的 `sp-vision-25-python`，确保运行时加载到的是最新源码
- 这一轮之后，`uv run pytest` 和典型 diagnose 命令都恢复为稳定可用状态

### 19. UV editable 链路修正

- 将本地 `build_backend.py` 的 editable 构建改成源码链接式安装
- 现在 `uv run` 直接读取 `src/` 下的 Python 代码，不再依赖旧 wheel 拷贝
- 这能显著减少“改了源码但运行结果还是旧代码”的错觉和额外重装次数

### 20. diagnostics cpp 删除路线图

- 新增了 [docs/diagnostics_cpp_removal_roadmap.md](/home/cr/Codes/sp_vision_25/docs/diagnostics_cpp_removal_roadmap.md)
- 把 `diagnostics/` 下的 C++ 诊断程序分成“当前后端”与“未来可删”两类
- 明确了每个文件对应的 Python / `pybind11` 替代条件和删除前检查清单

### 20. TUI 仪表盘升级

- 将 `diagnose/tui.py` 从静态骨架升级为可刷新的仪表盘
- 现在 TUI 会展示 workspace 状态、pybind11 状态，以及 camera / gimbal / auto-aim 的二进制清单
- 为三个业务域补充了常用命令入口提示，后续可以继续在这里挂动态参数面板

### 21. 旧 diagnose 脚本删除

- 删除了 `diagnostics/camera/diagnose.sh`、`diagnostics/gimbal/diagnose.sh`、`diagnostics/auto_aim/diagnose.sh`
- 当前用户文档与调用示例已经迁移到 `sp-vision-diagnose` 入口
- 保留的 `diagnostics/*.cpp` 仍然是 Python diagnose 直接调用的 C++ 测试/诊断目标，不属于可删除的“旧东西”

### 22. gimbal Python 会话迁移

- 新增了 `bindings/io/gimbal_bindings.cpp`，把真实的 `io::Gimbal`、`GimbalState`、`GimbalRxStats`、`VisionToGimbal` 暴露给 Python
- 新增了 `src/sp_vision_25_python/diagnose/gimbal_session.py`，在 Python 侧实现 `snapshot`、`watch`、`control`、`script-control`
- `sp-vision-diagnose gimbal snapshot/watch/control/script-control` 现在直接走 Python + `pybind11`
- 删除了 `diagnostics/gimbal/gimbal_ui_test.cpp` 和对应的 CMake 目标，gimbal 的状态/控制链路从这里开始正式离开旧 C++ UI
- `gimbal list` 也同步收口，移除了不再使用的 `gimbal_ui_test`

### 23. gimbal 轴向诊断迁移

- 在 `gimbal_session.py` 中补齐了 `axis` 和 `manual-axis` 的 Python 实现
- `sp-vision-diagnose gimbal axis`、`sp-vision-diagnose gimbal manual-axis` 现在都直接走 Python 会话，不再依赖独立 C++ 诊断二进制
- 删除了 `diagnostics/gimbal/gimbal_axis_diag_test.cpp`、`diagnostics/gimbal/gimbal_manual_axis_diag_test.cpp`
- 对应的 CMake 目标也一并移除，gimbal 的高频交互诊断链路继续向 Python 收口

### 24. gimbal 二进制清单收口

- `gimbal list` 现在只展示仍然保留的 C++ 后端：`gimbal_link_diag_test` 和 `gimbal_serial_probe`
- `gimbal_axis_diag_test`、`gimbal_manual_axis_diag_test`、`gimbal_ui_test` 都已经从清单和构建中移除
- 这一步把 gimbal 的 Python / C++ 边界进一步收紧，后续只剩 link/probe 两条更底层的诊断链路还需要继续迁移
