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
