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
