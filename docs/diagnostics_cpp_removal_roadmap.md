# `diagnostics/*.cpp` Removal Roadmap

这份文档说明 `diagnostics/` 下现有 C++ 诊断程序如何逐步迁移到 Python / `pybind11`，以及每个文件在什么条件下才可以安全删除。

当前结论先写在前面：

- 这些文件**现在还不能直接删除**
- 它们大多仍然是 `sp-vision-diagnose` 的后端执行体
- 只有当对应的 Python 命令和 `pybind11` 会话能力完全替代后，才可以删源码、删 CMake 目标、删文档引用

## 1. 现存文件清单

- `diagnostics/gimbal/gimbal_link_diag_test.cpp`
- `diagnostics/gimbal/gimbal_serial_probe.cpp`
- `diagnostics/auto_aim/auto_aim_ui_test.cpp`
- `diagnostics/auto_aim/auto_aim_ui_tune.cpp`

## 2. 当前职责定位

### 2.1 Gimbal 诊断组

这些程序当前仍承担：

- 串口连通性检查
- 原始帧统计
- 云台姿态读取
- 控制模式交互
- 脚本化控制

### 2.2 Auto Aim 诊断组

这些程序当前仍承担：

- 装甲板在线画框
- 装甲板在线调参与快照
- Power Rune 离线回放
- Power Rune 调参
- 运行时参数导出

## 3. 迁移总原则

Python 不重写核心行为，只做控制面、界面和编排。

因此这批 C++ 诊断程序要么：

- 变成纯后端会话对象，由 Python TUI 驱动
- 要么彻底被 `pybind11` 暴露出来的 C++ runtime/session 替代

不能做的事情：

- 在 Python 里手写一份不同逻辑的“仿真版诊断”
- 在 C++ 和 Python 之间保留两套不一致的业务实现

## 4. 文件级路线图

### 4.1 `diagnostics/gimbal/gimbal_ui_test.cpp`

当前状态：

- 已由 Python 侧的 `GimbalSession` + `pybind11` 绑定接管
- `sp-vision-diagnose gimbal snapshot/watch/control/script-control` 现在走 Python
- 源文件和 CMake 目标已删除

后续建议：

- 继续把 `quick/rxonly/proto/probe/scan` 逐步转到 Python

### 4.2 `diagnostics/gimbal/gimbal_link_diag_test.cpp`

当前问题：

- 负责串口多端口扫描、原始帧统计、协议有效性判断

替代方向：

- Python 侧实现端口枚举和探测流程
- 核心串口收发仍然复用 `io::Gimbal`

删除条件：

- `sp-vision-diagnose gimbal quick/rxonly/proto/scan` 能完整替代
- Python 侧能够直接给出等价的统计/判定结果

### 4.3 `diagnostics/gimbal/gimbal_serial_probe.cpp`

当前问题：

- 偏底层字节流探测

替代方向：

- Python 探测层直接读取串口统计与原始帧样本
- 保留 `io::Gimbal` 作为真实通信后端

删除条件：

- `sp-vision-diagnose gimbal probe/probe-raw` 完全替代

### 4.4 `diagnostics/auto_aim/auto_aim_ui_test.cpp`

当前问题：

- 既负责 UI，又负责 snapshot，又负责 runtime 驱动

替代方向：

- 提取无 UI 的 `AutoAimDiagnoseSession`
- Python TUI / CLI 直接消费 snapshot
- 运行时逻辑由 `pybind11` 暴露的 runtime/session 接口承接

删除条件：

- `sp-vision-diagnose auto-aim armor-box/armor-intent/armor-rec/armor-offline/rune-box/rune-rec/rune-online/rune-online-mpc` 已完全由 Python + `pybind11` 接管

### 4.5 `diagnostics/auto_aim/auto_aim_ui_tune.cpp`

当前问题：

- C++ 中直接加载/修改/导出 YAML
- 运行时调参与持久化写回耦合

替代方向：

- 运行时 patch 交给 C++ runtime/session 接口
- YAML 读写和落盘交给 Python
- 这个文件应当停止扩展，并逐步缩成兼容层

删除条件：

- Python 版 `auto-aim tune` / `rune-tune` 完整接管
- Python 能直接从 runtime snapshot 生成可保存 patch

## 5. 建议的替换顺序

### 阶段 A：先替 gimbal

优先完成：

- `io::Gimbal` 的 Python 绑定
- `GimbalDiagnoseSession`
- `sp-vision-diagnose gimbal` 的 Python 实现

目标：

- 先让 gimbal 相关的 5 个诊断 cpp 失去独立必要性

### 阶段 B：再替 auto_aim

优先完成：

- `AutoAimDiagnoseSession`
- runtime snapshot 绑定
- Python 版 `armor-box` / `armor-tune` / `rune-tune`

目标：

- 让 `auto_aim_ui_test.cpp` 和 `auto_aim_ui_tune.cpp` 退出主路径

### 阶段 C：最后删源码

当满足以下三个条件时，才可以删：

1. `CMakeLists.txt` 不再 `add_executable(...)` 它们
2. `src/sp_vision_25_python/diagnose/actions.py` 不再调用对应二进制
3. 文档和测试不再引用这些可执行文件

## 6. 删除前检查清单

每删一个文件前，建议确认：

- [ ] Python diagnose 命令已经存在
- [ ] Python diagnose 测试通过
- [ ] 对应的 pybind11 / session 接口已经存在
- [ ] CMake 已移除该目标
- [ ] 文档中已不再把它写成当前入口
- [ ] `./build.sh` 通过
- [ ] `uv run pytest` 通过
