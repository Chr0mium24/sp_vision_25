# `diagnostics/*.cpp` Removal Roadmap

这份文档说明 `diagnostics/` 下现有 C++ 诊断程序如何逐步迁移到 Python / `pybind11`，以及每个文件在什么条件下才可以安全删除。

当前结论先写在前面：

- `gimbal` 相关诊断程序已经全部迁到 Python / `pybind11`
- `auto_aim_ui_tune.cpp` 也已经迁移并删除
- 这份路线图当前仅保留为历史记录，真正需要删的 `diagnostics/*.cpp` 诊断程序已经清空

## 1. 迁移总原则

Python 不重写核心行为，只做控制面、界面和编排。

因此这批 C++ 诊断程序要么：

- 变成纯后端会话对象，由 Python TUI 驱动
- 要么彻底被 `pybind11` 暴露出来的 C++ runtime/session 替代

不能做的事情：

- 在 Python 里手写一份不同逻辑的“仿真版诊断”
- 在 C++ 和 Python 之间保留两套不一致的业务实现

## 2. 最终收口条件

当满足以下三个条件时，才可以删任何仍在服役的诊断 C++ 文件：

1. `CMakeLists.txt` 不再 `add_executable(...)` 它们
2. `src/sp_vision_25_python/diagnose/actions.py` 不再调用对应二进制
3. 文档和测试不再引用这些可执行文件

## 3. 删除前检查清单

每删一个文件前，建议确认：

- [ ] Python diagnose 命令已经存在
- [ ] Python diagnose 测试通过
- [ ] 对应的 pybind11 / session 接口已经存在
- [ ] CMake 已移除该目标
- [ ] 文档中已不再把它写成当前入口
- [ ] `./build.sh` 通过
- [ ] `uv run pytest` 通过
