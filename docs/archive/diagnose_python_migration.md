# Diagnose Python 化方案

这份文件原本用于规划 Python diagnose 控制面重构，现在已经退役。

## 当前结论

- Python 负责 CLI、TUI、配置编辑和会话编排
- C++ 负责实时运行时、硬件访问和核心算法
- 白盒能力优先复用 pybind11 暴露的真实 C++ 类型

当前现行说明见：

- [../architecture.md](../architecture.md)
- [../testing.md](../testing.md)
- [../diagnose/camera.md](../diagnose/camera.md)
- [../diagnose/gimbal.md](../diagnose/gimbal.md)
- [../diagnose/auto_aim.md](../diagnose/auto_aim.md)
