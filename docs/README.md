# sp_vision_25 文档

`docs/` 现在只保留当前可执行的说明文档。迁移过程中的方案、路线图和日志已经收口到 `docs/archive/`，避免和现行文档混在一起。

## 文档导航

- [环境与构建](./setup.md)
- [项目结构与架构](./architecture.md)
- [测试与联调](./testing.md)
- [标定流程](./calibration.md)
- Diagnose
  - [相机](./diagnose/camera.md)
  - [云台](./diagnose/gimbal.md)
  - [自瞄](./diagnose/auto_aim.md)
- [历史归档](./archive/README.md)

## 适用范围

本文档基于当前仓库布局：

```text
cpp/        C++ 运行时、算法、bindings
python/     Python 工程、CLI、校准工具
tests/      C++ 与 Python 测试
configs/    运行配置
assets/     模型、布局、示例数据
logs/       调试输出与运行日志
```

如果你只想快速开始，建议按这个顺序阅读：

1. [环境与构建](./setup.md)
2. [测试与联调](./testing.md)
3. 需要时再看对应 Diagnose 文档
