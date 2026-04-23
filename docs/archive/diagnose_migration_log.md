# Diagnose 迁移日志

这份文件只保留迁移已经完成的结论，不再记录逐日施工细节。

## 已完成结果

- `sp-vision-diagnose` 成为统一 diagnose 入口
- gimbal 高频交互链路已经收口到 Python 会话
- Python 工程已经迁移到 `python/`
- 旧的 `python_tests/` 已并入 `python/tests`
- 当前使用文档已经拆到：
  - [../diagnose/camera.md](../diagnose/camera.md)
  - [../diagnose/gimbal.md](../diagnose/gimbal.md)
  - [../diagnose/auto_aim.md](../diagnose/auto_aim.md)

如果你需要当前命令说明，请不要再参考旧迁移日志。
