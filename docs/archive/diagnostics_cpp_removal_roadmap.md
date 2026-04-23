# 旧诊断 C++ 收缩路线图

这份路线图保留为历史备注。

## 当前状态

- 旧的 diagnose shell 脚本已经退出主流程
- gimbal 诊断主入口已经迁到 Python
- 少量 `build/bin/diag/auto_buff/*` 后端仍被在线打符命令使用

因此当前仓库的维护重点已经不是“继续写删除路线图”，而是保持现行入口文档准确。

请优先参考：

- [../testing.md](../testing.md)
- [../diagnose/auto_aim.md](../diagnose/auto_aim.md)
