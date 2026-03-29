# gimbal `diagnose.sh control` 参考包

这个目录整理了运行 `diagnostics/gimbal/diagnose.sh control` 相关的最小源码和配置上下文，按原项目目录结构保留，便于查找相对路径。

包含内容：
- `diagnostics/gimbal/diagnose.sh`
  - `control` 动作的入口脚本
- `diagnostics/gimbal/gimbal_ui_test.cpp`
  - `control` 实际调用的主程序源码
- `io/gimbal/`
  - 串口协议定义与收发实现
- `io/serial/`
  - 串口库源码
- `tools/`
  - `gimbal_ui_test` 和 `io::Gimbal` 直接依赖的工具文件
- `configs/standard3.yaml`
  - 默认配置文件，里面有 `com_port`
- `CMakeLists.txt`、`build.sh`
  - 目标定义和常用构建入口

说明：
- 当前仓库里没有现成的 `build/bin/diag/gimbal/gimbal_ui_test` 二进制，所以这里只复制了源码和配置，没有复制可执行文件。
- 这个目录是“参考包”，用于看链路、查协议、补环境和二次整理；不是完整独立工程。
