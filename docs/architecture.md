# 项目结构与架构

本文档说明当前仓库的目录分工，以及 C++ / Python 在项目里的边界。

## 1. 顶层目录

```text
cpp/        C++ 核心代码
python/     Python 工程
configs/    YAML 配置
assets/     模型、布局、示例输入
docs/       使用文档
scripts/    环境初始化、自启与辅助脚本
logs/       调试图与运行日志
build/      本地构建产物
```

## 2. C++ 部分

`cpp/` 是运行时核心，负责硬件访问、实时算法和 pybind11 绑定。

```text
cpp/
├── CMakeLists.txt
├── apps/       主程序入口
├── io/         相机、串口、云台、ROS2 等 I/O
├── tasks/      auto_aim、auto_buff、omniperception
├── tests/      C++ 测试与联调程序
├── tools/      通用基础库
└── bindings/   pybind11 绑定
```

职责划分：

- `io/`：设备访问与协议封装
- `tasks/`：业务算法和运行时流程
- `tools/`：公共数学、协议、并发、运行时工具
- `bindings/`：把真实 C++ 类型暴露给 Python

## 3. Python 部分

`python/` 是单独的 Python 工程，负责控制面、诊断命令、标定工具和 Python 单测。

```text
python/
├── pyproject.toml
├── tests/
├── uv.lock
├── build_backend.py
└── src/
    └── sp_vision_25_python/
        ├── calibration/
        └── diagnose/
```

职责划分：

- `diagnose/`：命令入口、状态查看、设备探测、会话编排
- `calibration/`：标定采集与标定计算流程
- pybind11 模块：复用真实 C++ 类型和算法，不维护 Python 版替身实现

## 4. 测试部分

测试已经分别放回各自子工程：

```text
cpp/tests/
python/tests/
```

- `cpp/tests/`：C++ smoke test、设备联调程序、离线算法测试
- `python/tests/`：Python CLI 测试与 pybind11 行为测试

## 5. 配置、资源与日志

- `configs/`：运行配置和标定配置
- `assets/`：模型、样例输入、布局文件
- `assets/layouts/`：PlotJuggler 等布局资源
- `logs/`：运行日志和调试输出
- `logs/auto_aim/patterns`：装甲板图案调试输出
- `logs/auto_aim/imgs`：检测模型调试图输出

## 6. 当前设计原则

- C++ 保持实时链路和硬件基座
- Python 负责控制面和工具链，不重写核心算法
- 文档优先描述“现在怎么用”，迁移历史单独归档
