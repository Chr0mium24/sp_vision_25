# 环境与构建

本文档说明当前仓库的依赖、编译方式和 Python 工作区使用方法。

## 1. 系统环境

- Ubuntu 22.04
- CMake + GCC/G++
- OpenCV、fmt、yaml-cpp、Eigen、spdlog
- 工业相机 SDK：
  - MindVision SDK，或
  - HikRobot MVS SDK
- 可选：
  - OpenVINO
  - ROS2 相关依赖

## 2. 安装基础依赖

```bash
sudo apt install -y \
    git \
    g++ \
    cmake \
    can-utils \
    libopencv-dev \
    libfmt-dev \
    libeigen3-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libusb-1.0-0-dev \
    nlohmann-json3-dev \
    openssh-server \
    screen
```

工业相机 SDK 和 OpenVINO 需要按各自官方安装包或安装说明单独准备。

## 3. 构建 C++ 工程

```bash
cmake -S cpp -B build
cmake --build build -j"$(nproc)"
```

构建完成后，常见输出位置如下：

- `build/bin/apps/...`：主程序
- `build/bin/tests/...`：C++ 测试与联调程序
- `build/bin/diag/...`：少量仍保留的诊断后端
- `build/python/`：pybind11 模块输出

## 4. 准备 Python 工作区

Python 相关内容现在统一位于 `python/`：

```text
python/
├── pyproject.toml
├── uv.lock
├── build_backend.py
└── src/sp_vision_25_python/
```

初始化环境：

```bash
uv --project python sync
```

运行 Python 测试：

```bash
uv --project python run pytest python/tests
```

查看 diagnose 状态：

```bash
uv --project python run sp-vision-diagnose status
uv --project python run sp-vision-diagnose bindings
```

如果你的机器上 `uv` 缓存目录需要单独指定，也可以这样运行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv --project python run sp-vision-diagnose status
```

## 5. 一次性初始化脚本

仓库提供了环境初始化脚本：

```bash
bash scripts/init_env.sh
```

如果需要同时安装本地相机 SDK 安装包：

```bash
bash scripts/init_env.sh \
    --mindvision-installer /path/to/MindVisionSDK.sh \
    --hikrobot-installer /path/to/HikRobotSDK.deb
```

## 6. 自启脚本

桌面环境自启使用：

```bash
scripts/autostart.sh
```

后台守护脚本使用：

```bash
scripts/watchdog.sh
```

运行日志默认写入 `logs/*.screenlog`。
