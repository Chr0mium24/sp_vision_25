# Camera Diagnose

本文档说明 `sp-vision-diagnose camera` 的当前用法。

## 1. 基本命令

```bash
uv --project python run sp-vision-diagnose camera help
```

常用动作：

- `list`
- `info`
- `release`
- `tune`
- `quick`
- `detect`
- `window`
- `save`
- `usb`
- `usb-detect`
- `thread`
- `handeye`

## 2. 查看设备与环境

```bash
uv --project python run sp-vision-diagnose camera info
```

作用：

- 列出 `/dev/video*`
- 如果系统存在 `v4l2-ctl`，同时打印设备详情

## 3. 工业相机快速联通

```bash
uv --project python run sp-vision-diagnose camera quick configs/standard3.yaml
```

适合先确认：

- SDK 已安装
- 配置里的相机名正确
- 能稳定取流

## 4. 释放相机占用

```bash
sudo uv --project python run sp-vision-diagnose camera release
sudo uv --project python run sp-vision-diagnose camera release --vidpid=2bdf:0001 --force
```

用途：

- 停掉占用工业相机的容器或进程
- 清理相关 USB 句柄

如果你使用了独立缓存目录，记得把环境变量放在 `sudo` 后面：

```bash
sudo env UV_CACHE_DIR=/tmp/uv-cache uv --project python run sp-vision-diagnose camera release --force
```

## 5. 调参窗口

```bash
uv --project python run sp-vision-diagnose camera tune configs/standard3.yaml --scale=0.7
```

交互命令：

- `e` 或 `e <num>`：设置 `exposure_ms`
- `g` 或 `g <num>`：设置 `gain`
- `r`：手动重载窗口
- `p`：打印当前参数
- `q`：退出

如果你想看到窗口进程日志：

```bash
uv --project python run sp-vision-diagnose camera tune configs/standard3.yaml --scale=0.7 --show-log
```

## 6. 预览、检测与保存

```bash
uv --project python run sp-vision-diagnose camera window configs/standard3.yaml --scale=0.7
uv --project python run sp-vision-diagnose camera detect configs/standard3.yaml
uv --project python run sp-vision-diagnose camera save configs/standard3.yaml --output-folder=assets/camera_captures
```

说明：

- `window`：单纯预览窗口
- `detect`：带检测链路
- `save`：在窗口里按 `s` 保存图像

## 7. USB 相机

```bash
uv --project python run sp-vision-diagnose camera usb configs/sentry.yaml --name=video0 -d
uv --project python run sp-vision-diagnose camera usb-detect configs/sentry.yaml --name=video0 -d
```

## 8. 线程与手眼测试

```bash
uv --project python run sp-vision-diagnose camera thread configs/ascento.yaml
uv --project python run sp-vision-diagnose camera handeye configs/handeye.yaml -d
```

## 9. 常见问题

- 无图像：检查 SDK、`camera_name` 和设备权限
- 相机被占用：先执行 `camera release`
- 无 GUI 环境：优先使用 `quick` 或关闭显示参数
