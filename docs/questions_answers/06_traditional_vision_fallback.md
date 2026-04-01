# 06 现在有回退传统视觉方法的功能吗

## 结论

有，但不是“完整切回纯传统视觉主链路”，而是“YOLO 后接传统方法做二次矫正”。

这是一个部分回退，不是全量回退。

## 当前实际行为

当前 `Runtime` 入口固定调用的是：

```cpp
auto armors = yolo_.detect(...)
```

见 `tasks/auto_aim/auto_aim_runtime.cpp:15-19`。

也就是说，主入口默认还是先走 YOLO。

## 传统方法在哪里介入

YOLOv5 内部有这个配置：

```cpp
use_traditional_ = yaml["use_traditional"].as<bool>();
```

见 `tasks/auto_aim/yolos/yolov5.cpp:27-29`。

然后在每个装甲板结果上，如果打开了 `use_traditional`，会调用传统检测器去做二次角点修正：

```cpp
if (use_traditional_) detector_.detect(*it, bgr_img);
```

见 `tasks/auto_aim/yolos/yolov5.cpp:183-185`。

## 这意味着什么

现在的“回退传统视觉”更准确地说是：

- YOLO 负责找目标
- 传统方法负责补角点/修角点

而不是：

- YOLO 完全失效时，主程序切换成纯 `Detector::detect` 路线

## 是否存在纯传统检测器

存在。

仓库里有单独的 `Detector` 类：

- `tasks/auto_aim/detector.hpp`
- `tasks/auto_aim/detector.cpp`

但当前标准运行时 `Runtime` 没有提供一个配置项，把 `YOLO::detect` 整体换成 `Detector::detect`。

## 所以答案分两层

### 1. 有没有传统方法能力

有。

### 2. 有没有“一键回退到纯传统视觉主链路”

当前没有现成总开关。

## 对 `standard3` 的现状判断

`configs/standard3.yaml` 里：

```yaml
use_traditional: true
```

所以 `standard3` 现在已经启用了“YOLO + 传统角点修正”的混合模式。

## 如果你要的是真正纯传统回退

那需要再做一层运行时切换，例如增加：

```yaml
detector_backend: yolo | traditional | hybrid
```

然后在 `Runtime::step` 里按配置决定调用：

- `yolo_.detect(...)`
- `detector_.detect(...)`
- 或者先 YOLO 后传统修正

## 结论复述

当前仓库已经有传统视觉参与链路，但它更像“辅助手段”，不是完整的“纯传统 fallback 模式”。
