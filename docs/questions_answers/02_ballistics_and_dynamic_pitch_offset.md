# 02 弹速修改与动态 Pitch 偏移

## 当前实现

当前弹道模型是“理想抛体 + 一个常量 `pitch_offset`”，没有空气阻力，也没有距离相关残差项。

核心代码在：

- `tools/trajectory.cpp`
- `tasks/auto_aim/aimer.cpp`
- `tasks/auto_aim/planner/planner.cpp`

`tools::Trajectory` 只根据 `v0 / d / h` 解飞行时间和抬头角：

```cpp
auto a = g * d * d / (2 * v0 * v0);
auto b = -d;
auto c = a + h;
```

见 `tools/trajectory.cpp:9-30`。

随后 `Aimer` 和 `Planner` 都只是在理想解上再加一个固定 `pitch_offset`：

- `tasks/auto_aim/aimer.cpp:118-122`
- `tasks/auto_aim/planner/planner.cpp:172-176`

## 当前已经支持的调试能力

虽然没有“距离相关偏移模型”，但仓库已经支持在线调：

- 弹速 `bullet_speed`
- yaw/pitch 的实时 offset delta
- 导出新 YAML
- 记录调试日志

见：

- `diagnostics/auto_aim/auto_aim_ui_tune.cpp:364-431`
- `diagnostics/auto_aim/auto_aim_ui_tune.cpp:557-579`
- `diagnostics/auto_aim/auto_aim_ui_tune.cpp:700-706`
- `diagnostics/auto_aim/auto_aim_ui_tune.cpp:726-733`

也就是说，现在能做“在线粗调”，但还不能做“按距离自动补偿”。

## 直接回答

你的判断是对的：纯理想物理计算通常不够，应该再叠加一个经验残差模型。

## 最推荐的改法

建议把最终 pitch 改成：

```text
pitch_final = pitch_ideal(v0, d, h) + pitch_bias(d)
```

其中 `pitch_bias(d)` 不要一开始就上复杂空气阻力，先做一个可拟合、可解释、可调试的距离残差模型。

优先级建议：

1. 分段线性插值
2. 二次多项式
3. 更复杂空气阻力模型

## 为什么优先分段线性

如果你现在手上就是这种数据：

- `(1m, +1deg)`
- `(2m, +2deg)`

那分段线性插值最直接：

- 采点容易
- 调试直观
- 不会像高阶多项式那样在边界发散
- 适合不同枪管、不同供弹状态分别建表

## 一个实用建模方案

先在配置里新增类似字段：

```yaml
pitch_bias_points:
  - [1.0, 1.0]
  - [2.0, 2.0]
  - [3.0, 2.8]
```

语义是：

- 第 1 列：水平距离 `d`，单位 m
- 第 2 列：额外补偿角，单位 degree

运行时流程改成：

1. 用 `Trajectory` 算理想 `pitch`
2. 用当前距离 `d` 查表插值得到 `bias_deg`
3. 转成弧度后叠加到 `pitch`

## 调参流程

建议实测流程：

1. 先固定一个近似正确的 `bullet_speed`
2. 在 `1m / 2m / 3m / 4m / 5m` 做静态打点
3. 记录每个距离下命中所需额外 pitch 偏移
4. 拟合成 `pitch_bias(d)`
5. 最后只保留很小的全局 `pitch_offset`

这样职责就会变成：

- `bullet_speed`：主要决定整体弹道曲率
- `pitch_bias(d)`：吸收非理想空气阻力、枪口状态、测速误差
- `pitch_offset`：只留给整体零点微调

## 结论

现状是“理想弹道 + 常量偏移”，不够。

最稳妥的下一步不是直接上复杂空气阻力，而是先加一个“距离到 pitch 残差”的配置化模型，优先用分段线性插值。
