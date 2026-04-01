# 重复实现、参考工程分叉与“空间检测”原理

本文回答两个问题：

1. 仓库里哪些地方存在“同一职责写了不止一份”
2. 你说的“空间检测”在这个工程里到底对应什么

## 1. 先说结论

仓库里确实存在不少“重复入口”或“平行实现”，但它们不全是坏的重复。

主要可以分成四类：

1. 生产入口和调试入口同时存在
2. 同一算法的 GUI 调试版和无界面版同时存在
3. 测试程序和正式程序都各自带了一份最小闭环
4. 两套参考电控工程长期分叉，名字相同但语义已经不完全一样

真正会影响你排障效率的，是第 4 类。

## 2. 视觉侧有哪些重复入口

## 2.1 自瞄主循环出现了多份

同类入口包括：

- `src/standard.cpp`
- `diagnostics/auto_aim/auto_aim_ui_test.cpp`
- `diagnostics/auto_aim/auto_aim_ui_tune.cpp`
- `tests/auto_aim/auto_aim_test.cpp`

它们都在做一条近似的链路：

1. 读图
2. 读姿态
3. 调 `auto_aim::Runtime`
4. 画调试信息或发命令

为什么会这样：

- `src/standard.cpp` 是正式运行入口
- `auto_aim_ui_test.cpp` 是工程调试入口
- `auto_aim_ui_tune.cpp` 是偏参数调试入口
- `tests/auto_aim/auto_aim_test.cpp` 是测试/回归入口

对排障的影响：

- 你看到“类似代码写了几遍”是正常的
- 但如果要查“真正部署到车上的路径”，优先看 `src/standard.cpp`
- 如果要查“我能不能导出日志”，优先看 `diagnostics/auto_aim/auto_aim_ui_test.cpp`

## 2.2 发云台命令的调试入口也有多份

同类入口包括：

- `diagnostics/gimbal/gimbal_ui_test.cpp`
- `diagnostics/gimbal/gimbal_axis_diag_test.cpp`
- `tests/gimbal/gimbal_test.cpp`
- `tests/planner/planner_test.cpp`

它们都直接或间接调用：

- `io::Gimbal::send(...)`

为什么会这样：

- 有的是手工控制姿态
- 有的是做轴向诊断
- 有的是验证 planner

对排障的影响：

- 如果你只想证明“协议链路通不通”，看 `gimbal_ui_test.cpp`
- 如果你只想证明“某个轴是不是走反了”，看 `gimbal_axis_diag_test.cpp`

## 2.3 重投影调试实现也分散在多处

同类文件包括：

- `diagnostics/auto_aim/auto_aim_ui_test.cpp`
- `diagnostics/auto_aim/auto_aim_ui_tune.cpp`
- `tests/auto_aim/auto_aim_test.cpp`
- `src/mt_auto_aim_debug.cpp`
- `src/sentry_debug.cpp`
- `src/uav_debug.cpp`
- `src/auto_aim_debug_mpc.cpp`

它们都在复用：

- `auto_aim::Solver::reproject_armor(...)`

为什么会这样：

- 每个入口都想现场看到“解出来的 3D 目标重新投回图像后对不对”
- 这是非常常见的视觉调试方式

这类重复不是协议问题，但会让新人误以为“同一个功能写了很多遍”。

更准确的说法是：

- 核心算法实现只有一份，在 `tasks/auto_aim/solver.cpp`
- 但显示和调用入口有很多份

## 2.4 打符链路也有平行调试实现

同类文件包括：

- `src/auto_buff_debug.cpp`
- `src/auto_buff_debug_mpc.cpp`
- `tests/auto_buff/auto_power_rune_test.cpp`

这些文件共享的核心是：

- `tasks/auto_buff/*`

这是另一套任务链路，和你现在的自瞄上车问题关系不大，但结构模式是类似的。

## 3. 参考电控工程里的重复才是你最该关心的

你给的两套参考工程里，很多文件是“同名、同职责、但实现语义已经分叉”的状态。

最关键的一组是：

- `Robo_USB.c`
- `Robo_AA.c`
- `Robo_gimbal.c`
- `Robo_Control.c`
- `Robo_Shoot.c`

这不是简单的“复制粘贴没删掉”，而是两套车端工程各自演化后的结果。

因此工程上最重要的原则是：

- 不要默认“两个工程名字一样，协议含义也一样”

你现在已经看到的典型分叉包括：

1. 入站 `pitch` 是否翻符号
2. 枪管仰角到底挂在 `pitch` 还是 `roll`
3. 回传给视觉的 `feedback_pitch` 到底是不是物理仰角

## 4. “空间检测”在这个仓库里最接近什么

这个仓库里没有一个正式模块就叫“空间检测”。

如果按工程含义来理解，最接近的是两部分：

1. `tasks/omniperception/`
2. `tasks/auto_aim/solver.cpp` 里的重投影校验

## 5. `tasks/omniperception/` 负责的是什么

它不是 SLAM，也不是地图建模。

它更接近：

- 多相机辅助感知
- 多视角目标发现
- 目标切换时的空间方向估计

关键文件：

- `tasks/omniperception/perceptron.cpp`
- `tasks/omniperception/decider.cpp`
- `src/sentry_multithread.cpp`

### 5.1 `Perceptron::parallel_infer`

作用：

- 四路 USB 相机并行读图
- 每路分别跑一份 YOLO
- 把检测结果压入 `DetectionResult` 队列

这一步得到的是：

- 哪个相机看到了目标
- 目标在这个相机图像里的归一化位置

### 5.2 `Decider::delta_angle`

作用：

- 根据装甲板在图像里的 `center_norm`
- 结合相机视场角 `FOV`
- 再加上左右/后相机的安装偏角
- 估算出“目标相对当前车体/云台大概在左多少、上多少”

这本质上是一种非常工程化的空间估计：

- 它不是精确三维重建
- 而是“把图像里的点，换算成应该往哪个方向转云台”

### 5.3 `Decider::armor_filter / sort / decide`

作用：

- 过滤掉非敌方、无敌、禁打目标
- 对多相机结果按优先级排序
- 生成辅助的 `yaw / pitch` 转向命令

因此如果你把“空间检测”理解成“多相机找目标并判断空间方向”，那这里就是最接近的实现。

## 6. `reproject_armor / world2pixel` 负责的是什么

关键文件：

- `tasks/auto_aim/solver.cpp`

关键函数：

- `reproject_armor(...)`
- `world2pixel(...)`

它们做的是另一种“空间一致性检查”。

直观理解：

1. 视觉先从图像里识别出装甲板
2. 解算器把它估成三维空间里的一个目标
3. 然后再把这个三维目标投回二维图像
4. 看投回去的位置和原始图像里的装甲板是否对得上

如果对得上，说明：

- 坐标系
- 相机内参
- 外参
- 解算结果

大体是一致的。

如果对不上，说明至少有一层空间关系错了。

所以从工程角度说：

- `omniperception` 更像“多相机空间发现和切目标”
- `reproject_armor/world2pixel` 更像“空间一致性校验”

## 7. 给不会 C++ 的人的一句话解释

如果把图像识别想成“看到了目标”，那这里的“空间检测”不是再做一次识别，而是在回答两个更工程的问题：

1. 目标在车的哪个方向
2. 这个方向换算和坐标系是不是自洽

对应到代码里就是：

- `tasks/omniperception/*` 负责“方向在哪边”
- `tasks/auto_aim/solver.cpp` 的重投影负责“这个方向算得对不对”

## 8. 你后面排障时的阅读优先级

如果你只想快速定位上车问题，按下面顺序看：

1. `docs/engineering_debug/01_transport_chain_and_logs.md`
2. `docs/engineering_debug/02_reference_vehicle_differences.md`
3. `diagnostics/gimbal/gimbal_ui_test.cpp`
4. `diagnostics/auto_aim/auto_aim_ui_test.cpp`
5. `references/.../Robo_USB.c`
6. `references/.../Robo_AA.c`
7. `references/.../Robo_gimbal.c`

如果你想理解“为什么这里到处像是重复代码”，再回来看本文档。
