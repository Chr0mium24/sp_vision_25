# 30. `tasks/auto_aim/` 目录

## 30.1 目录职责

`tasks/auto_aim/` 是标准自瞄算法主体，负责把图像转成可执行控制量。内部又分成四层：

- 检测层：`Detector`、`Classifier`、`YOLO*`
- 解算层：`Solver`
- 估计层：`Tracker`、`Target`
- 决策层：`Aimer`、`Shooter`、`Planner`

## 30.2 文件职责

| 文件 | 作用 |
| --- | --- |
| `tasks/auto_aim/CMakeLists.txt` | 定义 `auto_aim` 对象库，并引入 `planner/tinympc`。 |
| `tasks/auto_aim/armor.hpp` | 装甲板相关枚举、`Lightbar`、`Armor` 结构声明。 |
| `tasks/auto_aim/armor.cpp` | 灯条/装甲板构造逻辑，负责从关键点或分类结果生成 `Armor`。 |
| `tasks/auto_aim/classifier.hpp` | 图案分类器声明。 |
| `tasks/auto_aim/classifier.cpp` | 使用 OpenCV DNN 或 OpenVINO 对装甲板图案分类。 |
| `tasks/auto_aim/detector.hpp` | 传统灯条-装甲板检测器声明。 |
| `tasks/auto_aim/detector.cpp` | 基于轮廓、几何筛选、图案裁切的传统装甲板检测。 |
| `tasks/auto_aim/yolo.hpp` | YOLO 统一包装层与抽象基类。 |
| `tasks/auto_aim/yolo.cpp` | 根据配置选择 `YOLOV5/YOLOV8/YOLO11`。 |
| `tasks/auto_aim/yolos/yolov5.hpp` | YOLOv5 检测器声明。 |
| `tasks/auto_aim/yolos/yolov5.cpp` | YOLOv5 推理、解析、可选 ROI/传统方法融合。 |
| `tasks/auto_aim/yolos/yolov8.hpp` | YOLOv8 检测器声明。 |
| `tasks/auto_aim/yolos/yolov8.cpp` | YOLOv8 推理、关键点解析、类型与图案后处理。 |
| `tasks/auto_aim/yolos/yolo11.hpp` | YOLO11 检测器声明。 |
| `tasks/auto_aim/yolos/yolo11.cpp` | YOLO11 推理、关键点排序、类别与类型筛选。 |
| `tasks/auto_aim/solver.hpp` | 装甲板位姿解算器声明。 |
| `tasks/auto_aim/solver.cpp` | PnP 解算、坐标变换、yaw 优化、重投影。 |
| `tasks/auto_aim/target.hpp` | 单目标状态估计器声明。 |
| `tasks/auto_aim/target.cpp` | 装甲板多面体状态 EKF、装甲匹配、观测模型。 |
| `tasks/auto_aim/tracker.hpp` | 目标跟踪器声明。 |
| `tasks/auto_aim/tracker.cpp` | 跟踪状态机、目标初始化、目标更新、切目标逻辑。 |
| `tasks/auto_aim/aimer.hpp` | 传统自瞄瞄准器声明。 |
| `tasks/auto_aim/aimer.cpp` | 选取可打装甲板、做时间补偿和弹道迭代，结合 `tools::Trajectory` 输出随距离/高差变化的 yaw/pitch。 |
| `tasks/auto_aim/shooter.hpp` | 开火判定器声明。 |
| `tasks/auto_aim/shooter.cpp` | 根据命令连续性、回授误差和自动开火开关决定是否真的开火。 |
| `tasks/auto_aim/voter.hpp` | 计票器声明。 |
| `tasks/auto_aim/voter.cpp` | 对装甲板类型/身份进行计数投票。 |
| `tasks/auto_aim/auto_aim_runtime.hpp` | 单线程自瞄运行时总封装。 |
| `tasks/auto_aim/auto_aim_runtime.cpp` | `YOLO -> Tracker -> Aimer` 的一站式调用入口。 |

## 30.3 多线程子目录 `multithread/`

| 文件 | 作用 |
| --- | --- |
| `tasks/auto_aim/multithread/mt_detector.hpp` | 多线程/异步推理检测器声明。 |
| `tasks/auto_aim/multithread/mt_detector.cpp` | 预创建 OpenVINO `InferRequest`，异步提交图像并取回装甲板结果。 |
| `tasks/auto_aim/multithread/commandgener.hpp` | 后台决策发送线程声明。 |
| `tasks/auto_aim/multithread/commandgener.cpp` | 异步消费最新目标，调用 `Aimer/Shooter` 并通过 `CBoard` 发送控制。 |

## 30.4 规划子目录 `planner/`

| 文件 | 作用 |
| --- | --- |
| `tasks/auto_aim/planner/planner.hpp` | MPC 规划器声明，定义 `Plan` 和轨迹矩阵类型。 |
| `tasks/auto_aim/planner/planner.cpp` | 生成目标轨迹、调用 TinyMPC、输出带速度和加速度的控制计划。 |

## 30.5 `planner/tinympc/`

这是第三方 MPC 求解器源码，项目把它作为静态库嵌入。

| 文件 | 作用 |
| --- | --- |
| `tasks/auto_aim/planner/tinympc/CMakeLists.txt` | 构建 `tinympcstatic`。 |
| `tasks/auto_aim/planner/tinympc/admm.hpp` | ADMM 求解流程接口声明。 |
| `tasks/auto_aim/planner/tinympc/admm.cpp` | ADMM 主求解实现。 |
| `tasks/auto_aim/planner/tinympc/codegen.hpp` | 代码生成辅助声明。 |
| `tasks/auto_aim/planner/tinympc/codegen.cpp` | TinyMPC 代码生成辅助实现。 |
| `tasks/auto_aim/planner/tinympc/error.hpp` | 错误定义。 |
| `tasks/auto_aim/planner/tinympc/rho_benchmark.hpp` | `rho` 调参/基准测试接口。 |
| `tasks/auto_aim/planner/tinympc/rho_benchmark.cpp` | `rho` 更新与基准测试逻辑。 |
| `tasks/auto_aim/planner/tinympc/tiny_api.hpp` | 对外求解器 API。 |
| `tasks/auto_aim/planner/tinympc/tiny_api.cpp` | API 实现。 |
| `tasks/auto_aim/planner/tinympc/tiny_api_constants.hpp` | 常量定义。 |
| `tasks/auto_aim/planner/tinympc/types.hpp` | 求解器、缓存、工作区、设置结构定义。 |

## 30.6 核心函数

### 运行总入口

- `Runtime::step`：自瞄最短主链路入口，顺序是 `set_R_gimbal2world -> detect -> track -> aim`。

### 检测层

- `YOLO::detect`：统一入口，隐藏具体模型实现差异。
- `YOLOV5/8/11::detect`：OpenVINO 推理入口。
- `YOLOV5/8/11::postprocess`：解析输出张量，生成 `Armor`。
- `Detector::detect`：传统检测入口，适合测试、回退或辅助过滤。
- `Classifier::classify/ovclassify`：图案识别，补齐编号信息。

### 解算层

- `Solver::set_R_gimbal2world`：用四元数刷新世界坐标系旋转矩阵。
- `Solver::solve`：把 `Armor.points` 解成 `xyz/ypr/ypd`。
- `Solver::reproject_armor`：把世界坐标里的装甲板再投影回图像，供调试和优化。
- `Solver::optimize_yaw`：在搜索区间内最小化重投影误差，修正装甲板 yaw。

### 跟踪层

- `Tracker::track`：跟踪主入口。
- `Tracker::state_machine`：`lost/detecting/tracking/temp_lost/switching` 状态机。
- `Tracker::set_target`：根据首个装甲板初始化目标模型。
- `Tracker::update_target`：将观测装甲板融合进已有 EKF 状态。
- `Target::predict`：按常速度/常角速度模型预测目标状态。
- `Target::update`：决定观测对应哪一块装甲板，再做 EKF 更新。
- `Target::armor_xyza_list`：从整车状态反推出所有装甲板的位置和角度。
- `Target::h_jacobian`：观测方程雅可比，是 EKF 关键部分。

### 决策层

- `Aimer::aim`：考虑图像处理延迟、目标未来位置和弹道飞行时间，输出最终命令角。
- `Aimer::choose_aim_point`：在多块装甲板里选“当前最值得打的”那一块。
- `Shooter::shoot`：做保守开火判定，避免命令突变时误射。
- `Planner::plan`：MPC 版本决策器，输出位置、速度、加速度和 fire 标志。
- `Planner::get_trajectory`：从目标状态生成参考轨迹。

### 弹道 pitch 是怎么来的

这里的 `pitch` 不是固定值，也不是只由图像上下偏差直接决定。

实际链路是：

1. `Solver::solve` 先把目标解成三维位置 `xyz`
2. `Aimer::aim` 取当前瞄准点，计算水平距离 `d` 和高差 `h`
3. `tools::Trajectory(v0, d, h)` 根据子弹速度、距离和高差求飞行时间与弹道 `pitch`
4. `Aimer::aim` 再加上 `pitch_offset` 输出最终命令

所以：

- 目标更远时，`pitch` 一般会更大
- 目标更高时，`pitch` 也会更大
- 如果目标在运动，`Aimer` 还会根据飞行时间继续迭代预测目标未来位置

当前实现属于“基础弹道模型”：

- 已考虑重力
- 已考虑飞行时间
- 当前 `tools::Trajectory` 默认不考虑空气阻力

### 多线程

- `MultiThreadDetector::push/pop/debug_pop`：异步推理输入输出接口。
- `CommandGener::generate_command`：后台线程里连续生成并发送命令。

## 30.7 关键数据结构

- `Armor`：从图像检测结果一路扩展到 3D 结果的核心载体。
- `Target`：整车级状态估计对象，内部持有 11 维 EKF。
- `AimPoint`：Aimer 选中的具体瞄准点。
- `RuntimeInput/RuntimeOutput`：主链路输入输出协议。
- `Plan`：MPC 规划结果，包含控制量和导数信息。
