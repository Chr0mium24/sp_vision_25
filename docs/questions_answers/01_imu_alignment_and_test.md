# 01 IMU 调整与测试

## 结论

这个测试仓库里其实已经有雏形了，不需要从零再写一版“发几组 IMU 数据”的程序。

现有最接近你需求的是：

- `diagnostics/gimbal/gimbal_axis_diag_test.cpp`
- `diagnostics/gimbal/gimbal_manual_axis_diag_test.cpp`

其中 `gimbal_axis_diag_test.cpp` 已经会：

- 给云台发送几组 yaw/pitch 小步进命令
- 读取回传姿态
- 枚举候选 `R_gimbal2imubody`
- 评分哪个矩阵最符合“纯 yaw 只主要影响 yaw、纯 pitch 只主要影响 pitch”

这正对应“使机器对齐、最后以步兵为准”的需求。

## 代码依据

`R_gimbal2world` 的计算方式是：

```cpp
R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
```

见：

- `tasks/auto_aim/solver.cpp:48-52`

而现有诊断程序已经会对候选矩阵做评估：

- 采样原始姿态：`diagnostics/gimbal/gimbal_axis_diag_test.cpp:58-77`
- 用候选 `R_gimbal2imubody` 重新变换：`diagnostics/gimbal/gimbal_axis_diag_test.cpp:80-107`
- 发送 yaw/pitch 步进计划：`diagnostics/gimbal/gimbal_axis_diag_test.cpp:122-173`
- 穷举所有合法旋转候选：`diagnostics/gimbal/gimbal_axis_diag_test.cpp:190-225`

## 推荐做法

建议把“步兵”作为机械和坐标约定的基准机型，其它车都对齐到它。

实操顺序：

1. 在步兵上运行 `gimbal_axis_diag_test`，先找出最合理的 `R_gimbal2imubody`。
2. 再用 `gimbal_manual_axis_diag_test` 手动拨动云台做二次确认。
3. 把确认后的矩阵写回对应配置。
4. 其它车型参照步兵的轴定义，不要每台车自定义一套坐标语义。

## 需要补写的新东西吗

如果你的意思是“伪造 IMU 数据流，直接喂给算法”，当前仓库没有现成的假 IMU 注入层。

但从工程收益看，先用现有诊断程序完成轴对齐更合适，因为：

- 真实问题通常出在机械安装和轴定义，不在算法本身
- 真实闭环下测出来的矩阵更可信
- 现有代码已经覆盖了“发送几组动作并评分”的关键部分

## 对 `standard3` 的判断

`configs/standard3.yaml` 和 `configs/handeye.yaml` 目前都用了同一套：

```yaml
R_gimbal2imubody: [0, -1, 0, 1, 0, 0, 0, 0, 1]
```

这说明当前仓库已经把“步兵为准”的结果写进配置了，但仍需要靠诊断程序再确认一次是否和当前车体装配一致。
