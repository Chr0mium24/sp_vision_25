# 标定迁移日志

## 2026-04-22
- 新增 `sp-vision-calibration` Python 入口，统一承载 capture / camera / handeye / robotworld-handeye / split-video。
- 新增 `src/sp_vision_25_python/calibration/` 包，采集阶段复用现有 `Camera` / `CBoard` pybind11 绑定。
- 相机内参、手眼标定、robot-world hand-eye 标定改用 `cv2.calibrateCamera` / `cv2.calibrateHandEye` / `cv2.calibrateRobotWorldHandEye`。
- 删除 `calibration/*.cpp` 旧二进制实现，并从 `CMakeLists.txt` 移除对应 `add_executable`。
- 更新 `docs/calibration_workflow.md` 和 `readme.md`，把标定入口统一为 `sp-vision-calibration`。
- 补充 Python 单测，覆盖旋转工具、标定点生成和 CLI 冒烟测试。
