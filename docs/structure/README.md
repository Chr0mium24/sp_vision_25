# sp_vision_25 结构文档索引

这组文档的目标不是重复 `readme.md`，而是把项目拆成“目录职责、文件职责、核心函数、核心数据结构、数据流、Mermaid 图”六个视角，方便新人顺着代码往下看。

## 建议阅读顺序

1. `01_overview.md`
2. `10_tools.md`
3. `20_io.md`
4. `30_tasks_auto_aim.md`
5. `31_tasks_auto_buff_and_omniperception.md`
6. `40_entrypoints.md`
7. `50_calibration_configs_assets_docs_scripts.md`
8. `60_tests_and_diagnostics.md`
9. `70_core_functions.md`
10. `71_core_data_structures.md`
11. `72_data_flow.md`
12. `mermaid/`

## 约定

- “核心模块”指实际参与运行链路的代码。
- `io/serial`、`io/hikrobot/include`、`io/mindvision/include`、`tasks/auto_aim/planner/tinympc` 中有明显第三方代码；文档会按“在本项目里的作用”描述，不逐行解释实现。
- 某些文件未直接参与默认编译链路，例如 `tasks/auto_buff/buff_predict.hpp`，但仍然记录其用途和历史位置。

