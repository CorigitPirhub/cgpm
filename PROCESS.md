# PROCESS

## 2026-03-11 / S2 Pipeline Repair

### 事件
- 项目结构重整与结果清理后，`S2` 多个 runner 因旧路径与旧依赖失效而断链。
- 典型故障为：`processes/s2` 写盘失败、`output/s2_stage` 读取失效、直接运行时 `run_benchmark` 导入失败。

### 已修复
- `S2` runner 输出根统一迁移到 `output/s2/`。
- `S2` 阶段运行目录统一迁移到 `output/s2_stage/`。
- 补齐多个 `run_s2_rps_*.py` 的直接运行导入路径。
- 新增稳定原生基线入口：`experiments/s2/run_s2_native_geometry_chain.py`。

### 当前有效结论
- `111_native_geometry_chain_direct` 已恢复可运行，当前复跑结果约为：`Acc=4.452 cm`, `Comp-R=66.87%`。
- `116_occupancy_entropy_gap_activation` 仍是当前可运行的 Oracle 上界。
- `125_hybrid_papg_constrained` 可运行，但未恢复到历史最佳数值。

### 当前失效链
以下历史链条在当前 clean rerun 下已失效，不再作为当前主结论来源：
- `80_ray_penetration_consistency`
- `93_spatial_neighborhood_density_clustering`
- `97_global_map_anchoring`
- `99_manhattan_plane_completion`

补充诊断：
- `80` 的直接控制根 `72/80` 在当前代码下已经生成 `commit` 非零但 `surface_points = 0` 的空表面结果；
- `93/97/99` 进一步依赖这些空中间产物，因此只是级联失效，而不是独立 bug；
- 当前结论是：这些链条同时具有“非独立中间产物依赖”与“当前核心库语义不兼容”两类问题，应视为历史诊断链，而不是当前主线可修复资产。

### 状态判断
- `S2` 仍未通过。
- 绝对禁止进入 `S3`。
