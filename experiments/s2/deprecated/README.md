# S2 Deprecated Chains

以下变体已在 `2026-03-11` 的 clean rerun 中确认失效，不再作为当前 `S2` 主线结论来源：

- `80_ray_penetration_consistency`
- `93_spatial_neighborhood_density_clustering`
- `97_global_map_anchoring`
- `99_manhattan_plane_completion`

说明：

- 它们当前仍保留在原脚本中，原因是部分脚本函数仍被其他诊断脚本复用。
- 当前废弃的是**历史变体链路**，而不是整份 carrier script。
- 当前可运行的 `S2` 起点已重置为：
  - `111_native_geometry_chain_direct`
  - `116_occupancy_entropy_gap_activation`
  - `125_hybrid_papg_constrained`
