# S2 static anchored distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | history_anchor_mean | history_anchor_pre_mean | surface_anchor_mean | surface_anchor_pre_mean | surface_distance_mean | surface_distance_pre_mean | dynamic_shell_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 72_local_geometric_conflict_resolution_static_anchor_control | 234 | 3 | 30 | 201 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.000 |
| 74_static_history_weight_boosting | 252 | 4 | 33 | 215 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.000 |
| 75_dynamic_shell_masking | 234 | 3 | 28 | 203 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.000 |
| 76_surface_persistent_anchoring | 244 | 3 | 29 | 212 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.000 |

重点检查：TB 是否突破 `>=8`，Comp-R 是否恢复到 `>=70.5%`，Noise 是否明显下降。
