# S2 hybrid optimization distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | history_anchor_mean | history_anchor_pre_mean | surface_anchor_mean | surface_anchor_pre_mean | surface_distance_mean | surface_distance_pre_mean | dynamic_shell_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 72_local_geometric_conflict_resolution | 234 | 3 | 30 | 201 | 0.246 | 0.253 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.000 |
| 77_hybrid_boost_conflict | 244 | 4 | 29 | 211 | 0.247 | 0.253 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.000 |
| 78_conservative_anchoring | 252 | 3 | 32 | 217 | 0.246 | 0.253 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.001 |
| 79_feature_weighted_topk | 224 | 3 | 29 | 192 | 0.250 | 0.253 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.000 |

重点检查：特征统计必须非零；TB 是否提升到 `>=6`；Ghost 是否控制在 `<=25`。
