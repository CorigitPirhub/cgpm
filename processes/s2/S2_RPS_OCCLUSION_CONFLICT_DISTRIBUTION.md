# S2 occlusion conflict distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | occlusion_order_mean | local_conflict_mean | front_residual_mean | front_residual_drop_mean | dyn_risk_mean | dyn_risk_drop_mean | competition_mean | selectivity_topk_drop_sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 68_rear_front_score_competition | 204 | 2 | 24 | 178 | 0.031 | 0.062 | 0.094 | 0.092 | 0.000 | 0.035 | 0.250 | 177 |
| 71_occlusion_order_consistency | 244 | 4 | 31 | 209 | 0.032 | 0.077 | 0.094 | 0.090 | 0.000 | 0.042 | 0.258 | 137 |
| 72_local_geometric_conflict_resolution | 234 | 4 | 24 | 206 | 0.032 | 0.112 | 0.094 | 0.091 | 0.000 | 0.040 | 0.258 | 146 |
| 73_front_residual_aware_suppression | 257 | 4 | 34 | 219 | 0.032 | 0.084 | 0.094 | 0.090 | 0.000 | 0.045 | 0.261 | 124 |

重点检查：Ghost 是否压回 `<=15`，TB 是否恢复到 `>=8`，以及新时空特征均值是否拉开。
