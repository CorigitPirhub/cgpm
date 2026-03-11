# S2 HSSA distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | true_background_sum | ghost_sum | hole_or_noise_sum | added_support_points_sum | tb_support_score_mean | noise_support_score_mean | tb_neighbor_count_mean | noise_neighbor_count_mean | tb_history_reactivate_mean | noise_history_reactivate_mean | tb_noise_correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80_ray_penetration_consistency | 4 | 19 | 96 | 0 | 0.210 | 0.146 | 4.250 | 6.469 | 0.230 | 0.208 | 0.991 |
| 92_multi_view_ray_support_aggregation | 5 | 19 | 97 | 2 | 0.219 | 0.147 | 4.000 | 6.433 | 0.230 | 0.209 | 0.991 |
| 93_spatial_neighborhood_density_clustering | 13 | 19 | 96 | 9 | 0.241 | 0.146 | 3.385 | 6.469 | 0.230 | 0.208 | 0.991 |
| 94_historical_tsdf_consistency_reactivation | 8 | 19 | 96 | 4 | 0.232 | 0.146 | 3.625 | 6.469 | 0.230 | 0.208 | 0.991 |

重点检查：`support_score(TB) > support_score(Noise)` 是否成立，以及 `TB` 是否突破 `6`。
