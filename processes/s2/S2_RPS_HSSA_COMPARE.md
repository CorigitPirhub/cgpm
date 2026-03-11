# S2 HSSA compare

| variant | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | added_support_points_sum | bonn_tb_support_score_mean | bonn_noise_support_score_mean | tb_noise_correlation | bonn_comp_r_5cm | bonn_acc_cm | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 80_ray_penetration_consistency | 23.20 | 4 | 19 | 96 | 0 | 0.210 | 0.146 | 0.991 | 70.20 | 4.317 | control |
| 92_multi_view_ray_support_aggregation | 23.23 | 5 | 19 | 97 | 2 | 0.219 | 0.147 | 0.991 | 70.21 | 4.317 | abandon |
| 93_spatial_neighborhood_density_clustering | 23.59 | 13 | 19 | 96 | 9 | 0.241 | 0.146 | 0.991 | 70.20 | 4.318 | abandon |
| 94_historical_tsdf_consistency_reactivation | 22.45 | 8 | 19 | 96 | 4 | 0.232 | 0.146 | 0.991 | 70.21 | 4.317 | abandon |
