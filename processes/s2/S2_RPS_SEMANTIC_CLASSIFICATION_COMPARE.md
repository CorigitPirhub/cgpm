# S2 semantic classification compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_occlusion_order_mean | bonn_local_conflict_mean | bonn_front_residual_mean | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 68_rear_front_score_competition | 0.9358 | 93.12 | 4.3168 | 70.39 | 23.55 | 204 | 2 | 24 | 0.031 | 0.062 | 0.094 | control |
| 71_occlusion_order_consistency | 0.9358 | 93.12 | 4.3081 | 70.43 | 22.49 | 244 | 4 | 31 | 0.032 | 0.077 | 0.094 | abandon |
| 72_local_geometric_conflict_resolution | 0.9358 | 93.12 | 4.3956 | 68.81 | 27.27 | 234 | 4 | 24 | 0.032 | 0.112 | 0.094 | abandon |
| 73_front_residual_aware_suppression | 0.9358 | 93.12 | 4.3030 | 70.52 | 21.44 | 257 | 4 | 34 | 0.032 | 0.084 | 0.094 | abandon |
