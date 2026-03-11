# S2 static anchored compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_history_anchor_mean | bonn_surface_anchor_mean | bonn_surface_distance_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 72_local_geometric_conflict_resolution_static_anchor_control | 0.9358 | 93.12 | 4.3094 | 70.43 | 22.10 | 234 | 3 | 30 | 201 | 0.000 | 0.000 | 0.0000 |
| 74_static_history_weight_boosting | 0.9358 | 93.12 | 4.3079 | 70.43 | 21.91 | 252 | 4 | 33 | 215 | 0.000 | 0.000 | 0.0000 |
| 75_dynamic_shell_masking | 0.9358 | 93.12 | 4.3937 | 68.82 | 26.43 | 234 | 3 | 28 | 203 | 0.000 | 0.000 | 0.0000 |
| 76_surface_persistent_anchoring | 0.9358 | 93.12 | 4.3908 | 68.81 | 26.70 | 244 | 3 | 29 | 212 | 0.000 | 0.000 | 0.0000 |
