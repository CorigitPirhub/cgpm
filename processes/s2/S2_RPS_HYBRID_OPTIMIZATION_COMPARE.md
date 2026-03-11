# S2 hybrid optimization compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_history_anchor_mean | bonn_surface_anchor_mean | bonn_dynamic_shell_mean | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 72_local_geometric_conflict_resolution | 0.9358 | 93.12 | 4.3094 | 70.43 | 22.10 | 234 | 3 | 30 | 201 | 0.246 | 1.000 | 0.000 | control |
| 77_hybrid_boost_conflict | 0.9358 | 93.12 | 4.3085 | 70.41 | 21.83 | 244 | 4 | 29 | 211 | 0.247 | 1.000 | 0.000 | abandon |
| 78_conservative_anchoring | 0.9358 | 93.12 | 4.3074 | 70.44 | 21.37 | 252 | 3 | 32 | 217 | 0.246 | 1.000 | 0.001 | abandon |
| 79_feature_weighted_topk | 0.9358 | 93.12 | 4.3126 | 70.42 | 22.24 | 224 | 3 | 29 | 192 | 0.250 | 1.000 | 0.000 | abandon |
