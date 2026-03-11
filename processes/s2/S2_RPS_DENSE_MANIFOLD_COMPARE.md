# S2 dense manifold compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 108 | 1 | 10 | 97 | control |
| 50_dense_background_propagation | 0.9358 | 93.12 | 4.2421 | 70.86 | 13.75 | 67 | 1 | 7 | 59 | abandon |
| 51_geometry_guided_manifold_completion | 0.9358 | 93.12 | 4.2428 | 70.84 | 14.41 | 88 | 2 | 12 | 74 | abandon |
| 52_dual_scale_manifold_fusion | 0.9358 | 93.12 | 4.2429 | 70.82 | 14.71 | 112 | 1 | 13 | 98 | abandon |
