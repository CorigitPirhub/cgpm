# S2 surface constrained compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 108 | 1 | 10 | 97 | control |
| 53_surface_adjacent_propagation | 0.9358 | 93.12 | 4.2421 | 70.84 | 15.17 | 94 | 3 | 13 | 78 | abandon |
| 54_normal_guided_manifold_extension | 0.9358 | 93.12 | 4.2426 | 70.84 | 14.77 | 96 | 3 | 12 | 81 | abandon |
| 55_surface_constrained_ray_projection | 0.9358 | 93.12 | 4.2425 | 70.83 | 14.51 | 101 | 3 | 14 | 84 | abandon |
