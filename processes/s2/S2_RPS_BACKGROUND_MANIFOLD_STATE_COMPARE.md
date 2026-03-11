# S2 background manifold state compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 108 | 1 | 10 | 97 | control |
| 48_stable_background_memory_state | 0.9358 | 93.12 | 4.2407 | 70.86 | 13.50 | 29 | 1 | 3 | 25 | abandon |
| 49_relaxed_manifold_guided_generation | 0.9358 | 93.12 | 4.2413 | 70.88 | 13.50 | 39 | 1 | 4 | 34 | abandon |
