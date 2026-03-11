# S2 history-visible obstructed compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 1 | 10 | 97 | control |
| 46_history_background_only_admission | 0.9358 | 93.12 | 4.2389 | 70.86 | 13.29 | 0 | 0 | 0 | abandon |
| 47_history_visible_obstructed_manifold | 0.9358 | 93.12 | 4.2389 | 70.86 | 13.29 | 0 | 0 | 0 | abandon |
