# S2 space redirect compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 1 | 10 | 97 | control |
| 43_history_guided_background_location | 0.9358 | 93.12 | 4.2429 | 70.78 | 14.86 | 5 | 26 | 162 | abandon |
| 44_history_plus_ghost_suppress | 0.9358 | 93.12 | 4.2438 | 70.76 | 15.22 | 4 | 25 | 159 | abandon |
| 45_visual_evidence_anchor_strict | 0.9358 | 93.12 | 4.2429 | 70.78 | 14.86 | 5 | 26 | 162 | abandon |
