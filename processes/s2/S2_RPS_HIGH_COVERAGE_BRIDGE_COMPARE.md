# S2 high coverage bridge compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 108 | 1 | 10 | 97 | control |
| 59_relaxed_occlusion_tunneling | 0.9358 | 93.12 | 4.2417 | 70.86 | 15.02 | 65 | 2 | 9 | 54 | abandon |
| 60_cone_based_rear_projection | 0.9358 | 93.12 | 4.2417 | 70.88 | 15.02 | 65 | 2 | 9 | 54 | abandon |
| 61_hybrid_confidence_gating | 0.9358 | 93.12 | 4.2415 | 70.88 | 15.02 | 65 | 2 | 9 | 54 | abandon |
