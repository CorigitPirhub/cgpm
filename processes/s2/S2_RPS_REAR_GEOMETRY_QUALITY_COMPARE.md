# S2 rear geometry quality compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision | handoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 108 | 1 | 10 | 97 | control | space_redirect_tested_no_gain_keep_38 |
| 40_bonn_geometry_aligned_admission | 0.9358 | 93.12 | 4.2435 | 70.77 | 15.22 | 144 | 2 | 16 | 126 | abandon | superseded_by_space_redirect_round |
| 41_bonn_geometry_occlusion_admission | 0.9358 | 93.12 | 4.2433 | 70.77 | 15.22 | 149 | 3 | 18 | 128 | abandon | superseded_by_space_redirect_round |
| 42_bonn_geometry_density_gate | 0.9358 | 93.12 | 4.2434 | 70.77 | 15.22 | 148 | 2 | 18 | 128 | abandon | superseded_by_space_redirect_round |

补充说明：后续 `43/44/45` 的 space-redirect 复核表明，本轮几何质量候选没有超过 `38_bonn_state_protect`。
