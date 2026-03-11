# S2 occlusion bridge compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 108 | 1 | 10 | 97 | control |
| 56_temporal_occlusion_tunneling | 0.9358 | 93.12 | 4.2409 | 70.89 | 15.02 | 57 | 2 | 9 | 46 | abandon |
| 57_historical_surface_rear_projection | 0.9358 | 93.12 | 4.2408 | 70.89 | 15.02 | 58 | 3 | 9 | 46 | abandon |
| 58_ghost_aware_surface_inpainting | 0.9358 | 93.12 | 4.2416 | 70.87 | 14.87 | 58 | 2 | 9 | 47 | abandon |
