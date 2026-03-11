# S2 multi-candidate generation compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_extract_rear_selected_sum | bonn_bridge_rear_synth_updates_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 108 | 1 | 10 | 97 | 585 | 0 | control |
| 62_dense_patch_projection | 0.9358 | 93.12 | 4.2498 | 70.82 | 15.02 | 378 | 7 | 53 | 318 | 1594 | 17616 | abandon |
| 63_multi_hypothesis_depth_sampling | 0.9358 | 93.12 | 4.2468 | 70.79 | 15.17 | 250 | 3 | 23 | 224 | 1092 | 9819 | abandon |
| 64_patch_depth_hybrid_generation | 0.9358 | 93.12 | 4.2509 | 70.80 | 16.48 | 442 | 8 | 60 | 374 | 1925 | 36612 | abandon |
