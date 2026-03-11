# S2 ghost capped selectivity compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_extract_rear_selected_sum | bonn_rear_selectivity_drop_sum | bonn_rear_selectivity_topk_drop_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64_patch_depth_hybrid_generation | 0.9358 | 93.12 | 4.2509 | 70.80 | 16.48 | 442 | 8 | 60 | 374 | 442 | 0 | 0 | control |
| 65_ghost_risk_prediction_filter | 0.9358 | 93.12 | 4.2509 | 70.80 | 16.48 | 442 | 8 | 60 | 374 | 442 | 0 | 0 | abandon |
| 66_geometry_constrained_admission | 0.9358 | 93.12 | 4.2489 | 70.74 | 15.72 | 555 | 15 | 84 | 456 | 555 | 0 | 0 | abandon |
| 67_topk_selective_generation | 0.9358 | 93.12 | 4.3075 | 70.40 | 22.16 | 231 | 3 | 38 | 190 | 442 | 0 | 211 | abandon |
