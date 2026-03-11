# S2 discriminative fusion compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_front_score_mean | bonn_rear_score_mean | bonn_rear_gap_mean | bonn_competition_mean | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 67_topk_selective_generation | 0.9358 | 93.12 | 4.3086 | 70.37 | 22.43 | 231 | 3 | 41 | 0.174 | 0.419 | 0.2450 | 0.280 | control |
| 68_rear_front_score_competition | 0.9358 | 93.12 | 4.3148 | 70.33 | 23.52 | 204 | 2 | 32 | 0.174 | 0.414 | 0.2405 | 0.249 | abandon |
| 69_depth_gap_validation | 0.9358 | 93.12 | 4.4522 | 67.40 | 35.21 | 0 | 0 | 0 | 0.000 | 0.000 | 0.0000 | 0.000 | abandon |
| 70_fused_discriminator_topk | 0.9358 | 93.12 | 4.4522 | 67.40 | 35.21 | 0 | 0 | 0 | 0.000 | 0.000 | 0.0000 | 0.000 | abandon |
