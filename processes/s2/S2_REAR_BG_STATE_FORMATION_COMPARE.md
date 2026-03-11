# S2 rear/bg state formation compare

协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | tum_rear_cand_nz | tum_bg_w_nz | tum_bg_cand_nz | bonn_rear_cand_nz | bonn_bg_w_nz | bonn_bg_cand_nz | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 14_rearbg_control | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 2855 | 0 | 0 | 2937 | 0 | 0 | control |
| 23_rear_candidate_support_rescue | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 2855 | 0 | 0 | 2937 | 0 | 0 | abandon |
| 24_joint_bg_state_coformation | 1.0058 | 73.93 | 2.8868 | 83.53 | -7.12 | 2855 | 2855 | 2855 | 2937 | 2937 | 2937 | abandon |
| 25_rear_bg_coupled_formation | 0.9400 | 91.90 | 2.8867 | 83.57 | -7.56 | 2855 | 2855 | 2855 | 2937 | 2937 | 2937 | iterate |
