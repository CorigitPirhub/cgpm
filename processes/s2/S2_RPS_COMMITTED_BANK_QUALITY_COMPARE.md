# S2 committed rear-bank quality compare

协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_w_nz | bonn_rear_selected | bonn_commit_w_sum | bonn_commit_rho_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 30_rps_commit_geom_bg_soft_bank | 0.9413 | 91.93 | 2.8868 | 83.57 | -7.56 | 56 | 9 | 4.102 | 2.619 | control |
| 31_rps_commit_quality_bank_mid | 0.9412 | 91.93 | 2.8868 | 83.57 | -7.56 | 57 | 9 | 4.791 | 4.566 | abandon |
| 32_rps_commit_quality_bank_geom | 0.9411 | 91.93 | 2.8868 | 83.57 | -7.56 | 57 | 9 | 5.053 | 5.211 | abandon |
| 33_rps_commit_quality_bank_push | 0.9411 | 91.93 | 2.8868 | 83.57 | -7.56 | 57 | 9 | 5.143 | 5.707 | abandon |
