# S2 downstream sync/export chain compare

协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | tum_bank_front | tum_bank_bg | tum_bank_rear | bonn_bank_front | bonn_bank_bg | bonn_bank_rear | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 14_downstream_control | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 24913 | 0 | 0 | 24491 | 0 | 0 | control |
| 17_banked_export_compete | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 24913 | 0 | 0 | 24491 | 0 | 0 | abandon |
| 18_banked_dual_compete_export | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 24913 | 0 | 0 | 24491 | 0 | 0 | abandon |
| 19_banked_nonpersistent_export | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 24913 | 0 | 0 | 24491 | 0 | 0 | abandon |
