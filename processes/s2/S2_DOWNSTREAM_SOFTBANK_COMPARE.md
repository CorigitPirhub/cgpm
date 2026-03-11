# S2 downstream soft rear-bank export compare

协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | tum_bank_rear | tum_soft_bank_on | bonn_bank_rear | bonn_soft_bank_on | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 14_softbank_control | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 0 | 0 | 0 | 0 | control |
| 20_soft_rear_bank_export | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 0 | 0 | 0 | 0 | abandon |
| 21_soft_rear_bank_dual_compete | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 0 | 0 | 0 | 0 | abandon |
| 22_soft_rear_bank_nonpersistent | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | 0 | 0 | 0 | 0 | abandon |
