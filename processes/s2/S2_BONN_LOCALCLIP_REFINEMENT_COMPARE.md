# S2 Bonn local clipping / calibration refinement compare

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | decision |
|---|---:|---:|---:|---:|---:|---|
| 14_bonn_localclip_drive_recheck | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | control |
| 15_bonn_localclip_band_relax | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | abandon |
| 16_bonn_localclip_pfv_rearexpand | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | abandon |
