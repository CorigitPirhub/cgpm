# S2 committed rear-bank competition compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_selected_sync_sum | bonn_extract_rear_selected_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 30_rps_commit_geom_bg_soft_bank | 0.9358 | 93.12 | 4.2389 | 70.87 | 13.25 | 35 | 7 | control |
| 34_bonn_compete_bgscore | 0.9358 | 93.12 | 4.2389 | 70.87 | 13.25 | 35 | 7 | abandon |
| 35_bonn_compete_softgap | 0.9358 | 93.12 | 4.2389 | 70.87 | 13.25 | 35 | 7 | abandon |
| 36_bonn_compete_softgap_support | 0.9358 | 93.12 | 4.2389 | 70.87 | 13.25 | 35 | 7 | abandon |
