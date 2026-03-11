# S2 committed rear-bank admission compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_extract_hard_commit_on_sum | bonn_extract_rear_enabled_sum | bonn_extract_rear_selected_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 30_rps_commit_geom_bg_soft_bank | 0.9358 | 93.12 | 4.2389 | 70.87 | 13.25 | 7 | 7 | 7 | control |
| 37_bonn_admission_gate_relax | 0.9358 | 93.12 | 4.2431 | 70.78 | 15.00 | 193 | 193 | 193 | abandon |
| 38_bonn_state_protect | 0.9358 | 93.12 | 4.2435 | 70.83 | 15.47 | 108 | 108 | 108 | iterate |
| 39_bonn_admission_gate_plus_protect | 0.9358 | 93.12 | 4.2431 | 70.78 | 15.11 | 202 | 202 | 202 | abandon |
