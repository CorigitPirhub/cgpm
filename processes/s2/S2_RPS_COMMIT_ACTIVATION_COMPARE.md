# S2 RPS commit activation compare

协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | tum_rear_w_nz | tum_rps_active_nz | bonn_rear_w_nz | bonn_rps_active_nz | decision | handoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 25_rps_commit_control | 0.9400 | 91.90 | 2.8867 | 83.57 | -7.56 | 0 | 0 | 0 | 0 | control | activation_phase_only |
| 28_rps_commit_geom_bg_soft | 0.9400 | 91.90 | 2.8867 | 83.57 | -7.56 | 108 | 108 | 55 | 55 | abandon | activation_phase_only |
| 29_rps_commit_geom_bg_mid | 0.9400 | 91.90 | 2.8867 | 83.57 | -7.56 | 4 | 4 | 2 | 2 | abandon | activation_phase_only |
| 30_rps_commit_geom_bg_soft_bank | 0.9413 | 91.93 | 2.8868 | 83.57 | -7.56 | 108 | 108 | 56 | 56 | iterate | extract_admission_score_gate_bottleneck_confirmed |

补充说明：后续 admission-phase 复核表明，`30` 的最窄流失点是 Bonn extract admission 的 `score_gate`；当前唯一继续配置已升级为 `38_bonn_state_protect`。
