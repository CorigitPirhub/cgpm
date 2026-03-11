# S2 current-code drift compare

协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | note |
|---|---:|---:|---:|---:|---:|---|
| `05_anchor_noroute_recheck600` | 0.9292 | 68.50 | 2.8865 | 83.60 | -7.56 | current-code early anchor_noroute control |
| `14_bonn_localclip_drive_recheck` | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | historical active branch label under current code |
| `15_bonn_localclip_band_relax` | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | zero-delta vs `14` |
| `16_bonn_localclip_pfv_rearexpand` | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | zero-delta vs `14` |
| `25_rear_bg_coupled_formation` | 0.9400 | 91.90 | 2.8867 | 83.57 | -7.56 | bg state restored |
| `30_rps_commit_geom_bg_soft_bank` | 0.9413 | 91.93 | 2.8868 | 83.57 | -7.56 | current-code best iterate; rear commit chain activated |

结论：
- current-code `14/15/16` 已整体塌缩到与 `05` 近乎等价的行为族；
- 第三轮 `25_rear_bg_coupled_formation` 首次把 `bg state` 恢复为全量非零，并把 `TUM Comp-R` 提到 `91.90%`；
- 本轮 `30_rps_commit_geom_bg_soft_bank` 又首次把 `rear commit chain` 真正打通到 export 读出（`rear_selected > 0`）；
- 因此 residual drift 的更窄主瓶颈已更新为：`committed rear bank scale / quality still too weak to move Bonn metrics`。
