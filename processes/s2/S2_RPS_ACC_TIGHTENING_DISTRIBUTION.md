# S2 Acc tightening distribution report

日期：`2026-03-11`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | bonn_acc_cm | comp_r | ghost_reduction | tb | ghost | noise | plane_dist_before | plane_dist_after | snapped_points_sum | tb_noise_correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99_manhattan_plane_completion | 4.346 | 70.08 | 28.28 | 39 | 20 | 31 | 0.2191 | 0.2191 | 0 | -0.225 |
| 101_manhattan_plane_projection_hard_snapping | 4.346 | 70.08 | 28.58 | 39 | 19 | 32 | 0.2511 | 0.2472 | 16 | -0.254 |
| 102_scale_drift_correction | 4.734 | 68.67 | 29.98 | 38 | 15 | 37 | 0.0397 | 0.3590 | 90 | -0.260 |
| 103_local_cluster_refinement | 4.346 | 70.08 | 28.00 | 39 | 22 | 29 | 0.2476 | 0.2475 | 49 | -0.419 |
