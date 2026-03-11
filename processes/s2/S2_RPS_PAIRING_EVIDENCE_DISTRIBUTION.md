# S2 pairing evidence distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | protected_sum | snapped_sum | comp_r | acc_cm | tb_noise_correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80_ray_penetration_consistency | 119 | 4 | 19 | 96 | 0 | 0 | 70.20 | 4.317 | 0.991 |
| 89_front_back_surface_pairing_guard | 119 | 4 | 22 | 93 | 115 | 82 | 70.16 | 4.318 | 0.992 |
| 90_background_plane_evidence_accumulation | 119 | 4 | 27 | 88 | 115 | 72 | 68.89 | 4.322 | 0.993 |
| 91_occlusion_depth_hypothesis_tb_protection | 119 | 4 | 19 | 96 | 115 | 8 | 70.20 | 4.317 | 0.991 |

重点检查：TB 是否回升到 `4+` 并进一步逼近 `6+`；Noise 减少是否伴随 TB 上升。
