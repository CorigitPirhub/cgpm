# S2 dense manifold distribution report

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum |
|---|---:|---:|---:|---:|
| 38_bonn_state_protect | 108 | 1 | 10 | 97 |
| 50_dense_background_propagation | 67 | 1 | 7 | 59 |
| 51_geometry_guided_manifold_completion | 88 | 2 | 12 | 74 |
| 52_dual_scale_manifold_fusion | 112 | 1 | 13 | 98 |

检查重点：rear 总量是否恢复到 `80-100`，TB 是否达到 `>=10`，Ghost 是否仍 `<=15`。
