# S2 history-visible obstructed distribution report

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum |
|---|---:|---:|---:|---:|
| 38_bonn_state_protect | 108 | 1 | 10 | 97 |
| 46_history_background_only_admission | 0 | 0 | 0 | 0 |
| 47_history_visible_obstructed_manifold | 0 | 0 | 0 | 0 |

目标检查：`true_background >= 20` 且 `ghost <= 15` 才算空间分布真正接近本轮目标。
