# S2 background manifold state distribution report

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum |
|---|---:|---:|---:|---:|
| 38_bonn_state_protect | 108 | 1 | 10 | 97 |
| 48_stable_background_memory_state | 29 | 1 | 3 | 25 |
| 49_relaxed_manifold_guided_generation | 39 | 1 | 4 | 34 |

检查重点：rear 总量是否恢复到 `>=100`，以及 `true_background` 是否提升且 `ghost <= 15`。
