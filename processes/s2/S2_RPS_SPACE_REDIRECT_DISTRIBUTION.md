# S2 rear-point space redirect distribution report

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum |
|---|---:|---:|---:|---:|
| 38_bonn_state_protect | 108 | 1 | 10 | 97 |
| 43_history_guided_background_location | 193 | 5 | 26 | 162 |
| 44_history_plus_ghost_suppress | 188 | 4 | 25 | 159 |
| 45_visual_evidence_anchor_strict | 193 | 5 | 26 | 162 |

空间重定向是否有效，重点看两点：
- `true_background_sum` 是否显著高于 `38`；
- 同时 `ghost_sum` 是否没有明显恶化。
