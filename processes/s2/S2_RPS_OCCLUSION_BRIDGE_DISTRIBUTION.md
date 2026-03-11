# S2 occlusion bridge distribution report

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum |
|---|---:|---:|---:|---:|
| 38_bonn_state_protect | 108 | 1 | 10 | 97 |
| 56_temporal_occlusion_tunneling | 57 | 2 | 9 | 46 |
| 57_historical_surface_rear_projection | 58 | 3 | 9 | 46 |
| 58_ghost_aware_surface_inpainting | 58 | 2 | 9 | 47 |

重点检查：True Background 是否突破 `10`，且 Ghost 是否保持在 `<=10`。
