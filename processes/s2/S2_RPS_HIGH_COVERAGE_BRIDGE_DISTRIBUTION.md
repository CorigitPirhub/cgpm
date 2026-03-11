# S2 high coverage bridge distribution report

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum |
|---|---:|---:|---:|---:|
| 38_bonn_state_protect | 108 | 1 | 10 | 97 |
| 59_relaxed_occlusion_tunneling | 65 | 2 | 9 | 54 |
| 60_cone_based_rear_projection | 65 | 2 | 9 | 54 |
| 61_hybrid_confidence_gating | 65 | 2 | 9 | 54 |

重点检查：rear 是否恢复到 `>=80`，TB 是否达到 `>=8`，Ghost 是否仍 `<=10`。
