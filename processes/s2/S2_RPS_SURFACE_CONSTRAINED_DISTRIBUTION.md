# S2 surface constrained distribution report

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum |
|---|---:|---:|---:|---:|
| 38_bonn_state_protect | 108 | 1 | 10 | 97 |
| 53_surface_adjacent_propagation | 94 | 3 | 13 | 78 |
| 54_normal_guided_manifold_extension | 96 | 3 | 12 | 81 |
| 55_surface_constrained_ray_projection | 101 | 3 | 14 | 84 |

重点检查：True Background 是否有倍数级提升，以及 Ghost 是否仍控制在 `<=12`。
