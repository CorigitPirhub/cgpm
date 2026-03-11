# S2 topology constraint distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | thickness_kept | thickness_dropped | normal_kept | normal_dropped | convergence_kept | convergence_dropped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80_ray_penetration_consistency | 248 | 4 | 30 | 214 | 0.113 | 0.192 | 0.741 | 0.792 | 0.161 | 0.182 |
| 83_minimum_thickness_topology_filter | 126 | 4 | 16 | 106 | 0.384 | 0.054 | 0.826 | 0.739 | 0.254 | 0.137 |
| 84_front_back_normal_consistency | 127 | 4 | 18 | 105 | 0.383 | 0.053 | 0.928 | 0.847 | 0.266 | 0.132 |
| 85_occlusion_ray_convergence_constraint | 125 | 4 | 22 | 99 | 0.383 | 0.055 | 0.940 | 0.694 | 0.729 | 0.424 |

| variant | tb_thickness | noise_thickness | tb_normal | noise_normal | tb_convergence | noise_convergence |
|---|---:|---:|---:|---:|---:|---:|
| 80_ray_penetration_consistency | 0.184 | 0.117 | 0.688 | 0.726 | 0.179 | 0.167 |
| 83_minimum_thickness_topology_filter | 0.371 | 0.384 | 0.965 | 0.808 | 0.179 | 0.253 |
| 84_front_back_normal_consistency | 0.371 | 0.384 | 0.944 | 0.931 | 0.179 | 0.268 |
| 85_occlusion_ray_convergence_constraint | 0.371 | 0.383 | 0.965 | 0.943 | 0.433 | 0.744 |

重点检查：被剔除点与被保留点的 `thickness / normal / convergence` 是否明显分离；TB 是否突破 `4`。
