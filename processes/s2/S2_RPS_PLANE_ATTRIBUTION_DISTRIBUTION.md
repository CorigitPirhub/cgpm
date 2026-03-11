# S2 plane attribution distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | thickness_kept | thickness_dropped | normal_kept | normal_dropped | convergence_kept | convergence_dropped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80_ray_penetration_consistency | 248 | 3 | 23 | 222 | 0.092 | 0.156 | 0.715 | 0.796 | 0.129 | 0.226 |
| 86_rear_plane_clustering_snapping | 42 | 0 | 5 | 37 | 0.385 | 0.058 | 0.803 | 0.729 | 0.254 | 0.142 |
| 87_front_mask_guided_back_projection | 34 | 0 | 3 | 31 | 0.385 | 0.070 | 0.798 | 0.732 | 0.304 | 0.138 |
| 88_occlusion_depth_hypothesis_validation | 10 | 0 | 3 | 7 | 0.386 | 0.102 | 0.691 | 0.743 | 0.047 | 0.166 |

| variant | tb_thickness | noise_thickness | tb_normal | noise_normal | tb_convergence | noise_convergence |
|---|---:|---:|---:|---:|---:|---:|
| 80_ray_penetration_consistency | 0.245 | 0.111 | 0.623 | 0.729 | 0.239 | 0.161 |
| 86_rear_plane_clustering_snapping | 0.184 | 0.112 | 0.688 | 0.727 | 0.179 | 0.164 |
| 87_front_mask_guided_back_projection | 0.245 | 0.114 | 0.623 | 0.725 | 0.239 | 0.164 |
| 88_occlusion_depth_hypothesis_validation | 0.184 | 0.110 | 0.688 | 0.728 | 0.179 | 0.162 |

重点检查：吸附前后 TB 是否上升、Noise 是否下降，以及 kept/dropped 的平面归属统计是否明显分离。
