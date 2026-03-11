# S2 topology constraint compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_thickness_mean | bonn_normal_consistency_mean | bonn_ray_convergence_mean | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 80_ray_penetration_consistency | 26.84 | 4 | 30 | 214 | 0.113 | 0.741 | 0.161 | control |
| 83_minimum_thickness_topology_filter | 22.49 | 4 | 16 | 106 | 0.384 | 0.826 | 0.254 | abandon |
| 84_front_back_normal_consistency | 21.86 | 4 | 18 | 105 | 0.383 | 0.928 | 0.266 | abandon |
| 85_occlusion_ray_convergence_constraint | 22.30 | 4 | 22 | 99 | 0.383 | 0.940 | 0.729 | abandon |
