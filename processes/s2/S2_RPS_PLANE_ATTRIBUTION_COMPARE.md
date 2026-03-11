# S2 plane attribution compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_thickness_mean | bonn_normal_consistency_mean | bonn_ray_convergence_mean | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 80_ray_penetration_consistency | 20.85 | 3 | 23 | 222 | 0.092 | 0.715 | 0.129 | control |
| 86_rear_plane_clustering_snapping | 29.64 | 0 | 5 | 37 | 0.385 | 0.803 | 0.254 | abandon |
| 87_front_mask_guided_back_projection | 30.95 | 0 | 3 | 31 | 0.385 | 0.798 | 0.304 | abandon |
| 88_occlusion_depth_hypothesis_validation | 34.36 | 0 | 3 | 7 | 0.386 | 0.691 | 0.047 | abandon |
