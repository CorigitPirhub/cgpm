# S2 ray consistency compare

协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_noise_ratio | bonn_penetration_mean | bonn_observation_support_mean | bonn_static_coherence_mean | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 72_local_geometric_conflict_resolution | 22.64 | 234 | 4 | 27 | 203 | 0.868 | 0.099 | 0.183 | 0.823 | control |
| 80_ray_penetration_consistency | 22.66 | 119 | 4 | 15 | 100 | 0.840 | 0.338 | 0.168 | 0.882 | abandon |
| 81_unobserved_space_veto | 22.18 | 231 | 4 | 25 | 202 | 0.874 | 0.106 | 0.218 | 0.821 | abandon |
| 82_static_neighborhood_coherence | 22.70 | 115 | 4 | 18 | 93 | 0.809 | 0.335 | 0.167 | 1.000 | abandon |
