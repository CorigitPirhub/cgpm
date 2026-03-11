# S2 deep explore compare

| variant | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | completion_added_sum | tb_noise_correlation | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 97_global_map_anchoring | 68.90 | 34.06 | 10 | 5 | 22 | 0 | -0.786 | control |
| 98_geodesic_support_diffusion | 70.08 | 28.00 | 40 | 20 | 30 | 49 | -0.327 | iterate |
| 99_manhattan_plane_completion | 70.08 | 28.28 | 39 | 20 | 31 | 49 | -0.225 | iterate |
| 100_cluster_view_inpainting | 70.08 | 28.28 | 34 | 24 | 32 | 49 | -0.115 | iterate |
