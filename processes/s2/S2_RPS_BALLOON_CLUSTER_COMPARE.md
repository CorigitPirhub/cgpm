# S2 balloon cluster compare

| variant | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | tb_noise_correlation | bonn_ghost_reduction_vs_tsdf | kept_rear_points_sum | dropped_rear_points_sum | tb_cluster_cluster_fitting_error_mean | noise_cluster_cluster_fitting_error_mean | tb_cluster_geodesic_smoothness_mean | noise_cluster_geodesic_smoothness_mean | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 93_spatial_neighborhood_density_clustering | 13 | 19 | 96 | 0.991 | 23.59 | 128 | 0 | 0.0023 | 0.0074 | 0.1782 | 0.0543 | control |
| 95_geodesic_balloon_consistency | 13 | 19 | 96 | 0.991 | 23.59 | 128 | 0 | 0.0023 | 0.0074 | 0.1782 | 0.0543 | abandon |
| 96_support_cluster_model_fitting | 12 | 6 | 23 | -0.756 | 31.04 | 41 | 87 | 0.0023 | 0.0074 | 0.1782 | 0.0543 | iterate |
| 97_global_map_anchoring | 10 | 5 | 22 | -0.786 | 34.06 | 37 | 91 | 0.0023 | 0.0074 | 0.1782 | 0.0543 | iterate |
