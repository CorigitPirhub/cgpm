# S2 deep explore distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | true_background_sum | ghost_sum | hole_or_noise_sum | comp_r | ghost_reduction | completion_added_sum | tb_noise_correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 97_global_map_anchoring | 10 | 5 | 22 | 68.90 | 34.06 | 0 | -0.786 |
| 98_geodesic_support_diffusion | 40 | 20 | 30 | 70.08 | 28.00 | 49 | -0.327 |
| 99_manhattan_plane_completion | 39 | 20 | 31 | 70.08 | 28.28 | 49 | -0.225 |
| 100_cluster_view_inpainting | 34 | 24 | 32 | 70.08 | 28.28 | 49 | -0.115 |
