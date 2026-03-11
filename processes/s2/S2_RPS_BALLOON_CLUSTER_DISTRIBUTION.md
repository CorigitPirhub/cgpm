# S2 balloon cluster distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | true_background_sum | ghost_sum | hole_or_noise_sum | tb_noise_correlation | tb_cluster_fit | noise_cluster_fit | tb_cluster_geo | noise_cluster_geo | retained_clusters_sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 93_spatial_neighborhood_density_clustering | 13 | 19 | 96 | 0.991 | 0.0023 | 0.0074 | 0.1782 | 0.0543 | 0 |
| 95_geodesic_balloon_consistency | 13 | 19 | 96 | 0.991 | 0.0023 | 0.0074 | 0.1782 | 0.0543 | 0 |
| 96_support_cluster_model_fitting | 12 | 6 | 23 | -0.756 | 0.0023 | 0.0074 | 0.1782 | 0.0543 | 1 |
| 97_global_map_anchoring | 10 | 5 | 22 | -0.786 | 0.0023 | 0.0074 | 0.1782 | 0.0543 | 1 |

重点检查：`tb_noise_correlation < 0.9` 是否成立，以及 `TB` 是否仍保持 `>= 8`。
