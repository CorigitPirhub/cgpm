# S2 ray consistency distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | noise_ratio | penetration_mean | thickness_proxy_mean | observation_support_mean | static_coherence_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 72_local_geometric_conflict_resolution | 234 | 4 | 27 | 203 | 0.868 | 0.099 | 0.112 | 0.183 | 0.823 |
| 80_ray_penetration_consistency | 119 | 4 | 15 | 100 | 0.840 | 0.338 | 0.414 | 0.168 | 0.882 |
| 81_unobserved_space_veto | 231 | 4 | 25 | 202 | 0.874 | 0.106 | 0.120 | 0.218 | 0.821 |
| 82_static_neighborhood_coherence | 115 | 4 | 18 | 93 | 0.809 | 0.335 | 0.383 | 0.167 | 1.000 |

| variant | tb_penetration | noise_penetration | tb_observation | noise_observation | tb_static_coherence | noise_static_coherence |
|---|---:|---:|---:|---:|---:|---:|
| 72_local_geometric_conflict_resolution | 0.156 | 0.105 | 0.246 | 0.183 | 0.750 | 0.801 |
| 80_ray_penetration_consistency | 0.321 | 0.339 | 0.186 | 0.169 | 1.000 | 0.860 |
| 81_unobserved_space_veto | 0.156 | 0.112 | 0.286 | 0.219 | 0.750 | 0.800 |
| 82_static_neighborhood_coherence | 0.321 | 0.336 | 0.186 | 0.167 | 1.000 | 1.000 |

重点检查：TB 是否达到 `>=6`；Noise 占比是否降到 `< 0.75`；`ghost_reduction_vs_tsdf` 是否保持 `>=22%`。

注：`thickness_proxy_mean` 使用 `penetration_free_span_mean`，表示 rear 点前方可追溯空洞/穿透长度的平均值。
