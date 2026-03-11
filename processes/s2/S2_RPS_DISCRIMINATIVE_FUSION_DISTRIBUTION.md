# S2 discriminative fusion distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | front_score_mean | rear_score_mean | rear_gap_mean | competition_mean | selectivity_topk_drop_sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 67_topk_selective_generation | 231 | 3 | 41 | 187 | 0.174 | 0.419 | 0.2450 | 0.280 | 211 |
| 68_rear_front_score_competition | 204 | 2 | 32 | 170 | 0.174 | 0.414 | 0.2405 | 0.249 | 177 |
| 69_depth_gap_validation | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.0000 | 0.000 | 0 |
| 70_fused_discriminator_topk | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.0000 | 0.000 | 0 |

重点检查：Ghost 是否压回 `<=15`，TB 是否恢复到 `>=8`，以及 `rear_score/front_score` 是否出现可分离均值。
