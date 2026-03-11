# S2 ghost capped selectivity distribution report

日期：`2026-03-10`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | extract_score_ready_sum | selectivity_pre_sum | selectivity_kept_sum | selectivity_drop_sum | selectivity_topk_drop_sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64_patch_depth_hybrid_generation | 442 | 8 | 60 | 374 | 442 | 0 | 0 | 0 | 0 |
| 65_ghost_risk_prediction_filter | 442 | 8 | 60 | 374 | 442 | 442 | 442 | 0 | 0 |
| 66_geometry_constrained_admission | 555 | 15 | 84 | 456 | 555 | 555 | 555 | 0 | 0 |
| 67_topk_selective_generation | 231 | 3 | 38 | 190 | 442 | 442 | 231 | 0 | 211 |

重点检查：Ghost 是否压回 `<=15`，TB 是否保住 `>=6`，rear 是否仍 `>=150`。
