# S2 multi-candidate generation distribution report

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | extract_rear_selected_sum | extract_score_ready_sum | extract_support_protected_sum | extract_fail_score_sum | bridge_rear_synth_updates_sum | bridge_rear_synth_w_sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 38_bonn_state_protect | 108 | 1 | 10 | 97 | 585 | 108 | 201 | 93 | 0 | 0.00 |
| 62_dense_patch_projection | 378 | 7 | 53 | 318 | 1594 | 378 | 481 | 108 | 17616 | 28.29 |
| 63_multi_hypothesis_depth_sampling | 250 | 3 | 23 | 224 | 1092 | 250 | 352 | 106 | 9819 | 25.06 |
| 64_patch_depth_hybrid_generation | 442 | 8 | 60 | 374 | 1925 | 442 | 589 | 154 | 36612 | 40.44 |

重点检查：rear 是否恢复到 `>=100`，TB 是否突破 `>=10`，Ghost 是否仍 `<=12`。
