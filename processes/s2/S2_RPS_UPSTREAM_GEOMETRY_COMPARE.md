# S2 upstream geometry compare

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | mean_distance_to_plane_before | mean_distance_to_plane_after | depth_bias_offset_m | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 99_manhattan_plane_completion | 4.346 | 70.08 | 28.28 | 39 | 0.0980 | 0.0980 | +0.000 | reference |
| 104_depth_bias_minus1cm | 4.238 | 70.86 | 13.45 | 0 | 0.0980 | 0.0746 | -0.010 | progress |
| 105_depth_bias_plus1cm | 4.238 | 70.81 | 13.20 | 0 | 0.0980 | 0.0983 | +0.010 | progress |
