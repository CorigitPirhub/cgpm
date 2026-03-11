# S2 Acc tightening compare

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | tb_noise_correlation | mean_distance_to_plane_before | mean_distance_to_plane_after | orthogonality_drift_before | orthogonality_drift_after | snapped_points_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 99_manhattan_plane_completion | 4.346 | 70.08 | 28.28 | -0.225 | 0.2191 | 0.2191 | 0.5806 | 0.5806 | 0 | control |
| 101_manhattan_plane_projection_hard_snapping | 4.346 | 70.08 | 28.58 | -0.254 | 0.2511 | 0.2472 | 0.5696 | 0.5696 | 16 | abandon |
| 102_scale_drift_correction | 4.734 | 68.67 | 29.98 | -0.260 | 0.0397 | 0.3590 | 0.5397 | 0.5397 | 90 | abandon |
| 103_local_cluster_refinement | 4.346 | 70.08 | 28.00 | -0.419 | 0.2476 | 0.2475 | 0.5775 | 0.5775 | 49 | abandon |
