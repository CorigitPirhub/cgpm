# S2 geometry chain coupling compare

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | tb_noise_correlation | upstream_rear_points_sum | donor_rear_points_sum | donor_cluster_selected_sum | donor_cluster_retained_sum | donor_patch_points_sum | projected_points_sum | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 99_manhattan_plane_completion_raw | 4.346 | 70.08 | 39 | 20 | 31 | -0.225 | nan | nan | nan | nan | nan | nan | control |
| 104_depth_bias_minus1cm_raw | 4.238 | 70.86 | 0 | 0 | 0 | nan | 0 | 0 | 0 | 0 | 0 | 0 | control |
| 108_geometry_chain_coupled_direct | 4.233 | 70.86 | 39 | 20 | 31 | -0.225 | 0 | 90 | 0 | 0 | 0 | 0 | iterate |
| 109_geometry_chain_coupled_projected | 4.233 | 70.84 | 39 | 19 | 32 | -0.115 | 0 | 90 | 0 | 0 | 0 | 21 | iterate |
| 110_geometry_chain_coupled_conservative | 4.237 | 70.83 | 10 | 5 | 22 | -0.786 | 0 | 90 | 0 | 37 | 0 | 0 | abandon |
