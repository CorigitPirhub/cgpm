# S2 write-time target synthesis 对比表

注：本表为 `historical archive`。本轮已补回其缺失协议项：`frames=5 / stride=3 / seed=7 / max_points_per_frame=600`。

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | hit_2of4_partial | pass_comp | decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 00_no_synthesis_rps | 3.5187 | 99.47 | 5.0188 | 74.23 | 34.98 | 1 | False | baseline |
| 01_weak_legacy_wdsg | 1.2180 | 84.10 | 2.9948 | 82.57 | -5.79 | 2 | False | weak |
| 02_anchor_synthesis | 1.2346 | 83.27 | 2.9750 | 82.77 | -5.35 | 2 | False | abandon |
| 03_counterfactual_synthesis | 1.1866 | 82.20 | 2.9572 | 82.60 | -7.56 | 2 | False | abandon |
| 04_energy_synthesis | 1.1581 | 77.80 | 2.9345 | 82.77 | -6.67 | 2 | False | abandon |
| 05_anchor_noroute | 0.9292 | 68.50 | 2.8865 | 83.60 | -7.56 | 2 | False | abandon |
| 06_counterfactual_noroute | 0.9300 | 68.23 | 2.8888 | 83.57 | -7.56 | 2 | False | abandon |
| 07_anchor_lite_noroute | 1.3713 | 78.23 | 2.8887 | 83.57 | -7.56 | 2 | False | abandon |
| 08_anchor_ultralite_noroute | 2.7237 | 98.90 | 4.0597 | 81.90 | 20.71 | 1 | False | abandon |
| 09_anchor_ultralite_bonn_noadaptive | 2.7237 | 98.90 | 3.4861 | 82.50 | 17.52 | 1 | False | abandon |
| 10_anchor_ultralite_bonn_relaxed | 2.7237 | 98.90 | 3.7586 | 83.97 | 32.09 | 1 | False | abandon |
| 11_gcclip_anchor_ultralite | 2.7237 | 98.90 | 4.0597 | 81.90 | 20.71 | 1 | False | abandon |
| 12_gcclip_anchor_ultralite_bonn_relaxed | 2.7237 | 98.90 | 4.0633 | 81.93 | 20.72 | 1 | False | abandon |
| 13_bonn_localclip_soft | 2.6832 | 99.03 | 3.8291 | 83.27 | 15.42 | 1 | False | abandon |
| 14_bonn_localclip_drive | 2.6008 | 99.13 | 3.7550 | 83.87 | 35.69 | 1 | False | iterate |
