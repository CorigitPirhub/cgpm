# S2 Bonn-side family-specific calibration 对比

## 1. historical archive
historical 协议页原先只写了：
- `Bonn / slam / rgbd_bonn_balloon2 / frames=5 / stride=3 / seed=7`

本轮已补全其缺失协议项：
- `max_points_per_frame=600`

historical 记录如下：

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | decision |
|---|---:|---:|---:|---:|---:|---|
| `08_anchor_ultralite_noroute` | 2.724 | 98.90 | 4.060 | 81.90 | 20.71 | superseded |
| `09_anchor_ultralite_bonn_noadaptive` | 2.724 | 98.90 | 3.486 | 82.50 | 17.52 | abandon |
| `10_anchor_ultralite_bonn_relaxed` | 2.724 | 98.90 | 3.759 | 83.97 | 32.09 | superseded |
| `13_bonn_localclip_soft` | 2.683 | 99.03 | 3.829 | 83.27 | 15.42 | abandon |
| `14_bonn_localclip_drive` | 2.601 | 99.13 | 3.755 | 83.87 | 35.69 | iterate |

## 2. current-code canonical recheck
协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | decision |
|---|---:|---:|---:|---:|---:|---|
| `14_bonn_localclip_drive_recheck` | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | control |
| `15_bonn_localclip_band_relax` | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | abandon |
| `16_bonn_localclip_pfv_rearexpand` | 0.9355 | 68.53 | 2.8864 | 83.57 | -8.00 | abandon |

## 3. current-code 结论
- historical 表中的 `14` 不能再直接视为 current-code canonical；
- current-code 下 `15/16` 对 `14` 没有任何有效增益；
- 因此当前不应继续沿 `16` 做 Bonn-side calibration 微调；
- 下一步应先恢复让 `14` 区别于 `05_anchor_noroute` 的实现收益链条。
