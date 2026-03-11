# S1 RoDyn-SLAM 核心数据集口径确认表

| sequence | dataset_kind | status | frames | stride | chamfer | fscore | comp_r_5cm |
|---|---|---|---:|---:|---:|---:|---:|
| rgbd_dataset_freiburg3_walking_xyz | tum | ok | 5 | 3 | 2.4972329568705995 | 0.000954122604754711 | 0.00048 |
| rgbd_dataset_freiburg3_walking_static | tum | ok | 5 | 3 | 0.9293168242390886 | 0.01053280434881537 | 0.00536 |
| rgbd_dataset_freiburg3_walking_halfsphere | tum | ok | 10 | 2 | 0.27202830805285055 | 0.18309961313675252 | 0.11004 |
| rgbd_dataset_freiburg1_xyz | tum | ok | 10 | 2 | 0.17306761867905107 | 0.27695649875571315 | 0.2206 |
| rgbd_bonn_balloon | bonn | ok | 5 | 3 | 18.4270057031937 | 0.0 | 0.0 |
| rgbd_bonn_balloon2 | bonn | ok | 5 | 3 | 15.014358881759797 | 0.0 | 0.0 |
| rgbd_bonn_crowd2 | bonn | ok | 5 | 3 | 18.297317684499156 | 0.0 | 0.0 |

## 说明修正（2026-03-09）
- 本页是外部 baseline 的 core-dataset protocol check 记录，不是 `S2 dev quick current-code canonical`；
- 其 `frames/stride` 仅用于确认 `RoDyn-SLAM` 在核心数据集上的本地 runnable / eval closure；
- 后续若做阶段结论或主线 compare，不得把本页协议直接替代 `S2 current-code canonical`。
