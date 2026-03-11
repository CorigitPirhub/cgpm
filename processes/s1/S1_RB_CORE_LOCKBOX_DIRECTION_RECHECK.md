# S1 RB-Core 锁箱方向复验

开发门槛：`TUM / oracle / rgbd_dataset_freiburg3_walking_xyz / frames=5 / stride=3 / seed=7 / max_points_per_frame=3000`
锁箱复验：`TUM / oracle / rgbd_dataset_freiburg3_walking_static / frames=5 / stride=3 / seed=7 / max_points_per_frame=3000`

判定规则：若某 baseline 在 `chamfer`、`fscore`、`comp_r_5cm` 三项上同时不差于当前 mainline (`egf`)，且至少一项严格更优，则视为“全面支配”；否则记为 `mixed/not-dominating`。

| method | dev_chamfer | dev_fscore | dev_comp_r_5cm | dev_vs_egf | lockbox_chamfer | lockbox_fscore | lockbox_comp_r_5cm | lockbox_vs_egf |
|---|---:|---:|---:|---|---:|---:|---:|---|
| egf | 0.05327758634396815 | 0.7584985332859999 | 1.0 | reference | 0.05534376549234313 | 0.7240532062415014 | 1.0 | reference |
| tsdf | 0.12349528560528517 | 0.33445035272140333 | 0.20096 | mixed/not-dominating | 0.12427687491679978 | 0.3285418198836665 | 0.19656 | mixed/not-dominating |
| dynaslam | 2.3902842158306665 | 0.0014682068444348483 | 0.0008 | mixed/not-dominating | 2.7994530192550755 | 0.0 | 0.0 | mixed/not-dominating |
| rodyn_slam | 2.4972329568705995 | 0.000954122604754711 | 0.00048 | mixed/not-dominating | 0.9293168242390886 | 0.01053280434881537 | 0.00536 | mixed/not-dominating |

结论：开发门槛全面支配者=['none']；锁箱全面支配者=['none']。
按 `RB-Core` 全面支配判据，当前对比方向未翻转。
`RoDyn-SLAM` 已替换旧的 `StaticFusion` 进入当前正式 `RB-Core`；在当前 smoke / lockbox gate 下，它未对 `EGF` 构成全面支配。
