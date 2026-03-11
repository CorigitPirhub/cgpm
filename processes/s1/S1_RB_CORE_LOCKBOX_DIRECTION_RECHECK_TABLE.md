# S1 RB-Core 锁箱方向复验表

协议：`TUM / oracle / rgbd_dataset_freiburg3_walking_static / frames=5 / stride=3 / seed=7 / max_points_per_frame=3000`

| method | status | surface_points | chamfer | fscore | comp_r_5cm |
|---|---|---:|---:|---:|---:|
| egf | ok | 189547 | 0.05534376549234313 | 0.7240532062415014 | 1.0 |
| tsdf | ok | 266 | 0.12427687491679978 | 0.3285418198836665 | 0.19656 |
| dynaslam | ok | 11282 | 2.7994530192550755 | 0.0 | 0.0 |
| rodyn_slam | ok | 126 | 0.9293168242390886 | 0.01053280434881537 | 0.00536 |
