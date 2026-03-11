# S1 RB-Core 开发门槛烟雾子集对比表

协议：`TUM / oracle / rgbd_dataset_freiburg3_walking_xyz / frames=5 / stride=3 / seed=7 / max_points_per_frame=3000`

| method | status | surface_points | chamfer | fscore | comp_r_5cm |
|---|---|---:|---:|---:|---:|
| egf | ok | 166116 | 0.05327758634396815 | 0.7584985332859999 | 1.0 |
| tsdf | ok | 261 | 0.12349528560528517 | 0.33445035272140333 | 0.20096 |
| dynaslam | ok | 11445 | 2.3902842158306665 | 0.0014682068444348483 | 0.0008 |
| rodyn_slam | ok | 77 | 2.4972329568705995 | 0.000954122604754711 | 0.00048 |
