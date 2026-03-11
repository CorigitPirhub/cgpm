# S2 锁箱复验记录

## 当前已完成复验
### `08_anchor_ultralite_noroute` on `TUM walking_static`
协议：`TUM / oracle / rgbd_dataset_freiburg3_walking_static / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

结果：
- `Acc = 2.955 cm`
- `Comp-R = 99.13%`
- `ghost_ratio = 0.1519`

解释：
- 相对其在 `TUM walking_xyz` 上的 `Comp-R = 98.9%`，锁箱方向没有翻到“明显 completeness 崩塌”；
- 说明 `anchor_ultralite_noroute` 至少在 `TUM family` 内呈现出一致的保守方向。

## 尚未完成部分
- `Bonn` lockbox (`balloon / crowd2`) 的同口径复验本轮未补齐；
- 因此当前锁箱结论只能算 **partial lockbox recheck**。

## 结论
- `TUM` 锁箱方向：`partial-pass`
- `Bonn` 锁箱方向：`pending`
- 整体：**尚不足以支撑 `S2` 正式通过。**
