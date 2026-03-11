# S2 rear-point spatial distribution diagnosis

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
控制组：`38_bonn_state_protect`

## 1. 诊断目的
- 分析 `38` 下真正导出的 rear points 究竟落在 Ghost Region、True Background 还是 Hole/Noise 区域。
- 判断“admission 已增加，但几何质量转化不足”是否来自 rear point 空间分布无效。

## 2. 38 配置下的 Bonn rear-point 分布
### `rgbd_bonn_balloon2`
- rear points: `28`
- true background: `1` (3.57%)
- ghost region: `6` (21.43%)
- holes / noise: `21` (75.00%)

### `rgbd_bonn_balloon`
- rear points: `18`
- true background: `0` (0.00%)
- ghost region: `0` (0.00%)
- holes / noise: `18` (100.00%)

### `rgbd_bonn_crowd2`
- rear points: `62`
- true background: `0` (0.00%)
- ghost region: `4` (6.45%)
- holes / noise: `58` (93.55%)

## 3. family-mean 结论
- total rear points: `108`
- total true background: `1`
- total ghost region: `10`
- total holes / noise: `97`
- 若 `hole_or_noise` 占比高，说明 admission 增加后的 rear points 仍然缺乏稳定背景锚定，更多是在填无效区域。
- 若 `ghost_region` 占比高，说明 rear points 仍落在动态物体历史区域，几何一致性不足。
- 若 `true_background` 增长但 `Comp-R` 仍不动，则说明点的空间覆盖或连通性仍不足。
