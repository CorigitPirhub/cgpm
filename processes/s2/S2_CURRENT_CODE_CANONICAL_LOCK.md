# S2 current-code canonical 锁定页

版本：`2026-03-09`

## 1. canonical 协议
当前 `S2 dev quick` 明确锁定为：
- `frames=5`
- `stride=3`
- `seed=7`
- `max_points_per_frame=600`

锁定理由：
- historical `S2` 输出统一呈现 `reference_points=3000`；
- 结合当前评测实现，这与 `5 x 600` 一致；
- 因此 historical `S2` 文档先前缺失的关键协议项已补回：`max_points_per_frame=600`。

## 2. current-code canonical 对比
### `05_anchor_noroute_recheck600`
- `TUM Acc = 0.9292 cm`
- `TUM Comp-R = 68.50%`
- `Bonn Acc = 2.8865 cm`
- `Bonn Comp-R = 83.60%`
- `Bonn ghost_reduction_vs_tsdf = -7.56%`

### `14_bonn_localclip_drive_recheck`
- `TUM Acc = 0.9355 cm`
- `TUM Comp-R = 68.53%`
- `Bonn Acc = 2.8864 cm`
- `Bonn Comp-R = 83.57%`
- `Bonn ghost_reduction_vs_tsdf = -8.00%`

### `25_rear_bg_coupled_formation`
- `TUM Acc = 0.9400 cm`
- `TUM Comp-R = 91.90%`
- `Bonn Acc = 2.8867 cm`
- `Bonn Comp-R = 83.57%`
- `Bonn ghost_reduction_vs_tsdf = -7.56%`

### `30_rps_commit_geom_bg_soft_bank`
- `TUM Acc = 0.9413 cm`
- `TUM Comp-R = 91.93%`
- `Bonn Acc = 2.8868 cm`
- `Bonn Comp-R = 83.57%`
- `Bonn ghost_reduction_vs_tsdf = -7.56%`

### `15/16/17-29`
- 全部已在 current-code canonical 下判定为 `abandon`

## 3. 锁定结论
- current-code 下，historical `14` 已不再是唯一继续配置；
- 经第三轮 `rear/bg state formation` 与本轮 `rps commit activation` 实验后，当前唯一值得继续的 `iterate` 候选更新为：
  - `30_rps_commit_geom_bg_soft_bank`
- 但它仍不是 `accept` 候选，`S2` 仍未通过。

## 4. 阶段状态
- 当前更准确的状态是：`re-baselined / rear-bg-formation partially restored / not-pass`
- `S2` **绝对不能进入 `S3`**。

## 5. 后续限制
在恢复 `historical 05 -> 14` 的实现收益链条前：
- 禁止再继续堆 `16` 方向的 Bonn-only calibration 小修补；
- 禁止把当前 historical `S2` 表格继续当作 current-code canonical；
- 后续所有 `S2` compare 必须显式写出：`max_points_per_frame=600`。
