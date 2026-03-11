# S2 Hybrid Integration Analysis

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 对比结果

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_ghost_sum | bonn_rear_true_background_sum |
|---|---:|---:|---:|---:|
| 122_evidential_visibility_deficit | 4.391 | 74.10 | 3 | 372 |
| 124_hybrid_evidential_activation_strict | 4.238 | 71.64 | 5 | 62 |
| 125_hybrid_papg_constrained | 4.273 | 72.87 | 3 | 195 |
| 116_oracle_upper_bound | 4.120 | 76.14 | 23 | 515 |

## 2. 解释

- `124` 把 ghost 压到 `5`，但 `Comp-R` 只有 `71.64%`；说明单纯严格过滤会把高召回 proxy 的收益一并裁掉。
- `125` 把 `Comp-R` 保在 `72.87%`，ghost 只有 `3`；它是当前 GT-free 路线中最平衡的版本。
- 但 `125` 的 `Acc=4.273` 仍高于门槛 `4.200 cm`，说明即使几何一致性约束生效，Proxy 带来的位置偏差还没有被完全消除。

## 3. S2 最终形态

- 当前 GT-free 最终形态已经清晰：`PAPG + Visibility Deficit + Occupancy Activation` 是正确主线。
- 但本轮尚未把该主线推进到可交付状态，因为 `Acc` 仍差最后一小段。
- 因此 `S2` 仍未 fully pass，但技术路径已打通到“只差最后精度收敛”的状态。
