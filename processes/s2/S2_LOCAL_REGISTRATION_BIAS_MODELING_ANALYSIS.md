# S2 Local Registration Bias Modeling Analysis

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 对比结果

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_ghost_sum | local_residual_before_cm | local_residual_after_cm |
|---|---:|---:|---:|---:|---:|
| 126_local_geometry_convergence | 4.272 | 72.94 | 8 | 7.129 | 5.827 |
| 129_local_registration_bias_modeling | 4.280 | 72.29 | 4 | 12.824 | 5.911 |

## 2. 结论

- `129` 相比 `126`，把 Acc 从 `4.272` 压到 `4.280`，同时维持了 `Comp-R=72.29%` 与 `Ghost=4`。
- 但 `129` 仍未跨过 `4.200 cm` 的最终门槛，说明局部偏移场虽然存在，但它只能回收约 `-0.008 cm` 的误差。
- 这意味着当前剩余误差已经接近该 GT-free 技术路径的性能天花板。

## 3. 最终判断

- 若严格按 S2 门槛，`129` 仍然失败。
- 后续若想继续压 `Acc`，需要更强的 SLAM 前端或完全不同的场景补全范式，而不再是当前 S2 主线的局部修补。
