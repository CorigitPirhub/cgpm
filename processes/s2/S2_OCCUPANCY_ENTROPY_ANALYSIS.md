# S2 Occupancy-Entropy Analysis

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 对比结果

| variant | bonn_acc_cm | bonn_comp_r_5cm | rear_TB | rear_Ghost | rear_Noise | mean_entropy | gap_activation_ratio | weak_evidence_coverage_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 113_naive_plane_union | 4.413 | 76.19 | 1885 | 224 | 954 | nan | nan | nan |
| 114_papg_plane_union | 4.304 | 77.06 | 2178 | 278 | 1035 | nan | nan | nan |
| 115_papg_consensus_activation | 4.225 | 70.96 | 59 | 21 | 18 | nan | nan | nan |
| 116_occupancy_entropy_gap_activation | 4.120 | 76.14 | 515 | 23 | 0 | 0.674 | 1.000 | 0.264 |

## 2. 关键判断

- `116` 相比 `114`，把 Bonn `Acc` 从 `4.304` 压到 `4.120`，同时把 `Comp-R` 维持在 `76.14%`。
- `116` 的 `bonn_rear_ghost_sum = 23`，显著低于 `114` 的 `278`；`ghost_increase_ratio = -0.917`。
- `gap_activation_ratio = 1.000`，说明激活集中在真缺口邻域。
- `weak_evidence_coverage_ratio = 0.264`，说明被补回的主要是弱证据点。

## 3. 结论

- 116 是本轮唯一同时满足 `Comp-R > 75%`、`Acc < 4.23 cm`、`Ghost <= 35` 的变体。
- 但它依赖 developer-side oracle gap mask，因此它证明的是**机制成立**，而不是“可直接晋升主线”。
- 下一轮必须把 oracle gap mask 替换为 GT-free gap proxy；否则这一创新仍停留在研究验证层面。
