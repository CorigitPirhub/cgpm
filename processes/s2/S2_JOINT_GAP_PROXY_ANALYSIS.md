# S2 Joint Gap Proxy Analysis

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 对比结果

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_ghost_sum | proxy_recall | proxy_precision |
|---|---:|---:|---:|---:|---:|
| 118_plane_extrapolation_closure | 4.199 | 72.17 | 92 | 0.261 | 0.128 |
| 120_joint_proxy_geometric_fusion | 4.184 | 71.12 | 167 | 0.043 | 0.042 |
| 121_joint_proxy_score_weighted | 4.209 | 71.61 | 41 | 0.132 | 0.241 |
| 116_oracle_upper_bound | 4.120 | 76.14 | 23 | - | - |

## 2. 结论

- `120` 没有超过 `118`：`proxy_recall=0.043` 低于 `118` 的 `0.261`，同时 `Ghost=167` 仍偏高。
- `121` 进一步压低了 recall 到 `0.132`，说明 score-weighted 融合在当前特征质量下更像是强裁剪器，而不是高质量 proxy。
- 这意味着三种信号虽然互补，但当前可用的 `Visibility / Plane / Entropy` 估计仍然太粗糙，简单联合后仍无法逼近 `116`。

## 3. 距离 Oracle 的剩余差距

- `116` 的 Bonn `Acc=4.120`、`Comp-R=76.14`、`Ghost=23`；
- 最接近的 GT-free 单点仍是 `118`，但 recall 只有 `0.261`；
- `120/121` 进一步说明：当前差距不在激活规则，而在输入信号本身还不够强。

## 4. 阶段判断

- 联合 proxy 已经完成构建，但**未成功**达到工程可用门槛。
- `S2` 仍不具备“无 GT 通关”的可能性；当前还缺一个更强的 GT-free gap localization 信号。
