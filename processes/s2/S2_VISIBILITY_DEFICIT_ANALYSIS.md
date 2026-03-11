# S2 Visibility Deficit Analysis

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 对比结果

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_ghost_sum | proxy_recall | proxy_precision |
|---|---:|---:|---:|---:|---:|
| 118_plane_extrapolation_closure | 4.199 | 72.17 | 92 | 0.261 | 0.128 |
| 120_joint_proxy_geometric_fusion | 4.184 | 71.12 | 167 | 0.043 | 0.042 |
| 121_joint_proxy_score_weighted | 4.209 | 71.61 | 41 | 0.132 | 0.241 |
| 122_evidential_visibility_deficit | 4.391 | 74.10 | 3 | 0.519 | 0.309 |
| 123_ray_deficit_accumulation | 4.398 | 74.06 | 9 | 0.469 | 0.282 |
| 116_oracle_upper_bound | 4.120 | 76.14 | 23 | - | - |

## 2. 关键结论

- `122` 把 `proxy_recall` 提到 `0.519`，明显超过 `118` 的 `0.261`，同时 `proxy_precision=0.309` 也高于门槛。
- `123` 的 `proxy_recall=0.469`、`proxy_precision=0.282` 同样过线，且 `Comp-R=74.06%` 高于 `118`。
- 两个新信号都把 `Ghost` 压到远低于 `118`：`122 Ghost=3`，`123 Ghost=9`。

## 3. 相比 118/120/121 的实质提升

- 相比最佳单点 `118`，`122` 在 `Recall / Precision / Ghost` 三者上同时上移；
- 相比 `120/121`，`122/123` 的提升来自更强的射线证据，而不是更复杂的融合规则；
- 这证明当前确实已经构建出了更可信的 GT-free visibility deficit 信号。

## 4. 距离 Oracle 116 还有多远

- `116` 仍然在系统指标上更优：Bonn `Acc=4.120`, `Comp-R=76.14`, `Ghost=23`。
- `122` 与 `123` 虽然把 GT-free recall 拉到了 `0.519` / `0.469`，但 `Acc` 仍高于 Oracle 约 `0.24` / `0.22 cm`。
- 这表明：缺口信号已经够用，但后续还需要把该信号更深地耦合回 occupancy-entropy 激活与 geometry chain。

## 5. 阶段判断

- 结论：**成功构建了更强但 GT-free 的 visibility deficit 信号**。
- `122/123` 已具备成为后续补全主线输入的资格。
- `S2` 仍未 fully pass，但 GT-free gap localization 已从“不可用”提升到“可继续主线化”。
