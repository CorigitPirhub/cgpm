# S2 GT-Free Gap Proxy Analysis

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. Oracle 上界 vs GT-Free 下界

- Oracle 上界 `116`：Bonn `Acc=4.120 cm`, `Comp-R=76.14%`, `Ghost=23`
- GT-free 下界 `114`：Bonn `Acc=4.304 cm`, `Comp-R=77.06%`, `Ghost=278`

## 2. 三个 GT-Free Proxy

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_ghost_sum | proxy_recall | proxy_precision |
|---|---:|---:|---:|---:|---:|
| 117_frustum_unobserved_proxy | 4.136 | 71.24 | 152 | 0.083 | 0.042 |
| 118_plane_extrapolation_closure | 4.199 | 72.17 | 92 | 0.261 | 0.128 |
| 119_entropy_guided_proxy | 4.152 | 71.17 | 107 | 0.058 | 0.040 |

## 3. 结论

- 三个 GT-free proxy 中，`118_plane_extrapolation_closure` 最接近可用：`Acc=4.199`, `Comp-R=72.17%`，但 `Ghost=92` 仍高于安全线。
- `117` 与 `119` 都未复现 116 的 coverage 收益，`proxy_recall` 分别只有 `0.083` 与 `0.058`，说明单纯用 frustum 或 entropy 仍难以定位大部分真缺口。
- `118` 的 `proxy_recall=0.261` 虽然高于其他 GT-free proxy，但距离目标 `0.6` 仍有明显差距；这说明 plane-hole closure 只覆盖了局部结构化缺口。

## 4. 是否找到可替代 Oracle 的方案

- 结论：**尚未找到**。
- 原因不是 occupancy+entropy 机制无效，而是 GT-free gap localization 仍然过弱。
- 下一轮应继续设计：`visibility deficit + plane closure + entropy` 的联合 proxy，而不是退回到全图 union。
