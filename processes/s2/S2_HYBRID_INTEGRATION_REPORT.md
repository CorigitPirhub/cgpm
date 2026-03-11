# S2 Hybrid Integration Report

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 集成策略

- `124`：保持 `122` 的 GT-free evidential proxy mask，但用更严格的 occupancy / entropy / geometry 门槛做激活。
- `125`：使用 `123` 式更保守的 visibility proxy，并增加 PAPG 平面一致性约束（仅保留高 `|n_z|`、足够平面范围、近 gap band 的候选）。

## 2. 关键结果

- `114`：Acc=`4.304`, Comp-R=`77.06`, Ghost=`224`
- `122`：Acc=`4.391`, Comp-R=`74.10`, Ghost=`3`
- `124`：Acc=`4.238`, Comp-R=`71.64`, Ghost=`5`
- `125`：Acc=`4.273`, Comp-R=`72.87`, Ghost=`3`
- `116` Oracle：Acc=`4.120`, Comp-R=`76.14`, Ghost=`23`

## 3. 结论

- `124` 证明：单纯收紧激活门槛确实能显著压低 ghost，但代价是 completeness 大幅回退。
- `125` 证明：PAPG 平面一致性约束能在维持 `Comp-R > 73%` 的同时压住 ghost，但 Acc 仍未恢复到 `4.20 cm` 以下。
- 当前混合集成已经逼近可交付形态，但还没有完全打通 `Acc / Comp-R` 双赢。
