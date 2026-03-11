# S2 geometry chain coupling analysis

日期：`2026-03-11`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_GEOMETRY_CHAIN_COUPLING_COMPARE.csv`

## 1. 断裂诊断
- 真上游参考 `104` 的核心症状是 `upstream_rear_points_sum=0`；这不是 cluster 阈值被击穿，而是 upstream rerun 本身根本没有把 downstream rear-completion stage 执行出来。
- 原始 `99` 则依赖 `TB=39` 的 rear 完成链才能成立。

## 2. 修复方案
- 将 `104` 的 corrected front/surface geometry 作为上游输入，显式接入 `99` 的 rear donor / patch completion，构成单向 `upstream -> completion` 数据流。
- 不再依赖环境态或隐式全局状态；上游几何与下游 rear donor 在 runner 内显式拼接。

## 3. 结果
- 原始 `99`：Bonn `Acc=4.346 cm`, `Comp-R=70.08%`, `TB=39`。
- 原始 `104`：Bonn `Acc=4.238 cm`, `Comp-R=70.86%`, `TB=0`。
- 最佳集成候选 `109_geometry_chain_coupled_projected`：Bonn `Acc=4.233 cm`, `Comp-R=70.84%`, `TB=39`。
- 若集成后 `TB >= 20` 且 `Comp-R >= 70%`，则说明链路修复成功；若同时 `Acc <= 4.238 cm`，则说明上游校正确实叠加到了最终几何上。

## 4. 阶段判断
- 即便本轮链路修复成功，只要 Bonn `Acc` 仍高于 `3.10 cm`，`S2` 仍未通过，绝对不能进入 `S3`。
