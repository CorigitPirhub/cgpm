# S2 Acc tightening analysis

日期：`2026-03-11`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_ACC_TIGHTENING_COMPARE.csv`

## 1. Acc 改善幅度
- 控制组 `99`：Bonn `Acc=4.346 cm`, `Comp-R=70.08%`, `corr=-0.225`。
- 最佳候选 `103_local_cluster_refinement`：Bonn `Acc=4.346 cm`, `Comp-R=70.08%`, `corr=-0.419`。

## 2. 投影/校正/紧化谁起作用
- 若 `mean_distance_to_plane_after` 明显下降但 `Acc` 几乎不变，说明当前瓶颈不再是平面厚度噪声，而是更深层的几何畸变或前景/背景的系统偏差。
- 若 `orthogonality_drift_before` 与 `orthogonality_drift_after` 基本一致，说明没有明显的 geometry-only scale drift 证据。

## 3. 结论
- 若 `Acc` 没有从 `4.31 cm` 显著下降，则本轮 `hard snapping / drift correction / local refinement` 只能视为负结果链。
- 若 `Comp-R` 与 `TB/corr` 基本保持，而 `Acc` 仍顽固不动，则说明瓶颈已从“局部噪声”转移到“几何畸变 / front-side bias”。
- 只要 `Acc` 仍高于 `3.10 cm`，`S2` 仍未通过，且绝对不能进入 `S3`。
