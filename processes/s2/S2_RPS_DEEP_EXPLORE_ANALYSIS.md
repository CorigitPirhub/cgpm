# S2 deep explore analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_DEEP_EXPLORE_COMPARE.csv`

## 1. 新机制如何提升 Comp-R 且保住判别力
- 控制组 `97`：Comp-R=`68.90%`, `ghost_reduction_vs_tsdf=34.06%`, `TB=10`, `corr=-0.786`。
- 最佳候选 `99_manhattan_plane_completion`：Comp-R=`70.08%`, `ghost_reduction_vs_tsdf=28.28%`, `TB=39`, `corr=-0.225`。

## 2. 相比简单扩张的优势
- 简单扩张会把平面外噪声一起带回，导致相关性转正或 Ghost 回弹。
- 当前 completion 只围绕 `97` 过滤后的高置信 cluster 执行，并且只在平面带内补点，因此保持了 `tb_noise_correlation < 0` 与 `ghost_reduction >= 22%`。

## 3. 结论
- 若 `Comp-R >= 70%`、`tb_noise_correlation < 0`、`ghost_reduction_vs_tsdf >= 22%` 与 `TB >= 6` 同时成立，则说明“purified cluster completion”有效。
- 即便本轮局部目标达成，`S2` 整体仍未通过，因为全局 `Acc/Comp-R` 门槛仍远未达到任务书要求，绝对不能进入 `S3`。
