# S2 rear geometry quality analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_REAR_GEOMETRY_QUALITY_COMPARE.csv`

## 1. 结果概览
- `38_bonn_state_protect`: Bonn `ghost_reduction_vs_tsdf = 15.47%`, `Comp-R = 70.83%`, `rear_true_background_sum = 1`, `rear_ghost_sum = 10`, decision=`control`
- `40_bonn_geometry_aligned_admission`: Bonn `ghost_reduction_vs_tsdf = 15.22%`, `Comp-R = 70.77%`, `rear_true_background_sum = 2`, `rear_ghost_sum = 16`, decision=`abandon`
- `41_bonn_geometry_occlusion_admission`: Bonn `ghost_reduction_vs_tsdf = 15.22%`, `Comp-R = 70.77%`, `rear_true_background_sum = 3`, `rear_ghost_sum = 18`, decision=`abandon`
- `42_bonn_geometry_density_gate`: Bonn `ghost_reduction_vs_tsdf = 15.22%`, `Comp-R = 70.77%`, `rear_true_background_sum = 2`, `rear_ghost_sum = 18`, decision=`abandon`

## 2. 几何质量策略是否有效
- 控制组 `38`：Bonn `ghost_reduction_vs_tsdf = 15.47%`
- 本轮三个 geometry-quality 候选中的最强者是 `41_bonn_geometry_occlusion_admission`，但其 Bonn `ghost_reduction_vs_tsdf = 15.22%`，仍低于控制组。
- 结论：本轮几何质量改进没有任何一个候选能够超过 `38`。

## 3. Ghost 与 Comp-R 是否同步提升
- 控制组 Bonn `Comp-R = 70.83%`；三个候选都只有 `70.77%` 左右，未超过控制组。
- 结论：本轮没有出现 Ghost 与 Comp-R 的同步提升，说明几何质量转化仍不充分。

## 4. 若未达标，主要原因是什么
- 新增 rear points 仍有相当部分落在 ghost region，说明 admission 质量仍不足。

## 5. 当前阶段判断
- 本轮若未同时达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过。
- 当前不能进入 `S3`。

## 6. 后续 space-redirect 复核（2026-03-09）
后续 `43/44/45` 的 space-redirect 复核进一步确认：
- 虽然 `true_background` rear points 可从 `1` 提到 `4-5`；
- 但 `ghost` rear points 同时从 `10` 膨胀到 `25-26`；
- Bonn `ghost_reduction_vs_tsdf` 与 `Comp-R` 都没有超过 `38`。

统一引用：`processes/s2/S2_RPS_SPACE_REDIRECT_ANALYSIS.md`
