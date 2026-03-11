# S2 high coverage bridge analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_HIGH_COVERAGE_BRIDGE_COMPARE.csv`

## 1. 覆盖量是否恢复
- 控制组 `38`: rear=`108`, TB=`1`, Ghost=`10`
- 本轮最佳候选 `59_relaxed_occlusion_tunneling`: rear=`65`, TB=`2`, Ghost=`9`
- rear 覆盖量仍未恢复到目标区间。
- True Background 仍未达到本轮目标。
- Ghost 仍保持在低位。

## 2. 指标是否改善
- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = 15.47%`, `Comp-R = 70.83%`
- 本轮最佳候选 `59_relaxed_occlusion_tunneling`: Bonn `ghost_reduction_vs_tsdf = 15.02%`, `Comp-R = 70.86%`
- ghost 指标没有超过控制组。
- Comp-R 有提升。

## 3. 若未达标，原因判断
- 若桥接覆盖量恢复但 TB 仍低，说明扩张方向仍没有命中足够多的被遮挡背景。
- 若 Ghost 仍被压住但指标不升，说明当前恢复的覆盖量仍不足以改变整体重建。
- 若数量恢复靠近 38 但 TB 不升，说明当前更多是在恢复 hole/noise，而不是恢复背景。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
