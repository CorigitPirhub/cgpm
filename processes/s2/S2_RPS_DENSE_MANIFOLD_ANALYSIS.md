# S2 dense manifold analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_DENSE_MANIFOLD_COMPARE.csv`

## 1. 稠密流形是否成功构建
- 控制组 `38`: rear=`108`, TB=`1`, Ghost=`10`
- 本轮最佳候选 `52_dual_scale_manifold_fusion`: rear=`112`, TB=`1`, Ghost=`13`
- rear 总量已恢复到可用区间。
- True Background 覆盖仍明显不足。

## 2. 指标是否提升
- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = 15.47%`, `Comp-R = 70.83%`
- 本轮最佳候选 `52_dual_scale_manifold_fusion`: Bonn `ghost_reduction_vs_tsdf = 14.71%`, `Comp-R = 70.82%`
- ghost 指标没有超过控制组。
- Comp-R 仍无明显提升。

## 3. 若未达标，原因判断
- 若 TB 仍很低，说明局部传播虽然恢复了流形密度，但没有把支持场有效延展到被遮挡背景。
- 若 Ghost 同时升高，说明传播过程仍把动态区域当成了可传播背景。
- 若指标没有提升，则说明当前 dense manifold 仍不足以替代显式背景表面。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
