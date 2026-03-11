# S2 surface constrained analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_SURFACE_CONSTRAINED_COMPARE.csv`

## 1. 表面约束是否成功
- 控制组 `38`: TB=`1`, Ghost=`10`, Hole/Noise=`97`
- 本轮最佳候选 `53_surface_adjacent_propagation`: TB=`3`, Ghost=`13`, Hole/Noise=`78`
- 表面约束至少提升了 True Background 覆盖。
- Ghost 仍高于本轮目标。

## 2. 指标是否达到新高度
- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = 15.47%`, `Comp-R = 70.83%`
- 本轮最佳候选 `53_surface_adjacent_propagation`: Bonn `ghost_reduction_vs_tsdf = 15.17%`, `Comp-R = 70.84%`
- ghost 指标没有超过控制组。
- Comp-R 有提升。

## 3. 若未达标，原因判断
- 若 TB 仍然很低，说明表面约束虽然限制了扩散方向，但支持仍未跨越遮挡缝隙。
- 若 rear 总量显著下降，则说明约束过硬，传播覆盖面不足。
- 若 Ghost 仍升高，说明表面掩码仍不足以区分动态表面与真实背景表面。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
