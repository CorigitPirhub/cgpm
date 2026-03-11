# S2 occlusion bridge analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_OCCLUSION_BRIDGE_COMPARE.csv`

## 1. 跨缝隙桥接是否成功
- 控制组 `38`: TB=`1`, Ghost=`10`, Hole/Noise=`97`
- 本轮最佳候选 `57_historical_surface_rear_projection`: TB=`3`, Ghost=`9`, Hole/Noise=`46`
- True Background 有提升，说明桥接至少部分覆盖了被遮挡背景。
- Ghost 误入得到了抑制。

## 2. 指标是否改善
- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = 15.47%`, `Comp-R = 70.83%`
- 本轮最佳候选 `57_historical_surface_rear_projection`: Bonn `ghost_reduction_vs_tsdf = 15.02%`, `Comp-R = 70.89%`
- ghost 指标没有超过控制组。
- Comp-R 有提升。

## 3. 若未达标，原因判断
- 若 TB 提升有限，说明桥接距离或桥接目标选择仍不足以跨越真实遮挡缝隙。
- 若 Ghost 增长，说明桥接仍会错误穿过动态区域并落到伪背景。
- 若指标不提升，则说明当前桥接信号仍不足以替代直接观测的背景定位。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
