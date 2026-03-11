# S2 background manifold state analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_BACKGROUND_MANIFOLD_STATE_COMPARE.csv`

## 1. 是否解决了“零点”问题
- 控制组 `38`: rear_points=`108`
- 本轮最佳候选 `48_stable_background_memory_state`: rear_points=`29`
- 结论：稳定背景流形状态仍未能恢复足够的 rear points。

## 2. True Background / Ghost 变化
- 控制组 `38`: TB=`1`, Ghost=`10`, Hole/Noise=`97`
- 最佳候选 `48_stable_background_memory_state`: TB=`1`, Ghost=`3`, Hole/Noise=`25`
- True Background 点数没有提升。
- Ghost 点数仍控制在可接受范围内。

## 3. 指标是否提升
- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = 15.47%`, `Comp-R = 70.83%`
- 最佳候选 `48_stable_background_memory_state`: Bonn `ghost_reduction_vs_tsdf = 13.50%`, `Comp-R = 70.86%`
- Ghost 指标没有超过控制组。
- Comp-R 有提升。

## 4. 若未达标，原因判断
- 若 rear 总量恢复但 TB 仍不高，说明稳定流形状态仍不够准确，更多是在恢复量而非恢复位置。
- 若 manifold 引导仍把点投到 hole/noise，则说明历史背景状态本身还没有学成可靠 manifold。
- 若指标不提升，则说明当前阶段仍未真正把 rear points 转化成有效背景重建。

## 5. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
