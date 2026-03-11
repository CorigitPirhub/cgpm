# S2 history-visible obstructed analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_HISTORY_VISIBLE_OBSTRUCTED_COMPARE.csv`

## 1. True Background / Ghost 变化
- 控制组 `38`: TB=`1`, Ghost=`10`, Hole/Noise=`97`
- True-background 最强候选 `46_history_background_only_admission`: TB=`0`, Ghost=`0`, Hole/Noise=`0`
- 结论：新的约束没有达成目标分布；True Background 仍远未到 20，或 Ghost 已明显超出允许范围。

## 2. 指标是否得到推动
- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = 15.47%`, `Comp-R = 70.83%`
- 指标最强候选 `46_history_background_only_admission`: Bonn `ghost_reduction_vs_tsdf = 13.29%`, `Comp-R = 70.86%`
- Ghost 指标没有超过控制组。
- `Comp-R` 只有微小数值波动，但在 rear points 被完全清零的前提下，这个变化不具备方法学正面意义。

## 3. 若未达标，原因判断
- 若 TB 上升但 Ghost 同步大幅上升，说明“历史可见 + 当前遮挡”约束仍与动态区域耦合。
- 若 TB 仍很低，说明当前 rear generation 仍在历史 manifold 之外生成。
- 若指标没有提升，则说明该约束尚未实质性改善 Rear Points 的空间有效性。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
