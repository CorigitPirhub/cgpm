# S2 space redirect analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_SPACE_REDIRECT_COMPARE.csv`

## 1. 空间分布是否得到重定向
- 控制组 `38`: true background=`1`, ghost=`10`, hole/noise=`97`
- True-background 最强候选 `43_history_guided_background_location`: true background=`5`, ghost=`26`, hole/noise=`162`
- 结论：空间重定向未真正成功；虽然部分候选提高了 True Background 点数，但 Ghost 或 Hole/Noise 代价同步上升。

## 2. 指标是否得到实质性推动
- 指标最强候选 `44_history_plus_ghost_suppress`: Bonn `ghost_reduction_vs_tsdf = 15.22%`, `Comp-R = 70.76%`
- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = 15.47%`, `Comp-R = 70.83%`
- 结论：本轮没有实质性推动 Ghost 指标。
- `Comp-R` 仍然没有被拉起来。

## 3. 若未达标，主要原因
- 若 `true_background` 上升但 `ghost` 同时上升，说明重定向信号仍与动态区域耦合过强。
- 若 `hole/noise` 仍占主导，说明历史背景锚定仍不足，Rear Points 还在填补无效空洞。

## 4. 阶段判断
- 若未同时达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
