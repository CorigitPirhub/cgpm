# S2 ghost capped selectivity analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_GHOST_CAPPED_SELECTIVITY_COMPARE.csv`

## 1. Ghost 是否被拦截
- 基数组 `64`: rear=`442`, TB=`8`, Ghost=`60`
- 本轮最接近目标的候选 `67_topk_selective_generation`: rear=`231`, TB=`3`, Ghost=`38`
- selectivity pre/keep/drop: `442 / 231 / 0`，Top-K 额外裁剪=`211`
- Ghost 仍未压回控制线。
- True Background 被明显误伤。
- 覆盖率仍维持在安全区间。

## 2. 指标是否站稳
- 基数组 `64`: Bonn `ghost_reduction_vs_tsdf = 16.48%`, `Comp-R = 70.80%`
- 候选 `67_topk_selective_generation`: Bonn `ghost_reduction_vs_tsdf = 22.16%`, `Comp-R = 70.40%`
- extract `score_ready`: `442 -> 442`
- extract `support_protected`: `589 -> 589`
- ghost reduction 站上了 `16.5%`。
- Comp-R 仍无实质改善。

## 3. 诊断结论
- 若 `selectivity_drop_sum` 很大但 Ghost 仍高，说明当前风险分数没有精准对准 Ghost，而是在同时裁掉 TB 与 hole/noise。
- 若 Top-K 明显压低 rear 总量但 Ghost 仍不够低，说明排序分数本身缺乏选择性，而不是阈值太宽。
- 若 Geometry floor 一启用就把 rear 和 TB 一起砍掉，说明当前 `phi_geo` 对被遮挡背景仍不够可靠，不能作为硬门槛单独使用。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
