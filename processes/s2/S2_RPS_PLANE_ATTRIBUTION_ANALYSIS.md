# S2 plane attribution analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_PLANE_ATTRIBUTION_COMPARE.csv`

## 1. 平面吸附效果验证
- 控制组 `80`: TB=`3`, Noise=`222`, Ghost=`23`
- 三个 plane attribution 变体都没有保住任何 TB：`86/87/88` 的 `TB` 全部降到 `0`
- `best noise cleanup` `88_occlusion_depth_hypothesis_validation`: thickness kept/dropped=`0.386/0.102`, normal kept/dropped=`0.691/0.743`, convergence kept/dropped=`0.047/0.166`

## 2. 是否打破 TB=4
- 控制组 `80`: TB=`3`
- plane attribution 三个变体 `86/87/88`: TB=`0/0/0`
- 结论：本轮不但没有打破 `TB=4` 的死锁，反而把剩余 TB 一并裁掉，说明当前平面归属假设过强。

## 3. Noise 是否被转化为 TB
- 控制组 `80`: Noise=`222`
- `best noise cleanup` `88_occlusion_depth_hypothesis_validation`: Noise=`7`
- 若 Noise 下降而 TB 归零，说明本轮更多是在删除或收缩错误点，而不是把它们成功吸附到真实背景表面。

## 4. 吸附是否影响 Acc
- 控制组 `80`: Bonn `Acc = 4.310 cm`, `ghost_reduction_vs_tsdf = 20.85%`
- 最佳候选 `88_occlusion_depth_hypothesis_validation`: Bonn `Acc = 4.443 cm`, `ghost_reduction_vs_tsdf = 34.36%`
- `Acc` 没有灾难性恶化，但 `Comp-R` 明显回落到 `67-69%`，说明当前平面吸附更偏向保守清理而非背景回填。

## 5. 阶段判断
- 若未达到 `TB > 6`、`hole_or_noise_sum < 180`、`Ghost <= 25` 与 `ghost_reduction_vs_tsdf >= 22%`，则 `S2` 仍未通过，绝对不能进入 `S3`。
