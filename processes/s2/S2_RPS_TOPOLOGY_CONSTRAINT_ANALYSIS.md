# S2 topology constraint analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_TOPOLOGY_CONSTRAINT_COMPARE.csv`

## 1. 拓扑特征有效性
- 控制组 `80`: thickness=`0.113`, normal=`0.741`, convergence=`0.161`
- `best TB` `83_minimum_thickness_topology_filter`: thickness=`0.384`, normal=`0.826`, convergence=`0.254`
- `best noise cleanup` `85_occlusion_ray_convergence_constraint`: thickness kept/dropped=`0.383/0.055`, normal kept/dropped=`0.940/0.694`, convergence kept/dropped=`0.729/0.424`

## 2. 是否清理了 Noise
- 控制组 `80`: TB=`4`, Ghost=`30`, Noise=`214`, `ghost_reduction_vs_tsdf=26.84%`
- `best noise cleanup` `85_occlusion_ray_convergence_constraint`: TB=`4`, Ghost=`22`, Noise=`99`, `ghost_reduction_vs_tsdf=22.30%`

## 3. TB 是否突破
- 控制组 `80`: TB=`4`
- 最佳候选 `83_minimum_thickness_topology_filter`: TB=`4`
- 若 `TB` 仍停在 `4`，说明拓扑约束只能净化已有候选，尚不能把隐藏在 `Noise` 中的点重新定位到真实背景表面。

## 4. 为什么拓扑约束比标量特征更有效
- `thickness` 直接编码前后表面间的最小物理间隔，能优先打掉贴着遮挡物后方的薄壁伪影。
- `front-back normal consistency` 检查后方点是否能与静态背景法向量构成连续表面，避免孤立浮点混入。
- `ray convergence` 要求多个相邻 rear 候选在厚度与法向上达成一致，比单点分数更接近真实遮挡拓扑。

## 5. 阶段判断
- 若未达到 `TB > 6`、`hole_or_noise_sum < 180`、`Ghost <= 25` 与 `ghost_reduction_vs_tsdf >= 22%`，则 `S2` 仍未通过，绝对不能进入 `S3`。
