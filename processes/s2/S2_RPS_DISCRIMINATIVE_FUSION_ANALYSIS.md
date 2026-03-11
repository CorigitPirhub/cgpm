# S2 discriminative fusion analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_DISCRIMINATIVE_FUSION_COMPARE.csv`

## 1. 判别特征是否分离了 TB/Ghost
- 基组 `67`: rear=`231`, TB=`3`, Ghost=`41`, front=`0.174`, rear=`0.419`, gap=`0.2450`, comp=`0.280`
- 本轮最佳候选 `68_rear_front_score_competition`: rear=`204`, TB=`2`, Ghost=`32`, front=`0.174`, rear=`0.414`, gap=`0.2405`, comp=`0.249`
- 退化候选：`69_depth_gap_validation, 70_fused_discriminator_topk` 直接把 rear 裁到 `0`，其 ghost 指标不具备可用性。
- 判别融合没有提高后前竞争均值。
- 平均 front_score 被压低。

## 2. 指标是否站稳 22%
- 基组 `67`: Bonn `ghost_reduction_vs_tsdf = 22.43%`, `Comp-R = 70.37%`
- 候选 `68_rear_front_score_competition`: Bonn `ghost_reduction_vs_tsdf = 23.52%`, `Comp-R = 70.33%`
- ghost_reduction_vs_tsdf 站稳了 `22%`。
- 仍未同时满足 `Ghost <= 15` 和 `TB >= 8`。

## 3. 诊断结论
- 若 `competition_mean` 提升而 Ghost 仍高，说明单纯 `rear-front` 竞争还不能覆盖动态残留的复杂模式。
- 若 `rear_gap_mean` 收窄但 TB 下降，说明 gap 约束更像是在裁覆盖，而不是识别正确背景。
- 若 `ghost_reduction_vs_tsdf` 继续保持在 `22%+`，但 Ghost Count 仍下不来，说明当前指标改善主要来自总量裁剪，而非真正的 TB/Ghost 分类正确率提升。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
