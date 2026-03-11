# S2 hybrid optimization analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_HYBRID_OPTIMIZATION_COMPARE.csv`

## 1. 特征统计验证
- 控制组 `72`: history_anchor=`0.246`, surface_anchor=`1.000`, dynamic_shell=`0.000`
- `77_hybrid_boost_conflict`: history_anchor=`0.247`, surface_anchor=`1.000`, dynamic_shell=`0.000`
- `79_feature_weighted_topk`: history_anchor=`0.250`, surface_anchor=`1.000`, dynamic_shell=`0.000`
- 关键特征统计已成功输出为非零值。

## 2. 组合策略是否奏效
- 控制组 `72`: rear=`234`, TB=`3`, Ghost=`30`, Bonn `Comp-R = 70.43%`, `ghost_reduction_vs_tsdf = 22.10%`
- 最佳 TB 恢复 `77_hybrid_boost_conflict`: rear=`244`, TB=`4`, Ghost=`29`, Bonn `Comp-R = 70.41%`, `ghost_reduction_vs_tsdf = 21.83%`
- 唯一守住 `22%` 红线的 hybrid `79_feature_weighted_topk`: rear=`224`, TB=`3`, Ghost=`29`, Bonn `Comp-R = 70.42%`, `ghost_reduction_vs_tsdf = 22.24%`
- 结论：没有任何 hybrid 同时满足 `TB >= 6`、`ghost_reduction_vs_tsdf >= 22%` 和 `Ghost <= 25`。

## 3. 顾此失彼原因
- 若 history boosting 提高了 TB 但守不住 `22%` 红线，说明持久性本身不能区分“真实背景”与“长期残留伪影”。
- 若 dynamic-shell masking 压住了 Ghost 却拖垮 Comp-R，说明当前抑制范围仍过宽，把大量合法背景也一起裁掉了。
- 若 feature-weighted Top-K 只能维持 `22%` 却拉不回 TB，说明当前 `history_anchor + dynamic_shell + rear_score` 仍缺少足够的 true-background 判别力。
- 若 surface anchoring 统计长期接近饱和（`surface_anchor_mean = 1.000`），说明当前 surface-distance 特征区分度几乎为零，后验筛选只能做有限补救。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
