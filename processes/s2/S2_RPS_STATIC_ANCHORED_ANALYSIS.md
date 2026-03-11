# S2 static anchored analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_STATIC_ANCHORED_COMPARE.csv`

## 1. 历史锚定是否有效提升了 TB
- 基组 `72`: rear=`234`, TB=`3`, Ghost=`30`, Noise=`201`
- 本轮最佳候选 `74_static_history_weight_boosting`: rear=`252`, TB=`4`, Ghost=`33`, Noise=`215`
- 基组 history/surface=`0.000/0.000`, 候选=`0.000/0.000`
- 基组 surface_distance=`0.0000`, 候选=`0.0000`
- 历史锚定提升了 TB。
- Noise 没有被有效清理。

## 2. Comp-R 是否恢复
- 基组 `72`: Bonn `Comp-R = 70.43%`, `ghost_reduction_vs_tsdf = 22.10%`
- 候选 `74_static_history_weight_boosting`: Bonn `Comp-R = 70.43%`, `ghost_reduction_vs_tsdf = 21.91%
- Bonn Comp-R 仍未恢复到 `70.5%`。
- Ghost 仍超出 `25`。

## 3. 诊断结论
- 若 `history_anchor_mean` 与 `surface_anchor_mean` 上升，但 TB 仍不升，说明静态持久性只是在筛掉极端噪声，尚未把候选准确拉回真实背景表面。
- 若 `surface_distance_mean` 下降但 Comp-R 不升，说明当前“表面吸附”更多是在做保守裁剪，而非恢复真实背景覆盖。
- 若 `dynamic_shell_mean` 下降但 Ghost 仍高，说明 Ghost 源并不只来自动态壳层，仍存在更深层的错误 rear synthesis。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
