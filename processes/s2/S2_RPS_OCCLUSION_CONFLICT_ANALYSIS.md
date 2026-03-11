# S2 occlusion conflict analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_OCCLUSION_CONFLICT_COMPARE.csv`

## 1. 新特征是否区分了 TB 和 Ghost
- 基组 `68`: rear=`204`, TB=`2`, Ghost=`24`, order=`0.031`, conflict=`0.062`, residual=`0.094`
- 本轮最佳候选 `72_local_geometric_conflict_resolution`: rear=`234`, TB=`4`, Ghost=`24`, order=`0.032`, conflict=`0.112`, residual=`0.094`
- 基组 kept/drop front_residual=`0.094/0.092`, dyn_risk=`0.000/0.035`
- 候选 kept/drop front_residual=`0.094/0.091`, dyn_risk=`0.000/0.040`
- `occlusion_order` 均值提升。
- `local_conflict` 没有明显下降。

## 2. 是否解决“指标虚高但分布恶化”
- 基组 `68`: Bonn `ghost_reduction_vs_tsdf = 23.55%`, `Comp-R = 70.39%`
- 候选 `72_local_geometric_conflict_resolution`: Bonn `ghost_reduction_vs_tsdf = 27.27%`, `Comp-R = 68.81%`
- ghost_reduction_vs_tsdf 保持在 `22%+`。
- 仍未同时达到 `Ghost <= 15` 与 `TB >= 8`。

## 3. 诊断结论
- 若 `occlusion_order` 上升但 TB 不升，说明当前时序顺序特征仍偏向“保守裁剪”，没有真正识别被遮挡背景。
- 若 `local_conflict` 降低但 Ghost 仍高，说明冲突检测只裁掉了少量浅层残留，未覆盖主要 Ghost 来源。
- 若保留/剔除点在 `front_residual` 或 `dyn_risk` 上均值差异仍然很小，说明当前动态上下文特征没有形成有效分类边界。
- 若 `front_residual` 压制后仍主要降低 TB，说明当前前景残留信号与 TB 深度区高度耦合，仍缺少可分离特征。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
