# S2 multi-candidate generation analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_MULTI_CANDIDATE_GENERATION_COMPARE.csv`

## 1. 覆盖率是否提升
- 控制组 `38`: rear=`108`, TB=`1`, Ghost=`10`, extract_selected=`585`
- 本轮最佳候选 `64_patch_depth_hybrid_generation`: rear=`442`, TB=`8`, Ghost=`60`, extract_selected=`1925`
- bridge 合成更新数：`36612`，合成权重和：`40.44`
- rear 覆盖量达到了本轮目标。
- True Background 仍未突破 10。
- Ghost 超出了本轮允许上界。

## 2. 指标是否改善
- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = 15.47%`, `Comp-R = 70.83%`
- 本轮最佳候选 `64_patch_depth_hybrid_generation`: Bonn `ghost_reduction_vs_tsdf = 16.48%`, `Comp-R = 70.80%`
- extract `support_protected`: `201 -> 589`
- extract `score_ready`: `108 -> 442`
- extract `fail_score`: `93 -> 154`
- ghost 指标有净提升。
- Comp-R 仍无明显提升。

## 3. 诊断结论
- 若 `bridge_rear_synth_updates_sum` 明显增加但 `extract_score_ready_sum` 没有同步恢复，说明新增 rear state 仍在 admission/score_gate 被拦截。
- 若 `rear_points_sum` 增加但 `true_background_sum` 不增，说明多点生成主要放大了 hole/noise，而非命中真实背景。
- 若 `true_background_sum` 上升且 `ghost_sum` 仍受控，则说明多点生成开始把 bridge 方向转化为有效背景覆盖。

## 4. 阶段判断
- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。
