# S2 rear/bg state formation 分析

日期：`2026-03-09`

## 1. 本轮目标
本轮主线从 `export-side sensitivity` 上移到：
- `phi_rear`
- `phi_rear_cand`
- `rho_rear`
- `rho_rear_cand`
- `rho_bg`

目标是在 export 之前先恢复 `rear/bg state formation`，再观察这些 state 能否传导到最终表面。

## 2. 候选设计
协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

候选：
- `23_rear_candidate_support_rescue`
- `24_joint_bg_state_coformation`
- `25_rear_bg_coupled_formation`

对比表：`processes/s2/S2_REAR_BG_STATE_FORMATION_COMPARE.md`

## 3. 结果摘要
### `23_rear_candidate_support_rescue`
- `rho_rear_cand_sum` 增长，但：
  - `rear_w_nonzero = 0`
  - `bg_w_nonzero = 0`
- reconstruction 指标无可用变化
- 结论：`abandon`

### `24_joint_bg_state_coformation`
- 首次把 `bg_w_nonzero / bg_cand_w_nonzero` 从 `0` 拉到全量非零：
  - TUM：`0 -> 2855`
  - Bonn：`0 -> 2937`
- TUM：
  - `Comp-R: 68.53% -> 73.93%`
  - `ghost_reduction_vs_tsdf: -58.65% -> -52.07%`
- 但 TUM `Acc` 变差：`0.9355 -> 1.0058 cm`
- 结论：说明 `bg state formation` 是有效方向，但当前版本仍不够好，`abandon`

### `25_rear_bg_coupled_formation`
- 保持 `bg_w_nonzero / bg_cand_w_nonzero` 全量非零；
- 同时把 TUM `Comp-R` 从 `68.53%` 拉到 `91.90%`；
- TUM `Acc` 仅轻微回退：`0.9355 -> 0.9400 cm`；
- TUM `ghost_reduction_vs_tsdf` 显著改善：`-58.65% -> -15.76%`；
- Bonn：
  - `Acc` 基本不变：`2.8864 -> 2.8867 cm`
  - `Comp-R` 基本不变：`83.57% -> 83.57%`
  - `ghost_reduction_vs_tsdf` 仅略改善：`-8.00% -> -7.56%`
- 结论：`iterate`
- 说明：该候选随后在 commit activation 轮被 `30_rps_commit_geom_bg_soft_bank` supersede，但仍是当前 active line 的直接起点。

## 4. 机制判断
本轮最重要的新结论：
- 不是所有 `rear/bg state formation` 都无效；
- `25` 证明：只要 `bg state` 在 export 前被真正建立，downstream 链就会对上游差异重新变得敏感；
- 但 export 诊断同时显示：
  - `bg_selected` 已显著非零；
  - `rear_selected` 仍为 `0`；
- 因此当前主瓶颈已进一步收缩为：
  - `phi_rear_cand / rho_rear_cand -> phi_rear / rho_rear` 的 committed rear-bank activation

## 5. 阶段判断
- 本轮已完成“方法设计 -> 实验验证 -> 结果分析 -> 候选淘汰”闭环；
- 当前唯一值得继续的 `iterate` 候选是：
  - `25_rear_bg_coupled_formation`
- 但 `S2` 仍然**不能通过，也绝对不能进入 `S3`**。

## 6. 下一步建议
若下一轮继续 `S2`，唯一合理主线应为：
- `rear candidate -> committed rear bank activation`
- 也就是专攻：
  - `phi_rear_cand / rho_rear_cand`
  - `rps_commit_score / rps_active`
  - `phi_rear / rho_rear`

而不是再继续做纯 `bg` 扩张或 export-only 竞争。
