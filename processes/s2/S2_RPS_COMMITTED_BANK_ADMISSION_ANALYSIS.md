# S2 committed rear-bank admission analysis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_COMMITTED_BANK_ADMISSION_COMPARE.csv`

## 1. 结果概览
- `30_rps_commit_geom_bg_soft_bank`: Bonn `extract_hard_commit_on_sum = 7`, `extract_rear_enabled_sum = 7`, `extract_rear_selected_sum = 7`, `ghost_reduction_vs_tsdf = 13.25%`, decision=`control`
- `37_bonn_admission_gate_relax`: Bonn `extract_hard_commit_on_sum = 193`, `extract_rear_enabled_sum = 193`, `extract_rear_selected_sum = 193`, `ghost_reduction_vs_tsdf = 15.00%`, decision=`abandon`
- `38_bonn_state_protect`: Bonn `extract_hard_commit_on_sum = 108`, `extract_rear_enabled_sum = 108`, `extract_rear_selected_sum = 108`, `ghost_reduction_vs_tsdf = 15.47%`, decision=`iterate`
- `39_bonn_admission_gate_plus_protect`: Bonn `extract_hard_commit_on_sum = 202`, `extract_rear_enabled_sum = 202`, `extract_rear_selected_sum = 202`, `ghost_reduction_vs_tsdf = 15.11%`, decision=`abandon`

## 2. 哪个变体提升了 `bonn_extract_rear_selected_sum`
- **最大 admission 提升**来自 `39_bonn_admission_gate_plus_protect`：
  - 控制组 `bonn_extract_rear_selected_sum = 7`
  - `39` 提升到 `202`
  - 同时 `bonn_extract_hard_commit_on_sum` 从 `7` 提升到 `202`
- 但 **当前 best iterate** 不是 `39`，而是 `38_bonn_state_protect`：
  - `38` 的 `bonn_extract_rear_selected_sum = 108`
  - 但它在 Bonn `ghost_reduction_vs_tsdf` 上取得了本轮最佳值 `15.47%`
  - 因此说明“更多 admission”并不自动等于“更好的最终几何质量”

## 3. 是否转化为指标改善
- 控制组 Bonn `ghost_reduction_vs_tsdf = 13.25%`
- `38_bonn_state_protect = 15.47%`（本轮最佳，较控制组提升 `2.22` 个百分点）
- `39_bonn_admission_gate_plus_protect = 15.11%`（admission 最大，但弱于 `38`）
- 控制组 Bonn `Comp-R = 70.87%`，`38 = 70.83%`，`39 = 70.78%`
- 结论：admission 提升已经开始转化为 Bonn ghost 改善，但尚未转化为 `Comp-R` 提升；过强的 admission 反而会带来几何质量副作用。

## 4. 是否达到 S2 门槛
- TUM mean(best): `Acc = 0.9358 cm`, `Comp-R = 93.12%`
- Bonn mean(best): `Acc = 4.2431 cm`, `Comp-R = 70.78%`, `ghost_reduction_vs_tsdf = 15.11%`
- 结论：本轮若未同时触达 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。

## 5. 失败归因与下一轮最合理方向
- 当前 admission 设计已增加 extract 阶段 rear 候选，但 downstream geometry 质量或 extract filter 仍在限制最终指标转化。
- 综合判断：若未过线，更接近 admission 设计仍不够对症，而不是简单的参数未收敛。
