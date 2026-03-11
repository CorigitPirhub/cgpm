# S2 恢复 `05 -> 14` 失活链条报告

日期：`2026-03-09`

## 1. 任务目标
本轮目标不是继续做 `16` 微调，而是先恢复 historical `05_anchor_noroute -> 14_bonn_localclip_drive` 的实现收益链条，并确认当前仓库中是否还存在类似的“上游有差异、下游被抹平”的问题。

## 2. 已确认的 current-code canonical
本轮固定并沿用的 `S2 dev quick canonical`：
- `frames=5`
- `stride=3`
- `seed=7`
- `max_points_per_frame=600`

该 canonical 已锁定在：
- `processes/s2/S2_CURRENT_CODE_CANONICAL_LOCK.md`

## 3. 核心诊断结论
### 3.1 `05` 与 `14` 在 write-time target 层并非完全相同
本轮在 `summary.json` 中加入了 `ptdsf_diag`，用于统计 `write_time_dual_surface_targets()` 的内部行为。

current-code `05/14@600` 的关键诊断：
- `anchor_nonempty` 与 `anchor_nontrivial` 均大量非零；
- `anchor_delta_sum / static_delta_sum / transient_delta_sum / rear_delta_sum` 在 `05` 与 `14` 之间显著不同；
- 例如在 `TUM walking_xyz` 上：
  - `05`: `static_delta_sum ≈ 15.31`, `transient_delta_sum ≈ 8.27`, `rear_delta_sum ≈ 12.54`
  - `14`: `static_delta_sum ≈ 7.86`, `transient_delta_sum ≈ 0.96`, `rear_delta_sum ≈ 6.17`

结论：
- `14` 的 write-time synthesis 并没有完全失效；
- `05` 与 `14` 在 target 生成层已经明显不同。

### 3.2 差异在 downstream write/export 链条被抹平
尽管 `ptdsf_diag` 显示 target 层有显著差异，但 current-code canonical compare 仍表现为：
- `05` 与 `14` 的 `surface_points` 完全相同；
- `TUM / Bonn` 的 `Acc / Comp-R` 几乎完全相同；
- `15/16` 相对 `14` 也都是 zero-delta。

结论：
- 当前主失活点不在 `target synthesis` 本身；
- 而是在后续 `write -> sync -> export` 链条，导致上游差异无法投影到最终导出表面。

## 4. 本轮排查与修复尝试
本轮做过以下四类“最小修复”尝试，并全部按 canonical 口径复验：
1. `source/near cell` richer fallback
2. `rho_rear` soft keep under `rps_hard_commit_enable`
3. `dyn_prob` re-injection into `d_score`
4. `no-route` separation-aware internal response

结果：
- 这些尝试都**没有**把 historical `05 -> 14` 收益链条恢复到可接受程度；
- 个别尝试只带来极小波动，甚至方向不稳定；
- 因此这些行为性尝试**未被接受为新的 canonical 逻辑**。

最终保留的只有：
- canonical 协议修正
- `ptdsf_diag` 诊断计数
- current-code drift 文档化

## 5. 类似问题的充分评估
本轮已确认至少存在三类“类似问题”：

### 5.1 协议层类似问题
- historical `S2` 文档漏写 `max_points_per_frame=600`
- 说明仓库中存在“结果表看起来同口径，实际上协议漏项”的风险

### 5.2 target-to-export 失活问题
- `05/14` 在 target 层差异显著；
- 但最终 `surface_points` 与 reconstruction metrics 几乎不变；
- 说明当前 `sync/export` 链条存在明显的“差异抹平”问题

### 5.3 whole-family zero-delta 问题
- `14/15/16` 在 current-code canonical 下几乎完全等价；
- 说明不仅 `14` 本身有问题，整个 Bonn-side fine-tune family 当前都被同一 downstream 瓶颈卡住

## 6. 当前阶段判断
- 已完成：`restore-chain` 方向的系统排查与修复尝试；
- 但**historical `05 -> 14` 收益链条尚未恢复**；
- `S2` 当前仍然**不能通过，也绝对不能进入 `S3`**。

## 7. 下一步建议
如果下一轮继续 `S2`，不应再做 Bonn-side 小微调，而应直接转向：
1. 追踪 `phi_static / phi_rear / phi_bg -> _sync_legacy_channels() -> extract_surface_points()` 这条 downstream 链；
2. 定位究竟是哪一层把 `05/14` 的 target 差异抹平；
3. 只有恢复 downstream sensitivity 后，`14/15/16` 的 family calibration 才有继续优化意义。
