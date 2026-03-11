# S2 downstream chain attack 报告

日期：`2026-03-09`

## 1. 任务目标
本轮严格按新的任务书口径，只围绕：
- `phi_static`
- `phi_rear`
- `phi_bg`
- `_sync_legacy_channels()`
- `extract_surface_points()`

这条 downstream 链推进 `S2`，目标是验证 historical `05 -> 14` 的差异是否能在下游链条重新被放大为最终指标增益。

## 2. 方法设计来源
本轮方法设计参考了“多表示/多图层分离后应保持 readout bank identity，而不是过早塌缩成单一表面”的思路；该判断与下列 primary-source 方法的经验一致：
- Panoptic Multi-TSDFs（CVPR 2019）：https://openaccess.thecvf.com/content_CVPR_2019/html/McCormac_SceneGraph_Mapping_Using_Semantic_Instance_Information_CVPR_2019_paper.html
- ReFusion（ICRA 2019）：https://ieeexplore.ieee.org/document/8793760
- PG-SLAM（ICRA 2024）：https://openreview.net/forum?id=EN4t83x2de

本轮没有回到已被治理页明确退场的 `CSR-XMap / OTV / XMem / OBL / PFV-bank` 家族。

## 3. Round-A：banked downstream export competition
实现：
- 暴露 `rps_bank_*` 参数链；
- 在 `persistent_surface_readout()` 中记录 `ptdsf_export_diag`；
- 候选：
  - `17_banked_export_compete`
  - `18_banked_dual_compete_export`
  - `19_banked_nonpersistent_export`

结果表：`processes/s2/S2_DOWNSTREAM_EXPORT_CHAIN_COMPARE.md`

结论：
- 三个候选全部 `abandon`；
- `front_selected` 大量非零，但 `bg_selected = 0`、`rear_selected = 0`；
- `rear_score_sum = 0`、`rear_sep_sum = 0`；
- 说明当前不是“bank competition 不够强”，而是 `rear/bg` bank 在进入竞争前就已缺席。

## 4. Round-B：soft/candidate rear-bank export
实现：
- 新增 `rps_soft_bank_export_*` 配置链；
- 允许在 downstream readout 中尝试用 `phi_rear_cand` 代替 `phi_rear` 参与 soft rear-bank export；
- 候选：
  - `20_soft_rear_bank_export`
  - `21_soft_rear_bank_dual_compete`
  - `22_soft_rear_bank_nonpersistent`

结果表：`processes/s2/S2_DOWNSTREAM_SOFTBANK_COMPARE.md`

结论：
- 三个候选仍全部 `abandon`；
- `tum_bank_rear = 0`、`bonn_bank_rear = 0`；
- `tum_soft_bank_on = 0`、`bonn_soft_bank_on = 0`；
- 说明不仅 committed rear-bank 没有进入 downstream 竞争，连 candidate rear-bank 也没有在 current-code canonical 下形成可用 export entry。

## 5. Round-C：rear/bg state formation before export
实现：
- 新增 `rps_candidate_rescue_*`；
- 新增 `joint_bg_state_*`；
- 新增 `rear_bg_state_diag` 汇总；
- 候选：
  - `23_rear_candidate_support_rescue`
  - `24_joint_bg_state_coformation`
  - `25_rear_bg_coupled_formation`

结果表：`processes/s2/S2_REAR_BG_STATE_FORMATION_COMPARE.md`

结论：
- `23`：只拉高了 `rho_rear_cand_sum`，但没有形成可导出的 rear/bg state，`abandon`；
- `24`：首次把 `bg_w / bg_cand_w` 从 `0` 拉到全量非零，并把 TUM `Comp-R` 从 `68.53%` 提到 `73.93%`，但 `Acc` 回退过大，`abandon`；
- `25`：在保持 `bg state` 全量非零的同时，把 TUM `Comp-R` 进一步推到 `91.90%`，成为当前唯一新的 `iterate` 候选。

## 6. 关键诊断结论
### 6.1 `05` 与 `14` 的差异确实存在于 target synthesis 层
此前 `ptdsf_diag` 已证明：
- `anchor_delta_sum / static_delta_sum / transient_delta_sum / rear_delta_sum`
在 `05` 与 `14` 之间显著不同。

### 6.2 downstream 链经历了三层收缩定位
- `17-22` 说明：纯 export-side family 全部 zero-delta；
- `23` 说明：仅抬高 `rho_rear_cand` 还不够；
- `24/25` 说明：只要 `bg state` 被真正建立，downstream 链就会恢复一部分敏感性；
- 但 `rear_selected` 仍为 `0`，因此当前更窄主瓶颈已更新为：
  - `rear candidate -> committed rear bank activation`

## 7. 对 S2 的含义
- 本轮已完成“设计 -> 验证 -> 分析 -> 淘汰”闭环；
- 产出了唯一新的 `iterate` 候选：`25_rear_bg_coupled_formation`；
- 但 `S2` 仍然**不能通过，也绝对不能进入 `S3`**。

## 8. 对类似问题的进一步评估
当前已明确至少还存在以下同类风险：
1. `downstream chain sensitivity dead`
   - target 差异存在，但 sync/export 不敏感
2. `front+background collapse with missing rear bank`
   - `bg` 已可恢复，但 `rear` 仍未形成有效 committed bank
3. `whole-family zero-delta`
   - 只要方法作用点仍停留在纯 downstream export family，当前很大概率都将 zero-delta

## 9. 下一步建议
如果下一轮继续 `S2`，唯一合理主线应为：
- 以 `25_rear_bg_coupled_formation` 为当前唯一 `iterate` 候选；
- 专攻 `phi_rear_cand / rho_rear_cand -> phi_rear / rho_rear` 的 committed rear-bank activation；
- 不再继续堆 downstream-only export 竞争小改。
