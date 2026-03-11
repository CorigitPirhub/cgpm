# S2 动态抑制状态说明页

日期：`2026-03-09`
定位：本页用于统一解释当前项目在 `S2` 阶段出现的核心现象：
- 为什么项目总体上“动态抑制”仍是优势方向；
- 为什么 current-code `S2 dev quick canonical` 下又会表现出 `Bonn ghost_reduction_vs_tsdf` 明显不过线；
- 为什么这两件事并不矛盾。

## 1. 现象
当前最容易引发误解的现象是：
- 历史 `P10 / S2` 尝试中，dynamic suppression 一直被视为当前项目最有希望保留的优势方向；
- 但 current-code `S2 dev quick canonical`（`frames=5 / stride=3 / seed=7 / max_points_per_frame=600`）下，Bonn 指标却表现很差：
  - `14_bonn_localclip_drive_recheck`：`ghost_reduction_vs_tsdf = -8.00%`
  - `25_rear_bg_coupled_formation`：`ghost_reduction_vs_tsdf = -7.56%`
  - `30_rps_commit_geom_bg_soft_bank`：`ghost_reduction_vs_tsdf = -7.56%`
- 因此表面上看，似乎是“动态抑制方向突然不行了”。

## 2. 根因
结论：**不是 dynamic suppression 这个研究方向突然失效，而是 historical 效果在 current-code 的特定链条中没有被完整继承，并且长期被 downstream readout/export 机制掩盖。**

更准确地说，当前问题由三层原因叠加：

### 2.1 协议层原因：historical 与 current-code 口径曾不一致
historical `S2` 文档曾漏写关键协议项：
- `max_points_per_frame=600`

这导致：
- 旧文档中的一些数字属于 `historical archive`；
- current-code 下重新跑时，如果不补全协议，得到的结果并不能直接与历史冻结数值等价比较。

### 2.2 机制层原因：target synthesis 差异存在，但没有顺利传到最终导出
`ptdsf_diag` 已证明：
- current-code `05` 与 `14` 在 `write_time_dual_surface_targets()` 层仍存在显著差异；
- 也就是说，target synthesis 本身没有“完全坏掉”。

但这些差异在以下链条中被逐步抹平：
- `phi_static / phi_rear / phi_bg`
- `_sync_legacy_channels()`
- `extract_surface_points()`

因此长期表现成：
- `front_selected` 大量非零；
- `rear_selected / bg_selected / soft_bank_on` 长时间为零；
- 最终 reconstruction 表面几乎只剩 front-dominant readout。

### 2.3 当前更窄的主瓶颈：rear bank 规模与质量不足
本轮第三、四轮恢复实验说明：
- `25_rear_bg_coupled_formation` 已首次把 `bg_w / bg_cand_w` 恢复成全量非零；
- `30_rps_commit_geom_bg_soft_bank` 又首次把：
  - `rps_commit_score`
  - `rps_active`
  - `phi_rear / rho_rear`
  - `rear_selected`
 这条 rear commit 链打通；
- 但 Bonn 指标仍未起量。

这说明当前更精确的瓶颈已收缩为：
> committed rear bank 的有效规模与几何质量仍不足以转化为 Bonn ghost suppression 优势。

## 3. 证据链
### 证据 A：historical 与 current-code 不能直接混用
- 参考：`processes/s2/S2_CURRENT_CODE_CANONICAL_LOCK.md`
- 结论：current-code `S2 dev quick canonical` 必须显式写为：
  - `frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

### 证据 B：`05` 与 `14` 在 target 层确实不同
- 参考：`processes/s2/S2_RESTORE_05_TO_14_CHAIN_REPORT.md`
- 结论：`anchor_delta_sum / static_delta_sum / transient_delta_sum / rear_delta_sum` 在 `05` 与 `14` 之间显著不同。

### 证据 C：downstream 链曾长期只剩 front export
- 参考：
  - `processes/s2/S2_DOWNSTREAM_EXPORT_CHAIN_COMPARE.md`
  - `processes/s2/S2_DOWNSTREAM_SOFTBANK_COMPARE.md`
  - `processes/s2/S2_DOWNSTREAM_CHAIN_ATTACK_REPORT.md`
- 结论：`17-22` 全部 zero-delta，且 `rear_selected / bg_selected / soft_bank_on` 长期为 `0`。

### 证据 D：`bg state` 已被恢复
- 参考：`processes/s2/S2_REAR_BG_STATE_FORMATION_ANALYSIS.md`
- 结论：`25_rear_bg_coupled_formation` 首次把 `bg_w / bg_cand_w` 从 `0` 拉到全量非零，并把 TUM `Comp-R` 从 `68.53%` 提到 `91.90%`。

### 证据 E：rear commit 链已被打通，但 Bonn 仍未起量
- 参考：`processes/s2/S2_RPS_COMMIT_ACTIVATION_ANALYSIS.md`
- 结论：`30_rps_commit_geom_bg_soft_bank` 首次实现：
  - `rear_w_nonzero > 0`
  - `rps_active_nonzero > 0`
  - `rear_selected > 0`
  但 Bonn `ghost_reduction_vs_tsdf` 仍停留在 `-7.56%`。

## 4. 当前统一结论
因此，当前应统一采用如下说法：

> dynamic suppression 仍然是项目最有希望保留的优势方向；
> 当前表现不佳，不是因为方向本身突然失效，而是因为 historical 效果没有被完整继承到 current-code canonical，并且在一段时间内被 downstream readout/export 链掩盖；
> 现在这条链已经从 `target synthesis` -> `bg formation` -> `rear commit activation` 逐步恢复，但距离真正把 Bonn 指标拉到 `S2` 门槛仍有明显差距。

## 5. 当前阶段判断
- `S0`：完成（historical governance stage）
- `S1`：完成（不回滚）
- `S2`：**未通过，绝对不能进入 `S3`**

当前唯一 best iterate：
- `30_rps_commit_geom_bg_soft_bank`

## 6. 后续文档引用建议
后续若任何文档需要解释“为什么 dynamic suppression 明明是优势方向，但 current-code canonical 又表现不好”，统一引用本页即可：
- `processes/s2/S2_DYNAMIC_SUPPRESSION_STATUS_EXPLANATION.md`
