# S2 RPS commit activation 分析

日期：`2026-03-09`

## 1. 目标
本轮以 `25_rear_bg_coupled_formation` 为唯一控制组，专攻：
- `rps_commit_score`
- `rps_active`
- `phi_rear`
- `rho_rear`

即 `rear candidate -> committed rear bank activation`。

## 2. 机制设计
详见：`processes/s2/S2_RPS_COMMIT_ACTIVATION_DESIGN.md`

核心设计：
- `rps_commit_score` 同时考虑：
  - 证据强度：`rho_rear_cand`、`phi_rear_cand_w`
  - 几何一致性：`phi_rear_cand` 与 `phi_geo / phi_static / phi_bg` 的对齐程度
  - 背景支撑：`rho_bg / phi_bg_w`
  - 前景惩罚：`wod_front / wod_shell / q_dyn_obs / assoc_risk`
- `rps_active` 在 `score/age/rho/weight` 条件满足时触发：
  - `phi_rear_cand -> phi_rear`
  - `rho_rear_cand -> rho_rear`

## 3. 受控实验
协议固定：`frames=5 / stride=3 / seed=7 / max_points_per_frame=600`

- 控制组：`25_rps_commit_control`
- 实验组：
  - `28_rps_commit_geom_bg_soft`
  - `29_rps_commit_geom_bg_mid`
  - `30_rps_commit_geom_bg_soft_bank`

对比表：`processes/s2/S2_RPS_COMMIT_ACTIVATION_COMPARE.md`

## 4. 结果摘要
### `28_rps_commit_geom_bg_soft`
- 已成功激活 committed rear bank：
  - TUM `rear_w_nonzero = 108`
  - Bonn `rear_w_nonzero = 55`
  - `rps_active_nonzero` 同步非零
- 但最终 reconstruction 指标与控制组几乎完全一致
- 结论：`abandon`

### `29_rps_commit_geom_bg_mid`
- rear activation 过弱：
  - TUM `rear_w_nonzero = 4`
  - Bonn `rear_w_nonzero = 2`
- reconstruction 指标同样没有实质变化
- 结论：`abandon`

### `30_rps_commit_geom_bg_soft_bank`
- 同时完成：
  - committed rear-bank activation
  - rear-bank export participation
- 关键状态：
  - TUM `rear_w_nonzero = 108`, `rps_active_nonzero = 108`, `rear_selected = 21`
  - Bonn `rear_w_nonzero = 56`, `rps_active_nonzero = 56`, `rear_selected = 9`
- reconstruction 指标：
  - TUM `Acc = 0.9413 cm`
  - TUM `Comp-R = 91.93%`
  - Bonn `Acc = 2.8868 cm`
  - Bonn `Comp-R = 83.57%`
  - Bonn `ghost_reduction_vs_tsdf = -7.56%`
- 结论：`iterate`

## 5. 机制判断
本轮最重要的新结论：
- `rear candidate -> committed rear bank activation` 已经被真实打通；
- `30` 证明：`phi_rear_cand / rho_rear_cand -> phi_rear / rho_rear` 不再只是状态层“伪激活”，而已经进入 export 读出；
- 但即便如此，Bonn 主指标仍未起量，说明：
  - 当前 rear activation 规模仍偏小；
  - 或 rear bank 几何本身还不够准，无法转化为有效的 Bonn ghost suppression。

## 6. 是否达到 S2 目标
结论：**未达到。**

对照本轮成功判定标准：
- [ ] `Bonn ghost_reduction_vs_tsdf >= 22%`（当前 `-7.56%`）
- [ ] `TUM Comp-R >= 98%`（当前 `91.93%`）
- [x] `Acc` 未结构性崩塌（相对 `25` 回退远小于 10%）
- [ ] 至少一个数据集对 `RB-Core` 呈现明确净正优势（当前无法诚实宣称）

## 7. 当前最佳配置
当前唯一 best iterate：
- `30_rps_commit_geom_bg_soft_bank`

它相对 `25` 的价值不在于直接过线，而在于：
- 首次把 committed rear-bank activation 真正打通到 export 读出；
- 把 current-code 主瓶颈进一步收缩成“rear bank scale / quality 不足”，为下一轮继续冲线提供更清晰的方向。

## 8. 下一步建议
若下一轮继续 `S2`，唯一合理主线应为：
- 围绕 `30_rps_commit_geom_bg_soft_bank`，继续扩大并稳定 rear bank 的有效规模与几何质量；
- 重点继续攻坚：
  - `rps_commit_score` 的 evidence / geometry / bg support 权重
  - `rps_active` 的保持与衰减
  - `phi_rear` 与 `phi_geo / phi_static / phi_bg` 的几何一致性
- 不再回到 `25` 之前的 `17-29` 已淘汰分支。

## 9. 后续 competition-phase 复核（2026-03-09）
后续 `34/35/36` competition-only 复核已进一步确认：
- 当前 Bonn 真正进入 extract competition 的 rear 候选数量很少；
- 但这些样本一旦进入 competition，已经 `7/7` 全部胜出；
- 因此当前主瓶颈已从“rear bank scale / quality 不足”进一步修正为：
  - `too few committed rear states survive into extract competition`
  - 而不是最终 front-vs-rear competition boundary 本身。

统一引用：`processes/s2/S2_RPS_COMMITTED_BANK_COMPETITION_DIAGNOSIS.md`
