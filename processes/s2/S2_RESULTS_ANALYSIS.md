# S2 结果分析
# S2 结果分析

统一说明页：`processes/s2/S2_DYNAMIC_SUPPRESSION_STATUS_EXPLANATION.md`

## 1. historical 结果回顾
historical `S2` 已形成的主要认识仍然成立，且应被视为 archive：
- `anchor / counterfactual / energy`：`Acc` 可下降，但会明显伤害 `Comp-R`
- `anchor_noroute / counterfactual_noroute`：说明 route/spg 联动会放大 completeness 损失
- `anchor_ultralite_noroute`：第一次把 `TUM Comp-R` 拉回高位
- `10_anchor_ultralite_bonn_relaxed`：首次把 Bonn `ghost_reduction_vs_tsdf` 推到 `32%+`
- `13/14`：证明 Bonn-specific local clipping 在 historical 代码状态下确实有效

但这些结论目前只能被视为 **historical archive**，不能再直接当作 current-code canonical。

## 2. 本轮 re-baseline 结论
本轮首先定位并修正了协议漂移：
- historical `S2` dev quick 实际协议应为：
  - `frames=5`
  - `stride=3`
  - `seed=7`
  - `max_points_per_frame=600`

修正后，current-code `14/15/16` 的结果为：
- `14_bonn_localclip_drive_recheck`: `0.9355 / 68.53 / 2.8864 / 83.57 / -8.00%`
- `15_bonn_localclip_band_relax`: 与 `14` 等价
- `16_bonn_localclip_pfv_rearexpand`: 与 `14` 等价

格式：`TUM Acc / TUM Comp-R / Bonn Acc / Bonn Comp-R / Bonn ghost_reduction_vs_tsdf`

## 3. residual drift 的核心判断
修正协议后，residual drift 仍然存在，而且已经被明显定位：
- current-code `14` 不再呈现 historical `14` 的行为；
- current-code `14` 的行为几乎与 `05_anchor_noroute_recheck600` 等价：
  - `05`: `0.9292 / 68.50 / 2.8865 / 83.60 / -7.56%`
  - `14`: `0.9355 / 68.53 / 2.8864 / 83.57 / -8.00%`
- 二者差异已经小到可以忽略，这说明：
  - current-code 中原本让 `05 -> 14` 成立的收益链条已经失活；
  - 后续 `15/16` 之所以 zero-delta，并不是它们想法不够细，而是上游 delayed synthesis / dynamic cue 链条当前没有真正工作。

## 4. 漂移来源定位
本轮把 drift 分成两类：

### 4.1 协议漂移
- 文档漏写了 `max_points_per_frame=600`
- 导致上一轮误用 `1500/frame`，出现 `reference_points=7500` 的伪漂移

### 4.2 实现漂移
修正协议后仍然存在，定位到：
- `egf_dhmap3d/P10_method/ptdsf.py`
- `egf_dhmap3d/modules/updater.py`
- `egf_dhmap3d/modules/pipeline.py`

最直接的可观察症状是：
- historical `14` 的 `dynamic_score_mean` 为正；
- current-code `14` 的 `dynamic_score_mean(TUM/Bonn)` 全部掉到 `0.0`；
- `14/15/16` 在 current-code 下完全失去可分辨性。

## 5. 对 `16` 方向的判断
结论：**当前不继续 `16`。**

理由：
- `16` 在 current-code canonical 下没有比 `14` 更好的任何有效指标；
- 当前主矛盾不再是“Bonn-only 微调不够细”，而是“historical `14` 的生效链条已经失活”；
- 继续沿 `16` 只会得到重复的 zero-delta 实验。

## 6. 阶段判断
- 本轮已完成：`current-code re-baseline + canonical refresh + drift localization`
- `S2` 仍然**没有通过**
- 当前**绝对不能进入 `S3`**

## 7. 下一步
下一步唯一合理主线：
1. 以 `05_anchor_noroute_recheck600` 与 `14_bonn_localclip_drive_recheck` 的等价性为靶点
2. 恢复让 `historical 05 -> 14` 成立的实现收益链条
3. 只有在 `14` 再次显著区别于 `05` 之后，才重新讨论 `16` 这类 Bonn-only calibration 微调

## 8. 本轮 `05 -> 14` restore-chain 结论
详见：`processes/s2/S2_RESTORE_05_TO_14_CHAIN_REPORT.md`

补充结论：
- current-code `14` 与 `05` 在 `write_time_dual_surface_targets()` 的 target 层并非完全相同；
- 但这些差异没有成功传导到最终导出表面；
- 因此当前 residual drift 的更精确定位应为：`target synthesis alive, downstream write/export sensitivity dead`；
- 这也解释了为什么 `15/16` 在 current-code canonical 下会呈现 zero-delta。

## 9. 本轮 downstream chain attack 结论
详见：`processes/s2/S2_DOWNSTREAM_CHAIN_ATTACK_REPORT.md`

本轮先后验证了两组 downstream-only family：
- `17/18/19`: banked downstream export competition
- `20/21/22`: soft / candidate rear-bank export

共同结论：
- 全部 `abandon`；
- `front_selected` 大量非零，但 `rear_selected / bg_selected / soft_bank_on` 全为 `0`；
- 因此当前更精确的主矛盾不是“bank competition 不够强”，而是“rear/bg bank 在 export 前已不存在有效 entry”。

## 10. 本轮 rear/bg state formation 恢复结果
详见：`processes/s2/S2_REAR_BG_STATE_FORMATION_ANALYSIS.md`

本轮 `23/24/25` 的关键结论：
- `23` 仅拉高了 `rho_rear_cand_sum`，但没有形成可导出的 rear/bg state，`abandon`；
- `24` 首次把 `bg_w / bg_cand_w` 从 `0` 拉成全量非零，并带来 `TUM Comp-R` 的明确提升，但 `Acc` 回退过大，`abandon`；
- `25` 保住了 `24` 的 `bg state formation`，同时把 `TUM Comp-R` 从 `68.53%` 推到 `91.90%`，成为 current-code 下唯一新的 `iterate` 候选；
- 但 `rear_selected` 仍为 `0`，因此当前主瓶颈已从“rear/bg state formation 完全缺失”进一步收缩为“rear candidate -> committed rear bank activation”。
