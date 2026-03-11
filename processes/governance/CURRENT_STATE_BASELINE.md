# 当前状态基线页

版本：`2026-03-09`
作用：作为阶段 `S0` 的一页式现状基线；自 `2026-03-09` 起，本页同时承担 `S0/S1/S2` 阶段状态重评估后的顶层现状说明。

## 1. 项目定位

当前项目是一个面向动态场景的 3D 局部建图研究原型，主线结构应概括为：

> `evidence-gradient + dual-state disentanglement + delayed branch + write-time target synthesis`

当前最大研究阻塞项仍是：
- `P10`
- 主矛盾仍是：`Acc / Comp-R / ghost suppression` 的统一优化

## 2. Canonical source of truth

当前需要区分两类 canonical：

### 2.1 对外主表 canonical
- `output/summary_tables/paper_main_table_local_mapping.csv`
- `output/summary_tables/local_mapping_main_metrics_toptier.csv`
- `output/summary_tables/dual_protocol_multiseed_significance.csv`

### 2.2 `S2 dev quick` current-code canonical
- 协议：`frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
- 锁定页：`processes/s2/S2_CURRENT_CODE_CANONICAL_LOCK.md`

### 2.3 S0 historical freeze snapshot
- `output/freeze_snapshots/S0_2026-03-08_summary_tables/`
- 现应视为 `historical governance archive`，不再直接等同 current-code canonical。

## 3. 当前正式主表现状

按当前 `5-seed canonical` 口径：
- TUM dynamic mean：`Acc = 4.1655 cm`, `Comp-R = 100.0%`
- Bonn dynamic mean：`Acc = 6.1481 cm`, `Comp-R = 77.21%`

结论：
- 当前正式主表下，项目还不具备直接投稿顶刊/顶会主口径；
- 最大短板是：
  - `Acc`
  - Bonn `Comp-R`
  - Bonn dynamic suppression 强度

## 4. 当前 P10 专项差距

当前 `P10` 门槛：
- TUM oracle：`Acc <= 1.80 cm`, `Comp-R >= 95%`
- Bonn slam：`Acc <= 2.60 cm`, `Comp-R >= 95%`
- `ghost_reduction_vs_tsdf >= 35%`

当前最接近结果：
- TUM 最优 `Acc = 2.789 cm`，还差 `0.989 cm`
- Bonn 最优 `Acc = 3.349 cm`，还差 `0.749 cm`
- Bonn 最优 `ghost_reduction_vs_tsdf = 15.4%`，还差约 `19.6` 个百分点

## 5. 当前 active mainline

当前应保留的 active mainline：
- `evidence-gradient`
- `dual-state disentanglement`
- `delayed branch`
- `write-time target synthesis`

当前应降级为 diagnostics chain 的内容：
- delayed/export 末端微调分支
- 已判负的单模块 veto / memory / cross-map 家族

## 6. 当前阶段总状态

结论：
- `S0`：已完成（historical governance stage）
- `S1`：已完成（baseline floor / protocol closure 有效，不回滚）
- `S2`：已完成 current-code re-baseline 与 drift localization，但**未通过，绝对不能进入 `S3`**

说明：
- `S0` 的 freeze snapshot 现在属于 historical archive；
- `S1` 交接给 `S2` 的 active direction 仍保留，但 current-code superiority 需在 `S2` 重新建立；
- 当前项目真正的主阻塞项位于 `S2` downstream write/export sensitivity，而不是 `S0/S1` 治理或 baseline 接入。

对应总评估页：`processes/governance/S0_S1_S2_STAGE_REASSESSMENT_2026-03-09.md`
对应总评估页：`processes/governance/S0_S1_S2_STAGE_REASSESSMENT_2026-03-09.md`
动态抑制状态统一说明页：`processes/s2/S2_DYNAMIC_SUPPRESSION_STATUS_EXPLANATION.md`
