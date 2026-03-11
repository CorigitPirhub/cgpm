# S1 结果分析与第一轮候选筛选

## 1. 当前 active reference system 与 candidate 池

### Active reference system
- 名称：`current-egf-mainline`
- 用途：作为当前本地可运行主线与 `RB-Core` 面板对比时的固定参考系统
- 说明：它用于 `S1` 的 baseline closure / lockbox recheck，不计作 `S2` 的新方法候选

### Candidate-B
- 名称：`downstream delayed/export continuation`
- 状态：`abandon`
- 原因：已有大量历史分支表明边际收益衰减，不能再作为主线继续扩写

### Candidate-C
- 名称：`delayed-branch write-time target synthesis`
- 状态：`accept`
- 原因：当前最可能带来新增信息增益的上游瓶颈点，值得在 `S2` 作为唯一主创新方向推进

## 2. 更新版 RB-Core 开发门槛结论
- 数据：TUM `walking_xyz`, `oracle`, `frames=5`, `stride=3`, `seed=7`, `max_points_per_frame=3000`
- 表：`processes/s1/S1_RB_CORE_COMPARE_TUM_SMOKE.csv`
- 结论：
  - `EGF` 对 `TSDF`、`DynaSLAM family` 与 `RoDyn-SLAM family` 均未被全面支配；
  - 说明更新后的 `RB-Core` 已完成最小本地闭环。

## 3. 锁箱方向复验
- 数据：TUM `walking_static`, `oracle`, `frames=5`, `stride=3`, `seed=7`, `max_points_per_frame=3000`
- 表：
  - `processes/s1/S1_RB_CORE_LOCKBOX_DIRECTION_RECHECK.csv`
  - `processes/s1/S1_RB_CORE_LOCKBOX_DIRECTION_RECHECK.md`
- 判据：若某 baseline 在 `chamfer / fscore / comp_r_5cm` 三项上同时不差于 `EGF`，且至少一项严格更优，则视为“全面支配”；否则记为 `mixed/not-dominating`
- 结论：
  - 锁箱集上没有任何 `RB-Core` baseline 对 `EGF` 构成全面支配；
  - 更新后的 `RB-Core` 方向复验通过。

## 4. RoDyn-SLAM 核心数据集口径确认
- 表：`processes/s1/S1_RODYN_CORE_PROTOCOL_CHECK.csv`
- 覆盖：
  - TUM walking 三序列
  - Bonn all3：`balloon / balloon2 / crowd2`
  - static sanity：`freiburg1_xyz`
- 结论：
  - `RoDyn-SLAM` 已在当前 `cgpm` 主环境完成本地 runnable smoke；
  - 当前 smoke 口径下，Bonn 三序列几何结果仍明显偏弱，说明它是一个有效但不强势的 recent baseline；
  - 这恰好适合作为当前项目的 modern dynamic dense 对标线，而不会误伤归因边界。

## 5. 实验链充分性判断
- 仅有 `TSDF + DynaSLAM + RoDyn-SLAM` 三条线，不足以直接代表最终顶刊/顶会实验链的充分性；
- 因此 `S1` 末尾必须额外冻结 `RB-S1+`：
  - `NICE-SLAM`
  - `4DGS-SLAM`（2025）
- 在完成上述冻结后，`S1` 可视为“主命题收敛 + 评价闭环建立”已达标。

## 6. S1 阶段性判断
- 当前已完成：
  - 主命题收敛
  - 协议卡
  - 核心数据集接入确认
  - 更新版 `RB-Core` 本地闭环
  - `RoDyn-SLAM` 核心数据集口径确认
  - `RB-S1+` 冻结
  - 第一轮候选筛选
  - 锁箱方向复验
- 当前唯一 `accept` 候选：`delayed-branch write-time target synthesis`
- 阶段判断：**`S1` 已完成且不回滚；但进入 `S2` 后，active direction 的 current-code superiority 仍需在 `S2` 重新建立。**

## 7. 说明修正（2026-03-09）
- `S2` 的 current-code re-baseline 不会回滚 `S1` 的阶段完成判定；
- 但 `Candidate-C = delayed-branch write-time target synthesis` 现在应被理解为 historical handoff label，而不是 current-code superiority 已自动延续的证据；
- 后续若引用本页，必须同时引用 `processes/governance/S0_S1_S2_STAGE_REASSESSMENT_2026-03-09.md`。
