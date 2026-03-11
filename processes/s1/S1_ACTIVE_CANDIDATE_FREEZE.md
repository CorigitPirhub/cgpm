# S1 唯一 Active Candidate 冻结页

版本：`2026-03-08`

## 唯一 active candidate
- `delayed-branch write-time target synthesis`

补充说明：
- `current-egf-mainline` 仅作为 `S1` 的 `RB-Core` 对比参考系统，不计入 `S2` 方法候选名额。

## 冻结理由
- 当前所有 downstream delayed/export 微调分支都已证明边际收益有限；
- 当前最合理的主创新突破点位于 delayed branch 写入期，而非导出期。

## 当前冻结约束
- 后续 `S2` 只允许围绕 `write-time target synthesis` 设计 `1-3` 个候选版本
- 不允许再次扩写新的 export-side 独立分支
- 不允许同时漂移 delayed/export 多个模块后再做单模块归因

## 说明修正（2026-03-09）
- 本页冻结的是 `S1 -> S2` 的 historical handoff label；
- 它仍然说明 `write-time target synthesis` 是 `S1` 收敛出的唯一继续研究方向；
- 但它不再自动等同于“current-code 下该 candidate 的 superiority 已被持续保持”；
- current-code 下是否仍成立，必须以 `S2` 的 re-baseline / canonical compare 为准。
