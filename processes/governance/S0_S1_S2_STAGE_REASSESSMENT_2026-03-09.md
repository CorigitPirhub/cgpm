# S0 / S1 / S2 阶段重评估（2026-03-09）

## 结论总览
- `S0`：**完成，但其 `2026-03-08` freeze snapshot 现在应视为 historical governance archive，不再等同 current-code canonical**
- `S1`：**完成，且不需要回滚；但其 handoff candidate 说明必须改写为“historical accept/handoff label”，不能直接视为 current-code superiority 已成立**
- `S2`：**未完成 / 未通过 / 绝对不能进入 `S3`**

## 一、S0 的重评估
### 仍然成立的部分
- 治理动作已完成：
  - canonical source of truth 冻结
  - 历史任务书归档
  - 主线/支线裁剪
  - 现状一页式治理页建立
- 因此 `S0` 作为“治理阶段”仍然是完成的。

### 需要修正的部分
- `S0` 形成的 `2026-03-08` freeze snapshot 不能再被表述为 current-code canonical；
- 经过后续 `S2 current-code re-baseline`，它现在更准确地属于：
  - `historical governance archive`
  - 而不是 current-code executable truth。

### 结论
- `S0` 不回滚；
- 但后续文档必须避免把 `S0` freeze snapshot 继续当作 current-code 主口径引用。

## 二、S1 的重评估
### 仍然成立的部分
- 主命题收敛已完成；
- `RB-Core` 本地接入与 baseline floor 建立已完成；
- `RB-S1+` 冻结已完成；
- 开发/锁箱 protocol card 已建立；
- 因此 `S1` 作为“主命题收敛 + 评价闭环建立阶段”仍然成立。

### 需要修正的部分
- `S1` 文档中凡是写到 `frames=5 / stride=3 / seed=7` 的 gate/protocol 子协议，必须把 `max_points_per_frame=3000` 写显式；
- `S1` 中“`delayed-branch write-time target synthesis` = 唯一 active candidate / accept candidate”这一表述，当前应改写为：
  - 这是 historical handoff label；
  - 它允许 `S2` 继续围绕该研究方向推进；
  - 但不等于 current-code 下该 candidate 的 superiority 已经保持成立。

### 结论
- `S1` 不回滚；
- 但其 handoff 叙事必须从“已证明当前有效”修正为“已完成方向收敛，current-code superiority 需在 `S2` 重新建立”。

## 三、S2 的重评估
### 当前已完成的部分
- `current-code re-baseline`
- `canonical refresh`
- `protocol drift` 定位
- `historical 05 -> 14` 失活链条排查
- `15/16` Bonn-only 微调的 current-code 复验与淘汰

### 当前未完成的部分
- 未恢复 historical `05 -> 14` 的实现收益链条；
- 未恢复 `14` 与 `05` 的 current-code 差异传导到最终导出表面；
- 未满足 `S2` 量化门槛；
- 因此不能进入 `S3`。

### 更精确的 current-code 判断
- current-code `14` 与 `05` 在 `write-time target synthesis` 层仍有显著差异；
- 但这些差异在 downstream `write -> sync -> export` 链被抹平；
- 因此当前 `S2` 的主矛盾已从“继续调 Bonn clipping”切换为“恢复 downstream sensitivity”。

## 四、统一说明原则
从本页起，活跃文档中所有与阶段状态相关的说明必须遵守：
1. 区分 `historical archive` 与 `current-code canonical`
2. 阶段“完成”与“handoff label 仍可用”不是同一件事
3. `S2` 当前唯一真实结论是：
   - 已完成 re-baseline 和 drift localization
   - 但阶段未通过，绝对不能进入 `S3`
