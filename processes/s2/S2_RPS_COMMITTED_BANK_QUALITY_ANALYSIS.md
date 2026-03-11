# S2 committed rear-bank quality enhancement analysis

日期：`2026-03-09`
协议：`frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
控制组：`30_rps_commit_geom_bg_soft_bank_recheck`
对比表：`processes/s2/S2_RPS_COMMITTED_BANK_QUALITY_COMPARE.csv`

## 1. 控制组复核
本轮先对 `30_rps_commit_geom_bg_soft_bank` 做了 current-code recheck，结果与既有冻结记录一致：
- TUM：`Acc = 0.9413 cm`，`Comp-R = 91.93%`
- Bonn：`Acc = 2.8868 cm`，`Comp-R = 83.57%`
- Bonn：`ghost_reduction_vs_tsdf = -7.56%`

说明本轮新增质量增强开关没有改动 `30` 的原始行为；后续 `31/32/33` 属于严格受控增量。

## 2. 对比结果
核心现象非常一致：
- `31/32/33` 都显著提高了 committed rear-bank 的 internal state；
- 但 Bonn 最终指标完全不变，`rear_selected` 也没有增长。

关键数字：
- 控制组 `30`（recheck）Bonn：
  - `rear_w_nonzero = 56`
  - `rear_selected = 9`
  - `commit_w_sum = 4.102`
  - `commit_rho_sum = 2.619`
  - `ghost_reduction_vs_tsdf = -7.56%`
- `31` Bonn：
  - `rear_w_nonzero = 57`
  - `rear_selected = 9`
  - `commit_w_sum = 4.791`
  - `commit_rho_sum = 4.566`
  - `ghost_reduction_vs_tsdf = -7.56%`
- `32` Bonn：
  - `rear_w_nonzero = 57`
  - `rear_selected = 9`
  - `commit_w_sum = 5.053`
  - `commit_rho_sum = 5.211`
  - `ghost_reduction_vs_tsdf = -7.56%`
- `33` Bonn：
  - `rear_w_nonzero = 57`
  - `rear_selected = 9`
  - `commit_w_sum = 5.143`
  - `commit_rho_sum = 5.707`
  - `ghost_reduction_vs_tsdf = -7.56%`

TUM 侧则几乎完全稳定：
- `Comp-R` 始终保持在 `91.93%`
- `Acc` 只有 `0.9413 -> 0.9411 cm` 级别微小波动

## 3. 结论
### 3.1 本轮方法没有推动 Bonn 主指标
虽然 internal state 被增强，但：
- Bonn `rear_selected` 固定停在 `9`
- Bonn `ghost_reduction_vs_tsdf` 固定停在 `-7.56%`
- Bonn `Comp-R / Acc` 也没有任何可用变化

因此 `31/32/33` 必须全部判为 `abandon`。

### 3.2 当前瓶颈进一步收窄
本轮最重要的负结果不是“方法完全无效”，而是：
> committed rear bank 的 transfer / rho / score 已经被显著拉高，但这些新增 committed state 仍然没有转化为更多 Bonn rear-surface export。

也就是说，当前更准确的瓶颈已经从：
- “rear bank 没有被激活”
收缩为：
- “rear bank 虽然被进一步写强了，但仍未跨过 Bonn 下游 front-vs-rear bank competition 的最终边界”。

### 3.3 对 S2 的含义
这轮实验说明：
- 单纯继续增加 committed rear-bank 的 internal mass / rho，已经不够；
- 只在 commit 形成侧继续加力，不足以推动 Bonn 主表面发生变化；
- `S2` 仍然没有通过，且仍然绝对不能进入 `S3`。

## 4. 当前唯一继续配置
当前唯一继续配置保持不变：
- `30_rps_commit_geom_bg_soft_bank`

原因：
- `31/32/33` 虽增强了内部状态，但对 Bonn 最终指标是严格 zero-delta；
- 因此它们不是新的 active configuration，只能作为“证明当前瓶颈位置”的负结果链。

## 5. 本轮统一判断
- `30_rps_commit_geom_bg_soft_bank`：`iterate / keep`
- `31_rps_commit_quality_bank_mid`：`abandon`
- `32_rps_commit_quality_bank_geom`：`abandon`
- `33_rps_commit_quality_bank_push`：`abandon`

统一结论：
> 本轮已完成 committed rear-bank quality enhancement 的受控实现与 canonical 验证；结论是 internal state 增强属实，但 Bonn 最终表面与 ghost 指标 zero-delta，故 S2 仍未通过。
