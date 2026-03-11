# S1 方法概述（1页版）

## 主命题

论文主命题固定为：

> `evidence-gradient + dual-state disentanglement + delayed geometry synthesis`

## 结构概述

### 1. 状态层解耦
- `dual-state` 把静态与动态污染从状态层分离；
- `evidence-gradient` 提供几何/证据联合约束；
- 目标是避免单纯后置删点对 `Acc` 的天然无力。

### 2. Delayed branch
- 对 conflict samples 不直接写入 committed background；
- 而是进入 delayed branch，形成 conflict-isolated geometry path；
- delayed branch 的存在不是为了多一张图，而是为了让冲突几何拥有独立的生存路径。

### 3. 当前未解决的核心问题
- 近期多轮 tri-map/export 微调已经表明：
  - delayed-only 能打开；
  - promotion 回流能抑制；
  - delayed export 能参与；
  - 但最终收益仍停留在极小 mixed change。
- 因此，当前最值得继续投入的上游方向是：
  - `write-time target synthesis`
  - 让 delayed branch 在写入期就拥有 delayed-specific surface target。

## Supporting modules
- supporting module 1: `Acc` line（`lzcd / stcg / dualch / geo debias` family）
- supporting module 2: delayed/export diagnostics chain（仅作 supporting evidence，不再作为独立主贡献）

## Appendix-only / diagnostics chain
- OTV / CSR-XMap / XMem / OBL / CMCT / CGCC / PFV-sharp / PFV-bank
- quantile / hold / residency export / local replacement / competition replacement / banked readout / persistent bank

## 当前 S1 要完成的决策
- 固定 active mainline
- 固定开发/锁箱协议
- 建立 RB-Core 对比面板
- 从若干候选方向中只保留一个 `accept` 候选进入 `S2`
