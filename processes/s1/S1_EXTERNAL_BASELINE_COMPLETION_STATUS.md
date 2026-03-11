# S1 外部基线完成度表

版本：`2026-03-08`

说明：
- 本表只记录**当前仍保留在主线计划中**、且属于“已经接入”或“未来将接入”的外部基线；
- 已被移除或归档的基线不再出现在本表；
- 状态分为：`已完成` / `部分完成` / `未开始`。

| baseline_family | role | current_stage_status | completion_level | current_scope | next_required_action |
|---|---|---|---|---|---|
| `DynaSLAM family` | classic dynamic RGB-D baseline | 已有本地 runner/adapter，已完成 TUM smoke/dev/lockbox | `部分完成` | TUM 本地链路已通；Bonn 尚未完成当前本地 runner 闭环 | 补 Bonn 侧 protocol-aligned 本地 compare，进入更完整 baseline matrix |
| `RoDyn-SLAM family` | recent dynamic dense baseline (2024) | 已完成本地接入与核心数据集 protocol check | `已完成` | TUM walking 三序列 + Bonn all3 + static sanity 已完成 smoke/eval 级口径确认 | 在后续阶段扩展到更强 budget / 更稳定 compare |
| `NICE-SLAM` | representation / neural dense baseline | 已有真实外部输出链，但只覆盖部分核心序列 | `部分完成` | 当前最强是 TUM `walking_xyz` 的真实输出链；尚未覆盖全核心数据集 | 扩展到更多核心序列并进入统一 compare matrix |
| `4DGS-SLAM` | 2025 recent dynamic dense / 4DGS family baseline | 已冻结为 `RB-S1+` 目标线，尚未本地接入 | `未开始` | 仅完成 baseline freeze，不存在本地 protocol-aligned 结果 | 启动代码接入 / 输出接入，并完成核心数据集口径确认 |
| `D4DGS-SLAM` | `4DGS-SLAM` fallback | 尚未接入，仅作为备用方案保留 | `未开始` | 仅在任务书中作为 fallback 存在 | 若 `4DGS-SLAM` 阻塞，则切换为备用接入线 |
| `NID-SLAM` | recent dynamic dense fallback | 尚未接入，仅作为 `RoDyn-SLAM` fallback 保留 | `未开始` | 当前未启用 | 仅当 `RoDyn-SLAM` 失效或需要补强时再启动 |
| `TSDF++ / Panoptic Multi-TSDFs` | representation / multi-surface family | 仍处于 literature baseline 规划状态 | `未开始` | 尚未形成本地或可信输出链 | 在后续阶段完成至少一条对齐线 |

## 当前阶段判断
- 在 `S1` 末尾，真正达到“已完成接入并完成核心数据集口径确认”的外部基线，当前只有：`RoDyn-SLAM`；
- `DynaSLAM` 与 `NICE-SLAM` 都属于**部分完成**：已具备可用资产，但尚未覆盖完整核心数据集 compare；
- `4DGS-SLAM / D4DGS-SLAM / NID-SLAM / TSDF++ / Panoptic Multi-TSDFs` 目前都还属于后续阶段要继续推进的外部基线。

## 当前对 `S2` 直接可用的基线

按“是否已经完成本地接入、并且当前就能为 `S2` 提供稳定 compare floor”划分：

### 可直接用于 `S2` 的基线
- `TSDF`
  - 说明：原生本地基线，canonical 与本地 benchmark 都已稳定。
- `RoDyn-SLAM family`
  - 说明：已完成当前核心数据集 smoke / eval 级口径确认，可作为 `S2` 的 recent dynamic dense baseline floor。
- `DynaSLAM family`（限当前 `TUM` 开发/锁箱协议直接可用）
  - 说明：当前在 `TUM` 侧已有本地 runner/adapter 与 gate compare，可直接用于 `S2` 的 `TUM` 向对比；
  - 限制：`Bonn` 侧尚未形成同等强度的本地闭环，因此不能单独作为 `S2` 全核心数据集的完整 baseline。

### 仅部分可用于 `S2` 的基线
- `NICE-SLAM`
  - 说明：当前已有真实外部输出链，可用于 `TUM walking_xyz` 等局部 representation/neural dense sanity compare；
  - 限制：尚未完成全核心数据集覆盖，因此当前更适合作为 `S2` 的补充线，而不是主 compare matrix 的唯一 representation baseline。

### 当前还不能直接用于 `S2` 的基线
- `4DGS-SLAM`
- `D4DGS-SLAM`
- `NID-SLAM`
- `TSDF++ / Panoptic Multi-TSDFs`

原因：这些线当前仍处于 frozen target / planning 阶段，尚未完成本地 protocol-aligned 接入与口径确认。
