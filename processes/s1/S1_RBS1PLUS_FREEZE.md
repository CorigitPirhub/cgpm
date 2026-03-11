# S1 RB-S1+ 冻结页

版本：`2026-03-08`

## 冻结目标
`RB-S1+` 用于把 `S1` 从“只有 3 条 baseline 的闭环”提升到对后续 `S2/S3` 可用的稳定 baseline floor。

## 当前冻结组成
- `TSDF`：底线 / classical dense reconstruction
- `DynaSLAM family`：classic dynamic RGB-D baseline
- `RoDyn-SLAM family`：recent dynamic dense baseline（2024）
- `NICE-SLAM`：representation / neural dense baseline（已有真实输出链）
- `4DGS-SLAM`：2025 recent dynamic dense / 4DGS family 目标线

## 解释
- 仅有 `TSDF + DynaSLAM + RoDyn-SLAM` 三条线，不足以支撑最终顶刊/顶会投稿；
- 但只要把 `NICE-SLAM` 与 `4DGS-SLAM` 作为 `S1` 末尾冻结的补强线，`S1` 就已具备进入 `S2` 的 baseline floor；
- 更完整的 literature matrix 仍需在 `S2-S5` 继续扩充。
