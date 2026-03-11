# S1 P10 开发/锁箱协议卡

版本：`2026-03-08`

## 开发集
- TUM / oracle / `rgbd_dataset_freiburg3_walking_xyz`
- Bonn / slam / `rgbd_bonn_balloon2`
- 推荐快速配置：`frames=20`, `stride=3`, `seed=7`, `max_points_per_frame=3000`

## 锁箱集
- TUM / oracle / `rgbd_dataset_freiburg3_walking_static`
- TUM / oracle / `rgbd_dataset_freiburg3_walking_halfsphere`
- Bonn / slam / `rgbd_bonn_balloon`
- Bonn / slam / `rgbd_bonn_crowd2`
- static sanity: `rgbd_dataset_freiburg1_xyz`

## 固定要求
- 协议不得漂移：`oracle` 与 `slam` 严格分开
- `slam` 绝不允许 `GT delta` 泄漏
- 开发阶段只允许在开发集上比较候选方法
- 任何要进入阶段结论的 config，必须在锁箱集上方向不翻转

## S1 RB-Core gate 共用子协议
- 为保证 `TSDF / DynaSLAM family / RoDyn-SLAM family` 三条线在同一口径上完成本地闭环，`S1` 额外固定一条共同 `TUM` 子协议：
  - dev gate：`TUM / oracle / rgbd_dataset_freiburg3_walking_xyz / frames=5 / stride=3 / seed=7 / max_points_per_frame=3000`
  - lockbox gate：`TUM / oracle / rgbd_dataset_freiburg3_walking_static / frames=5 / stride=3 / seed=7 / max_points_per_frame=3000`
- 该 gate 仅用于验证：
  - `RB-Core` 三条线本地可运行；
  - compare 表可复跑；
  - 锁箱方向按 `RB-Core` 全面支配判据不翻转。
- `RoDyn-SLAM` 的 core-dataset smoke / protocol confirmation 单独记录于：`processes/s1/S1_RODYN_CORE_PROTOCOL_CHECK.md`

## 当前唯一 active mainline
- `evidence-gradient + dual-state disentanglement + delayed geometry synthesis`

## 当前滚动强基线面板（RB-Core）
- `TSDF`
- `DynaSLAM family`
- `RoDyn-SLAM family`

## 当前冻结的 RB-S1+
- `RB-Core`
- `NICE-SLAM`（representation / neural dense）
- `4DGS-SLAM`（2025 recent dynamic dense 目标线）

## 说明修正（2026-03-09）
- 本卡描述的是 `S1` baseline/gate 协议，不是 `S2 dev quick current-code canonical`；
- `S2 dev quick canonical` 已单独锁定为：`frames=5 / stride=3 / seed=7 / max_points_per_frame=600`；
- 后续文档不得再把 `S1 gate` 与 `S2 canonical` 混写。
