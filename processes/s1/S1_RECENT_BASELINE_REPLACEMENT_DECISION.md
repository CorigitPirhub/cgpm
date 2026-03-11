# S1 recent baseline 替代决策页

版本：`2026-03-08`

## 1. StaticFusion 的处理结论
- `StaticFusion` 由于环境适配/构建阻塞明显，不再作为当前项目的正式 blocking baseline；
- 其论文级 related-work 对齐可以保留，但不再进入当前 `RB-Core`。

## 2. recent dynamic dense 替代线
- 正式替代基线：`RoDyn-SLAM`（RA-L 2024）
- fallback：`NID-SLAM`

选择原因：
- 与当前项目同属 dynamic dense RGB-D / neural dense mapping 语境，更贴近当前审稿人会参考的现代对标；
- 官方代码可在本机 `cgpm` 环境完成依赖满足；
- 已在本地完成核心数据集 smoke + protocol check。

## 3. 2025 更近基线（替代原 `V3D-SLAM` 保留位）
- 新的 `2025` 优先对标线：`4DGS-SLAM`
- fallback：`D4DGS-SLAM`

使用原则：
- `4DGS-SLAM` 作为 `2025` recent dynamic dense / 4DGS family 代表进入 `RB-S1+`；
- 若本地接入阻塞，则记录阻塞并切换到 `D4DGS-SLAM` 或保留为 paper-level controlled comparison target；
- `V3D-SLAM` 不再保留为当前 `S1` 任务目标。

## 4. 当前冻结结论
- `RB-Core = TSDF + DynaSLAM + RoDyn-SLAM`
- `RB-S1+` 需在此基础上补入：
  - `NICE-SLAM`（已有真实输出链，作为 representation/neural dense 线）
  - `4DGS-SLAM`（2025 recent dynamic dense 线）
