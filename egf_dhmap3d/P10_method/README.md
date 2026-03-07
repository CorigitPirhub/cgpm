# P10 Methods

本目录收纳 `TASK_LOCAL_TOPTIER.md` 中 `P10` 阶段已经代码化的主要方法簇，目标是在**不改变运行逻辑**的前提下，把原先散落在 `Updater / VoxelHash / Pipeline` 中的尝试按方法家族拆分整理。

当前迁移策略：
- 将已经具备独立方法边界的 helper / state-update / readout 函数抽到对应文件；
- 原类中的同名方法保留一层转发壳，确保调用路径与外部接口不变；
- 少数仍深度耦合在 `_integrate_measurement` / `extract_surface_points` 热路径内部的分支（如部分 `DCCM / WDSG / dual-map` 内联决策）暂不强拆，以避免引入逻辑漂移。
