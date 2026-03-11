# S2 committed rear-bank admission diagnosis

日期：`2026-03-09`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
控制组：`30_rps_commit_geom_bg_soft_bank`

## 1. 诊断目标
- 追踪 committed rear states 在 `commit -> active -> sync -> extract` 各阶段的数量流失。
- 定位最主要的丢失环节，判断当前瓶颈是否位于 active 维护、hard-commit admission，还是 extract 前的准入筛选。

## 2. Bonn all3 链路统计
### `rgbd_bonn_balloon2`
- committed cells: `56`
- active cells: `56`
- sync hard-commit on: `5` / sync rear enabled: `5`
- extract wr present: `56`
- extract hard-commit on: `2`
- extract rear enabled: `2`
- extract rear selected: `2`
- extract fail_active: `0` / fail_score: `54`
- extract support_mean: `0.0000` / active_like_mean: `0.0182`

### `rgbd_bonn_balloon`
- committed cells: `39`
- active cells: `39`
- sync hard-commit on: `4` / sync rear enabled: `4`
- extract wr present: `39`
- extract hard-commit on: `1`
- extract rear enabled: `1`
- extract rear selected: `1`
- extract fail_active: `0` / fail_score: `38`
- extract support_mean: `0.0000` / active_like_mean: `0.0126`

### `rgbd_bonn_crowd2`
- committed cells: `84`
- active cells: `84`
- sync hard-commit on: `23` / sync rear enabled: `23`
- extract wr present: `84`
- extract hard-commit on: `4`
- extract rear enabled: `4`
- extract rear selected: `4`
- extract fail_active: `0` / fail_score: `80`
- extract support_mean: `0.0000` / active_like_mean: `0.0275`

## 3. 控制组瓶颈判断
- 若 `committed_cells >> extract_hard_commit_on`，则说明 committed rear state 在 extract admission 之前已被 hard-commit gate 大量拦截。
- 若 `extract_hard_commit_on ≈ extract_rear_selected`，则说明一旦通过 admission，后续 competition 并不是主要问题。
- 因此本轮主线应优先提升 `extract_hard_commit_on` 与 `extract_rear_enabled`，而不是继续调 competition。
