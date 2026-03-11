# S2 Visibility Deficit Design

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 借鉴来源

| Paper | Venue | Year | 借鉴思想 |
|---|---|---:|---|
| Accurate Training Data for Occupancy Map Prediction in Automated Driving Using Evidence Theory | CVPR | 2024 | 用 evidence theory 将反射/传输沿射线转换成 occupied / free / unknown 质量。 |
| PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness | CVPR | 2024 | 不确定性不是副产品，而应直接成为体素激活的决策量。 |
| GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction | ICCV | 2023 | 必须先修正 pose consistency，再谈 dense gap localization。 |

这些工作只提供**思想借鉴**；本轮实现没有照抄其 Dempster-Shafer 合成公式、神经网络或特定网格结构。

## 2. 122 `Evidential Visibility Deficit`

对每个候选体素 `v`，基于每一帧的射线投影统计：
- `n_surface_hit(v)`：射线与深度表面一致的次数；
- `n_free_traversal(v)`：候选体素位于观测深度前方的次数；
- `n_unknown_traversal(v)`：该像素无有效深度，或候选体素位于观测表面后方的次数；
- `n_ray_should_see(v)`：该体素进入视锥且落在有效深度范围内的总次数。

定义归一化质量：

`m_occ = n_surface_hit / n_ray_should_see`

`m_free = n_free_traversal / n_ray_should_see`

`m_unobs = n_unknown_traversal / n_ray_should_see`

再定义 GT-free 的 `expected_visibility`：

`expected_visibility = clip((gap_dist_to_base - 0.03) / 0.09, 0, 1)`

最终可见性缺口分数：

`visibility_deficit_score(v) = expected_visibility * (m_unobs + 0.5 * m_free) * exp(-0.5 * m_occ)`

激活规则：`visibility_deficit_score > 0.085`。

解释：
- `m_unobs` 表示“应该有信息但没有”；
- `m_free` 表示“射线穿过但未形成稳定表面”，也可能对应漏建区域；
- `m_occ` 抑制已经被稳定观测到的区域；
- `expected_visibility` 只保留靠近当前 map 缺口带的体素。

## 3. 123 `Ray Deficit Accumulation`

定义：

`ray_deficit_ratio(v) = 1 - n_surface_hit(v) / n_ray_should_see(v)`

再结合 map gap 带：

`ray_deficit_score(v) = clip((gap_dist_to_base - 0.03) / 0.09, 0, 1) * ray_deficit_ratio(v)`

激活规则：`ray_deficit_score > 0.145`。

解释：
- 如果一个体素多次进入视锥，但很少被深度表面真正命中，则其 deficit 高；
- 与 `gap_dist_to_base` 结合后，只在当前地图的真实断裂带附近放大该 deficit。

## 4. 与现有主线对接

- `122/123` 都替代了 `118/120/121` 中的 GT-free gap proxy 部分；
- 输出仍然进入与 `116` 同源的 `occupancy + entropy + score-based activation` 风格流程，只是输入信号换成更强的 visibility deficit。
