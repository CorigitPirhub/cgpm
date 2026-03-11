# S2 Innovation Attack Analysis

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`
基线：`111_native_geometry_chain_direct`

## 1. 变体定义

- `113_naive_plane_union`：不做位姿校正，只把所有 plane-snapped 单元直接并入基线；这是“简单放宽阈值”的朴素对照。
- `114_papg_plane_union`：先做 Plane-Anchored Pose Graph，再并入 plane-snapped 单元；这是 Temporal Drift + Weak Evidence 的联合原型。
- `115_papg_consensus_activation`：在 `114` 基础上只保留 `>=2` 帧支持的 plane cells；这是保守高精度版本。

## 2. 关键对比

- 朴素方法 `113`：Bonn `Acc=4.413 cm`, `Comp-R=76.19%`, added `TB/Ghost/Noise=23/255/3043`。
- 创新联合原型 `114`：Bonn `Acc=4.304 cm`, `Comp-R=77.06%`, added `TB/Ghost/Noise=35/224/3091`。
- 保守版本 `115`：Bonn `Acc=4.225 cm`, `Comp-R=70.96%`, added `TB/Ghost/Noise=5/26/82`。

## 3. 结论

- 相比朴素方法，`114` 把 Bonn `Acc` 从 `4.413` 压到 `4.304`，同时把 `Comp-R` 维持在 `75%+`；这说明 Plane-Anchored Pose Graph 确实对弱证据补全有几何增益。
- 但 `114` 的 added ghost 仍高达 `224`，说明仅靠 plane support 仍不足以区分真实背景与动态边界伪支持。
- `115` 把 added ghost 压到 `26`，且 Bonn `Acc` 反而优于基线（`4.233 -> 4.225`），但 `Comp-R` 几乎不涨；这证明保守 consensus 只能修 geometry，不能补 completeness。

## 4. 为什么创新方法比朴素方法更合理

- `113` 只是把 plane-aligned 弱观测一股脑加入地图，没有解决 drift，因此单元位置本身就带偏差。
- `114` 先用 plane anchors 约束轨迹，再做单元累积，属于“先校正时空坐标，再累积弱证据”。这比降阈值更符合误差来源。
- `115` 进一步证明：只要多视图共识足够强，几何质量可以守住；问题在于当前弱证据模型还不够精细，无法在 `ghost` 与 `coverage` 之间同时过线。

## 5. 阶段判断

- 本轮最接近目标的是 `114_papg_plane_union`：Bonn `Comp-R=77.06%` 已跨过本轮 completeness 门槛，但 `Acc=4.304 cm` 仍未触达 `3.50 cm`。
- 结论不是“继续调阈值”，而是：下一轮必须把 `114` 升级为真正的 uncertainty-aware weak-evidence model，专门削减 dynamic-boundary 支持幻觉。
