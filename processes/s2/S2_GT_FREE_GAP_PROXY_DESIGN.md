# S2 GT-Free Gap Proxy Design

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 设计目标

- 目标不是重新发明 116 的激活公式；目标是把 `oracle gap mask` 替换为 GT-free gap proxy。
- 本轮三种 proxy 都严格禁止使用 `reference_points` 参与激活；`reference_points` 只用于离线评测 proxy recall / precision。

## 1.1 借鉴来源

- `GO-SLAM` / `Loopy-SLAM`：提供“先修正全局 pose consistency，再讨论 dense gap completion”的顺序性启发。
- `PaSCo`：提供 uncertainty-aware voxel activation 的思路，说明高不确定性区域应被显式建模，而不是被动裁掉。
- `BUOL`：提供 occupancy-aware lifting / fusion 的思路，说明不同信号应先统一到 occupancy 空间，再做后续决策。

本轮实现没有照搬这些工作的网络或推理器，只借鉴了：
- 几何一致性优先；
- occupancy / uncertainty 作为中间表示；
- 多信号联合，而不是单信号硬阈值。

## 2. 三种 GT-Free Proxy

### 117 `Frustum-Based Unobserved Proxy`
- 规则：`visible >= 4` 且 `0.01 <= gap_dist_to_base <= 0.03` 且 `score > 0.04`。
- 直觉：若候选点在多帧视锥内长期存在，但基线地图在该位置仍有局部 gap，则它更像“被漏掉的表面”而不是随机噪声。

### 118 `Plane Extrapolation & Closure Proxy`
- 把 base map 投影到每个高置信平面上，构造 2D occupancy grid。
- 若候选 cell 落在空网格，但周围 `5x5` 邻域已有足够 occupied cells，则视为 plane-hole closure 候选。
- 直觉：缺口通常表现为平面中的局部断裂，而不是任意方向的散点。

### 119 `Entropy-Guided Proxy`
- 规则：`0.008 <= gap_dist_to_base <= 0.03` 且 `entropy > 0.60` 且 `score > 0.04`。
- 直觉：高熵意味着状态尚不确定，但如果它又贴近当前 map 的 gap band，就值得被优先关注。

## 3. Oracle 对比方式

- Oracle gap 仍然定义为：candidate 到 `111` 与 reference 差集的最近距离 `< 0.05 m`。
- `proxy_recall`：Oracle gap candidate 被 proxy 覆盖的比例。
- `proxy_precision`：被 proxy 激活的 candidate 中，真实属于 Oracle gap 的比例。

## 4. 结论预期

- 若某个 GT-free proxy 不能同时做到 `Comp-R >= 75%` 与 `Ghost <= 50`，则它只能作为中间诊断，而不能替代 oracle 版本。
