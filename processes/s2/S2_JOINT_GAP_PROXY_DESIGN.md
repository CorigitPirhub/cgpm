# S2 Joint Gap Proxy Design

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 设计动机

- 单点 GT-free proxy 已证明：`Visibility` 太宽，`Plane Closure` 太窄，`Entropy` 缺乏方向性。
- 因此本轮不再把三种信号单独当作 proxy，而是构造成联合规则。

## 2. 变体 A：120 `Joint Proxy Geometric Fusion`

定义三类信号：
- `C_vis`：`visible >= 4` 且 `gap_dist_to_base` 落在局部 gap band (`0.008~0.03 m`)；
- `C_plane`：平面 2D occupancy 中存在 closure hole；
- `C_ent`：`entropy > 0.60`。

融合规则：
- 高置信：`C_vis ∧ C_plane`
- 中置信：`C_vis ∧ C_ent ∧ score > 0.04`
- 激活：`high_conf ∨ mid_conf`

## 3. 变体 B：121 `Joint Proxy Score Weighted`

特征：
- `f1 = I(C_vis)`
- `f2 = I(C_plane)`
- `f3 = I(C_ent)`
- `f4 = occupancy-confidence score`

评分函数：

`GapScore(v) = 0.25 f1 + 0.50 f2 + 0.25 f3 + 0.25 f4`

激活规则：
- `GapScore > 0.75`
- 且候选所属平面满足 `|n_z| > 0.85`，用于抑制非曼哈顿噪声。

## 4. 借鉴来源

- `GO-SLAM` / `Loopy-SLAM`：先做全局一致性，再讨论 dense gap completion。
- `PaSCo`：不确定性 / 熵应成为激活决策的一部分，而不是只做后验分析。
- `VisFusion`：多视角 visibility 与结构几何需要联合，而不是独立阈值化。

## 5. 评测方式

- 系统指标：`Acc / Comp-R / Ghost`
- Proxy 质量：`proxy_recall / proxy_precision`
- Oracle `116` 只作为上界；`118` 作为最佳单点下界。
