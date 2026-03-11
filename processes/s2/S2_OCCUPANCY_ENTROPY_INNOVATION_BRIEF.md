# S2 Occupancy-Entropy Innovation Brief

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`

## 1. 参考论文

| Title | Conf | Year | 借鉴的核心思想 |
|---|---|---:|---|
| GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction | ICCV | 2023 | 用全局优化统一 pose 与 dense map，说明漂移修复应先于密集补全。 |
| Loopy-SLAM: Dense Neural SLAM with Loop Closures | CVPR | 2024 | 用 loop / global consistency 先拉直轨迹，再回写 dense map。 |
| PaSCo: Uncertainty-Aware Panoptic Semantic Scene Completion | CVPR | 2024 | 显式输出 uncertainty，用不确定性而不是硬标签决定哪些体素值得激活。 |
| BUOL: Balanced Multimodal Fusion Using Occupancy-Aware Lifting for 3D Object Detection | CVPR | 2023 | 用 occupancy-aware lifting 把 2D/3D 证据统一到 occupancy 表达，启发我们用占据概率做弱观测融合。 |
| MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors | CVPR | 2025 | 结构先验应直接进入 SLAM/重建联合优化，而不是只做后处理。 |

这些工作只提供**思想借鉴**：
- `PAPG` 借鉴 GO-SLAM / Loopy-SLAM / MASt3R-SLAM 的“先校正 pose，再做 dense map”的顺序；
- `116` 的弱证据建模借鉴 PaSCo / BUOL 的 occupancy + uncertainty 思想，但实现完全重构为当前体素哈希 / plane cell 轻量版本。

## 2. 本项目的 116 设计

### 2.1 Occupancy Probability

对每个 plane cell `c` 定义：
- `h_c`：支持该 cell 的帧数；
- `m_c`：该 cell 可见但被前方深度反证的次数；

贝叶斯型后验：

`p_occ(c) = (alpha + h_c) / (alpha + beta + h_c + m_c)`，其中 `alpha=1, beta=2`。

### 2.2 Entropy

对每个 cell 计算二元熵：

`H(c) = -p_occ log p_occ - (1-p_occ) log (1-p_occ)`

再定义证据得分：

`s(c) = p_occ(c) * exp(-H(c))`

解释：
- `p_occ` 高说明支持更强；
- `exp(-H)` 抑制高不确定性 cell；
- 两者乘积是一个轻量、可解释的 occupancy-confidence score。

### 2.3 Gap-Only Activation

本轮 dev runner 用的是**oracle gap mask**：
- 先用基线 `111` 与 reference points 的差集得到 missing-reference neighborhood；
- 只允许落在该缺口邻域中的 weak cells 被激活；
- 激活规则为：`oracle_gap(c)=1 and s(c) > 0.08`。

注意：
- 这是一个**研究验证用 oracle 版本**，只用于证明 occupancy+entropy 机制在“真缺口”中是否有效；
- 它**不能直接晋升为主线配置**，因为 gap mask 仍然使用了开发集 reference；
- 但它能回答一个更关键的研究问题：如果 gap localization 足够准，occupancy+entropy 是否真的能兼顾 `Comp-R / Acc / Ghost`。

### 2.4 伪代码

```text
1. 从 111 提取 front planes
2. 用 PAPG 对关键帧位姿做 plane-anchored 校正
3. 重新投影所有 plane-consistent observations，形成 plane cells
4. 对每个 cell 统计 h_c, m_c，计算 p_occ(c), H(c), s(c)
5. 只在 oracle gap mask 内做激活：oracle_gap(c)=1 and s(c)>0.08
6. 将激活 cell 与 111 基线并集，得到 116
```

## 3. 本轮结论

- 116 证明：一旦 activation 被严格限制在真缺口中，弱观测不需要大幅放宽阈值，也能提高 `Comp-R` 且控制 ghost；
- 当前真正缺的不是“更多 plane cells”，而是一个**GT-free 的 gap proxy** 来替代 oracle gap mask；
- 因此下一轮的研究重点应从 plane union 转到“如何从当前 map / visibility / uncertainty 中近似 gap mask”。

## 4. 论文链接

- GO-SLAM: https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_GO-SLAM_Global_Optimization_for_Consistent_3D_Instant_Reconstruction_ICCV_2023_paper.html
- Loopy-SLAM: https://openaccess.thecvf.com/content/CVPR2024/html/Liso_Loopy-SLAM_Dense_Neural_SLAM_with_Loop_Closures_CVPR_2024_paper.html
- PaSCo: https://openaccess.thecvf.com/content/CVPR2024/html/Cetin_PaSCo_Uncertainty-Aware_Panoptic_Semantic_Scene_Completion_CVPR_2024_paper.html
- BUOL: https://openaccess.thecvf.com/content/CVPR2023/html/Xia_BUOL_Balanced_Multimodal_Fusion_Using_Occupancy-Aware_Lifting_for_3D_Object_CVPR_2023_paper.html
- MASt3R-SLAM: https://openaccess.thecvf.com/content/CVPR2025/papers/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.pdf
