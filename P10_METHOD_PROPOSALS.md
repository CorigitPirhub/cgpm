# P10 专项方法设计书

## Canonical 口径说明（2026-03-07）

> 本文是 `P10` 研究与结构设计文档，内部会保留大量 profile / probe / 历史轮次数字，用于诊断模块行为与记录负结果链；这些数字**不自动等同于当前项目正式对外主结论**。

当前正式对外数字统一以以下文件为准：
- `output/summary_tables/paper_main_table_local_mapping.csv`
- `output/summary_tables/local_mapping_main_metrics_toptier.csv`
- `output/summary_tables/dual_protocol_multiseed_significance.csv`
- `output/summary_tables/dual_protocol_multiseed_reconstruction_agg.csv`
- `output/summary_tables/dual_protocol_multiseed_dynamic_agg.csv`

按当前 `5-seed` canonical 口径：
- TUM `oracle` dynamic（3 条 walking 均值）：EGF `F-score=0.7903`，`Chamfer=0.05317`，`ghost_ratio=0.2566`，`Acc=4.1655 cm`，`Comp-R(5cm)=100.0%`
- Bonn `slam` dynamic（`balloon/balloon2/crowd2` 均值）：EGF `F-score=0.6463`，`Chamfer=0.10116`，`ghost_ratio=0.08613`，`Acc=6.1481 cm`，`Comp-R(5cm)=77.21%`
- 显著性（EGF vs TSDF）：TUM `p_fscore=4.81e-10`, `p_ghost=1.72e-24`；Bonn `p_fscore=1.06e-18`, `p_ghost=5.35e-05`

因此：
- 若本文中的历史 profile / probe 数字与上述 canonical 表冲突，以 canonical 表为准；
- 若本文讨论的是分支优劣、负结果、模块走向，则应把这些数字理解为**研究过程证据**，而不是最终投稿主表数字；
- 截至 `2026-03-07`，`P10` 仍未过线，这一点以 canonical 主表与 `TASK_LOCAL_TOPTIER.md` 的最新口径为准。

## 1. 目标与结论先行

`TASK_LOCAL_TOPTIER.md` 中的 `P10` 仍未过线，其核心矛盾不是参数空间没扫够，而是当前系统仍把两件本该分离的事情绑在同一条门控链上：

1. **几何去偏（Accuracy / Chamfer）**
   - 目标是把静态表面的零交叉位置拉回正确位置，减少表面变厚、整体偏移、局部毛刺。
2. **动态抑制（Ghost / ghost reduction）**
   - 目标是抑制动态残影、遮挡污染和瞬态错误表面，不让它们写入最终静态地图。

当前代码中，很多动态判别逻辑发生在提取端或写后判别端，而静态几何偏置已经在融合阶段写入 `phi/phi_geo`。因此，继续强化后置筛除通常只能“少取一些点”，不能真正回正零交叉；反过来，一旦增强 free-space 或动态惩罚，又容易误删背景，导致 `Comp-R` 下滑。

**本设计书的核心主张是：P10 要过线，必须从“参数调优”切换到“结构解耦”。**

更具体地说，最有希望的主线不是再调 `rho_thresh`、`phi_thresh`、`dyn_forget_gain`，而是新增具有独立论文贡献潜力的模块：

1. `PT-DSF`: Persistent-Transient Dual Surface Field  
   把静态持久面与瞬态动态面显式拆成两套状态。
2. `ZCBF`: Zero-Crossing Bias Field  
   在静态几何分支上单独估计局部零交叉偏置场，专门打 `Acc`。
3. `DCCM`: Delayed-Commit Contradiction Memory  
   把自由空间/矛盾证据改成“延迟提交”的时间确认机制，专门打 `ghost`。

这三者组合起来，构成一条完整且有独立性的 P10 方案线：

- `PT-DSF` 解决“为什么 `Acc` 和 `ghost` 会在当前结构上天然耦合”；
- `ZCBF` 解决“为什么表面即使点数足够，位置仍系统性偏移”；
- `DCCM` 解决“为什么激进清理虽然能压 ghost，却会误伤背景”。

---

## 2. 当前项目状态与 P10 症结

### 2.1 当前系统已经具备的能力

根据当前代码与结果表，系统已经具备：

- 体素哈希显式场表示：`egf_dhmap3d/core/voxel_hash.py`
- 梯度场 + 证据场融合：`egf_dhmap3d/modules/updater.py`
- 关联噪声建模与异方差观测：`egf_dhmap3d/modules/associator.py`
- 多类后置提取/筛选逻辑：`SNEF`, `STCG`, `two-stage`, `dual-layer`, `MOPC`
- 若干中后期实验算子：`LZCD`, `SSE-EM`, `VCR`, `RBI`, `ST-Mem` 等

这说明当前仓库不是“功能不够”，而是**算子分工仍未彻底理顺**。

### 2.2 P10 当前失败模式

结合 `P10_RESEARCH_ASSESSMENT.md` 与 `output/summary_tables/local_mapping_precision_profile*.csv`，当前失败模式可以概括为：

1. **`Comp-R` 很强，但 `Acc` 仍偏高**
   - 说明问题不是“没建满”，而是“建得偏”。
   - 表面零交叉存在系统性偏移/厚化。

2. **增强 dynamic gating 往往会损伤完整性**
   - 压 `ghost` 的同时容易让 `Comp-R` 或 `F-score` 下滑。
   - 尤其在 Bonn 等遮挡强、动态快的场景下更明显。

3. **后置删点对 `Acc` 无法形成决定性帮助**
   - 这与当前大量逻辑生效于提取端而非融合写入端一致。

因此，P10 的关键不是“让提取器更聪明”，而是**让静态几何和动态污染在状态层面分道扬镳**。

---

## 3. 文献映射：什么值得借鉴，什么不能直接照抄

以下文献不是要直接复现，而是用于回答两个问题：

1. 顶会/顶刊在动态建图里已经做到了什么；
2. 当前 EGF-DHMap 若要继续前进，应该把创新落在什么位置。

### 3.1 对象/实例分离路线

代表工作：

- `Co-Fusion` (ICRA 2017)
- `MID-Fusion` (ICRA 2019)
- `EM-Fusion` (ICCV 2019)
- `TSDF++` (ICRA 2021)
- `Panoptic Multi-TSDFs` (ICRA 2022)

共同特点：

- 通过对象级或多 TSDF/多地图表示，把动态对象与静态背景分开处理。
- 优点是解耦天然；缺点是通常依赖对象分割、实例追踪或对象图层管理。

对本项目的启发：

- **“多状态表示”是有效的**，但本项目不应复制“实例级 map 管理”这条线。
- 更适合 EGF 的是：**在体素级、无对象依赖的前提下做持久/瞬态分解。**

### 3.2 时空分解路线

代表工作：

- `3D LiDAR Mapping in Dynamic Environments Using a 4D Implicit Neural Representation` (CVPR 2024)
- `DIO: Decomposable Implicit 4D Occupancy-Flow World Model` (CVPR 2025)
- `DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction` (CVPR 2025)
- `WildGS-SLAM` (CVPR 2025)
- `DyGS-SLAM` (2025)

共同特点：

- 把“时间”纳入表示本身，而不是靠后处理删除动态点。
- 通过显式静/动态分解、时间依赖场或多分支表示来处理动态场景。

对本项目的启发：

- **动态残影本质上是时间上不稳定的表面假设。**
- EGF 已有 `rho` 和 `d_score`，但还缺少真正的“时态分层状态”。
- 因此，最自然的扩展是：在显式 SDF 体素场中引入 `persistent / transient` 双状态，而不是重写成完整 4D 神经场。

### 3.3 不确定度与概率融合路线

代表工作：

- `PSDF Fusion` (ECCV 2018)
- `Probabilistic Volumetric Fusion for Dense Monocular SLAM` (WACV 2023)
- `BundleFusion` (SIGGRAPH / TOG 2017)
- `ElasticFusion` (RSS 2015)

共同特点：

- 强调观测不确定度、离群观测与重积分的重要性。
- 若表面偏差已经写入体积，后续仅靠提取端筛选往往不够，需要“重估/重积分/再解释”。

对本项目的启发：

- `Acc` 问题需要一个**融合层面的几何去偏算子**，而不是继续在提取端做 hard prune。
- 因此引出 `ZCBF`：显式估计局部零交叉偏置，而非仅用点级筛除来间接改善精度。

### 3.4 结构/表示质量路线

代表工作：

- `Co-SLAM` (CVPR 2023)
- `SplaTAM` (CVPR 2024)
- `GS-SLAM` (CVPR 2024)
- `Benchmarking Implicit Neural Representation and Geometric Rendering in Real-Time RGB-D SLAM` (CVPR 2024)
- `PlanarSplatting` (CVPR 2025)
- `iS-MAP` (ACCV 2024)

共同特点：

- 高几何质量通常来自更强的表示能力，或者更专用的结构几何正则。
- 只做“动态点删掉”并不能自动得到更好的静态表面精度。

对本项目的启发：

- 若 P10 要真正追平甚至超过强静态基线，必须承认：**`Acc` 需要专门建模。**
- 这进一步支持 `ZCBF` 和 `SACR` 两条线。

### 3.5 本项目应坚持的独立性

本项目不应退化成以下几类“已有路线的工程集成”：

- 不应变成“语义分割 + TSDF/NeRF”的常规动态 SLAM 拼装；
- 不应完全切换到对象级 map 管理，失去 EGF 当前无实例依赖的优势；
- 不应简单模仿 4D neural field，而丢掉本项目已有的体素哈希、显式场、局部重建效率；
- 不应仅靠后处理删点去讲创新。

因此，P10 的独立创新应明确落在：

1. **显式 SDF 体素场内部的静/动态状态解耦**；
2. **几何专用的零交叉去偏**；
3. **时间确认式的负证据管理**。

---

## 4. 方案 A: PT-DSF
### Persistent-Transient Dual Surface Field

### 4.1 核心思想

对每个体素，不再只维护一套主表面状态，而是维护两类表面：

- `persistent`：长期稳定、应进入最终静态地图的表面
- `transient`：短时出现、与动态物体或遮挡相关的瞬态表面

当前仓库中已有 `phi_static / phi_transient` 与部分 dual-state 逻辑，但仍偏“辅助态”或“后置门控态”。

`PT-DSF` 的要求更强：

- 观测在写入时就分流；
- 静态输出只读取 `persistent` 主分支；
- 动态抑制不再直接通过全局门控干预几何主分支。

### 4.2 数学定义

对体素 `x`，维护状态：

- 持久表面：`(phi_p(x), W_p(x), rho_p(x))`
- 瞬态表面：`(phi_t(x), W_t(x), rho_t(x))`
- 体素动态责任：`r_t(x) in [0, 1]`
- 矛盾记忆：`c(x)`

给定观测 `z_k(x)` 及其残差 `e_k(x)`，先估计观测属于瞬态分支的责任：

\[
r_k(x) = \sigma\left(
\alpha_d d\_score(x)
+ \alpha_c c(x)
+ \alpha_f \text{free\_ratio}(x)
- \alpha_\rho \rho_p(x)
- \alpha_s \text{surf\_support}(x)
\right)
\]

然后用软分配更新：

\[
W_p' = W_p + (1-r_k) w_k, \quad
\phi_p' = \frac{W_p \phi_p + (1-r_k) w_k z_k}{W_p'}
\]

\[
W_t' = W_t + r_k w_k, \quad
\phi_t' = \frac{W_t \phi_t + r_k w_k z_k}{W_t'}
\]

最终静态导出面仅从 `phi_p` 提取；`phi_t` 仅用于动态分析、ghost 诊断与可视化。

### 4.3 独立创新点

这不是对象级多 TSDF，也不是实例分图。区别在于：

- 分解发生在**体素级表面状态**，而不是对象级地图；
- 不需要语义分割或实例 ID；
- 可以直接复用当前 `rho`、`d_score`、`surf/free evidence`；
- 与 EGF 的证据场故事自然兼容。

### 4.4 代码位点

- `egf_dhmap3d/core/voxel_hash.py`
  - 强化 `phi_static / phi_transient` 为正式主状态
  - 新增 `rho_persist / rho_transient`
  - 新增 `commit_age / rollback_age`
- `egf_dhmap3d/modules/updater.py`
  - 把观测写入从“单通道更新 + 写后判别”改成“责任估计 + 双通道更新”
- `egf_dhmap3d/modules/pipeline.py`
  - 提取阶段默认只读取 `persistent` 分支
- `egf_dhmap3d/core/config.py`
  - 新增 `ptdsf_*` 参数组

### 4.5 预期收益

- `ghost` 可以通过限制 `transient -> persistent` 转移来抑制
- `Acc` 不再被动态抑制逻辑直接改坏
- `Comp-R` 应比全局 hard prune 更稳

### 4.6 主要风险

- 若责任估计不稳，可能导致“状态漂移”
- 若 `persistent` 过保守，可能降低早期覆盖率

---

## 5. 方案 B: ZCBF
### Zero-Crossing Bias Field

### 5.1 核心思想

P10 当前更像是“表面位置偏了”，而不仅仅是“动态点多了”。

因此需要一个只服务于静态几何精度的专用算子：

- 在高置信静态区域上估计局部零交叉偏置 `b(x)`
- 用它修正静态面：

\[
\phi_p^{corr}(x) = \phi_p(x) - b(x)
\]

这个偏置场不直接参与动态判断，因此不会把“清 ghost”和“修几何”又绑回一起。

### 5.2 偏置场估计

只使用满足以下条件的样本参与估计：

- `rho_p(x)` 高于阈值
- `d_score(x)` 低于阈值
- `contradiction(x)` 低
- 处于零交叉窄带 `|phi_p(x)| <= tau_b`

对局部块 `B`，估计鲁棒偏置：

\[
b_B = \operatorname{WeightedMedian}\left( \{\phi_p(x)\}_{x \in B}, \{w(x)\}_{x \in B} \right)
\]

再经平滑扩散得到 `b(x)`：

\[
b(x) = \sum_{B \in \mathcal{N}(x)} \omega_{xB} b_B
\]

其中权重由：

- 证据强度
- 法向一致性
- 局部结构一致性

共同决定。

### 5.3 为什么不是 LZCD 的简单加强版

当前仓库已有 `LZCD`，但它更接近“局部 EMA 标量 bias 修正”。

`ZCBF` 比 `LZCD` 更强的地方在于：

- 以**块级偏置场**而不是点级 EMA 为主；
- 只在 `persistent` 几何分支上起作用；
- 允许引入结构一致性与邻域光滑；
- 与动态抑制逻辑彻底解耦。

换句话说，`LZCD` 更像轻量修补；`ZCBF` 是正式的几何去偏层。

### 5.4 代码位点

- `egf_dhmap3d/core/voxel_hash.py`
  - 新增 `bias_field`, `bias_confidence`, `bias_block_id`
- `egf_dhmap3d/modules/updater.py`
  - 新增局部块偏置收集与偏置场迭代更新
- `egf_dhmap3d/modules/pipeline.py`
  - surface extraction 前默认使用 `phi_p_corr`
- `egf_dhmap3d/core/config.py`
  - 新增 `zcbf_*` 参数组

### 5.5 预期收益

主攻指标：

- `Acc`
- `Chamfer`

预期副作用较小：

- `Comp-R` 不应显著下降
- `ghost` 变化应该很小或间接受益

### 5.6 主要风险

- 若偏置场估计样本被动态污染，可能误修正
- 若块划分过粗，可能损害细节

---

## 6. 方案 C: DCCM
### Delayed-Commit Contradiction Memory

### 6.1 核心思想

当前许多 dynamic cleaning 失败的原因是：

- 自由空间证据来得太快；
- destructive update 提交得太早；
- 遮挡与真正 ghost 在当前逻辑里很难区分。

`DCCM` 的核心改造是：

- 负证据先进入一个“延迟提交”的矛盾记忆层；
- 只有当该矛盾在时间上持续存在、并且后方有更可信静态表面支撑时，才允许对 `persistent` 几何产生 destructive 影响。

### 6.2 状态定义

对每个体素维护：

- `s(x)`: surface support
- `f(x)`: free-space support
- `o(x)`: rear-occluder / rear-surface support
- `c_age(x)`: contradiction age
- `q_commit(x)`: contradiction commit score

更新规则示意：

\[
q\_{commit}(x) = \sigma\left(
\beta_f f(x)
+ \beta_a c\_{age}(x)
+ \beta_o o(x)
- \beta_s s(x)
- \beta_\rho \rho_p(x)
\right)
\]

只有当 `q_commit(x)` 高于阈值并持续 `T_commit` 帧后，才允许：

- 降低 `W_p`
- 把部分状态迁移到 `transient`
- 或触发局部重积分 / rollback

### 6.3 与现有 STCG 的区别

`STCG` 更像“矛盾评分 + 提取时抑制”。

`DCCM` 更强调：

- 时间确认
- 写入前后的责任切换
- destructive action 的延迟提交

因此，它不是简单加一个 score，而是改变 negative evidence 的提交语义。

### 6.4 代码位点

- `egf_dhmap3d/core/voxel_hash.py`
  - 新增 `dccm_free`, `dccm_surface`, `dccm_rear`, `dccm_age`, `dccm_commit`
- `egf_dhmap3d/modules/updater.py`
  - ray/free update 先写入 DCCM 缓冲层
  - 仅在满足持续条件时触发 destructive commit
- `egf_dhmap3d/modules/pipeline.py`
  - 输出 DCCM 统计用于可视化

### 6.5 预期收益

主攻指标：

- `ghost reduction`
- `ghost_tail_ratio`

同时尽量保持：

- `Comp-R`
- `Background Recovery`

### 6.6 主要风险

- 提交延迟可能使 ghost 清除速度偏慢
- 若场景很短，瞬态残影可能来不及被完全确认

---

## 7. 方案 D: OMHS
### Occlusion-aware Multi-Hypothesis Surface

### 7.1 核心思想

对于局部块，不只存一条零交叉，而是允许最多两条：

- 前景瞬态面
- 背景持久面

这比 `PT-DSF` 更进一步，专门针对“动态遮挡导致的前后表面共存”问题。

### 7.2 适用场景

- `walking_static`
- Bonn `balloon` 一类强遮挡序列

### 7.3 优势与局限

优势：

- 对遮挡场景更直接
- 有潜力进一步提升背景恢复

局限：

- 实现复杂度明显高于 `PT-DSF`
- 提取阶段更复杂
- 更像 P10 后期增强项，而非首轮主线

---

## 8. 方案 E: NTRF
### Normal-space Transient Residual Field

### 8.1 核心思想

将动态污染建模为沿局部法向的瞬态残差，而不是直接污染完整 `phi`：

\[
\phi(x) = \phi_p(x) + \delta_n(x)
\]

其中 `delta_n` 只允许短期变化，并受法向一致性约束。

### 8.2 作用

- 抑制“表面变厚”
- 把动态污染限制在几何最相关的方向上

### 8.3 评价

- 方法上有一定新意
- 但落地复杂度高于收益确定性
- 优先级低于 `PT-DSF + ZCBF + DCCM`

---

## 9. 方案 F: SACR
### Structure-Aware Consensus Regularization

### 9.1 核心思想

在高置信静态区域引入结构一致性正则，只约束 `persistent` 分支：

- 局部平面一致性
- 邻域法向一致性
- 低曲率区域的平滑一致性

### 9.2 作用

- 面向 `Acc` 的后续增强
- 尤其适合室内静态场景的墙面、地面、桌面

### 9.3 评价

- 是很好的 `P10` 后续增强项
- 但独立作为主线的优先级不如 `ZCBF`
- 容易被质疑为“经典几何正则”，创新性略弱

---

## 10. 推荐执行顺序

### 阶段 1：主线方案

优先执行以下三项：

1. `PT-DSF`
2. `ZCBF`
3. `DCCM`

原因：

- `PT-DSF` 先把状态解耦问题解决
- `ZCBF` 直接面向 `Acc`
- `DCCM` 直接面向 `ghost`

### 阶段 2：增强方案

若阶段 1 达到大部分目标但仍差临门一脚，再考虑：

4. `OMHS`
5. `SACR`

### 阶段 3：备选研究分支

6. `NTRF`

这条线更适合作为高风险探索，而不是当前主线。

---

## 11. 验证计划

### 11.1 第一阶段验收

以 `P10` 当前硬门槛为主：

- TUM oracle:
  - `Acc <= 1.80 cm`
  - `Comp-R >= 95%`
- Bonn slam:
  - `Acc <= 2.60 cm`
  - `Comp-R >= 95%`
- 动态抑制：
  - `ghost_reduction_vs_tsdf >= 35%`

### 11.2 分模块验收策略

#### A. `PT-DSF`
先看：

- `ghost` 是否下降
- `Comp-R` 是否保持
- `Acc` 至少不劣化

若 `ghost` 有明显收益而 `Acc` 不恶化，则保留。

#### B. `ZCBF`
先看：

- `Acc`
- `Chamfer`

若 `Acc` 明显下降而 `Comp-R` 基本不变，则说明几何去偏链条有效。

#### C. `DCCM`
先看：

- `ghost_tail_ratio`
- `ghost_reduction_vs_tsdf`
- `Background Recovery`

若 `ghost` 下降且背景恢复不明显回退，则说明延迟提交策略有效。

### 11.3 论文级可视化

每个主方案都应补以下图：

- `persistent` vs `transient` 地图分层可视化
- `bias_field` 热图
- `dccm_commit_score` 热图
- `Acc / Comp-R / ghost` 三指标折线图

---

## 12. 预期论文叙事

若上述主线跑通，P10 可被重新叙述为一条完整的方法学故事：

1. **问题定义**：
   传统动态局部建图方法常把几何去偏与动态抑制绑定在同一门控链上，导致精度、完整性与 ghost 抑制难以兼得。

2. **核心思想**：
   在显式 SDF 体素场内进行职责解耦：
   - 用 `PT-DSF` 解耦静态持久面与瞬态动态面；
   - 用 `ZCBF` 专门回正静态零交叉；
   - 用 `DCCM` 管理延迟提交式负证据。

3. **贡献形式**：
   - 一种无对象依赖的双状态局部动态表面场；
   - 一种专用于显式 SDF 的零交叉偏置场修正方法；
   - 一种时间确认式矛盾证据提交机制。

4. **实验结论**：
   在保持高 `Comp-R` 和动态 ghost 优势的同时，显著压低 `Acc`，从而真正完成 P10 所要求的“精度-完整性-动态抑制”三角平衡。

---

## 13. 最终建议

当前最值得投入的路线不是继续加提取器规则，而是：

1. 先做 `PT-DSF`
2. 再做 `ZCBF`
3. 再接 `DCCM`

理由如下：

- `PT-DSF` 解决结构性耦合
- `ZCBF` 解决几何主偏差
- `DCCM` 解决动态残影清理的误伤问题

从论文价值看，这三者也最容易形成清晰且独立的贡献点；从工程连续性看，它们都能直接挂接到当前 `voxel_hash + updater + extraction` 主线，而不需要把项目整体重写成对象级 SLAM 或 4D neural field。

---

## 14. 本轮实现状态（2026-03-06）

本设计书对应的第一轮代码落地已经完成，且采用“默认关闭”的方式接入主工程，避免污染现有终版结果路径。

### 14.1 已落地模块

1. `PT-DSF`
   - `egf_dhmap3d/core/types.py`
   - `egf_dhmap3d/core/config.py`
   - `egf_dhmap3d/modules/updater.py`
   - `egf_dhmap3d/core/voxel_hash.py`

   已实现：
   - `rho_static / rho_transient` 双证据状态
   - `ptdsf_commit_age / ptdsf_rollback_age` 提交-回滚年龄
   - persistent-only surface readout 的提取端选项

2. `ZCBF`
   - `egf_dhmap3d/modules/updater.py`
   - `egf_dhmap3d/core/voxel_hash.py`

   已实现：
   - block-level 局部零交叉偏置统计
   - `zcbf_bias / zcbf_bias_conf` 状态写回
   - 提取阶段的偏置注入与 `phi_geo / phi_static` 小步修正

3. `DCCM`
   - `egf_dhmap3d/modules/updater.py`
   - `egf_dhmap3d/core/voxel_hash.py`

   已实现：
   - `dccm_free / dccm_surface / dccm_rear / dccm_commit` 时空矛盾记忆
   - ray/free-space 负证据的 delayed-commit 更新
   - 提取阶段基于 `dccm_commit` 的软抑制

### 14.2 首轮 smoke 结论

运行命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_a,p10_ptdsf_zcbf_dccm_b   --frames 20 --stride 2   --out_root output/post_cleanup/p10_structural_smoke   --summary_csv output/summary_tables/local_mapping_precision_profile_structural_smoke.csv   --figure assets/p10_structural_smoke.png   --force
```

本轮实际执行结果：
- `p10_ptdsf_zcbf_dccm_a` 已完整跑完，并整理为 `output/summary_tables/p10_structural_ptdsf_zcbf_dccm_smoke_a.csv`
- `p10_ptdsf_zcbf_dccm_b` 在 `TUM` 首段启动后手动终止，因为 `A` 已足以说明首版实现仍未形成冲线趋势

`profile A` 聚合结果：
- `TUM`: `mean_acc_cm=2.5981`, `mean_comp_r_5cm=99.9967`, `min_ghost_reduction_vs_tsdf=0.1826`
- `Bonn`: `mean_acc_cm=4.8190`, `mean_comp_r_5cm=84.3287`, `min_ghost_reduction_vs_tsdf=0.4623`

### 14.3 结果解释

这组结果说明：

1. **设计方向不是空想，代码链路已经成立**
   - 新状态字段、更新项和提取逻辑都已经可以稳定跑通 `TUM + Bonn` 的统一评估链路。

2. **但首版 `PT-DSF/ZCBF/DCCM` 还没有真正解开 `Acc` 与 `ghost` 的耦合**
   - `ZCBF` 去偏幅度仍偏小，未能把 `TUM/Bonn` 的零交叉位置显著拉回；
   - `DCCM` 的抑制更像“附加 readout 权重”，还不是足够强的 delayed-commit static readout，因此 `TUM` 侧 ghost 降幅只有 `18.3%`；
   - `PT-DSF` 当前更像“软分流缓存”，还没有形成真正意义上的“persistent dominates readout”。

3. **下一轮不该继续大范围扫参数，而应加强结构本身**
   - 强化 persistent-only 读出，使静态主分支真正支配最终表面；
   - 把 `ZCBF` 从局部 bias smoothing 提升为带可信度归一化的偏置场传播；
   - 让 `DCCM` 只作用于 transient 分支或竞争分支，而不是再回到几何主分支的门控链上。

### 14.4 第二轮结构回写（2026-03-06）

本轮已经按上面的三条主线直接修改代码，不再继续扩大参数搜索：

1. `PT-DSF` persistent readout 强化
   - 位置：`egf_dhmap3d/core/voxel_hash.py`
   - 变化：
     - 重新定义 persistent readout 的 `anchor/support` 权重，使 `phi_static + phi_geo` 成为最终表面的主来源；
     - transient 仅在 persistent 支路偏弱或 persistent/geometry 明显失配时允许极弱泄漏；
     - `legacy phi` 同步阶段直接使用带 `zcbf_bias_conf` 的 debiased persistent readout。

2. `ZCBF` 从局部 smoothing 升级为“观测置信归一化 + block propagation”
   - 位置：`egf_dhmap3d/modules/updater.py`
   - 变化：
     - block bias 观测权重不再只依赖 `phi_w`，而是加入 persistent 置信和 `rho` 归一化；
     - block-level 传播保留高置信 anchor block，同时让邻域 static-consistent block 吸收平滑偏置场。

3. `DCCM` 与几何主门控链隔离
   - 位置：`egf_dhmap3d/core/voxel_hash.py`
   - 变化：
     - 删除 `dccm_commit -> dyn_mix/drop_scale` 的主门控回流；
     - `DCCM` 现在只用于 transient veto 和竞争支路，不再直接改变 persistent geometry 的保留门限。

### 14.5 第二轮短 probe（已完成）

验证命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v3   --summary_csv output/summary_tables/p10_structural_probe_v3.csv   --figure assets/p10_structural_probe_v3.png   --force
```

产物：
- `output/post_cleanup/p10_structural_probe_v3/`
- `output/summary_tables/p10_structural_probe_v3.csv`
- `output/summary_tables/p10_structural_probe_v3_best.json`
- `output/summary_tables/p10_structural_probe_v3_by_sequence.csv`
- `assets/p10_structural_probe_v3.png`

结果汇总（短 probe，非终版验收）：
- `TUM(3 walking mean)`: `acc=3.724cm`, `fscore=0.8508`, `ghost_ratio=0.1237`。
- `TUM` 上相对 `TSDF` 的最差 `ghost_ratio` 降幅约为 `40.4%`，显著高于上一轮 `PT-DSF/ZCBF/DCCM` 首版 smoke 的 `18.3%`。
- `Bonn(all3 mean)`: `acc=4.955cm`, `fscore=0.6201`, `ghost_ratio=0.0386`, `ghost_tail_ratio=0.1185`。
- `Bonn` 上若看 `ghost_ratio` 会出现反常，因为短帧设置下 `TSDF` 的表面点过少、比例分母失真；但看 `ghost_tail_ratio`，EGF 相对 `TSDF` 的均值下降约 `52.6%`，与设计预期一致。

这轮结果支持两个判断：
1. 结构改造本身有效，尤其是 `TUM` 上的 ghost 抑制趋势已经明显抬升；
2. `Bonn` 上必须采用双口径 ghost 结论，不能再让单一 `ghost_ratio` 决定结构方案去留。

因此，第二轮结构线应继续保留，并作为后续 `P10` 冲线的正式起点；下一步的重点不再是调阈值，而是围绕当前解耦结构补一层更强的动态状态估计/竞争融合，同时在汇总端固定输出 `ghost_ratio + ghost_tail_ratio` 双字段。

### 14.6 `OMHS` 首轮实现与结论（2026-03-06）

按照方案 D，本轮已把 `OMHS` 以“遮挡感知的前/后层后验保护”形式正式打入代码：

- `egf_dhmap3d/core/types.py`
  - 新增 `omhs_front_conf / omhs_rear_conf / omhs_gap / omhs_active`；
- `egf_dhmap3d/core/config.py`
  - 新增 `surface.omhs_enable`；
- `egf_dhmap3d/modules/updater.py`
  - 新增 `_update_omhs_state()`，基于 `phi_static / phi_transient / phi_geo / dccm / dyn_prob / rho_static` 维护遮挡前层、后层及 gap；
- `egf_dhmap3d/core/voxel_hash.py`
  - 在 `persistent_surface_readout()` 中，以 `rear_support` 抑制 transient leak；
  - 在 `extract_surface_points()` 中新增 `omhs_enable`、rear keep 与后验 `omhs_rear_anchor`；
- `egf_dhmap3d/modules/pipeline.py`、`scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`
  - 全链路新增 `--surface_omhs_enable`。

验证命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v4_omhs   --summary_csv output/summary_tables/p10_structural_probe_v4_omhs.csv   --figure assets/p10_structural_probe_v4_omhs.png   --force
```

产物：
- `output/post_cleanup/p10_structural_probe_v4_omhs/`
- `output/summary_tables/p10_structural_probe_v4_omhs.csv`
- `output/summary_tables/p10_structural_probe_v4_omhs_best.json`
- `output/summary_tables/p10_structural_probe_v4_omhs_by_sequence.csv`
- `assets/p10_structural_probe_v4_omhs.png`

与 `v3` 的直接对照：
- `TUM` 聚合几乎不变：`acc 3.7237 -> 3.7259 cm`，`ghost_ratio 0.1237 -> 0.1241`；
- `Bonn` 聚合也仅为数值噪声级变化：`acc 4.9552 -> 4.9599 cm`，`Comp-R@5cm 64.76 -> 64.89`，`fscore 0.6201 -> 0.6206`；
- `ghost_tail_ratio` 的相对优势仍在，但新增 `OMHS` 没有带来明确抬升，说明收益几乎全部来自既有 `PT-DSF + ZCBF + DCCM` 主线。

这说明了一个关键事实：

> 当前 Bonn 瓶颈不是“后验提取时把背景删掉了”，而是“写入阶段 rear persistent surface 根本没有被稳定建立”。

换句话说，`OMHS` 的问题不是实现错误，而是其所在阶段不够上游：

1. 它在 **readout / extraction** 端做 rear anchor 保留；
2. 但 `P10` 需要修的，是 **update / assignment** 端的 front-rear 状态竞争；
3. 如果 rear persistent evidence 在写入时就没有被积累出来，那么提取端再聪明也只能保留一个“弱后验”。

因此，`OMHS` 当前版本的结论应明确写成：

- 是一个合理的辅助结构；
- 可以保留，用于避免极端遮挡下 rear surface 被二次裁掉；
- 但**不能作为 P10 过线主线**。

下一步应将方案 D 升级为：

- 从 `readout-side OMHS` 转向 `write-time occlusion decomposition`；
- 在单帧更新中显式分配 `front hit / rear hit / free-space contradiction`；
- 让 `rear persistent` 以独立状态写入，而不是寄希望于最终提取阶段去“救”回来。


### 14.7 `WOD` 首轮实现与结论（2026-03-06）

按照 `OMHS -> WOD` 的推进，本轮已把“遮挡职责分配”从 readout 端前移到了 update 端：

- `egf_dhmap3d/core/types.py`
  - 新增 `wod_front_conf / wod_rear_conf / wod_shell_conf`；
- `egf_dhmap3d/core/config.py`
  - 新增 `wod_*` 更新配置；
- `egf_dhmap3d/modules/updater.py`
  - 新增 `_write_time_occlusion_split()`；
  - 在 `_integrate_measurement()` 中把单次观测分成 `front / rear / shell` 三种职责，并分别作用于 `phi_static / phi_transient / phi_geo / phi_dyn / surf/free evidence`；
- `egf_dhmap3d/core/voxel_hash.py`
  - `wod_*` 状态接入遗忘与 persistent readout 辅助量；
- `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`
  - 全链路新增 `--wod_enable`。

验证命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v5_wod   --summary_csv output/summary_tables/p10_structural_probe_v5_wod.csv   --figure assets/p10_structural_probe_v5_wod.png   --force
```

产物：
- `output/post_cleanup/p10_structural_probe_v5_wod/`
- `output/summary_tables/p10_structural_probe_v5_wod.csv`
- `output/summary_tables/p10_structural_probe_v5_wod_by_sequence.csv`
- `assets/p10_structural_probe_v5_wod.png`

相对 `v4_omhs` 的聚合结果：
- `TUM`: `acc 3.7259 -> 3.7126 cm`, `fscore 0.8502 -> 0.8515`, `ghost_ratio 0.1241 -> 0.1244`；
- `Bonn`: `acc 4.9599 -> 4.9964 cm`, `Comp-R@5cm 64.89 -> 65.42`, `fscore 0.6206 -> 0.6201`, `ghost_ratio 0.0388 -> 0.0387`。

这说明：

1. `WOD` 的阶段位置是对的，`OMHS` 没有解决的“写入期 rear evidence 弱”问题，确实只能从 update 端修；
2. 但当前 `WOD` 仍然只是 **soft split + soft weighting**，它并没有把 rear surface 变成独立可读的状态，因此聚合增益仍然偏弱；
3. `P10` 的瓶颈从“是不是该前移到 update 端”收敛成了“是不是需要真正独立的 rear persistent state”。

因此，`WOD` 当前版本的结论应写为：
- 是主线必须保留的结构步骤；
- 但不能单独承担 `P10` 冲线任务；
- 下一步必须把 `rear` 从 soft weight 升级为真正的状态写入。

### 14.8 `RPS` 首轮实现与结论（2026-03-06）

基于上面的判断，本轮进一步实现了 `RPS`（Rear-Persistent Surface Buffer），把 `rear` 责任显式沉淀为第二表面状态：

- `egf_dhmap3d/core/types.py`
  - 新增 `phi_rear / phi_rear_w / rho_rear`；
- `egf_dhmap3d/core/config.py`
  - 新增 `rps_*` 配置；
- `egf_dhmap3d/modules/updater.py`
  - 在 geometry write 之后，依据 `WOD rear` 责任把观测写入 `phi_rear / rho_rear`；
- `egf_dhmap3d/core/voxel_hash.py`
  - 把 `phi_rear / rho_rear` 接入 `PT-DSF` 状态统计、persistent readout、遗忘与 stale 清理；
- `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`
  - 全链路新增 `--rps_enable` 与 profile `p10_ptdsf_zcbf_dccm_rps_a`。

验证命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_rps_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v6_rps   --summary_csv output/summary_tables/p10_structural_probe_v6_rps.csv   --figure assets/p10_structural_probe_v6_rps.png   --force
```

产物：
- `output/post_cleanup/p10_structural_probe_v6_rps/`
- `output/summary_tables/p10_structural_probe_v6_rps.csv`
- `output/summary_tables/p10_structural_probe_v6_rps_by_sequence.csv`
- `output/summary_tables/p10_structural_probe_v6_rps_vs_v5.csv`
- `assets/p10_structural_probe_v6_rps.png`

相对 `v5_wod` 的聚合结果：
- `TUM`: `acc 3.7126 -> 3.7429 cm`, `fscore 0.8515 -> 0.8478`, `ghost_ratio 0.1244 -> 0.1216`；
- `Bonn`: `acc 4.9964 -> 4.9992 cm`, `Comp-R@5cm 65.42 -> 64.89`, `fscore 0.6201 -> 0.6170`, `ghost_ratio 0.0387 -> 0.0386`。

按序列看，`RPS` 具有明显的条件敏感性：
- `rgbd_bonn_balloon2` 出现正增益：`acc 4.323 -> 4.300 cm`, `fscore 0.6967 -> 0.7015`, `ghost_tail_ratio 0.0607 -> 0.0587`；
- 但 `TUM(3 walking)` 与 `rgbd_bonn_balloon` 均出现轻度回退；
- 因而其收益并非“跨数据集稳定提升”，而是“对特定遮挡模式有效”。

这轮结果支持如下结论：

1. `RPS` 证明了“rear persistent state 写入”这个方向是可运行且可读出的；
2. 但 `always-on rear buffer` 仍然没有真正解开 `Acc` 与 ghost 的耦合，只是把耦合转移到了一个新状态上；
3. `RPS` 当前更适合作为**研究原型**，不适合作为当前主线终版配置。

因此，若继续冲 `P10`，下一步不应继续给 `RPS` 堆权重，而应升级成：
- `hard rear commit / surface-bank`；
- 仅在遮挡矛盾满足显式触发条件时提交 rear surface；
- 避免 `RPS` 这种 always-on second-surface 在 TUM 上带来的几何回退。

### 14.9 `HRC`（Hard Rear-Commit）首轮实现与结论（2026-03-06）

在 `RPS` 证明“rear persistent state 可以被状态化写入”之后，本轮进一步把它从 always-on buffer 升级为 `candidate -> commit -> active bank`：

- `egf_dhmap3d/core/types.py`
  - 新增 `phi_rear_cand / phi_rear_cand_w / rho_rear_cand / rps_commit_score / rps_commit_age / rps_active`；
- `egf_dhmap3d/core/config.py`
  - 新增 `rps_hard_commit_*` 配置；
- `egf_dhmap3d/modules/updater.py`
  - rear 责任不再直接写入 `phi_rear`，而是先进入 candidate，再依据遮挡矛盾与时序稳定性决定是否 commit；
- `egf_dhmap3d/core/voxel_hash.py`
  - 仅当 `rps_active` 与 commit score 满足条件时才让 rear bank 进入 `PT-DSF` 统计与 persistent readout；
- `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`
  - 全链路新增 `--rps_hard_commit_enable` 与 profile `p10_ptdsf_zcbf_dccm_hrc_a`。

验证命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_hrc_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v7_hrc   --summary_csv output/summary_tables/p10_structural_probe_v7_hrc.csv   --figure assets/p10_structural_probe_v7_hrc.png   --force
```

关键结果：
- 聚合上相对 `v5_wod`：`TUM acc 3.7126 -> 3.7232 cm`, `fscore 0.8515 -> 0.8501`, `ghost_ratio 0.1244 -> 0.1236`；
- `Bonn`: `acc 4.9964 -> 4.9775 cm`, `Comp-R@5cm 65.42 -> 64.64`, `fscore 0.6201 -> 0.6179`, `ghost_ratio 0.03866 -> 0.03830`；
- 逐序列表明：`HRC` 比 `RPS` 更稳，但仍未超过 `v5_wod`。

这轮结果说明：
1. `rear` 的“条件提交”比 `always-on` 更合理；
2. 但仅靠 commit 仍不能让主表面真正从 rear 几何中获益；
3. 因而瓶颈并非只是“rear bank 开得太多”，而是“rear bank 即使提交，仍很少成为最终主表面”。

### 14.10 `Surface-Bank`（Committed Surface-Bank Readout）首轮实现与结论（2026-03-06）

基于上面的判断，本轮又把 persistent readout 从 `front/rear soft mix` 改成 `front-bank vs rear-bank` 离散竞争：

- `egf_dhmap3d/core/config.py`
  - 新增 `rps_surface_bank_enable / rps_bank_*`；
- `egf_dhmap3d/core/voxel_hash.py`
  - 在 `_persistent_surface_readout()` 中先形成 front-bank，再让 committed rear-bank 进行显式竞争；
  - rear 只有在 score 明显胜出时才导出，否则完全不进入最终表面；
- `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`
  - 新增 `--rps_surface_bank_enable` 与 profile `p10_ptdsf_zcbf_dccm_sbank_a`。

验证命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_sbank_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v8_sbank   --summary_csv output/summary_tables/p10_structural_probe_v8_sbank.csv   --figure assets/p10_structural_probe_v8_sbank.png   --force
```

关键结果：
- 相对 `v7_hrc`：`TUM acc 3.7232 -> 3.7255 cm`, `fscore 0.8501 -> 0.8497`, `ghost_ratio 0.12358 -> 0.12364`；
- `Bonn`: `acc 4.9775 -> 4.9779 cm`, `Comp-R@5cm 64.64 -> 64.61`, `fscore 0.6179 -> 0.6176`, `ghost_ratio 0.03830 -> 0.03784`；
- `walking_xyz / walking_static` 与 `v7_hrc` 完全一致，`walking_halfsphere` 轻微回退，Bonn 仅有 `1e-4 ~ 1e-3` 级别波动。

这轮结果给出了一个非常关键的负结论：
1. 当 readout 已经被做成离散 bank 竞争后，结果仍几乎不变；
2. 因而 `P10` 的主瓶颈**不在 readout 污染**；
3. 下一步必须把结构继续前推到 update 端，在写入阶段直接形成 `front-transient / rear-static` 双几何状态，而不是继续在导出阶段选谁。


### 14.11 `WDSG`（Write-time Dual Surface Generation）首轮实现与结论（2026-03-06）

在 `Surface-Bank` 几乎无效之后，本轮把结构继续前推到 update 端：不再把同一个 `d_signed` 同时写入所有通道，而是在写入阶段直接合成前层瞬态与后层静态两个几何假设。

- `egf_dhmap3d/core/config.py`
  - 新增 `wdsg_*` 配置；
- `egf_dhmap3d/modules/updater.py`
  - 新增 `_measurement_view_components()` 与 `_write_time_dual_surface_targets()`；
  - `phi_static / phi_transient / phi_geo / phi_dyn / phi_rear` 不再共享同一 signed distance，而是按 front/rear 责任写入。

验证命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_wdsg_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v9_wdsg   --summary_csv output/summary_tables/p10_structural_probe_v9_wdsg.csv   --figure assets/p10_structural_probe_v9_wdsg.png   --force
```

关键结果：
- `TUM`: `Acc 3.7255 -> 0.9300 cm`, `Comp-R 99.63 -> 84.05`, `F-score 0.8497 -> 0.9120`, `ghost_ratio 0.1236 -> 0.3571`；
- `Bonn`: `Acc 4.9779 -> 5.5952 cm`, `Comp-R 64.61 -> 61.06`, `ghost_ratio 0.03784 -> 0.04088`。

结论：
1. write-time dual-surface generation 首次实质性解决了 `Acc`；
2. 但它把前层瞬态也一并写进了 persistent geometry，导致 `ghost` 大幅反弹；
3. 因而真正需要的不只是“双距离”，还要有“写入责任重路由”。

### 14.12 `WDSG-R`（Separation-Aware Routing）首轮实现与结论（2026-03-06）

针对 `WDSG` 的副作用，本轮进一步引入 separation-aware routing：利用 front/rear 分离度，把前层主导的观测从 `static/geo` 主线挪到 `transient/dyn`。

- `egf_dhmap3d/core/config.py`
  - 新增 `wdsg_route_*` 配置；
- `egf_dhmap3d/modules/updater.py`
  - 对 `w_static / w_geo / w_transient / dyn_obs` 做 separation-aware reroute。

验证命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_wdsgr_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v10_wdsgr   --summary_csv output/summary_tables/p10_structural_probe_v10_wdsgr.csv   --figure assets/p10_structural_probe_v10_wdsgr.png   --force
```

关键结果：
- `TUM`: `Acc=1.3887cm`, `Comp-R=91.79%`, `F-score=0.9369`, `ghost_ratio=0.2865`；
- 相对 `v9_wdsg`，`ghost_ratio` 明显下降，且 `F-score` 继续上升。

结论：
1. `WDSG-R` 证明 front/rear separation-aware routing 是必要的；
2. 但 `Comp-R` 仍偏低，说明为了压瞬态污染，static/geo 候选写入被一起压缩了；
3. 下一步需要把“候选几何”和“最终可导出静态表面”彻底拆开，而不是继续拿一条写入链同时承担两件事。

### 14.13 `SPG`（Static Promotion Gate）与 `No-Direct-Geo Export`（2026-03-06）

围绕上面的结论，本轮新增 `SPG`：

- `phi_geo` 继续承担高精度候选几何；
- 新增 `phi_spg / phi_spg_w / rho_spg / spg_score / spg_age / spg_active`，把时间一致的前层静态几何晋升到独立 front bank；
- 在 `SPG` 打开时，`phi_geo` 不再直接参与 persistent readout，只能经 `SPG` 晋升后导出；这一步在结构上把“几何去偏候选态”和“静态地图导出态”正式分离。

代码位点：
- `egf_dhmap3d/core/types.py`
  - 新增 `SPG` 状态字段；
- `egf_dhmap3d/core/config.py`
  - 新增 `spg_*` 配置；
- `egf_dhmap3d/modules/updater.py`
  - 新增 `_update_spg_state()`；
  - 在 `WDSG-R` 基础上放宽 candidate 路由，但把静态导出交给 `SPG`；
- `egf_dhmap3d/core/voxel_hash.py`
  - `persistent readout` 新增 `SPG` 优先导出逻辑；
  - `phi_geo` 在 `SPG` 模式下改为 non-exported candidate，仅在没有 static/promoted support 时做 fallback。

验证命令：

```bash
python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_wdsgr_spg_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v11_wdsgr_spg   --summary_csv output/summary_tables/p10_structural_probe_v11_wdsgr_spg.csv   --figure assets/p10_structural_probe_v11_wdsgr_spg.png   --force

python scripts/run_p10_precision_profile.py   --profiles p10_ptdsf_zcbf_dccm_wdsgr_spg_a   --frames 4 --stride 8 --max_points_per_frame 1500   --out_root output/post_cleanup/p10_structural_probe_v13_wdsgr_spg_nogeo   --summary_csv output/summary_tables/p10_structural_probe_v13_wdsgr_spg_nogeo.csv   --figure assets/p10_structural_probe_v13_wdsgr_spg_nogeo.png   --force
```

关键结果：
- `v11_wdsgr_spg`：`TUM Acc=1.1910cm`, `Comp-R=93.11%`, `F-score=0.9562`, `ghost_ratio=0.2954`；
- `v12_wdsgr_spgfb`：与 `v11` 近似，说明 readout 级“保守静态回退”不是主驱动；
- `v13_wdsgr_spg_nogeo`：`TUM Acc=1.2331cm`, `Comp-R=94.79%`, `F-score=0.9650`, `ghost_ratio=0.2853`。

这组结果给出三个清晰判断：
1. `SPG` 证明“候选几何 / 导出几何”分离是有效的，`Acc + Comp-R + F-score` 三项同时优于 `v10_wdsgr`；
2. `No-Direct-Geo Export` 是本轮最佳配置，说明真正污染静态地图的不是 `phi_static` 本身，而是**候选几何绕开晋升机制的直接导出**；
3. 但 `ghost_ratio` 只从 `0.295` 级别回落到 `0.285`，并未形成决定性突破，说明当前剩余耦合已经不在 persistent readout，而在**即时动态观测没有被单独状态化**。换句话说：历史残影基本被压住了，但“当前帧的动态前层表面是否应该进入静态局部图”仍缺少一个 observation-time 的瞬态 veto。

因此，下一步主线应从 `SPG` 继续前推到 observation/update 端，形成真正独立的“即时动态状态估计”，而不是继续在 readout 端堆软竞争规则。


## 15. 参考文献与入口链接

以下链接用于方法设计参考与文献映射，均为论文主页、官方仓库、arXiv 或 CVF OpenAccess 等一手来源：

- Co-Fusion project / paper: https://visual.cs.ucl.ac.uk/pubs/cofusion/index.html
- Co-Fusion PDF: https://www.martinruenz.de/media/pubs/ruenz17icra.pdf
- DynaSLAM project: https://bertabescos.github.io/DynaSLAM/
- DynaSLAM arXiv: https://arxiv.org/abs/1806.05620
- MID-Fusion arXiv: https://arxiv.org/abs/1812.07976
- EM-Fusion ICCV 2019 OpenAccess: https://openaccess.thecvf.com/content_ICCV_2019/html/Strecke_EM-Fusion_Dynamic_Object-Level_SLAM_With_Probabilistic_Data_Association_ICCV_2019_paper.html
- TSDF++ arXiv: https://arxiv.org/abs/2105.07468
- Panoptic Multi-TSDFs arXiv: https://arxiv.org/abs/2109.10165
- PSDF Fusion ECCV 2018 OpenAccess: https://openaccess.thecvf.com/content_ECCV_2018/html/Wei_Dong_Probabilistic_Signed_Distance_ECCV_2018_paper.html
- Probabilistic Volumetric Fusion for Dense Monocular SLAM (WACV 2023): https://arxiv.org/abs/2210.01276
- BundleFusion arXiv: https://arxiv.org/abs/1604.01093
- Co-SLAM CVPR 2023 OpenAccess PDF: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Co-SLAM_Joint_Coordinate_and_Sparse_Parametric_Encodings_for_Neural_Real-Time_CVPR_2023_paper.pdf
- Benchmarking INR and Geometric Rendering in Real-Time RGB-D SLAM (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Hua_Benchmarking_Implicit_Neural_Representation_and_Geometric_Rendering_in_Real-Time_RGB-D_CVPR_2024_paper.html
- SplaTAM CVPR 2024 OpenAccess PDF: https://openaccess.thecvf.com/content/CVPR2024/papers/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.pdf
- 3D LiDAR Mapping in Dynamic Environments Using a 4D Implicit Neural Representation (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/papers/Zhong_3D_LiDAR_Mapping_in_Dynamic_Environments_using_a_4D_Implicit_CVPR_2024_paper.pdf
- DIO CVPR 2025 OpenAccess: https://openaccess.thecvf.com/content/CVPR2025/html/Diehl_DIO_Decomposable_Implicit_4D_Occupancy-Flow_World_Model_CVPR_2025_paper.html
- DeSiRe-GS CVPR 2025 OpenAccess PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Peng_DeSiRe-GS_4D_Street_Gaussians_for_Static-Dynamic_Decomposition_and_Surface_Reconstruction_CVPR_2025_paper.pdf
- WildGS-SLAM arXiv: https://arxiv.org/abs/2504.03886
- iS-MAP ACCV 2024 OpenAccess: https://openaccess.thecvf.com/content/ACCV2024/html/Wang_iS-MAP_Neural_Implicit_Mapping_and_Positioning_for_Structural_Environments_ACCV_2024_paper.html
- PlanarSplatting CVPR 2025 OpenAccess: https://openaccess.thecvf.com/content/CVPR2025/html/Tan_PlanarSplatting_Accurate_Planar_Surface_Reconstruction_in_3_Minutes_CVPR_2025_paper.html


## 2026-03-06 OTV Probe Update

### Implemented
- Added `OTV (Observation-Time Transient Veto)` in `egf_dhmap3d/modules/updater.py`.
  - Independent transient state: `phi_otv`, `phi_otv_w`, `rho_otv`, `otv_score`, `otv_age`, `otv_active`.
  - Update-time routing: front-separated observations with rear static support are re-routed away from `phi_static` / `phi_geo` toward transient storage.
- Added `OTV` dynamic-context integration in `egf_dhmap3d/core/voxel_hash.py`.
- Added second-pass `front exclusion` variant based on `phi_otv` geometry itself.

### Focused Result
- Probe scene: `TUM rgbd_dataset_freiburg3_walking_xyz`, `60` frames, oracle protocol.
- Output roots:
  - `output/post_cleanup/p10_otv_probe_walking_xyz/`
  - `output/post_cleanup/p10_otv_probe_walking_xyz_tfe/`
- Comparison table:
  - `output/summary_tables/p10_otv_probe_walking_xyz_compare.csv`

### Outcome
- `OTV` write-time-only and `OTV + front exclusion` produced numerically identical results on the focused probe:
  - `F-score = 0.981691`
  - `ghost_ratio = 0.756281`
  - `ghost_tail_ratio = 0.340002`
- Conclusion: this branch is effectively inert with respect to the current ghost bottleneck.
- Interpretation: attaching another veto/exclusion state to the existing candidate/readout chain is not sufficient. The bottleneck is deeper than write-time routing; the exported geometry still comes from the same persistent candidate pool, so scalar dynamic side-states remain diluted.

### Decision
- Archive `OTV` as a negative structural result.
- Do not continue stacking threshold-style logic on top of `OTV`.
- Next valid direction should be a stronger architectural split, e.g. true dual-layer readout / explicit dynamic exclusion map rather than score-only veto.

## 2026-03-06 CSR-XMap Probe Update

### Implemented
- Added `CSR-XMap` extraction-side structural branch.
  - `CSR (Counterfactual Static Readout)`: static-only readout from `phi_static / phi_rear / phi_spg` plus geometry agreement-gated `phi_geo`.
  - `XMap (dynamic exclusion readout)`: transient/dynamic readout from `phi_transient / phi_dyn / phi_otv`.
  - Export-time rule changed from scalar dynamic score stacking to explicit geometric competition: when `XMap` is strong and geometrically separated, either replace the candidate with `CSR` or reject it.
- Wiring completed in:
  - `egf_dhmap3d/core/config.py`
  - `egf_dhmap3d/modules/pipeline.py`
  - `scripts/run_egf_3d_tum.py`
  - `scripts/run_benchmark.py`
  - `scripts/run_p10_precision_profile.py`
  - `egf_dhmap3d/core/voxel_hash.py`

### Focused Result
- Probe scene: `TUM rgbd_dataset_freiburg3_walking_xyz`, `60` frames, oracle protocol.
- Output root:
  - `output/post_cleanup/p10_csr_xmap_probe_walking_xyz/`
- Comparison table:
  - `output/summary_tables/p10_csr_xmap_probe_walking_xyz_compare.csv`

### Outcome
- `CSR-XMap` (`egf`) on the focused probe:
  - `F-score = 0.995147`
  - `Chamfer = 0.020893`
  - `ghost_ratio = 0.810235`
  - `ghost_tail_ratio = 0.338690`
- Previous `OTV` reference on the same probe:
  - `F-score = 0.981691`
  - `Chamfer = 0.024224`
  - `ghost_ratio = 0.756281`
  - `ghost_tail_ratio = 0.340002`
- `TSDF` reference:
  - `F-score = 0.913703`
  - `ghost_ratio = 0.687708`
  - `ghost_tail_ratio = 0.146341`

### Interpretation
- `CSR-XMap` **does improve geometry**: both `Chamfer` and `F-score` are better than the `OTV` branch.
- But it **does not improve the actual P10 bottleneck**:
  - `ghost_tail_ratio` is essentially unchanged (`0.3400 -> 0.3387`).
  - `ghost_ratio` becomes even worse (`0.7563 -> 0.8102`).
- Runtime cost also increases:
  - `OTV` probe wall time: about `784.9 s`
  - `CSR-XMap` probe wall time: about `845.9 s`
  - `extract_total_sec` rises from about `21.2 s` to `52.6 s`.

### Conclusion
- `CSR-XMap` is **not** the main fix for `P10`.
- The branch is still useful as a research result because it shows a clean separation:
  - export-time counterfactual static readout can improve geometric readout quality;
  - however, the long-tail ghost bottleneck is **not** primarily caused by readout-stage front/rear competition.
- Therefore the remaining bottleneck is deeper: the dynamic front needs an **independent persistent negative / exclusion state at write-time or state-time**, not just a stronger export-time competition rule.

### Decision
- Archive `CSR-XMap` as a second structural negative result for the ghost bottleneck.
- Keep its ideas available as an `Acc`-oriented branch, but do not use it as the main `P10` line.
- Next valid direction should move from readout competition to a stronger state split, e.g. true dual-map/static-dynamic state or explicit write-time exclusion occupancy.

## 2026-03-06 XMem / BECM / RCCM Probe Update

### Implemented
- Added `XMem (Exclusion Memory)` as a write-time reversible exclusion state.
  - State fields landed in `egf_dhmap3d/core/types.py`: `xmem_occ`, `xmem_free`, `xmem_score`, `xmem_age`, `xmem_active`.
  - Config wiring landed in `egf_dhmap3d/core/config.py`, `scripts/run_egf_3d_tum.py`, `scripts/run_benchmark.py`, `scripts/run_p10_precision_profile.py`.
  - Update-time routing landed in `egf_dhmap3d/modules/updater.py`.
  - Extraction-side hooks landed in `egf_dhmap3d/core/voxel_hash.py`.
- Then upgraded `XMem` into two stronger variants:
  - `BECM (Bifurcated Exclusion-Clear Memory)`: split `front exclusion` and `clear-lock` into separate state branches so free evidence no longer directly cancels exclusion.
  - `RCCM (Ray-Conditioned Clear Memory)`: inject line-of-sight free-space contradiction only into voxels that already carry front-dynamic memory, instead of enabling global aggressive ray-casting.

### Focused Result
- Probe scene: `TUM rgbd_dataset_freiburg3_walking_xyz`, `60` frames, oracle protocol.
- Output roots:
  - `output/post_cleanup/p10_xmem_probe_walking_xyz/`
  - `output/post_cleanup/p10_xmem_becm_probe_walking_xyz/`
  - `output/post_cleanup/p10_xmem_rccm_probe_walking_xyz/`
- Unified comparison table:
  - `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### Outcome
- `XMem v1` focused probe:
  - `F-score = 0.992518`
  - `Chamfer = 0.024049`
  - `ghost_ratio = 0.820857`
  - `ghost_tail_ratio = 0.366245`
  - `mapping_fps = 0.0542`
- `XMem + BECM clear-lock` focused probe:
  - `F-score = 0.992518`
  - `Chamfer = 0.024049`
  - `ghost_ratio = 0.820857`
  - `ghost_tail_ratio = 0.366245`
  - `mapping_fps = 0.0602`
- `XMem + RCCM targeted ray clear` focused probe:
  - `F-score = 0.992518`
  - `Chamfer = 0.024049`
  - `ghost_ratio = 0.820857`
  - `ghost_tail_ratio = 0.366245`
  - `mapping_fps = 0.0439`
- References on the same probe:
  - `OTV`: `ghost_tail_ratio = 0.340002`
  - `CSR-XMap`: `ghost_tail_ratio = 0.338690`
  - `TSDF`: `ghost_tail_ratio = 0.146341`

### Interpretation
- `XMem` improves neither `ghost_ratio` nor `ghost_tail_ratio`; both are worse than `OTV/CSR-XMap`.
- `BECM` and `RCCM` produced numerically identical reconstruction/dynamic metrics to `XMem v1` under the focused probe, despite adding more structure.
- This is a strong negative result, and it is informative:
  1. The remaining bottleneck is **not** just that free evidence was subtracted too early.
  2. The bottleneck is also **not** solved by a targeted clear state that still lives on top of the same single persistent geometry pool.
  3. In the current architecture, all these exclusion/clear states are still parasitic on a map whose dominant geometry is written into the same persistent channels; therefore the negative state never becomes a first-class background representation.
  4. The exact metric identity of `XMem -> BECM -> RCCM` indicates the main missing ingredient is a **true background/static state that can be written and read independently**, not another veto memory stacked on top of the existing one-map fusion.

### Decision
- Archive `XMem`, `BECM`, and `RCCM` as the third structural negative chain for the P10 ghost bottleneck.
- Do not continue stacking more veto/clear memories on the current single-map state.
- Next valid direction should be a genuinely stronger architectural split:
  - `explicit static background map / dynamic foreground map`, or
  - `dual write graph` where background geometry is committed into its own state and dynamic/front evidence never contaminates that state in the first place.

## 2026-03-07 OBL-3D Probe Update

### Implemented
- Added `OBL-3D (Occlusion-Buffered Background Layer)`.
  - A dedicated background geometry state inside each voxel: `phi_bg`, `phi_bg_w`, `rho_bg`, `obl_score`, `obl_age`, `obl_active`.
  - Update-time rule: under front/rear separation, rear/static support is written into the background layer while the same observation suppresses writes into the legacy static/geo channels.
  - Readout-time rule: background participates as a first-class persistent surface source instead of being a side-score attached to the same front surface.
- Added a stronger export variant:
  - `background_hard` arbitration, which hard-selects `phi_bg` when background confidence and front dynamic conflict are both high.

### Focused Result
- Probe scene: `TUM rgbd_dataset_freiburg3_walking_xyz`, `60` frames, oracle protocol.
- Output roots:
  - `output/post_cleanup/p10_obl_probe_walking_xyz/`
  - `output/post_cleanup/p10_obl_hardbg_probe_walking_xyz/`
- Unified comparison table:
  - `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### Outcome
- `OBL-3D` (`obl_soft`):
  - `F-score = 0.989473`
  - `Chamfer = 0.024445`
  - `ghost_ratio = 0.806074`
  - `ghost_tail_ratio = 0.368327`
  - `mapping_fps = 0.0575`
- `OBL-3D + background_hard` (`obl_hardbg`):
  - numerically identical reconstruction / dynamic metrics
  - slower: `mapping_fps = 0.0379`
- Reference:
  - `XMem ghost_tail_ratio = 0.366245`
  - `CSR-XMap ghost_tail_ratio = 0.338690`
  - `OTV ghost_tail_ratio = 0.340002`
  - `TSDF ghost_tail_ratio = 0.146341`

### Interpretation
- `OBL-3D` is not inert: mid-run state statistics changed, active voxel count increased, and dynamic score rose earlier than the `XMem` chain.
- But final metrics show that this is still not the `P10` fix.
- The important negative conclusion is stronger than before:
  1. a background layer that still lives **inside the same voxel map / same extraction graph** is not enough;
  2. even when the update operator writes background separately and export can hard-select it, the final surface is still constrained by the same candidate pool, same surface extraction path, and same active map support;
  3. therefore the bottleneck is no longer “missing background state”, but “missing background state with independent map-level persistence and readout”.

### Decision
- Archive `OBL-3D` and `background_hard` as another structural negative branch for the P10 ghost bottleneck.
- Do not continue extending single-map multi-channel readout.
- Next valid direction is now sharply defined:
  - `true dual-map / dual-voxel-hash architecture`, with a dedicated `background_map` and a separate `foreground/exclusion_map`.

## 2026-03-07 Dual-Map Probe Update

### Implemented
- Added `DMBG-3D (Dual-Map Background/Foreground Graph)`.
  - `background_map`: the only map used for association and final local-map export.
  - `foreground_map`: a separate voxel hash that absorbs front/transient writes.
  - Update-time routing now depends on `map_role`:
    - `background`: suppress front-dominant writes, keep rear/static writes, and commit a dedicated background state.
    - `foreground`: absorb front/transient writes while allowing only minimal static leakage.
- Added `BER (Background-Exclusive Readout)` for dual-map mode.
  - In dual-map extraction, candidate readout prioritizes `phi_bg` / `rho_bg` instead of mixing back into the old persistent channels whenever a valid background state exists.

### Focused Result
- Probe scene: `TUM rgbd_dataset_freiburg3_walking_xyz`, `60` frames, oracle protocol.
- Output roots:
  - `output/post_cleanup/p10_dualmap_probe_walking_xyz/`
  - `output/post_cleanup/p10_dualmap_ber_probe_walking_xyz/`
- Unified comparison table:
  - `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### Outcome
- `DMBG-3D` (`dual_map`):
  - `F-score = 0.996531`
  - `Chamfer = 0.020462`
  - `ghost_ratio = 0.813839`
  - `ghost_tail_ratio = 0.347614`
  - `mapping_fps = 0.0383`
- `DMBG-3D + BER` (`dual_map_ber`):
  - numerically identical to `dual_map`
  - slower: `mapping_fps = 0.0250`
- References:
  - `XMem chain`: `ghost_tail_ratio = 0.366245`
  - `CSR-XMap`: `ghost_tail_ratio = 0.338690`
  - `OTV`: `ghost_tail_ratio = 0.340002`

### Interpretation
- This is the first branch that is **not inert in state-time behavior**:
  - association stayed very high (`assoc_ratio_mean ≈ 0.9843`),
  - dynamic score in the exported background map decayed toward zero (`dynamic_score_mean ≈ 0.0403`),
  - geometry improved further (`Chamfer/F-score` both better than previous branches).
- Therefore the dual-map split is a structurally correct direction.
- However it still does not solve the P10 ghost bottleneck:
  - `ghost_tail_ratio` improved from `0.3662` to `0.3476`, but this is still far from the target line;
  - `BER` produced no further gain, which means the remaining issue is not just readout mixing.
- The remaining bottleneck is now narrower:
  1. the dual maps are separated in storage, but negative evidence is still mostly local to each map;
  2. background_map is cleaner, but it is not yet actively corrected by persistent contradiction transferred from foreground_map;
  3. therefore the next valid innovation is **cross-map contradiction transfer / foreground-to-background negative projection**, not another single-map gating trick.

### Decision
- Keep `DMBG-3D` as the new mainline, since it is the first structurally positive branch.
- Archive `BER` as a non-harmful but ineffective add-on.
- Next module should be:
  - `CMCT (Cross-Map Contradiction Transfer)` or equivalent, where stable foreground contradiction is explicitly projected back into background_map as negative background evidence.

## 2026-03-07 CMCT / Cross-Map Foreground Arbitration Update

### Implemented
- Added `CMCT (Cross-Map Contradiction Transfer)`.
  - After `foreground_map` update, stable foreground contradiction is projected back into `background_map` as explicit negative evidence.
  - Background cells receive `cmct_score`, `cmct_age`, `cmct_active`, and their static / geo / bg channels are softly decayed when cross-map contradiction is strong and local background support is weak.
- Added a stronger readout-side variant on top of `CMCT`:
  - `cross-map foreground arbitration`, which consults `foreground_map` at the same / neighboring voxel during background extraction and rejects weak background candidates when a strong foreground transient surface exists.

### Focused Result
- Probe scene: `TUM rgbd_dataset_freiburg3_walking_xyz`, `60` frames, oracle protocol.
- Output roots:
  - `output/post_cleanup/p10_cmct_probe_walking_xyz/`
  - `output/post_cleanup/p10_cmct_cfam_probe_walking_xyz/`
- Unified comparison table:
  - `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### Outcome
- `dual_map_cmct`:
  - `F-score = 0.996531`
  - `Chamfer = 0.020462`
  - `ghost_ratio = 0.813839`
  - `ghost_tail_ratio = 0.347614`
  - `mapping_fps = 0.0366`
- `dual_map_cmct_cfam`:
  - numerically identical to `dual_map_cmct`
  - `mapping_fps = 0.0380`
- These are also numerically identical to the previous `dual_map` branch.

### Interpretation
- `CMCT` is an informative negative result.
- It shows that even with two separate maps, **voxel-local contradiction transfer is still too weak** to move the final ghost metric.
- The companion readout-side foreground arbitration is also ineffective, which means the missing operation is not “same-voxel foreground veto”.
- The remaining bottleneck is now sharply localized:
  1. foreground/background are already separated in storage;
  2. local voxel transfer and same-voxel cross-map masking do not change final geometry;
  3. therefore the missing operator must live at the **geometry domain / line-of-sight domain**, not the local voxel domain.

### Decision
- Archive `CMCT` and `cross-map foreground arbitration` as a negative follow-up branch on top of `dual_map`.
- Keep `dual_map` as the structurally positive mainline.
- Next valid innovation should be a geometry-domain operator, e.g.:
  - `CGCC (Cross-Map Geometric Carving Corridor)`, where a stable foreground surface is dilated into a short corridor along the viewing direction and used to carve / veto background support behind it.

## 2026-03-08 CGCC Probe Update

### Implemented
- Added `CGCC (Cross-Map Geometric Carving Corridor)`.
  - A stable foreground transient surface is expanded into a short ray-aligned corridor using the current observation ray.
  - Background cells intersecting the corridor receive a dedicated `cgcc_score / cgcc_age / cgcc_active` state.
  - Their `phi_static_w / phi_geo_w / phi_bg_w / rho_static / rho_bg` are decayed according to corridor strength and local background support.
- `CGCC` is the first explicitly **geometry-domain** operator in the current chain; it is no longer voxel-local contradiction transfer.

### Focused Result
- Probe scene: `TUM rgbd_dataset_freiburg3_walking_xyz`, `60` frames, oracle protocol.
- Output root:
  - `output/post_cleanup/p10_cgcc_probe_walking_xyz/`
- Unified comparison table:
  - `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### Outcome
- `dual_map_cgcc`:
  - `F-score = 0.996531`
  - `Chamfer = 0.020462`
  - `ghost_ratio = 0.813839`
  - `ghost_tail_ratio = 0.347614`
  - `mapping_fps = 0.0229`
- These are numerically identical to `dual_map`.

### Interpretation
- This is another important negative result.
- It shows that even a ray-aligned geometric corridor carved from foreground into background is still insufficient, at least when the corridor is instantiated from per-frame accepted observations and applied as a local weight decay on background cells.
- Therefore the remaining bottleneck is even narrower than before:
  1. storage-level split works (`dual_map`);
  2. voxel-local cross-map transfer does not move the metric (`CMCT`);
  3. short local ray corridor decay also does not move the metric (`CGCC`);
  4. hence the missing piece is likely **persistent free-space world-modeling / explicit carving volume**, not just another short-range suppression operator.

### Decision
- Archive `CGCC` as a geometry-domain negative branch.
- Keep `dual_map` as the only structurally positive mainline so far.
- The next valid direction should move from local corridor decay to a true persistent free-space representation, e.g.:
  - `PFV (Persistent Free-space Volume)` or
  - `background-only carving volume` coupled to `background_map`.

## 2026-03-08 PFV Probe Update

### Implemented
- Added `PFV (Persistent Free-space Volume)` on top of `dual_map`.
- Unlike `raycast_clear`, `CMCT`, or `CGCC`, PFV does not directly treat free-space as a transient suppression cue. Instead, it accumulates a persistent free-space bank in `background_map` from stable background hits along the viewing ray, and export is forced to respect that bank.
- PFV therefore turns negative evidence into a first-class world state rather than a local decay heuristic.

### Focused Result
- Probe scene: `TUM rgbd_dataset_freiburg3_walking_xyz`, `60` frames, oracle protocol.
- Output root:
  - `output/post_cleanup/p10_pfv_probe_walking_xyz/`
- Unified comparison table:
  - `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### Outcome
- `dual_map_pfv`:
  - `F-score = 0.996541`
  - `Chamfer = 0.020461`
  - `ghost_ratio = 0.813677`
  - `ghost_tail_ratio = 0.346503`
  - `mapping_fps = 0.0289`
- Reference `dual_map`:
  - `F-score = 0.996531`
  - `Chamfer = 0.020462`
  - `ghost_tail_ratio = 0.347614`

### Interpretation
- PFV is the first module after `dual_map` that produces a measurable positive shift in the target metric, even though the gain is still small.
- This matters structurally:
  1. storage-level split (`dual_map`) was necessary;
  2. voxel-local contradiction transfer (`CMCT`) was not enough;
  3. local geometric corridor carving (`CGCC`) was not enough;
  4. once negative evidence is promoted into a **persistent free-space state**, the ghost metric starts to move again.
- Therefore the next valid direction should stay on the `PFV` line rather than returning to contradiction-only modules.

### Decision
- Keep `dual_map + PFV` as the new mainline.
- Archive `CMCT / CFAM / CGCC` as negative side branches.
- Next module should strengthen PFV itself, e.g. by:
  - foreground-aware PFV confidence projection,
  - longer-horizon free-space persistence,
  - PFV-guided association gating or extraction exclusivity.

## 2026-03-08 PFVP Probe Update

### Implemented
- Added `PFVP (PFV-guided Proposal Routing)`.
- Instead of blocking observations in the associator, PFVP keeps the associations but routes each accepted proposal into:
  - `background_map`,
  - `foreground_map`, or
  - both,
  based on persistent free-space confidence, current background static support, and foreground history.
- This is structurally different from `PFAG`:
  - `PFAG` rejects observations before update;
  - `PFVP` preserves observations but changes where they write.

### Focused Result
- Quick probe: `output/post_cleanup/p10_pfvp_quick/`
- Full probe: `output/post_cleanup/p10_pfvp_probe_walking_xyz_v2/`
- Unified comparison table:
  - `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### Outcome
- `dual_map_pfvp`:
  - `F-score = 0.997273`
  - `Chamfer = 0.020437`
  - `ghost_ratio = 0.816400`
  - `ghost_tail_ratio = 0.352354`
  - `mapping_fps = 0.0424`
- Reference `dual_map_pfv`:
  - `F-score = 0.996541`
  - `Chamfer = 0.020461`
  - `ghost_tail_ratio = 0.346503`

### Interpretation
- PFVP improves geometry slightly, but it still worsens the target dynamic metric.
- This is an important negative result because it separates two ideas:
  1. early hard rejection (`PFAG`) is too aggressive;
  2. even softer map routing (`PFVP`) still moves proposals away from the background map too early, which hurts late-frame ghost suppression.
- Therefore the next valid step is not “more proposal routing”.
- The stronger story is now:
  - `dual_map` is necessary,
  - `PFV` is beneficial,
  - but pushing PFV earlier than update/export degrades the target metric.

### Decision
- Keep `dual_map + PFV` as the mainline.
- Archive `PFVP` as a negative branch, alongside `PFAG`.
- Next valid module should strengthen PFV *within* the update/export path rather than earlier, e.g.:
  - longer-horizon PFV persistence,
  - PFV confidence sharpening,
  - background-only persistent free-space bank for export.

## 2026-03-08 PFVP Probe Update

### Implemented
- Added `PFVP (PFV-guided Proposal Routing)` on top of `dual_map + PFV`.
- PFVP keeps associations intact, but routes each accepted observation into:
  - `background_map`,
  - `foreground_map`, or
  - both,
  according to persistent free-space confidence, foreground history, and local background support.
- This is intentionally later than `PFAG`: it preserves observation availability and only changes the write target.

### Focused Result
- Quick probe:
  - `output/post_cleanup/p10_pfvp_quick/`
- Full probe:
  - `output/post_cleanup/p10_pfvp_probe_walking_xyz_v2/`
- Unified comparison table:
  - `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### Outcome
- `dual_map_pfvp`:
  - `F-score = 0.997273`
  - `Chamfer = 0.020437`
  - `ghost_ratio = 0.816400`
  - `ghost_tail_ratio = 0.352354`
  - `mapping_fps = 0.0424`
- Reference `dual_map_pfv`:
  - `F-score = 0.996541`
  - `Chamfer = 0.020461`
  - `ghost_tail_ratio = 0.346503`

### Interpretation
- PFVP improves geometry slightly but degrades the target dynamic metric.
- This confirms the stronger version of the earlier PFAG negative result:
  1. not only pre-association hard rejection is too early;
  2. even softer proposal routing still pushes PFV influence forward too much;
  3. late-frame dynamic suppression benefits from letting observations reach the update/export path before PFV dominates decisions.
- Therefore the best current story remains:
  - `dual_map` is necessary,
  - `PFV` is beneficial,
  - but PFV should stay inside update/export rather than move earlier into routing.

### Decision
- Keep `dual_map + PFV` as the mainline.
- Archive `PFVP` as a negative branch, together with `PFAG`.
- Next useful work should strengthen PFV *inside* the persistent free-space state itself, not earlier in the pipeline.

## 2026-03-08 PFV-Sharp Quick Probe Update

### Implemented
- Added a stronger PFV-internal variant with:
  - long-horizon PFV memory (`pfv_long`, `pfv_long_age`),
  - depth-aware PFV accumulation along the stable ray,
  - clustered PFV export sharpening.
- This keeps the innovation inside the update/export path and does not reintroduce early gating/routing.

### Quick Result
- Probe root:
  - `output/post_cleanup/p10_pfv_sharp_quick/`
- Scene: `rgbd_dataset_freiburg3_walking_xyz`
- Frames: `10`

### Outcome
- The quick probe is numerically indistinguishable from the current PFV branch on the 10-frame metric sheet:
  - same `surface_points`,
  - same `ghost_tail_ratio`,
  - same `F-score` to numerical noise.
- Therefore it was not promoted to a full 60-frame focused run.

### Interpretation
- This means the current PFV bottleneck is not simply “sharpen the PFV confidence harder” or “retain it longer” with the same local mechanism.
- The useful information here is architectural:
  - PFV itself is still the right family,
  - but the next gain likely requires a different *structure* inside PFV rather than a stronger scalar state.

### Decision
- Archive `PFV-sharp` as a low-yield quick branch.
- Keep `dual_map + PFV` unchanged as the mainline.
- The next valid PFV-side innovation should likely be a structural split such as:
  - `PFV bank decomposition` (near / mid / rear free-space banks), or
  - `persistent free-space exclusivity map` used only at export.

## 2026-03-08 PFV-Bank Quick Probe Update

### Implemented
- Added a structural `banked PFV` variant on top of the current PFV mainline:
  - near / mid / far PFV banks,
  - bank ages,
  - bank-aware export clustering.
- This is more structural than scalar sharpening because it tries to separate a true cleared foreground corridor from generic free-space support at different depths.

### Quick Result
- Probe root:
  - `output/post_cleanup/p10_pfv_banks_quick/`
- Scene: `rgbd_dataset_freiburg3_walking_xyz`
- Frames: `10`

### Outcome
- The 10-frame quick probe is numerically indistinguishable from the current PFV mainline on the exported metric sheet.
- Therefore it was not promoted to a full 60-frame focused run.

### Interpretation
- This indicates the remaining PFV bottleneck is not solved by simply decomposing the current free-space scalar into depth banks while keeping the same local update/export mechanism.
- The next PFV-side gain likely requires a different structural role for PFV, not just a richer decomposition of the same local signal.

### Decision
- Archive `PFV-bank` as another quick negative branch.
- Keep `dual_map + PFV` as the active mainline.


## 2026-03-07 PFV-Exclusive Export Map Attempt

### 模块
- `PFV-Exclusive (Persistent Free-space Exclusivity Map)`
- 定位：仍留在 `dual_map + PFV` 主线内部，不前移到 associator / routing；仅增强 update/export 内部的结构角色。

### 代码落地
- `egf_dhmap3d/core/types.py`
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/pfv.py`
- `egf_dhmap3d/core/voxel_hash.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### 设计要点
- 在现有 `PFV` 之上新增独立状态：
  - `pfv_exclusive`
  - `pfv_exclusive_age`
  - `pfv_exclusive_active`
- 更新期：把“长期 cleared corridor”单独记成 export-oriented exclusivity state，而不是继续把所有 free-space 证据都压回同一个 `pfv_score`。
- 导出期：让 `pfv_exclusive` 与 `static_anchor / rear_anchor` 发生显式竞争，尝试让 persistent free-space 对背景导出产生真正的排他作用。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_pfv_excl_probe_base/`
  - `output/post_cleanup/p10_pfv_excl_probe_excl_v3/`
  - `output/post_cleanup/p10_pfv_excl_bonn_base/`
  - `output/post_cleanup/p10_pfv_excl_bonn_excl/`
- 对比表：
  - `output/summary_tables/p10_pfv_exclusive_probe_walking_xyz_compare.csv`
  - `output/summary_tables/p10_pfv_exclusive_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `dual_map_pfv_base`: `F-score = 1.000000`, `Chamfer = 0.019050`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`
  - `dual_map_pfv_exclusive`: `F-score = 1.000000`, `Chamfer = 0.019050`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`
- Bonn `balloon2`:
  - `dual_map_pfv_base`: `F-score = 0.455234`, `Chamfer = 0.124902`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`
  - `dual_map_pfv_exclusive`: `F-score = 0.455234`, `Chamfer = 0.124902`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`

### 结论
- 本轮 `PFV-Exclusive` 尝试在 TUM / Bonn focused probe 上都**数值完全不变**。
- 这说明：
  1. 仅把 `PFV` 升级成“export-only exclusivity map”，即便已经允许它和 `static_anchor` 竞争，仍不足以改变最终导出结果；
  2. 当前 `dual_map + PFV` 的剩余瓶颈，已经不只是“导出端缺一个更强 veto”；
  3. 也就是说，问题更可能还在 `background_map` 的写入/提交阶段，而不是单纯导出阶段。

### 判定
- `PFV-Exclusive` 记为本轮新的**结构负结果**。
- 它不同于 `PFV-sharp / PFV-bank`：这次不是“更强标量”或“更多 bank”，而是一次真正的 export-role 改造；即便如此仍然不生效，因此可以更明确地排除“只改导出端就能解决”的路径。

### 下一步目标方案
- 保持 `dual_map + PFV` 为主线，不回退到 `PFVP / PFAG / contradiction-only` 路线。
- 下一步应转向：
  - **PFV-conditioned background commit delay / write suppression**
- 核心思路：
  1. 让 `PFV` 在 `background_map` 写入期直接影响 `phi_bg / rho_bg / phi_static` 的提交，而不是等到导出时再 veto；
  2. 仅对被 persistent free-space 长期覆盖、且缺少稳定背景支撑的体素延迟提交或降低写入权重；
  3. 保持该机制仍位于 update/export 内部，不前移到 associator / routing。


## 2026-03-07 PFV-Conditioned Background Commit Delay Attempt

### 模块
- `PFV-Conditioned Background Commit Delay`
- 定位：保持在 `dual_map + PFV` 主线内部，把 `PFV` 从 export-side veto 前移到 `background_map` 写入期的提交抑制，而不是前移到 associator / routing。

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/pfv.py`
- `egf_dhmap3d/modules/updater.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### 设计要点
- 新增 `pfv_commit_delay_*` 参数组。
- 在 `background_map` 的写入期，根据已有 `PFV` 持久自由空间状态，对：
  - `w_static`
  - `w_bg`
  - `w_geo`
  - `rho_static / rho_bg`
 进行条件减权。
- 目标是：让“长期被 cleared corridor 覆盖、但缺乏稳定背景支撑”的体素更晚提交，避免背景图在写入阶段就被污染。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_pfv_commitdelay_tum/`
  - `output/post_cleanup/p10_pfv_commitdelay_bonn/`
- 对比表：`output/summary_tables/p10_pfv_commit_delay_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `dual_map_pfv_base`: `F-score = 1.000000`, `Chamfer = 0.019050`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`
  - `dual_map_pfv_commit_delay`: `F-score = 1.000000`, `Chamfer = 0.019050`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`
- Bonn `balloon2`:
  - `dual_map_pfv_base`: `F-score = 0.455234`, `Chamfer = 0.124902`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`
  - `dual_map_pfv_commit_delay`: `F-score = 0.455234`, `Chamfer = 0.124902`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`

### 结论
- 本轮 `PFV-conditioned background commit delay` 在 TUM / Bonn focused probe 上仍然**数值完全不变**。
- 这说明：
  1. 仅把 `PFV` 从 export 侧前移到 `background_map` 写入期做 commit delay，仍不足以改变最终结果；
  2. 当前 `dual_map + PFV` 的剩余瓶颈，可能已经不在“缺一个更强的 PFV 抑制位置”，而在 `PFV` 本身还没有提供足够区分度的状态信息；
  3. 也就是说，继续沿“同一 PFV 信号换位置使用”这条线，边际收益已经很低。

### 判定
- `PFV-conditioned background commit delay` 记为本轮新的**结构负结果**。
- 到目前为止，以下 PFV-side 路线都未形成有效增益：
  - `PFV-sharp`
  - `PFV-bank`
  - `PFV-Exclusive`
  - `PFV-conditioned background commit delay`

### 下一步目标方案
- `dual_map + PFV` 仍是唯一合理主线，但下一步不应再做“同一 PFV 信号的更强阈值 / 更早位置 / 更晚位置”改写。
- 下一步应转向：
  - **PFV-conditioned write-time background routing with explicit alternate state**
- 核心思路：
  1. 不是简单减弱背景写入，而是把被 PFV 长期覆盖的候选显式路由到独立的 delayed background candidate state；
  2. 让 `background_map` 内部至少存在“committed background` 与 `delayed background candidate` 两种可区分状态；
  3. 只有在后续时序中重新获得稳定背景支撑时，candidate 才重新并入 committed background。


## 2026-03-07 Delayed Background Candidate State Attempt

### 模块
- `Delayed Background Candidate State`
- 定位：保持在 `dual_map + PFV` 主线内部；当 `PFV` 长期覆盖某个背景写入位置时，不直接写入 committed background，而是路由到同图内的 delayed background candidate state。

### 代码落地
- `egf_dhmap3d/core/types.py`
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/bg_candidate.py`
- `egf_dhmap3d/modules/updater.py`
- `egf_dhmap3d/core/voxel_hash.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### 设计要点
- 新增状态：
  - `phi_bg_cand`
  - `phi_bg_cand_w`
  - `rho_bg_cand`
  - `bg_cand_score`
  - `bg_cand_age`
  - `bg_cand_active`
- 写入期：若 `PFV` 对背景写入形成持续矛盾，则把该次观测优先写入 `bg_candidate`，并仅向 committed background 泄露少量质量。
- 提升期：当后续重新获得稳定背景支撑时，再把 candidate 以 soft promotion 的方式并入 committed background。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_pfv_bgcand_tum/`
  - `output/post_cleanup/p10_pfv_bgcand_bonn/`
- 对比表：`output/summary_tables/p10_pfv_bg_candidate_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `dual_map_pfv_base`: `F-score = 1.000000`, `Chamfer = 0.019050`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`
  - `dual_map_pfv_bg_candidate`: `F-score = 1.000000`, `Chamfer = 0.019050`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`
- Bonn `balloon2`:
  - `dual_map_pfv_base`: `F-score = 0.455234`, `Chamfer = 0.124902`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`
  - `dual_map_pfv_bg_candidate`: `F-score = 0.455234`, `Chamfer = 0.124902`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`

### 结论
- 本轮 `delayed background candidate state` 在 TUM / Bonn focused probe 上仍然**数值完全不变**。
- 这说明：
  1. 即便不再只是“减弱写入”，而是显式引入了 delayed candidate state，同图内的 candidate buffering 仍不足以改变最终结果；
  2. 当前瓶颈大概率已经不在“同一 background_map 内部还缺一个 state”，而在 map-level persistence / routing 仍不够独立；
  3. 继续在同一 `background_map` 内堆叠更多 PFV-side candidate state，预期收益已经很低。

### 判定
- `Delayed Background Candidate State` 记为本轮新的**结构负结果**。
- 到目前为止，以下 `dual_map + PFV` 内部强化分支均未形成有效 focused gain：
  - `PFV-sharp`
  - `PFV-bank`
  - `PFV-Exclusive`
  - `PFV-conditioned background commit delay`
  - `Delayed Background Candidate State`

### 下一步目标方案
- `dual_map + PFV` 仍是主线，但下一步不应再继续在**同一 background_map 内部**堆叠新 state。
- 下一步应转向：
  - **Tri-map background architecture**
- 核心思路：
  1. 将当前 `background_map` 分裂为：
     - `committed_background_map`
     - `delayed_background_map`
     - `foreground_map`
  2. 让 `PFV` 直接决定写入去向：被 persistent free-space 覆盖且缺少稳定支撑的观测写入 `delayed_background_map`；
  3. 只有当 delayed map 在后续时序中重新获得稳定背景支撑时，才通过显式 promotion 并回 committed background。


## 2026-03-07 Tri-Map Background Architecture Attempt

### 模块
- `Tri-map Background Architecture`
- 三图结构：
  - `committed_background_map`
  - `delayed_background_map`
  - `foreground_map`

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `egf_dhmap3d/P10_method/bg_candidate.py`
- `egf_dhmap3d/modules/pipeline.py`
- `egf_dhmap3d/modules/updater.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### 设计要点
- `PFV` 不再只决定“同一 background_map 内部怎么写”，而是直接决定写入去向：
  - 可靠背景 -> `committed_background_map`
  - 被 persistent free-space 长期覆盖且支撑不足 -> `delayed_background_map`
  - 动态前景 -> `foreground_map`
- delayed map 中的背景候选，只有在后续重新获得稳定背景支撑时，才 promotion 回 committed map。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_tum_v2/`
  - `output/post_cleanup/p10_trimap_bonn_v2/`
- 对比表：`output/summary_tables/p10_trimap_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `dual_map_pfv_base`: `Acc = 0.009441`, `Chamfer = 0.019050`, `F-score = 1.000000`, `Comp-R = 1.000000`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`
  - `tri_map_pfv_v2`: `Acc = 0.009360`, `Chamfer = 0.035427`, `F-score = 0.926335`, `Comp-R = 0.862778`, `ghost_ratio = 0.728791`, `ghost_tail_ratio = 0.096225`
  - 路由统计：`trimap_delayed_mean = 379.8`, `trimap_promoted_mean = 4240.25`
- Bonn `balloon2`:
  - `dual_map_pfv_base`: `Acc = 0.065994`, `Chamfer = 0.124902`, `F-score = 0.455234`, `Comp-R = 0.485667`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`
  - `tri_map_pfv_v2`: `Acc = 0.051250`, `Chamfer = 0.166409`, `F-score = 0.283288`, `Comp-R = 0.183722`, `ghost_ratio = 0.093995`, `ghost_tail_ratio = 0.417786`
  - 路由统计：`trimap_delayed_mean = 691.45`, `trimap_promoted_mean = 6916.9`

### 结论
- 这是当前 `PFV` 主线下第一条**真正产生显著数值变化**的 map-level 结构分支。
- 正向信号：
  1. `Acc` 在 TUM / Bonn 两侧都改善；
  2. `ghost_tail_ratio` 在 TUM / Bonn 两侧都下降；
  3. 说明“把 delayed background 从 committed background 真正分离出来”方向是有效的。
- 负向结果：
  1. `Comp-R` 明显下降；
  2. `F-score` 与 `Chamfer` 整体退化；
  3. `ghost_ratio` 反而变差。
- 这说明：tri-map 的核心结构方向是对的，但当前 promotion / rescue 机制太保守，导致 coverage 大量丢失。

### 判定
- `Tri-map Background Architecture` 不是无效分支，而是当前第一条“有正向机制信号、但未通过联合指标”的 `PFV` map-level 主线。
- 它比 `PFV-sharp / PFV-bank / PFV-Exclusive / PFV-conditioned background commit delay / delayed background candidate state` 更接近真正的下一代结构。

### 下一步目标方案
- 保持 `tri-map background architecture` 为下一轮主线。
- 下一步不再继续扩大 delayed routing，而应专注于：
  - **Comp-R recovery without reintroducing tail ghost**
- 具体方向：
  1. `promotion-aware rescue`: delayed map 中满足稳定支撑的体素，更积极地 promotion 回 committed background；
  2. `hole-only rescue`: 只在 committed map 局部缺口处使用 delayed map 做补洞，避免全局召回回弹；
  3. `promotion confidence gating`: 把 delayed->committed promotion 绑定到更明确的静态支撑，而不是简单 age / rho。


## 2026-03-07 Promotion-Aware Rescue + Hole-Only Rescue Attempt

### 模块
- `promotion-aware rescue`
- `hole-only rescue`
- 定位：建立在 `tri-map background architecture` 之上，目标是恢复 `Comp-R / F-score`，同时避免把 tail ghost 带回来。

### 代码落地
- `egf_dhmap3d/P10_method/tri_map.py`
- `egf_dhmap3d/modules/pipeline.py`
- `egf_dhmap3d/core/config.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### 设计要点
- `promotion-aware rescue`：
  - 当 committed map 局部存在 hole 时，降低 delayed->committed promotion 的阈值，并提高 promotion blend。
- `hole-only rescue`：
  - 导出期不直接全量使用 delayed map，而只在 committed map 局部洞区域尝试用 delayed map 补洞。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_rescue_tum/`
  - `output/post_cleanup/p10_trimap_rescue_bonn/`
- 对比表：`output/summary_tables/p10_trimap_rescue_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_pfv_v2`: `F-score = 0.926335`, `Chamfer = 0.035427`, `Comp-R = 0.862778`, `ghost_ratio = 0.728791`, `ghost_tail_ratio = 0.096225`
  - `tri_map_pfv_rescue`: `F-score = 0.926335`, `Chamfer = 0.035427`, `Comp-R = 0.862778`, `ghost_ratio = 0.728791`, `ghost_tail_ratio = 0.096225`
- Bonn `balloon2`:
  - `tri_map_pfv_v2`: `F-score = 0.283288`, `Chamfer = 0.166409`, `Comp-R = 0.183722`, `ghost_ratio = 0.093995`, `ghost_tail_ratio = 0.417786`
  - `tri_map_pfv_rescue`: `F-score = 0.283288`, `Chamfer = 0.166409`, `Comp-R = 0.183722`, `ghost_ratio = 0.093995`, `ghost_tail_ratio = 0.417786`

### 结论
- 本轮 `promotion-aware rescue + hole-only rescue` 在 focused probe 上**没有带来新的数值改善**。
- 这说明：
  1. tri-map 当前的主要问题并不只是“promotion 不够积极”或“导出没补洞”；
  2. rescue 逻辑没有改变 tri-map 当前的核心 trade-off：`Acc / ghost_tail` 改善，但 `Comp-R / F-score / ghost_ratio` 退化；
  3. 下一步若继续沿 tri-map 推进，应该直接作用于 delayed map 的生成/提交标准，而不是继续在 promotion / export-rescue 上微调。

### 判定
- `promotion-aware rescue + hole-only rescue` 记为本轮**低收益负结果**。
- 它不会推翻 tri-map 的方向判断，但说明 tri-map 的下一步不该再停留在“恢复层面的小修补”。

### 下一步目标方案
- 保持 `tri-map background architecture` 为主线。
- 下一步应转向：
  - **delayed-map write criterion redesign**
- 核心思路：
  1. 重新定义哪些观测应该进入 delayed map，而不是 committed map；
  2. 将 delayed routing 绑定到更明确的“front occupancy / background support conflict”，而不是当前较松的 PFV + foreground history 组合；
  3. 优先在写入生成端减少不必要的 delayed routing，再谈 promotion / rescue。


## 2026-03-07 Delayed-Map Write Criterion Redesign Attempt

### 模块
- `delayed-map write criterion redesign`
- 定位：不再在 tri-map 上继续堆 rescue，而是直接重写“什么样的观测应该进入 delayed map”。

### 代码落地
- `egf_dhmap3d/P10_method/tri_map.py`
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/modules/pipeline.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### 设计要点
- 将原先较激进的 delayed routing 改成“更明确的前景占据冲突判据”：
  - 使用 `PFV` + `foreground local conflict` + `background support` 的显式冲突带；
  - 只有在强冲突时才 delayed-only；
  - 中等冲突时走 `bg + delayed` 双写入；
  - 否则直接写 committed background。
- 目标是：减少误送 delayed map，恢复 `Comp-R / F-score / ghost_ratio`，同时尽量保留 tri-map 的 `Acc / ghost_tail` 改善。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_criterion_tum/`
  - `output/post_cleanup/p10_trimap_criterion_bonn/`
- 对比表：`output/summary_tables/p10_trimap_write_criterion_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_pfv_v2`: `F-score = 0.926335`, `Chamfer = 0.035427`, `Comp-R = 0.862778`, `ghost_ratio = 0.728791`, `ghost_tail_ratio = 0.096225`, `trimap_delayed_mean = 379.8`
  - `tri_map_pfv_write_criterion`: `F-score = 1.000000`, `Chamfer = 0.019050`, `Comp-R = 1.000000`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`, `trimap_delayed_mean = 0.0`
- Bonn `balloon2`:
  - `tri_map_pfv_v2`: `F-score = 0.283288`, `Chamfer = 0.166409`, `Comp-R = 0.183722`, `ghost_ratio = 0.093995`, `ghost_tail_ratio = 0.417786`, `trimap_delayed_mean = 691.45`
  - `tri_map_pfv_write_criterion`: `F-score = 0.455234`, `Chamfer = 0.124902`, `Comp-R = 0.485667`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`, `trimap_delayed_mean = 0.0`

### 结论
- 本轮 `write criterion redesign` 把 tri-map 几乎**完全退回了 `dual_map + PFV` 基线**：
  - `Comp-R / F-score / ghost_ratio` 全部回来了；
  - 但 tri-map 原本带来的 `Acc / ghost_tail_ratio` 改善也一起消失了；
  - `trimap_delayed_mean = 0.0` 直接说明 delayed routing 基本未被触发。
- 这说明：
  1. tri-map 的方向本身是有效的；
  2. 但当前这版 write criterion 过于保守，已经把 tri-map “关掉了”；
  3. 问题不在 tri-map 方向，而在 delayed routing 的冲突带没有落在正确区间。

### 判定
- `delayed-map write criterion redesign` 记为本轮**有信息增益但未形成增益结果**的分支：
  - 它不是简单负结果；
  - 它告诉我们 tri-map 的可用区间位于“当前 v2 过激”和“本轮 redesign 过保守”之间。

### 下一步目标方案
- 下一步应转向：
  - **conflict-band tri-map routing**
- 核心思路：
  1. 不走当前这种几乎关闭 delayed routing 的硬判据；
  2. 也不回到 v2 那种大规模 delayed routing；
  3. 而是构造一个中间带：
     - 强冲突 -> delayed-only
     - 中等冲突 -> bg + delayed 双写入
     - 弱冲突 -> committed-only
  4. 重点目标是把 `delayed_mean` 控制在非零但显著低于 `v2` 的范围内，尝试同时保住部分 `Acc / ghost_tail` 改善与 `Comp-R`。


## 2026-03-07 Conflict-Band Tri-Map Routing Attempt

### 模块
- `conflict-band tri-map routing`
- 定位：试图把 tri-map 的 delayed routing 从“过激”与“几乎关闭”之间拉回一个中间冲突带。

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### 设计要点
- 将 tri-map 写入准则明确分成三档：
  - 强冲突：`delayed-only`
  - 中等冲突：`bg + delayed` 双写入
  - 弱冲突：`committed-only`
- 目标是让 `trimap_delayed_mean` 落在 `v2` 与“criterion redesign 几乎为 0”之间，尝试同时保住部分 `Acc / ghost_tail` 改善与 `Comp-R`。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_conflictband_tum/`
  - `output/post_cleanup/p10_trimap_conflictband_bonn/`
- 对比表：`output/summary_tables/p10_trimap_conflict_band_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_pfv_v2`: `F-score = 0.926335`, `Chamfer = 0.035427`, `Comp-R = 0.862778`, `ghost_ratio = 0.728791`, `ghost_tail_ratio = 0.096225`, `trimap_delayed_mean = 379.8`
  - `tri_map_conflict_band`: `F-score = 0.926335`, `Chamfer = 0.035427`, `Comp-R = 0.862778`, `ghost_ratio = 0.728791`, `ghost_tail_ratio = 0.096225`, `trimap_delayed_mean = 379.8`
- Bonn `balloon2`:
  - `tri_map_pfv_v2`: `F-score = 0.283288`, `Chamfer = 0.166409`, `Comp-R = 0.183722`, `ghost_ratio = 0.093995`, `ghost_tail_ratio = 0.417786`, `trimap_delayed_mean = 691.45`
  - `tri_map_conflict_band`: `F-score = 0.283288`, `Chamfer = 0.166409`, `Comp-R = 0.183722`, `ghost_ratio = 0.093995`, `ghost_tail_ratio = 0.417786`, `trimap_delayed_mean = 691.45`

### 结论
- 本轮 `conflict-band tri-map routing` 与当前 `tri_map_pfv_v2` **数值完全一致**。
- 这说明：
  1. 当前实现下，所谓“冲突带”并没有真正改变 tri-map 的有效路由分布；
  2. delayed routing 的关键问题不在“多一个中间档”，而在冲突分数本身的构成仍然和 `v2` 等价；
  3. 因而下一步不该再继续微调 band，而应该重新定义冲突分数的来源。

### 判定
- `conflict-band tri-map routing` 记为本轮**无额外增益的等价分支**。
- 它没有提供新的有效改善，但进一步确认了：tri-map 的下一步必须重做冲突信号，而不是再细调路由形状。

### 下一步目标方案
- 下一步应转向：
  - **front-occupancy anchored tri-map routing**
- 核心思路：
  1. delayed routing 不再主要依赖当前 `PFV + foreground history + background support` 组合；
  2. 改成更明确的“前景占据成立且背景支撑不足”才进入 delayed map；
  3. 也就是说，先重做 conflict score 的物理语义，再谈 band / promotion / rescue。


## 2026-03-07 Front-Occupancy Anchored Tri-Map Routing Attempt

### 模块
- `front-occupancy anchored tri-map routing`
- 定位：不再以当前 `PFV + foreground history + background support` 组合作为 delayed routing 主信号，而改成“前景占据成立且背景支撑不足”才进入 delayed map。

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### 设计要点
- delayed routing 的主导信号从 `PFV` 转为 `front occupancy`：
  - 前景占据强 -> 才允许 delayed routing；
  - 背景支撑不足 -> 才真正进入 delayed map；
  - 否则保持 committed background。
- 目标是：减少 v2 中过于宽松的 delayed routing，同时避免 criterion redesign 那种“几乎完全关掉 tri-map”。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_frontocc_tum/`
  - `output/post_cleanup/p10_trimap_frontocc_bonn/`
- 对比表：`output/summary_tables/p10_trimap_front_occupancy_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_pfv_v2`: `F-score = 0.926335`, `Chamfer = 0.035427`, `Comp-R = 0.862778`, `ghost_ratio = 0.728791`, `ghost_tail_ratio = 0.096225`, `trimap_delayed_mean = 379.8`
  - `tri_map_front_occupancy`: `F-score = 1.000000`, `Chamfer = 0.019050`, `Comp-R = 1.000000`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`, `trimap_delayed_mean = 0.0`
- Bonn `balloon2`:
  - `tri_map_pfv_v2`: `F-score = 0.283288`, `Chamfer = 0.166409`, `Comp-R = 0.183722`, `ghost_ratio = 0.093995`, `ghost_tail_ratio = 0.417786`, `trimap_delayed_mean = 691.45`
  - `tri_map_front_occupancy`: `F-score = 0.455234`, `Chamfer = 0.124902`, `Comp-R = 0.485667`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`, `trimap_delayed_mean = 0.0`

### 结论
- 本轮 `front-occupancy anchored tri-map routing` 与此前的 `write criterion redesign` 一样，**把 tri-map 几乎完全关掉了**。
- 这说明：
  1. 单独依赖“前景占据成立”作为 delayed routing 主锚点过于保守；
  2. tri-map 需要的不是“更换主信号”，而是同时保留 `PFV` 与前景占据、并在两者之间形成真正的冲突融合；
  3. 当前 delayed routing 的问题不是“少了某一个锚点”，而是缺少一个能把 `PFV / front occupancy / background support` 融合成可调冲突带的统一分数。

### 判定
- `front-occupancy anchored tri-map routing` 记为本轮**有信息增益但未形成增益结果**的分支。
- 它进一步证明：tri-map 不能只靠 PFV，也不能只靠 front occupancy；下一步必须做两者的显式融合。

### 下一步目标方案
- 下一步应转向：
  - **hybrid conflict-score tri-map routing**
- 核心思路：
  1. 不是仅用 `PFV`；
  2. 也不是仅用 `front occupancy`；
  3. 而是构造统一 `conflict score = f(PFV, front occupancy, background support)`；
  4. 再在这个统一分数上做三段式路由：`committed-only / dual / delayed-only`。


## 2026-03-07 Hybrid Conflict-Score Tri-Map Routing Attempt

### 模块
- `hybrid conflict-score tri-map routing`
- 定位：用统一 `conflict score = f(PFV, front occupancy, background deficiency)` 取代单一 `PFV` 或单一 `front occupancy` 的 delayed routing 主信号。

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### 设计要点
- 统一冲突分数：
  - `PFV`
  - `front occupancy`
  - `assoc risk`
  - `background deficiency`
- 再在统一分数上做三段式路由：
  - `committed-only`
  - `dual`
  - `delayed-only`
- 目标是让 tri-map 不再完全依赖单一信号，同时避免当前 `v2` 的过度 delayed routing。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_hybrid_tum/`
  - `output/post_cleanup/p10_trimap_hybrid_bonn/`
- 对比表：`output/summary_tables/p10_trimap_hybrid_conflict_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_pfv_v2`: `F-score = 0.926335`, `Chamfer = 0.035427`, `Comp-R = 0.862778`, `ghost_ratio = 0.728791`, `ghost_tail_ratio = 0.096225`, `trimap_delayed_mean = 379.8`
  - `tri_map_hybrid_conflict`: `F-score = 1.000000`, `Chamfer = 0.019050`, `Comp-R = 1.000000`, `ghost_ratio = 0.560057`, `ghost_tail_ratio = 0.130459`, `trimap_delayed_mean = 0.0`
- Bonn `balloon2`:
  - `tri_map_pfv_v2`: `F-score = 0.283288`, `Chamfer = 0.166409`, `Comp-R = 0.183722`, `ghost_ratio = 0.093995`, `ghost_tail_ratio = 0.417786`, `trimap_delayed_mean = 691.45`
  - `tri_map_hybrid_conflict`: `F-score = 0.455234`, `Chamfer = 0.124902`, `Comp-R = 0.485667`, `ghost_ratio = 0.044268`, `ghost_tail_ratio = 0.472253`, `trimap_delayed_mean = 0.0`

### 结论
- 本轮 `hybrid conflict-score` 仍然**过于保守**，结果与“write criterion redesign / front-occupancy anchored”同类：
  - tri-map 基本被关掉；
  - `Comp-R / F-score / ghost_ratio` 回到基线；
  - `Acc / ghost_tail` 改善消失。
- `trimap_hybrid_mean` 虽然非零，但没有推动 delayed routing 进入有效区间。
- 这说明：
  1. 不是“缺少融合公式”；
  2. 而是当前 conflict score 的量纲和阈值仍没有把样本推到真正有用的中间带；
  3. 继续在当前 score 上调权重，预期收益有限。

### 判定
- `hybrid conflict-score tri-map routing` 记为本轮**有信息增益但未形成增益结果**的分支。
- 它证明了：简单把多个信号线性加权，还不足以得到有效 tri-map routing。

### 下一步目标方案
- 下一步应转向：
  - **support-gap calibrated tri-map routing**
- 核心思路：
  1. 不再直接用线性混合分数做 delayed 判定；
  2. 转而显式建模 `front support - background support` 的 gap；
  3. delayed routing 只在 gap 穿过一个稳定阈值区间时触发；
  4. 目标仍是把 delayed usage 控制在非零但低于 `v2` 的区间，同时保留部分 `Acc / ghost_tail` 改善与 `Comp-R`。

## 2026-03-07 Support-Gap Calibrated Tri-Map Routing Attempt

### 模块
- `support-gap calibrated tri-map routing`
- 定位：不再用线性 `hybrid conflict score` 直接做 delayed routing，而是显式计算 `front support - background support` 的 signed gap，再用 `PFV / assoc risk / background deficit` 只做小幅校准。

### 代码落地
- `egf_dhmap3d/P10_method/tri_map.py`
- `egf_dhmap3d/core/config.py`
- `scripts/run_egf_3d_tum.py`
- `egf_dhmap3d/modules/updater.py`

### 设计要点
- 主判据从“线性混合分数”改为：
  - `support_gap = front_support - background_support_mix`
- 其中：
  - `front_support` 仍由 `front occupancy / front history / local PFV-front` 构成；
  - `background_support_mix` 由 `bg_support` 与 `bg_rho` 混合；
  - `PFV / assoc risk / background deficit` 只作为 gap 的校准项，而不再主导 delayed 判定。
- 目标是：
  1. 让 gap 的符号直接表达“前景支撑是否真正压过背景支撑”；
  2. 避免 `hybrid conflict-score` 那种量纲混合后阈值难对齐的问题；
  3. 把 delayed routing 控制在“非零但明显低于 `v2`”的中间区间。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_supportgap_tum/`
  - `output/post_cleanup/p10_trimap_supportgap_bonn/`
- 为消除“旧 v2 输出与当前代码状态不完全同口径”的歧义，本轮额外补跑了当前代码下的同口径 baseline：
  - `output/post_cleanup/p10_supportgap_base_tum/`
  - `output/post_cleanup/p10_supportgap_base_bonn/`
- 对比表：`output/summary_tables/p10_trimap_support_gap_probe_tum_bonn_compare.csv`

### 结果
- 与 `legacy tri_map_pfv_v2` 比较：
  - TUM `walking_xyz`:
    - `tri_map_pfv_v2`: `F-score = 0.926335`, `Chamfer = 0.035427`, `Comp-R = 0.862778`, `ghost_ratio = 0.728791`, `ghost_tail_ratio = 0.096225`, `trimap_delayed_mean = 379.8`
    - `tri_map_support_gap`: `F-score = 0.999553`, `Chamfer = 0.020242`, `Comp-R = 1.000000`, `ghost_ratio = 0.607266`, `ghost_tail_ratio = 0.117678`, `trimap_delayed_mean = 0.0`
  - Bonn `balloon2`:
    - `tri_map_pfv_v2`: `F-score = 0.283288`, `Chamfer = 0.166409`, `Comp-R = 0.183722`, `ghost_ratio = 0.093995`, `ghost_tail_ratio = 0.417786`, `trimap_delayed_mean = 691.45`
    - `tri_map_support_gap`: `F-score = 0.630335`, `Chamfer = 0.099868`, `Comp-R = 0.701660`, `ghost_ratio = 0.142491`, `ghost_tail_ratio = 0.343565`, `trimap_delayed_mean = 0.0`
- 但在**当前代码同口径 baseline** 下，本轮更关键的判定是：
  - TUM：`dual_map_pfv_base_current` 与 `tri_map_support_gap` **逐项完全相同**；
  - Bonn：`dual_map_pfv_base_current` 与 `tri_map_support_gap` **逐项完全相同**。
- 新增路由统计显示：
  - TUM：`trimap_support_gap_mean = -0.308175`, `trimap_gap_score_mean = -0.258249`
  - Bonn：`trimap_support_gap_mean = -0.163471`, `trimap_gap_score_mean = -0.099534`
- 即：raw gap 与 calibrated gap 在两组 probe 上都整体落在负区间，没有把样本推入 delayed/dual 的有效带。

### 结论
- 本轮 `support-gap calibrated tri-map routing` 的**有效结论是负结果**：
  - 从当前代码同口径 baseline 看，它与 `dual_map + PFV` **数值完全等价**；
  - `trimap_delayed_mean = 0.0`、`trimap_dual_mean = 0.0`，tri-map 实际没有被激活；
  - 它属于又一个“有观测增益、但机制上退回基线”的分支。
- 这说明：
  1. 仅把 delayed routing 改写成 raw `front - background` gap 还不够；
  2. 当前 gap 的中心明显偏负，样本整体没有进入 tri-map 的可用工作带；
  3. 下一步不应只是继续微调阈值，而应先把 gap 做**零中心化 / 归一化 / bias-lift**，否则只会在“全关”和“过开”之间来回摆动。

### 判定
- `support-gap calibrated tri-map routing` 记为本轮**有信息增益但未形成收益**的分支。
- 它的核心价值在于明确暴露了：当前 `front_support - background_support` 的自然分布整体偏负，tri-map 下一步必须先做 gap 的中心校准，而不是继续直接调固定阈值。

### 下一步目标方案
- 下一步应转向：
  - **zero-centered normalized support-gap routing**
- 核心思路：
  1. 不直接对 raw gap 设阈值；
  2. 先对 `front_support - background_support` 做局部零中心化或 bias-lift；
  3. 再在归一化后的 gap 上做 `committed-only / dual / delayed-only` 三段路由；
  4. 目标是先把 `trimap_delayed_mean` 从 `0.0` 拉回到一个稳定非零区间，再观察是否还能保留部分 `ghost_tail / Acc` 改善而不过度损伤 `Comp-R`。

## 2026-03-07 Zero-Centered Normalized Support-Gap Routing Attempt

### 模块
- `zero-centered normalized support-gap routing`
- 定位：保留 `support-gap` 主线，但不再直接对 raw `front_support - background_support` 设阈值，而是先做：
  1. `background anchor` 缩放；
  2. `centered gap`；
  3. `normalized gap`；
  4. 再叠加小幅 `bias-lift`。

### 代码落地
- `egf_dhmap3d/P10_method/tri_map.py`
- `egf_dhmap3d/core/config.py`
- `scripts/run_egf_3d_tum.py`

### 设计要点
- 从上一轮得到的关键信号是：raw gap 在两组 probe 上整体偏负：
  - TUM: `-0.308175`
  - Bonn: `-0.163471`
- 因此本轮不再直接对 raw gap 做 hard threshold，而是：
  - 用 `bg_anchor_ratio < 1` 把背景支撑做零中心化；
  - 再用 `norm_floor + front + bg_anchor + deficit` 做归一化；
  - 最后叠加 `PFV / assoc / bg_deficit / front bonus` 的小幅 `bias-lift`。
- 同时，本轮还把 `bg_rho` 从 hard guard 中移出，改为让它通过 `bg_deficit` 间接进入 route score，避免再次出现“score 已经为正、但 hard guard 把所有样本全挡掉”的情况。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_zerocenter_tum/`
  - `output/post_cleanup/p10_trimap_zerocenter_bonn/`
- 对比表：`output/summary_tables/p10_trimap_zero_centered_support_gap_probe_tum_bonn_compare.csv`

### 结果
- 与当前代码同口径 baseline 比较，map-level 指标仍然**完全不变**：
  - TUM `walking_xyz`:
    - `dual_map_pfv_base_current`: `F-score = 0.999553`, `Chamfer = 0.020242`, `Comp-R = 1.000000`, `ghost_ratio = 0.607266`, `ghost_tail_ratio = 0.117678`
    - `tri_map_zero_centered_support_gap`: **完全相同**
  - Bonn `balloon2`:
    - `dual_map_pfv_base_current`: `F-score = 0.630335`, `Chamfer = 0.099868`, `Comp-R = 0.701660`, `ghost_ratio = 0.142491`, `ghost_tail_ratio = 0.343565`
    - `tri_map_zero_centered_support_gap`: **完全相同**
- 但机制统计上有了比上一轮更细的变化：
  - TUM:
    - `trimap_support_gap_mean = -0.308175`
    - `trimap_centered_gap_mean = -0.024127`
    - `trimap_norm_gap_mean = -0.038345`
    - `trimap_gap_bias_mean = 0.069926`
    - `trimap_gap_score_mean = 0.031581`
    - `trimap_dual_mean = 0.05`
    - `trimap_delayed_mean = 0.0`
  - Bonn:
    - `trimap_support_gap_mean = -0.163471`
    - `trimap_centered_gap_mean = -0.012112`
    - `trimap_norm_gap_mean = -0.019264`
    - `trimap_gap_bias_mean = 0.083937`
    - `trimap_gap_score_mean = 0.064673`
    - `trimap_dual_mean = 0.0`
    - `trimap_delayed_mean = 0.0`

### 结论
- 本轮 `zero-centered normalized support-gap routing` 相比上一轮**确实向前推进了一步**：
  - raw gap 被成功推近零中心；
  - route score 由负转正；
  - TUM 上首次出现了**非零 tri-map 激活**（`trimap_dual_mean = 0.05`）。
- 但它仍然**没有形成 map-level 收益**：
  - delayed routing 仍为 `0.0`；
  - Bonn 仍然完全没有激活；
  - TUM 的激活量也过小，尚不足以改变导出的最终指标。
- 这说明：
  1. 本轮已经证明“问题不在 raw gap 本身，而在固定阈值与固定预算太脆弱”；
  2. 零中心化让 score 进入了可用区，但**绝对阈值路由仍然太硬**；
  3. 下一步需要从“固定阈值”转向“分位数/预算约束”的路由方式，让 tri-map 在不同场景下都能保持一个稳定、可控的非零激活量。

### 判定
- `zero-centered normalized support-gap routing` 记为本轮**有信息增益、出现弱机制激活、但尚未形成指标收益**的分支。
- 它是目前 support-gap 主线上最接近“真正打开 tri-map”的版本，但离能稳定冲击 P10 指标还差一步“自适应路由预算”。

### 下一步目标方案
- 下一步应转向：
  - **quantile-calibrated support-gap routing**
- 核心思路：
  1. 继续使用当前 `zero-centered normalized support-gap score`；
  2. 不再只用固定绝对阈值；
  3. 每帧或每批次按 score 分位数 / capped budget 选择少量 top-conflict 候选进入 `dual` 或 `delayed`；
  4. 目标是把 `trimap_dual_mean / trimap_delayed_mean` 稳定拉到“小而非零”的区间，再观察是否能恢复部分 `ghost_tail / Acc` 改善而不过度伤害 `Comp-R`。

## 2026-03-07 Quantile-Calibrated Support-Gap Routing Attempt

### 模块
- `quantile-calibrated support-gap routing`
- 定位：保留上一轮的 `zero-centered normalized support-gap score`，但把最终 tri-map 激活从固定绝对阈值改为：
  - 先按每帧 score 分位数找 top-conflict；
  - 再施加 `capped budget`，保证 tri-map 使用量稳定、小幅、非零。

### 代码落地
- `egf_dhmap3d/P10_method/tri_map.py`
- `egf_dhmap3d/core/config.py`
- `scripts/run_egf_3d_tum.py`

### 设计要点
- 上一轮已经把 score 拉到接近可用区，但固定阈值导致：
  - TUM 只有极弱激活；
  - Bonn 基本仍为零。
- 本轮改成两阶段：
  1. 继续计算 `zero-centered normalized support-gap score`；
  2. 在 `soft_candidate` 上按分位数取 top tail；
  3. 再用 `soft/strong budget` 限制每帧进入 tri-map 的点数。
- 目标是：
  - 不再依赖某个固定绝对阈值；
  - 让 TUM/Bonn 都进入稳定非零 tri-map 使用区间；
  - 观察这种“受控非零使用”是否足以带来 map-level 指标变化。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_quantile_tum/`
  - `output/post_cleanup/p10_trimap_quantile_bonn/`
- 对比表：`output/summary_tables/p10_trimap_quantile_support_gap_probe_tum_bonn_compare.csv`

### 结果
- 与当前代码同口径 baseline 比较，最终 map-level 指标仍然**完全不变**：
  - TUM `walking_xyz`:
    - `dual_map_pfv_base_current`: `F-score = 0.999553`, `Chamfer = 0.020242`, `Comp-R = 1.000000`, `ghost_ratio = 0.607266`, `ghost_tail_ratio = 0.117678`
    - `tri_map_quantile_support_gap`: **完全相同**
  - Bonn `balloon2`:
    - `dual_map_pfv_base_current`: `F-score = 0.630335`, `Chamfer = 0.099868`, `Comp-R = 0.701660`, `ghost_ratio = 0.142491`, `ghost_tail_ratio = 0.343565`
    - `tri_map_quantile_support_gap`: **完全相同**
- 但机制统计上，本轮比 `zero-centered` 明显更进一步：
  - TUM:
    - `trimap_dual_mean = 9.15`
    - `trimap_promoted_mean = 77.95`
    - `trimap_quantile_soft_thresh_mean = 0.020248`
    - `trimap_quantile_soft_budget_mean = 42.75`
  - Bonn:
    - `trimap_dual_mean = 7.45`
    - `trimap_promoted_mean = 46.15`
    - `trimap_quantile_soft_thresh_mean = 0.019991`
    - `trimap_quantile_soft_budget_mean = 42.75`
- 同时也暴露出本轮的核心限制：
  - `trimap_delayed_mean` 仍然是 `0.0`；
  - `trimap_quantile_strong_budget_mean` 仍然是 `0.0`；
  - 即：quantile 确实把**dual branch** 打开了，但**delayed-only branch 仍未打开**。

### 结论
- 本轮 `quantile-calibrated support-gap routing` 的关键结论是：
  - 它已经成功把 tri-map 从“偶发弱激活”推进到了“稳定非零 dual 使用”；
  - TUM/Bonn 都进入了可重复的 tri-map 活跃状态；
  - 但由于所有激活都落在 `dual` 而不是 `delayed-only`，committed background 仍接收同一批测量，导致最终导出表面几乎不变。
- 这说明：
  1. 当前真正的瓶颈已经不再是“怎么打开 tri-map”；
  2. 而是“怎么让 top-conflict 样本真正离开 committed background”；
  3. 如果 strongest tail 仍只走 `dual`，tri-map 机制会被 committed 写回冲淡，最终指标不会动。

### 判定
- `quantile-calibrated support-gap routing` 记为本轮**有明显机制增益、但仍未形成指标收益**的分支。
- 它是目前 support-gap 主线上最有价值的一步：第一次让 TUM/Bonn 都进入稳定非零 tri-map 使用区间，但也把下一步的真正攻坚点钉死为“top tail delayed-only split”。

### 下一步目标方案
- 下一步应转向：
  - **top-tail delayed-only escalation under quantile routing**
- 核心思路：
  1. 保留本轮 quantile-calibrated candidate selection；
  2. 把 top tail 中最强冲突的一小部分从 `dual` 升级为 `delayed-only`；
  3. 中等冲突仍保持 `dual`；
  4. 目标是让 committed map 与 delayed map 真正出现结构性分叉，再观察是否能恢复 `Acc / ghost_tail` 改善而不过度破坏 `Comp-R / F-score`。

## 2026-03-08 Top-Tail Delayed-Only Escalation Attempt

### 模块
- `top-tail delayed-only escalation under quantile routing`
- 定位：保留上一轮 `quantile-calibrated support-gap routing` 的稳定非零 tri-map 激活，但把其中最强冲突的 top tail 从 `dual` 升级为 `delayed-only`，尝试让 committed map 与 delayed map 产生真正结构性分叉。

### 代码落地
- `egf_dhmap3d/P10_method/tri_map.py`
- `egf_dhmap3d/core/config.py`

### 设计要点
- 上一轮已经证明：
  - `dual` branch 能稳定打开；
  - 但 strongest tail 仍然没有离开 committed background；
  - 因此最终指标几乎不动。
- 本轮改动是：
  1. 先按 quantile 选出稳定的 `dual tail`；
  2. 再在这批 `dual tail` 内部按 top-tail score 做第二次筛选；
  3. 把其中最强的一小部分升级为 `delayed-only`；
  4. 其余仍保留 `dual`。
- 目标是：
  - 第一次真正打开 `delayed-only` 分支；
  - 验证“只要 strongest tail 真离开 committed map，是否就能产生 map-level 指标变化”。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_toptail_tum/`
  - `output/post_cleanup/p10_trimap_toptail_bonn/`
- 对比表：`output/summary_tables/p10_trimap_toptail_delayed_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `dual_map_pfv_base_current`: `F-score = 0.999553`, `Chamfer = 0.020242`, `Comp-R = 1.000000`, `ghost_ratio = 0.607266`, `ghost_tail_ratio = 0.117678`, `trimap_delayed_mean = 0.0`, `trimap_dual_mean = 0.0`
  - `tri_map_quantile_support_gap`: `F-score = 0.999553`, `Chamfer = 0.020242`, `Comp-R = 1.000000`, `ghost_ratio = 0.607266`, `ghost_tail_ratio = 0.117678`, `trimap_delayed_mean = 0.0`, `trimap_dual_mean = 9.15`
  - `tri_map_top_tail_delayed`: `F-score = 0.999553`, `Chamfer = 0.020246`, `Comp-R = 1.000000`, `ghost_ratio = 0.607141`, `ghost_tail_ratio = 0.117683`, `trimap_delayed_mean = 1.35`, `trimap_dual_mean = 7.75`
- Bonn `balloon2`:
  - `dual_map_pfv_base_current`: `F-score = 0.630335`, `Chamfer = 0.099868`, `Comp-R = 0.701660`, `ghost_ratio = 0.142491`, `ghost_tail_ratio = 0.343565`, `trimap_delayed_mean = 0.0`, `trimap_dual_mean = 0.0`
  - `tri_map_quantile_support_gap`: `F-score = 0.630335`, `Chamfer = 0.099868`, `Comp-R = 0.701660`, `ghost_ratio = 0.142491`, `ghost_tail_ratio = 0.343565`, `trimap_delayed_mean = 0.0`, `trimap_dual_mean = 7.45`
  - `tri_map_top_tail_delayed`: `F-score = 0.630482`, `Chamfer = 0.099874`, `Comp-R = 0.701800`, `ghost_ratio = 0.142210`, `ghost_tail_ratio = 0.343899`, `trimap_delayed_mean = 1.1`, `trimap_dual_mean = 6.35`

### 结论
- 本轮首次成功把 `delayed-only` 分支真正打开：
  - TUM: `trimap_delayed_mean = 1.35`
  - Bonn: `trimap_delayed_mean = 1.1`
- 这说明：
  1. `quantile-calibrated` 路由并非只会打开 `dual`；
  2. 通过 top-tail escalation，strongest tail 确实可以被结构性剥离出 committed background；
  3. 到这一步为止，tri-map 的“真正分叉机制”已经被验证可行。
- 但 map-level 收益仍然**没有形成明确突破**：
  - TUM 变化极小，接近数值微扰；
  - Bonn 出现了轻微 mixed change：`F-score / Comp-R / ghost_ratio` 有极小幅改善，但 `Chamfer / ghost_tail_ratio / Acc` 未同步改善；
  - 当前幅度仍不足以支撑“已冲破 P10 目标”的判断。
- 更关键的是：`trimap_promoted_mean` 仍然几乎不变（TUM `77.7`，Bonn `46.15`），说明 delayed-only tail 虽然被打开了，但很可能又被 promotion 很快吸回 committed map，造成结构分叉时间太短，最终导出仍被重新抹平。

### 判定
- `top-tail delayed-only escalation` 记为本轮**有明确机制突破、但仍未形成稳定指标突破**的分支。
- 它是当前 support-gap / tri-map 主线上最关键的一步：
  - 第一次同时在 TUM/Bonn 打开 `delayed-only`；
  - 证明“strongest tail delayed split”这条方向是对的；
  - 但下一步必须处理 promotion 回流过快的问题。

### 下一步目标方案
- 下一步应转向：
  - **escalation-aware promotion hold / delayed residency hysteresis**
- 核心思路：
  1. 保留本轮 `top-tail delayed-only` 选择机制；
  2. 对被 escalation 的 delayed tail 施加短时 residency hold，或提高其 promotion 阈值；
  3. 避免它们在写入 delayed map 后过快被 promotion 回 committed map；
  4. 目标是把已经打开的 delayed-only 分叉“保持住足够长时间”，再观察是否能把当前极小 mixed change 放大成真正可见的 P10 指标收益。

## 2026-03-08 Escalation-Aware Promotion Hold / Delayed Residency Hysteresis Attempt

### 模块
- `promotion hold / hysteresis`
- 定位：在上一轮 `top-tail delayed-only escalation` 已经打开 delayed-only 分支的基础上，专门抑制 escalation tail 被过快 promotion 回 committed map。

### 代码落地
- `egf_dhmap3d/modules/associator.py`
- `egf_dhmap3d/core/types.py`
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/modules/updater.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `scripts/run_egf_3d_tum.py`

### 设计要点
- 本轮给被 `top-tail delayed-only` 选中的 measurement 增加 escalation 标记与 hold frames；
- 在 delayed map 写入时，把 escalation tail 的 `hold / hysteresis / route_score` 写入 delayed cell；
- 在 `promote_delayed_background_map()` 内部：
  - hold 期间直接阻止 promotion；
  - hold 结束后仍施加一段 promotion hysteresis：
    - 提高 promotion threshold；
    - 降低 blend；
    - 逐帧衰减。
- 目标是：
  1. 把上一轮已经打开的 delayed-only 分叉保留更久；
  2. 防止 strongest tail 刚进入 delayed map 又被立刻吸回 committed map；
  3. 观察这种“延长 delayed residency”的做法，能否把上一轮的极小 mixed change 放大成更稳定的 P10 收益。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_hold_tum/`
  - `output/post_cleanup/p10_trimap_hold_bonn/`
- 对比表：`output/summary_tables/p10_trimap_promotion_hold_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_top_tail_delayed`: `F-score = 0.999553`, `Chamfer = 0.020246`, `Comp-R = 1.000000`, `ghost_ratio = 0.607141`, `ghost_tail_ratio = 0.117683`, `trimap_delayed_mean = 1.35`, `trimap_promoted_mean = 77.7`
  - `tri_map_promotion_hold`: `F-score = 0.999553`, `Chamfer = 0.020248`, `Comp-R = 1.000000`, `ghost_ratio = 0.607150`, `ghost_tail_ratio = 0.117681`, `trimap_delayed_mean = 1.35`, `trimap_promoted_mean = 73.4`
- Bonn `balloon2`:
  - `tri_map_top_tail_delayed`: `F-score = 0.630482`, `Chamfer = 0.099874`, `Comp-R = 0.701800`, `ghost_ratio = 0.142210`, `ghost_tail_ratio = 0.343899`, `trimap_delayed_mean = 1.1`, `trimap_promoted_mean = 46.15`
  - `tri_map_promotion_hold`: `F-score = 0.630482`, `Chamfer = 0.099874`, `Comp-R = 0.701800`, `ghost_ratio = 0.142210`, `ghost_tail_ratio = 0.343899`, `trimap_delayed_mean = 1.1`, `trimap_promoted_mean = 42.55`
- 新增 residency 统计：
  - TUM：`trimap_hold_blocked_mean = 4.5`, `trimap_hold_mean = 0.1640`, `trimap_hysteresis_mean = 0.1037`
  - Bonn：`trimap_hold_blocked_mean = 3.6`, `trimap_hold_mean = 0.1552`, `trimap_hysteresis_mean = 0.1037`
- 这说明 hold/hysteresis 的确在发挥作用：
  - delayed-only tail 没有消失；
  - promotion 回流被真实压低；
  - 但 map-level 指标并没有同步放大。

### 结论
- 本轮 `promotion hold / hysteresis` 的**机制结论是正的**：
  - 它成功降低了 escalation tail 的 promotion 回流；
  - 证明上一轮的判断是对的：promotion rebound 确实存在，而且可以被抑制。
- 但它的**结果结论仍然偏负**：
  - TUM 只出现极小数值扰动；
  - Bonn 基本与上一轮 top-tail delayed-only 完全同级；
  - 指标增量没有被进一步放大。
- 这说明：
  1. “把 strongest tail 保留在 delayed map 更久”本身并不足以转化成显著收益；
  2. 当前更大的瓶颈很可能已经从 promotion 回流，转移到**导出路径**；
  3. 也就是说，即使 delayed tail 被保留住，如果 extraction/export 仍几乎只读 committed map，那么 P10 指标仍然不会被明显改变。

### 判定
- `promotion hold / hysteresis` 记为本轮**有明确机制验证价值、但未形成收益放大**的分支。
- 它进一步缩小了搜索空间：当前最值得怀疑的瓶颈已不再是“路由”或“promotion”，而是 delayed map 对最终 surface export 的参与度过低。

### 下一步目标方案
- 下一步应转向：
  - **residency-gated delayed export participation**
- 核心思路：
  1. 只对 hold/hysteresis 仍活跃、且通过严格前景/支撑约束的 delayed tail，允许有限度参与 export；
  2. 不是全量 delayed export，更不是恢复旧的 hole-only rescue；
  3. 而是把“已经被 top-tail + hold 证实值得保留的 delayed-only subset”作为一个受控导出支路；
  4. 目标是验证：一旦 delayed tail 真能进入最终 export，当前已被验证的机制分叉能否终于转化为可见的 P10 指标收益。

## 2026-03-08 Residency-Gated Delayed Export Participation Attempt

### 模块
- `residency-gated delayed export participation`
- 定位：在 `top-tail delayed-only + promotion hold/hysteresis` 已经把 delayed-only tail 保留下来的前提下，允许其中一小部分仍处于 residency 活跃期的 delayed tail 受控参与最终 export。

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `egf_dhmap3d/modules/pipeline.py`
- `scripts/run_egf_3d_tum.py`

### 设计要点
- 本轮不是开放整个 delayed map 的 export；
- 而是只允许同时满足以下条件的 delayed subset 参与最终 export：
  1. 处于 `hold / hysteresis / escalated` 活跃期；
  2. delayed support 足够高；
  3. route score 足够高；
  4. committed map 在该位置附近没有很强的已导出支撑，或该点本来就不在 committed export 里。
- 目标是验证：如果 delayed tail 真能进入最终 export，当前已经建立的 tri-map 结构分叉是否终于会转化成可见的 P10 指标收益。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_export_tum/`
  - `output/post_cleanup/p10_trimap_export_bonn/`
- 对比表：`output/summary_tables/p10_trimap_residency_export_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_promotion_hold`: `F-score = 0.999553`, `Chamfer = 0.020248`, `Comp-R = 1.000000`, `ghost_ratio = 0.607150`, `ghost_tail_ratio = 0.117681`, `trimap_export_added = 0`
  - `tri_map_residency_export`: `F-score = 0.999553`, `Chamfer = 0.020242`, `Comp-R = 1.000000`, `ghost_ratio = 0.607266`, `ghost_tail_ratio = 0.117678`, `trimap_export_added = 18`
- Bonn `balloon2`:
  - `tri_map_promotion_hold`: `F-score = 0.630482`, `Chamfer = 0.099874`, `Comp-R = 0.701800`, `ghost_ratio = 0.142210`, `ghost_tail_ratio = 0.343899`, `trimap_export_added = 0`
  - `tri_map_residency_export`: `F-score = 0.630335`, `Chamfer = 0.099868`, `Comp-R = 0.701660`, `ghost_ratio = 0.142491`, `ghost_tail_ratio = 0.343565`, `trimap_export_added = 14`
- 说明 export 支路本身已经真正生效：
  - TUM：`trimap_export_added = 18`, `trimap_export_candidates = 165`, `trimap_export_residency = 24`
  - Bonn：`trimap_export_added = 14`, `trimap_export_candidates = 139`, `trimap_export_residency = 21`
- 但最终指标表现非常关键：
  - TUM：基本**回到当前 baseline**；
  - Bonn：也基本**回到当前 baseline**。

### 结论
- 本轮的结论非常有价值：
  - delayed tail 不仅能被路由出去、保留下来，而且现在**确实能进入最终 export**；
  - 但即便如此，最终指标仍几乎回到 baseline，说明“仅仅把 delayed tail 以附加点的形式加进 export”仍不足以改变主导表面的几何统计。
- 这意味着：
  1. 当前瓶颈已不再是“delayed tail 无法进入 export”；
  2. 而是 export 里 **committed surface 仍然主导几何**；
  3. delayed tail 作为附加点被加入时，只产生了极小或可忽略的统计影响。
- 因而，下一步不能再是“多加一点 delayed points”，而必须转向**export-time local replacement / shadow suppression**：
  - 在 delayed tail 参与 export 的局部邻域，对 committed export 做有控制的替换或抑制；
  - 让 delayed branch 不只是“附加”，而是真正参与最终表面的局部主导权竞争。

### 判定
- `residency-gated delayed export participation` 记为本轮**有明确链路打通价值、但结果仍未突破**的分支。
- 它进一步把问题钉死在 export-time 主导权上：当前需要的不再是更多 delayed 点，而是 delayed 点对 committed 点的局部替换权。

### 下一步目标方案
- 下一步应转向：
  - **export-time local replacement around delayed tail**
- 核心思路：
  1. 保留本轮 residency-gated delayed export subset；
  2. 对这些 delayed export 点的局部邻域，抑制或替换 nearby committed export points；
  3. 不是全局替换，而是仅在 delayed tail 覆盖的局部小球/索引邻域内做 controlled replacement；
  4. 目标是验证：一旦 delayed branch 获得局部表面主导权，当前已经打通的 tri-map 机制能否终于转化成真正可见的 P10 指标收益。

## 2026-03-08 Export-Time Local Replacement Around Delayed Tail Attempt

### 模块
- `export-time local replacement around delayed tail`
- 定位：不再只把 delayed tail 作为 export 的附加点，而是在 delayed tail 局部邻域内，受控抑制/替换 nearby committed export points，让 delayed branch 获得局部表面主导权。

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `scripts/run_egf_3d_tum.py`

### 设计要点
- 上一轮已经证明 delayed tail 可以进入 export，但只是“附加点”；
- 本轮进一步改为：
  1. 先选出 residency-active delayed export subset；
  2. 在其邻域内搜索 nearby committed export points；
  3. 若 committed 点局部支撑不够强，则在局部小球内删去少量 committed points；
  4. 再把 delayed tail 加入 export。
- 目标是：让 delayed tail 从“附加参与”变成“局部表面主导”，观察这是否能把 tri-map 机制分叉转化成真正可见的 P10 指标变化。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_replace_tum/`
  - `output/post_cleanup/p10_trimap_replace_bonn/`
- 对比表：`output/summary_tables/p10_trimap_local_replacement_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_residency_export`: `F-score = 0.999553`, `Chamfer = 0.020242`, `Comp-R = 1.000000`, `ghost_ratio = 0.607266`, `ghost_tail_ratio = 0.117678`, `trimap_export_added = 18`, `trimap_export_replaced = 0`
  - `tri_map_local_replacement`: `F-score = 0.999552`, `Chamfer = 0.020254`, `Comp-R = 1.000000`, `ghost_ratio = 0.607098`, `ghost_tail_ratio = 0.117708`, `trimap_export_added = 18`, `trimap_export_replaced = 62`
- Bonn `balloon2`:
  - `tri_map_residency_export`: `F-score = 0.630335`, `Chamfer = 0.099868`, `Comp-R = 0.701660`, `ghost_ratio = 0.142491`, `ghost_tail_ratio = 0.343565`, `trimap_export_added = 14`, `trimap_export_replaced = 0`
  - `tri_map_local_replacement`: `F-score = 0.630254`, `Chamfer = 0.099880`, `Comp-R = 0.701600`, `ghost_ratio = 0.142359`, `ghost_tail_ratio = 0.343798`, `trimap_export_added = 14`, `trimap_export_replaced = 42`

### 结论
- 本轮验证了一个非常关键的事实：
  - delayed tail 不仅能进入 export；
  - 它也**确实可以在局部邻域内替换 committed export points**；
  - replacement 统计已经非零且不小：
    - TUM：`62` 个 committed export points 被替换；
    - Bonn：`42` 个 committed export points 被替换。
- 但结果层面，这轮是一个**偏负的结构结果**：
  - replacement 确实让表面主导权发生了变化；
  - 但当前这种“半径邻域 + 硬替换”过于粗糙；
  - 指标表现变成了典型的几何扰动：
    - `ghost_ratio` 有轻微改善；
    - 但 `Chamfer / ghost_tail / F-score` 整体没有同步改善，甚至略有恶化。
- 这说明：
  1. 方向本身是对的——export-time 主导权确实是当前最后一道关键门；
  2. 但当前 replacement 机制太“硬”，缺少 delayed-vs-committed 的精细竞争；
  3. 不能只按半径删除 committed points，而必须让 delayed/committed 在局部做**一对一、带几何一致性约束的竞争替换**。

### 判定
- `export-time local replacement around delayed tail` 记为本轮**有关键链路验证价值、但当前实现过于粗糙，未形成收益**的分支。
- 它进一步明确了：P10 这条 tri-map 主线真正需要的不是“更多 delayed 点”，也不是“更大 replacement 半径”，而是**replacement-time competition scoring**。

### 下一步目标方案
- 下一步应转向：
  - **competition-scored local replacement**
- 核心思路：
  1. 保留当前 residency-gated delayed export subset；
  2. 不再按纯半径硬删 committed points；
  3. 而是在 delayed point 与 nearby committed point 之间构造局部竞争分数：
     - delayed residency / route score
     - committed static support
     - 几何距离
     - 法向一致性
  4. 只有 delayed 明确胜出时，才进行一对一或小规模替换；
  5. 目标是把本轮“表面真的会动”的证明，进一步变成“表面只在正确的地方动”，从而争取第一次稳定的 P10 指标正增益。

## 2026-03-08 Competition-Scored Local Replacement Attempt

### 模块
- `competition-scored local replacement`
- 定位：延续上一轮 `export-time local replacement`，但不再按“半径邻域 + 硬替换”删除 committed export points，而是对 delayed point 与 nearby committed point 逐点计算 competition score，只有 delayed 明确胜出时才替换。

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `scripts/run_egf_3d_tum.py`

### 设计要点
- 本轮在 replacement 时加入了更细的竞争分数：
  - delayed support
  - route score
  - residency strength
  - 法向一致性
  - committed static support
  - 局部距离惩罚
- 与上一轮相比，目标不是“替换更多 committed 点”，而是“只替换 delayed 确实胜出的 committed 点”。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_compete_tum/`
  - `output/post_cleanup/p10_trimap_compete_bonn/`
- 对比表：`output/summary_tables/p10_trimap_competition_replacement_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_local_replacement`: `F-score = 0.999552`, `Chamfer = 0.020254`, `ghost_ratio = 0.607098`, `ghost_tail_ratio = 0.117708`, `trimap_export_replaced = 62`
  - `tri_map_competition_replacement`: `F-score = 0.999553`, `Chamfer = 0.020247`, `ghost_ratio = 0.607093`, `ghost_tail_ratio = 0.117741`, `trimap_export_replaced = 32`, `trimap_export_compete_mean = 0.1236`, `trimap_export_normal_cos_mean = 0.9047`
- Bonn `balloon2`:
  - `tri_map_local_replacement`: `F-score = 0.630254`, `Chamfer = 0.099880`, `ghost_ratio = 0.142359`, `ghost_tail_ratio = 0.343798`, `trimap_export_replaced = 42`
  - `tri_map_competition_replacement`: `F-score = 0.630273`, `Chamfer = 0.099876`, `ghost_ratio = 0.142373`, `ghost_tail_ratio = 0.343714`, `trimap_export_replaced = 26`, `trimap_export_compete_mean = 0.1075`, `trimap_export_normal_cos_mean = 0.8668`
- 这说明：
  - replacement 从粗暴硬删变成了更节制的局部竞争替换；
  - 被替换的 committed 点数量明显下降；
  - 且替换对的法向一致性较高（TUM `0.90`，Bonn `0.87`）。

### 结论
- 本轮 `competition-scored local replacement` 是一个**比上一轮更合理的负结果**：
  - 它确实让 replacement 更精细；
  - 相比上一轮硬替换，TUM/Bonn 的几何扰动都更小；
  - Bonn 上也出现了比上一轮略更平衡的 mixed change。
- 但到结果层面，它仍然**没有形成净正收益**：
  - TUM 仍然只是极小扰动；
  - Bonn 虽然相对硬替换更稳，但仍未超越 `residency_export` / baseline；
  - 说明当前 export-time 替换这条线已经很接近“只能做微调”的上限。
- 这进一步说明：
  1. 当前 tri-map 主线的最后一个主要瓶颈不是“替换太粗”；
  2. 而是 delayed branch 本身承载的几何质量 / 几何位置，还不足以在局部竞争中稳定赢过 committed surface；
  3. 因此，再继续在 export 末端做 replacement trick，预期收益会越来越小。

### 判定
- `competition-scored local replacement` 记为本轮**有精细化机制收益、但未形成指标突破**的分支。
- 它基本宣告：当前 tri-map/export competition 这条子线已接近边际收益衰减区。

### 下一步目标方案
- 下一步应转向：
  - **delayed-branch geometry refinement before export competition**
- 核心思路：
  1. 不再只在 export 末端竞争；
  2. 先提升 delayed branch 自身的几何质量或几何定位稳定性；
  3. 再让它去和 committed branch 做 export competition；
  4. 换句话说，下一步重点应从“谁来主导 export”转向“delayed branch 先变得足够像一个高质量 surface bank”。

## 2026-03-08 Delayed-Branch Geometry Refinement Before Export Competition Attempt

### 模块
- `delayed-branch geometry refinement before export competition`
- 定位：不再继续强化 export 末端替换规则，而是在 delayed branch 自身先做局部几何 refinement（法向 + 零交叉点位），再把 refined delayed surface 送入 export competition。

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `scripts/run_egf_3d_tum.py`

### 设计要点
- 本轮在 delayed export 点参与 competition 之前，增加了一个 delayed-branch 局部 refinement：
  - 从 delayed map 邻域聚合 `g_mean` 做法向平滑；
  - 对 `phi_static / phi_bg / phi_geo` 做局部加权，构造 refined zero-crossing；
  - 最终得到 refined point / refined normal，再参加 delayed-vs-committed 的局部 competition。
- 目标是让 delayed branch 先变成一个更像“高质量 surface bank”的分支，再去争 export 主导权。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_refine_tum/`
  - `output/post_cleanup/p10_trimap_refine_bonn/`
- 对比表：`output/summary_tables/p10_trimap_delayed_refine_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_competition_replacement`: `F-score = 0.999553`, `Chamfer = 0.020247`, `ghost_ratio = 0.607093`, `ghost_tail_ratio = 0.117741`, `trimap_export_replaced = 32`
  - `tri_map_delayed_refine`: `F-score = 0.999553`, `Chamfer = 0.020247`, `ghost_ratio = 0.607056`, `ghost_tail_ratio = 0.117739`, `trimap_export_replaced = 31`, `trimap_delayed_refine_offset_mean = 0.00388`, `trimap_delayed_refine_normal_cos_mean = 0.9944`
- Bonn `balloon2`:
  - `tri_map_competition_replacement`: `F-score = 0.630273`, `Chamfer = 0.099876`, `ghost_ratio = 0.142373`, `ghost_tail_ratio = 0.343714`, `trimap_export_replaced = 26`
  - `tri_map_delayed_refine`: `F-score = 0.630273`, `Chamfer = 0.099876`, `ghost_ratio = 0.142373`, `ghost_tail_ratio = 0.343714`, `trimap_export_replaced = 26`, `trimap_delayed_refine_offset_mean = 0.00038`, `trimap_delayed_refine_normal_cos_mean = 0.9987`
- 说明 delayed branch refinement 在几何层面是有效的：
  - refinement offset 很小，说明它在做温和稳定化而不是大幅扭曲几何；
  - refinement 后的法向一致性非常高（TUM `0.9944`，Bonn `0.9987`）；
  - export competition 仍在正常工作。

### 结论
- 本轮 `delayed-branch geometry refinement` 的结论是：
  - 它成功把 delayed branch 做得更平滑、更稳定；
  - 但这种 refinement 并没有把 export competition 的结果明显推向净收益；
  - 指标变化仍停留在极小 mixed change 范围内。
- 这说明：
  1. delayed branch 的几何质量确实是一个问题，但当前的 refinement 还只是“局部平滑级别”的改进；
  2. delayed branch 与 committed branch 之间更深层的差异，可能不是点位/法向噪声，而是**surface field 本身的状态表达不够独立**；
  3. 继续只在 export 前做局部 refinement，预期收益也会逐渐变小。

### 判定
- `delayed-branch geometry refinement before export competition` 记为本轮**有稳定化收益、但未形成指标突破**的分支。
- 它进一步说明：如果要继续沿 delayed branch 深挖，下一步不能只做 point-level refinement，而应考虑 delayed branch 自身的 **surface field / readout state** 级增强。

### 下一步目标方案
- 下一步应转向：
  - **delayed-branch dedicated surface readout / banked field refinement**
- 核心思路：
  1. 不只对 delayed exported points 做后处理；
  2. 而是在 delayed branch 内部单独建立更稳定的 surface readout（例如 dedicated surface bank / refined persistent readout）；
  3. 让 delayed branch 在 export 之前就拥有更清晰的 surface representation；
  4. 再与 committed branch 做 competition。

## 2026-03-08 Delayed-Branch Dedicated Surface Readout / Banked Field Refinement Attempt

### 模块
- `delayed-branch dedicated surface readout / banked field refinement`
- 定位：不再只对 delayed export 点做 point-level refinement，而是给 delayed branch 一个更独立的 dedicated surface readout：
  - 先在 delayed map 内部做 banked field 读出；
  - 再把这个 delayed-specific surface representation 送入 export competition。

### 代码落地
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/P10_method/tri_map.py`
- `scripts/run_egf_3d_tum.py`

### 设计要点
- 本轮把 delayed branch 的 surface readout 从“通用 extractor + delayed postprocess”切换到：
  1. delayed map 内部单独计算 local banked field；
  2. 用 `phi_static / phi_bg / phi_geo` 做 delayed-specific field readout；
  3. 用 delayed 邻域做 dedicated normal / zero-crossing 估计；
  4. 再把这个 banked delayed surface 去做 export competition。
- 目标是让 delayed branch 在 export 之前就先拥有更独立的 surface representation，而不是继续依赖 committed-style extractor 再做末端修补。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_bank_tum/`
  - `output/post_cleanup/p10_trimap_bank_bonn/`
- 对比表：`output/summary_tables/p10_trimap_delayed_banked_readout_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_competition_replacement`: `F-score = 0.999553`, `Chamfer = 0.020247`, `ghost_ratio = 0.607093`, `ghost_tail_ratio = 0.117741`, `trimap_export_added = 18`, `trimap_export_replaced = 32`
  - `tri_map_delayed_bank_readout`: `F-score = 0.999553`, `Chamfer = 0.020247`, `ghost_ratio = 0.607114`, `ghost_tail_ratio = 0.117692`, `trimap_export_added = 4`, `trimap_export_replaced = 8`, `trimap_delayed_bank_points = 14`
- Bonn `balloon2`:
  - `tri_map_competition_replacement`: `F-score = 0.630273`, `Chamfer = 0.099876`, `ghost_ratio = 0.142373`, `ghost_tail_ratio = 0.343714`, `trimap_export_added = 14`, `trimap_export_replaced = 26`
  - `tri_map_delayed_bank_readout`: `F-score = 0.630477`, `Chamfer = 0.099874`, `ghost_ratio = 0.142234`, `ghost_tail_ratio = 0.343911`, `trimap_export_added = 2`, `trimap_export_replaced = 4`, `trimap_delayed_bank_points = 6`
- 统计上，本轮 delayed branch 的独立性确实变强了：
  - TUM：`trimap_delayed_bank_points = 14`, `trimap_delayed_bank_conf_mean = 0.3871`, `trimap_delayed_bank_residency_mean = 0.5363`
  - Bonn：`trimap_delayed_bank_points = 6`, `trimap_delayed_bank_conf_mean = 0.3494`, `trimap_delayed_bank_residency_mean = 0.2130`
- 说明 dedicated banked readout 的确让 delayed branch 变得更“克制”和更独立，而不是继续大规模动 export surface。

### 结论
- 本轮是一个很典型的“更干净但不更强”的结果：
  - delayed branch 的 surface readout 确实被独立出来了；
  - 干预点数明显减少；
  - 但最终指标并没有被显著抬升。
- 更具体地说：
  1. TUM 上它比前几轮更接近“无害扰动”，但没有形成净收益；
  2. Bonn 上出现了比 `competition replacement` 更接近 `top-tail delayed-only` 的 mixed positive pattern，但仍然很小，不足以构成真正突破；
  3. 这说明 delayed branch 的 dedicated readout 是对的方向，但当前 bank 还太弱、点太少，仍不足以主导局部表面统计。

### 判定
- `delayed-branch dedicated surface readout / banked field refinement` 记为本轮**有结构正确性增益、但未形成突破**的分支。
- 它说明 delayed branch 这条线如果继续走下去，应该从“更聪明地导出少量点”升级到“更强地积累一个 delayed-specific persistent surface bank”。

### 下一步目标方案
- 下一步应转向：
  - **persistent delayed surface bank accumulation**
- 核心思路：
  1. 不只在 export 时临时读 delayed field；
  2. 而是在 delayed branch 内部显式积累一个更稳定的 persistent surface bank；
  3. 让 delayed branch 拥有足够强的 surface mass，再去和 committed branch 做 export competition；
  4. 目标是从“少量干预”提升到“足够强的 delayed-specific geometry 主导权”。

## 2026-03-08 Persistent Delayed Surface Bank Accumulation Attempt

### 模块
- `persistent delayed surface bank accumulation`
- 定位：在上一轮 `delayed-branch dedicated surface readout / banked field refinement` 的基础上，不再只在 export 时临时读取 delayed field，而是在 delayed branch 内部显式积累一个 persistent surface bank，再从该 bank 做 delayed readout。

### 代码落地
- `egf_dhmap3d/core/types.py`
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/core/voxel_hash.py`
- `egf_dhmap3d/modules/updater.py`
- `egf_dhmap3d/P10_method/tri_map.py`

### 设计要点
- 本轮新增 delayed bank state：
  - `phi_delayed_bank`
  - `phi_delayed_bank_w`
  - `rho_delayed_bank`
  - `delayed_bank_conf / age / active`
- 在 delayed map 写入阶段做 bank accumulation；
- 在 delayed export readout 时优先读 persistent bank，而不是只依赖临时邻域读出。
- 目标是：让 delayed branch 具备真正的持久 surface mass，而不再只是 export 阶段的局部临时重建。

### focused probe
- TUM scene: `rgbd_dataset_freiburg3_walking_xyz`
- Bonn scene: `rgbd_bonn_balloon2`
- Frames: `20`
- Protocols: `oracle` (TUM) / `slam` (Bonn)
- 输出目录：
  - `output/post_cleanup/p10_trimap_pbank_tum/`
  - `output/post_cleanup/p10_trimap_pbank_bonn/`
- 对比表：`output/summary_tables/p10_trimap_persistent_delay_bank_probe_tum_bonn_compare.csv`

### 结果
- TUM `walking_xyz`:
  - `tri_map_delayed_bank_readout`: `F-score = 0.999553`, `Chamfer = 0.020247`, `ghost_ratio = 0.607114`, `ghost_tail_ratio = 0.117692`, `trimap_export_added = 4`, `trimap_export_replaced = 8`, `trimap_delayed_bank_points = 14`
  - `tri_map_persistent_delay_bank`: **数值基本完全相同**
- Bonn `balloon2`:
  - `tri_map_delayed_bank_readout`: `F-score = 0.630477`, `Chamfer = 0.099874`, `ghost_ratio = 0.142234`, `ghost_tail_ratio = 0.343911`, `trimap_export_added = 2`, `trimap_export_replaced = 4`, `trimap_delayed_bank_points = 6`
  - `tri_map_persistent_delay_bank`: **数值基本完全相同**

### 结论
- 本轮 `persistent delayed surface bank accumulation` 的结论是一个非常明确的负结果：
  - 在当前 focused probe 条件下，它与上一轮 `dedicated banked readout` 几乎完全等价；
  - 说明当前新增的 persistent bank state 还没有提供额外信息量；
  - delayed branch 的瓶颈并不是“缺少 bank 存储位”，而是这些 bank 中并没有被写入比现有 delayed readout 更强的几何内容。
- 换句话说：
  1. delayed branch 当前不是“没有记住自己”；
  2. 而是“记住的东西还不够有区分度”；
  3. 如果要继续深挖 delayed branch，就不能只做 persistence，还必须让写入 delayed bank 的 observation/geometry target 本身更强。

### 判定
- `persistent delayed surface bank accumulation` 记为本轮**无额外增益的结构验证分支**。
- 它进一步表明：当前 tri-map/delayed 主线如果继续推进，下一步必须转向 delayed branch 的 **write-time target synthesis / dedicated observation model**，而不是继续堆积存储或 export 侧技巧。

### 下一步目标方案
- 下一步应转向：
  - **delayed-branch write-time target synthesis**
- 核心思路：
  1. 不再只累积当前 delayed cell 已有的 `phi_static / phi_bg / phi_geo`；
  2. 而是在 delayed branch 写入期就专门构造 delayed-specific surface target；
  3. 让 delayed bank 写入的就不是 committed 的弱变体，而是 delayed branch 自己的几何假设；
  4. 目标是提升 delayed bank 的信息增益，再回到 export competition 看是否能真正带来可见的 P10 指标收益。
