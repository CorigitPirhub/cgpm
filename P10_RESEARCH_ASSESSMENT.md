# P10 阶段调研评估报告（代码审阅 + 顶会/顶刊文献映射）

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

## 1. 结论先行

当前 P10 未过线的核心原因，不是参数没调够，而是**算子职责错位**：

1. **Accuracy 偏高（更差）问题主要发生在融合阶段**，但目前多数抑制逻辑在提取阶段（SNEF/STCG 表面筛选）才生效。后处理可以降 ghost，却难以回正已写入体素的系统性零交叉偏置。
2. **Ghost 抑制与完整性耦合过强**：一旦强化全局筛除，Comp-R 很快塌陷；保持 Comp-R 时 ghost 降幅不足（Bonn 侧最明显）。
3. 以 `local_mapping_precision_profile` 为代表的多轮 sweep 已证明：在现有算子族内继续微调，边际收益很低，难以同时满足 P10 三条硬约束。

因此，P10 要过线必须走**模块级改造**（state/update/extraction 三处联动），而非继续参数扫面。

---

## 2. 当前代码与指标现状（已审阅证据）

### 2.1 P10 硬门槛（脚本硬编码）
- `scripts/run_p10_precision_profile.py:2599`：TUM oracle `acc_cm <= 1.80 && comp_r_5cm >= 95`
- `scripts/run_p10_precision_profile.py:2600`：Bonn slam `acc_cm <= 2.60 && comp_r_5cm >= 95`
- `scripts/run_p10_precision_profile.py:2601`：`min(ghost_reduction_vs_tsdf) >= 0.35`

### 2.2 当前最优结果（仍未过线）
来源：`output/summary_tables/local_mapping_precision_profile.csv`
- `lzcd_only_b`: TUM Acc=2.789, Bonn Acc=3.425, TUM/Bonn Comp-R≈100%, ghost 最小降幅=0.130
- `baseline_relaxed`: TUM Acc=2.911, Bonn Acc=3.438, ghost 最小降幅=0.137

来源：`output/summary_tables/local_mapping_precision_profile_step10_dyn_tight_quick12.csv`
- `baseline_joint_h_dyn05`: TUM Acc=2.458（有所下降），Bonn Acc=3.043（仍>2.60），ghost 最小降幅=0.210（仍<0.35）

来源：`output/summary_tables/p10_bonn_route_sweep.csv`
- Bonn 单侧可到 `acc=1.676, comp_r=96.18`，但 `ghost_reduction=27.3%` 仍低于 35%。

### 2.3 代码层瓶颈定位

1. 融合写入仍是单通道加权均值主导
- `egf_dhmap3d/modules/updater.py:83-99`
- `phi/phi_geo` 都是加权平均；动态矛盾主要以 `free/surf evidence` 累积，不会强约束“是否允许写入静态几何通道”。

2. 动态判别大多是“写后判别”
- `egf_dhmap3d/modules/updater.py:226-299`（`d_score` 与 `stcg_score`）
- 这些分数主要影响后续遗忘/提取门控，不能阻止错误观测先污染 `phi_geo`。

3. 提取端已非常复杂但治标不治本
- `egf_dhmap3d/core/voxel_hash.py:180-535`
- SNEF + two-stage + STCG + adaptive 都在点提取阶段发力，更多是在“删点/留点”上做权衡。

4. 关联端已有异方差，但仍缺“姿态不确定度入模”
- `egf_dhmap3d/modules/associator.py:111-167`
- 仅用局部梯度协方差与入射角/深度做观测噪声；未把 SLAM 位姿协方差传播到 `sigma_d`。

5. LZCD 已做但仍偏弱
- `egf_dhmap3d/modules/updater.py:301-413`
- 当前是体素局部 EMA 去偏（标量 bias），对“块级系统偏厚/偏移”校正能力有限。

---

## 3. 顶会/顶刊文献调研（与 P10 直接相关）

### 3.1 动态场景鲁棒建图主线

1. DynaSLAM (RA-L 2018)
- 核心：语义+几何动态剔除，静态背景重建/修补。
- 链接：https://arxiv.org/abs/1806.05620

2. Co-Fusion (ICRA 2017)
- 核心：多模型分割与独立跟踪，物体级融合。
- 链接：https://arxiv.org/abs/1706.06629

3. MID-Fusion (ICRA 2019)
- 核心：实例级八叉树体素映射，动态对象与背景解耦。
- 链接：https://arxiv.org/abs/1812.07976

4. StaticFusion (ICRA 2018)
- 核心：动态环境中静态背景建图与鲁棒跟踪。
- 链接：https://raluca-scona.github.io/docs/conference-papers/2018_icra_staticfusion.pdf

5. ReFusion (IROS 2019)
- 核心：利用配准残差 + 显式 free-space 建模识别动态。
- 链接：https://arxiv.org/abs/1905.02082

6. MaskFusion (ISMAR 2018)
- 核心：实例分割驱动对象级融合。
- 链接：https://arxiv.org/abs/1804.09194

7. RoDyn-SLAM (RA-L 2024)
- 核心：神经场下的动态/静态分解与鲁棒跟踪。
- 链接：https://arxiv.org/abs/2407.01303

### 3.2 几何融合与不确定度建模主线

1. KinectFusion (ISMAR 2011)
- TSDF 里程碑，确立了体素融合范式。
- 链接：https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/ismar_2011.pdf

2. Voxel Hashing (SIGGRAPH Asia 2013)
- 稀疏哈希体素，解决规模化。
- 代码/论文入口：https://github.com/niessner/VoxelHashing

3. ElasticFusion (RSS 2015)
- 高质量局部致密建图，置信驱动更新思想。
- 链接：https://www.roboticsproceedings.org/rss11/p01.html

4. BundleFusion (SIGGRAPH/TOG 2017)
- 通过重积分抑制历史融合误差。
- 链接：https://arxiv.org/abs/1604.01093

5. Voxblox (IROS 2017)
- 增量 TSDF/ESDF 融合与高效更新。
- 链接：https://arxiv.org/abs/1611.03631

6. PSDF Fusion (ECCV 2018)
- 概率化 SDF 融合，显式建模不确定度/离群。
- 链接：https://arxiv.org/abs/1807.11034

7. Probabilistic Volumetric Fusion (WACV 2023)
- 把深度与位姿不确定度传播到体融合。
- 链接：https://arxiv.org/abs/2210.01276

8. Panoptic Multi-TSDFs (ICRA 2022)
- 多 TSDF 子图 + 长期动态一致性。
- 链接：https://www.microsoft.com/en-us/research/publication/panoptic-multi-tsdfs-a-flexible-representation-for-online-multi-resolution-volumetric-mapping-and-long-term-dynamic-scene-consistency/

### 3.3 神经隐式/高保真局部映射参考

1. NICE-SLAM (CVPR 2022): https://arxiv.org/abs/2112.12130
2. Co-SLAM (CVPR 2023): https://arxiv.org/abs/2304.14377
3. Point-SLAM (CVPR 2023): https://arxiv.org/abs/2304.04278
4. Vox-Fusion (ISMAR 2022, 代码入口): https://github.com/zju3dv/Vox-Fusion

---

## 4. 与现有实现的“重复/空白”对照

### 4.1 已覆盖但仍偏工程化的点
- 局部去偏（LZCD）已存在，但是体素级偏置 EMA，尚未形成块级 MAP 去偏。
- 时空矛盾（STCG）已存在，但主要作用在提取门控，不在“写入判定”层闭环。
- SNEF 两阶段筛选已存在，但本质仍是后处理筛点。

### 4.2 文献已验证但你当前缺失的关键机制

1. **延迟提交/重积分机制**（BundleFusion 思路）
- 当前缺“先缓存后确认”的静态提交流程，动态污染仍会直接写入。

2. **显式概率融合（含位姿不确定度）**（PSDF/WACV 2023）
- 当前未把 pose covariance 显式合入 SDF 融合方差。

3. **多假设几何状态**（Panoptic Multi-TSDF / object-level mapping 类思想）
- 当前是单静态几何主通道，缺 `static vs transient` 双假设状态。

4. **动态判别前移到融合入口**（ReFusion/StaticFusion 的核心经验）
- 当前动态判别偏后置，导致“先污染后清理”。

---

## 5. P10 可行的模块化改造方案（非参数微调）

## 5.1 M1: 双假设体素状态 + 延迟静态提交（DH-ESF）

### 思想
为每体素维护两条几何假设：
- `phi_static, w_static`
- `phi_transient, w_transient`
并维护 `p_static`（静态后验概率）。

观测先进入 transient；只有在连续时序一致时，才“提交”到 static。这样动态/矛盾观测不会直接污染静态几何。

### 代码位点
- `egf_dhmap3d/core/types.py`: 新增 `phi_static/phi_transient/w_static/w_transient/p_static/commit_count`
- `egf_dhmap3d/modules/updater.py`: 改写 `_integrate_measurement`，增加 delayed-commit 逻辑
- `egf_dhmap3d/core/voxel_hash.py`: 提取默认读 `phi_static`，`phi_transient` 仅作辅助

### 预期收益
- Acc 明显下降（去除动态污染偏置）
- Comp-R 可维持（transient 保覆盖，static 保质量）
- ghost 可通过 `p_static` 低置信过滤提升

### 风险
- 状态维度翻倍，内存/更新复杂度上升

---

## 5.2 M2: 位姿不确定度传播的异方差融合（PUF）

### 思想
把位姿协方差 `Sigma_pose` 投影到法向方向，得到额外测量方差：
- `sigma_pose_n^2 = n^T Sigma_t n + J_rot Sigma_r J_rot^T`
- `sigma_total^2 = sigma_depth^2 + sigma_n^2 + sigma_pose_n^2`

用 `sigma_total` 重写 `w_obs`，降低位姿噪声导致的表面“厚化”。

### 代码位点
- `egf_dhmap3d/modules/pipeline.py`: 传递当前 pose covariance 到 updater
- `egf_dhmap3d/modules/updater.py`: `_integrate_measurement` 中融合权重改为 pose-aware
- `egf_dhmap3d/modules/associator.py`: 统一量纲/噪声项，避免重复计噪

### 预期收益
- 直接打 Acc（尤其 Bonn slam 侧）
- 对 Comp-R 影响可控（不是删点，而是重加权）

### 风险
- 需要检查协方差尺度标定，防止过度保守

---

## 5.3 M3: 块级鲁棒零交叉去偏（LZCD++）

### 思想
把现有体素 EMA 去偏升级为**块级 IRLS 优化**：
在局部块内同时估计 `(bias b, scale s)`，目标是最小化零交叉一致性残差 + 法向一致正则。

相比当前 `phi <- phi - gain*bias`，LZCD++ 可以修正“整块偏厚/偏移”的系统误差。

### 代码位点
- `egf_dhmap3d/modules/updater.py`: 新增 `_local_zero_crossing_debias_block_irls`
- `egf_dhmap3d/core/voxel_hash.py`: 缓存块邻接图/活跃块列表，降低每帧求解开销

### 预期收益
- TUM oracle Acc 端最关键模块（从 2.4~2.9 cm 向 1.x cm 逼近）

### 风险
- 若正则太弱，可能引入局部波纹；需配合块级平滑先验

---

## 5.4 M4: 反事实自由空间记忆门控（CFM-STCG）

### 思想
在 STCG 基础上增加“反事实可见性”记忆：
- 若某体素被多次穿透射线判为 free，且后方静态表面持续稳定，则该体素 `p_static` 快速衰减；
- 若该体素再次被观测到稳定表面，则允许恢复（防误杀）。

这比一次性 ray-clear 更稳，且不会强行清空导致 Comp-R 崩。

### 代码位点
- `egf_dhmap3d/modules/updater.py`: 扩展 `_raycast_clear` 为可逆记忆更新（free-hit / occ-hit）
- `egf_dhmap3d/core/types.py`: 新增 `free_hit_ema`, `occ_hit_ema`, `visibility_contradiction`
- `egf_dhmap3d/core/voxel_hash.py`: 提取时将 contradiction 作为 soft penalty，而非硬删

### 预期收益
- Ghost 降幅从 22~27% 提升到 >=35% 的最关键路径
- Comp-R 更容易守住 95%

### 风险
- 需要稳定的射线采样策略，否则噪声会被误记忆

---

## 5.5 M5（可选）: 动静解耦提取头（D-Head）

### 思想
提取时不再单头 hard gate，而是输出：
- `static_head`（主输出，投稿主表）
- `full_head`（调试与可解释性）

并在主表只评 static head，full head 用于分析覆盖度。

### 价值
- 明确“质量 vs 覆盖”的可解释边界，减小审稿歧义。

---

## 6. 对“创新性而非工程拼装”的评估

若按 `M1 + M2 + M3 + M4` 组合推进，可形成一条可写入论文方法章节的独立故事：

1. **状态创新**：双假设体素静/动态分量（不是单 TSDF）
2. **估计创新**：位姿不确定度入模的异方差融合
3. **几何创新**：块级鲁棒零交叉去偏（LZCD++）
4. **时序创新**：反事实自由空间记忆门控（可逆，不是硬删除）

这四项并非单参数增强，而是 state + update + extraction 的结构升级。

---

## 7. 建议执行顺序（面向 P10）

1. **阶段 A（先打 Acc）**
- 上 M2 + M3，固定现有 ghost 策略，目标：
  - TUM oracle `acc_cm <= 2.1`（中间门槛）
  - Comp-R 不低于 98%

2. **阶段 B（再打 Ghost）**
- 上 M1 + M4，保持 A 阶段 Acc 不显著回退，目标：
  - `min ghost reduction vs TSDF >= 0.30`（中间门槛）
  - Comp-R >= 95%

3. **阶段 C（联合收敛）**
- 小范围参数整定（仅 6~12 组）冲击 P10 最终门槛：
  - TUM `acc<=1.8`
  - Bonn `acc<=2.6`
  - ghost 降幅 `>=35%`

---

## 8. 验证设计（建议最小集）

1. 消融矩阵（至少 8 组）
- Baseline
- +M2
- +M3
- +M2+M3
- +M1
- +M4
- +M1+M4
- +M1+M2+M3+M4

2. 指标
- 主：`Acc(cm), Comp-R(5cm), ghost_ratio, ghost_reduction_vs_tsdf`
- 辅：`ghost_tail_ratio, precision_5cm, runtime_fps, memory`

3. 产物建议
- `output/summary_tables/p10_module_ablation.csv`
- `output/summary_tables/p10_module_pareto.csv`
- `assets/p10_module_acc_ghost_pareto.png`

---

## 9. 最终判断

1. 继续纯参数微调，P10 过线概率低（已被多轮 sweep 证伪）。
2. 具备过线潜力的路径是：
- **前移动态判别到融合入口（M1/M4）**
- **融合阶段显式去偏和不确定度传播（M2/M3）**
3. 这条路径同时具备：
- 工程可落地（与你现有代码结构兼容）
- 学术可叙事（可形成独立方法贡献）

