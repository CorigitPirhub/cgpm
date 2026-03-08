# 投稿可行性路线图 / 目标任务书

版本：`2026-03-08`
范围：基于当前仓库代码、主文档、canonical 主表、`P10` 专项实验链与顶刊/顶会同类论文对标，整理一份“宁冗勿缺”的投稿任务书。
目标：若本路线图中的**全部关键任务**被严格完成，则项目将达到**稳中稿顶刊/顶会水准**所需的最低完备条件。

---

## 0. 执行摘要

### 0.1 当前结论

当前项目是一个**研究深度较强、工程成熟度不错、动态局部建图效果明确优于 TSDF** 的 3D mapping 原型，但距离“稳中稿顶刊/顶会”仍有明显差距。

差距主要不在于：
- 工程不可运行；
- 机制不够丰富；
- 动态抑制完全没有效果；

差距主要在于：
- **P10 硬门槛未过**；
- **高层研究问题不新，必须靠更清晰的机制创新 + 更强实验闭环来立 paper**；
- **当前主表对比基线仍偏弱，缺少足够多的同类强基线**；
- **近几轮 P10 已验证出大量 downstream tri-map / delayed / export 机制，但收益幅度太小，边际收益开始衰减**。

### 0.2 最应该保留的论文主线

最值得保留的论文主线不是“把所有模块都讲一遍”，而是收敛成：

> **Evidence-gradient dynamic local mapping + state-level static/dynamic disentanglement + delayed branch for conflict-isolated geometry**

更具体地说，论文主线应围绕：
1. `dual_state / evidence-gradient` 如何把几何与动态污染分离；
2. `delayed branch` 为什么是必要的结构，而不是后置删点；
3. `write-time target synthesis` 如何赋予 delayed branch 真正独立的几何假设；
4. 为什么这种结构能同时兼顾 `Acc / Comp-R / ghost suppression`。

### 0.3 最应该砍掉的方向

下列方向应**退出主线叙事**，仅保留为附录/负结果链：
- `OTV`
- `CSR-XMap`
- `XMem / BECM / RCCM`
- `OBL-3D`
- `CMCT`
- `CGCC`
- `PFVP`
- `PFV-sharp`
- `PFV-bank`
- 最近多轮 `tri-map` 的 export-side 微调分支（`quantile / hold / residency export / local replacement / competition replacement / banked readout / persistent bank`）中的大多数应降级为**诊断链**，而非论文主贡献。

这些方向的共同问题是：
- 机制上有信息增益；
- 但结果层面只产生极小 mixed change；
- 作为主贡献不够“硬”。

### 0.4 最可能中的 venue 组合

按当前项目成熟度与问题设定，我建议的**现实投稿阶梯**：

#### A. 最稳妥主目标
- `RA-L + ICRA` 或 `RA-L + IROS`

理由：
- 题目属于机器人/SLAM/建图系统类；
- 允许较强系统工程 + 机制创新组合；
- 对“问题定义不新，但结构设计/实验闭环很扎实”的工作更友好；
- 当前项目最容易达到的“稳中稿”目标在这一档。

#### B. 进取目标
- `ICCV / CVPR`（3D Vision / SLAM / Dynamic Scene Reconstruction 方向）

前提：
- 必须有更干净的创新命题；
- 必须补足更强 literature baselines；
- 必须把 `P10` 指标硬伤显著推进；
- 必须把论文主线从“系统堆模块”压缩成“一个明确的新结构命题”。

#### C. 中长线目标
- `T-RO`

策略：
- 先用 `RA-L/ICRA` 或 `IROS` 版本稳定收敛；
- 再扩展成更完整的长期版本，补理论、更多基线、更多跨数据集验证、更多实时性/稳定性分析。

**结论**：若只选一个“最可能稳中”的主目标，优先选 `RA-L + ICRA`。 

---

## 1. 当前项目状态基线

### 1.1 当前 canonical 主表（对外正式口径）

以以下文件为正式主口径：
- `output/summary_tables/paper_main_table_local_mapping.csv`
- `output/summary_tables/local_mapping_main_metrics_toptier.csv`
- `output/summary_tables/dual_protocol_multiseed_significance.csv`

当前 `5-seed canonical` 口径（见 `README.md`）：
- TUM dynamic mean：`Acc = 4.1655 cm`, `Comp-R = 100.0%`
- Bonn dynamic mean：`Acc = 6.1481 cm`, `Comp-R = 77.21%`

这说明：
- 对外正式主表下，项目离顶刊主口径仍远；
- `TUM` 的 `Comp-R` 不是问题，但 `Acc` 仍高；
- `Bonn` 上 `Acc` 与 `Comp-R` 都是问题。

### 1.2 当前 P10 专项硬门槛

当前 `P10` 目标（见 `P10_METHOD_PROPOSALS.md:572`）：
- TUM oracle：`Acc <= 1.80 cm`, `Comp-R >= 95%`
- Bonn slam：`Acc <= 2.60 cm`, `Comp-R >= 95%`
- `ghost_reduction_vs_tsdf >= 35%`

### 1.3 当前距离 P10 过线还差多少

按 `precision profile` 当前最优行：
- `output/summary_tables/local_mapping_precision_profile.csv:2`
- `output/summary_tables/local_mapping_precision_profile.csv:3`
- `output/summary_tables/local_mapping_precision_profile.csv:4`

最接近结果：
- TUM 最优 `Acc = 2.789 cm`，距离 `1.80 cm` 还差 `0.989 cm`
- Bonn 最优 `Acc = 3.349 cm`，距离 `2.60 cm` 还差 `0.749 cm`
- Bonn 最优 `ghost_reduction_vs_tsdf = 15.4%`，距离 `35%` 还差约 `19.6` 个百分点
- `Comp-R` 两边实际上都已满足或远高于门槛

**所以，当前离线的核心缺口不是 `Comp-R`，而是：**
- `Acc`
- Bonn 动态抑制强度

---

## 2. 同类顶刊/顶会工作对标

下表仅保留与本项目最相关、最值得补做或补写的工作族谱。

### 2.1 经典 RGB-D / Dynamic SLAM / Background Reconstruction

1. **Co-Fusion** — real-time segmentation/tracking/fusion of multiple objects  
   论文：<https://arxiv.org/abs/1706.06629>

2. **StaticFusion** — dynamic environment 中的静态背景重建  
   论文 PDF：<https://raluca-scona.github.io/docs/conference-papers/2018_icra_staticfusion.pdf>

3. **MaskFusion** — object-aware / semantic / dynamic RGB-D SLAM  
   论文：<https://arxiv.org/abs/1804.09194>

4. **MID-Fusion** — octree-based object-level multi-instance dynamic RGB-D SLAM  
   论文：<https://arxiv.org/abs/1812.07976>

5. **EM-Fusion** — dynamic object-level SLAM with probabilistic data association  
   论文：<https://openaccess.thecvf.com/content_ICCV_2019/html/Strecke_EM-Fusion_Dynamic_Object-Level_SLAM_With_Probabilistic_Data_Association_ICCV_2019_paper.html>

6. **FlowFusion** — dynamic dense RGB-D SLAM based on optical flow  
   论文：<https://arxiv.org/abs/2003.05102>

7. **DynaSLAM** — tracking / mapping / inpainting in dynamic scenes  
   论文：<https://arxiv.org/abs/1806.05620>

8. **DynaSLAM II** — tightly coupled multi-object tracking + SLAM  
   论文：<https://arxiv.org/abs/2010.07820>

9. **PoseFusion2** — background reconstruction + human shape recovery  
   论文：<https://arxiv.org/abs/2108.00695>

### 2.2 表达层与多表面 / 多对象体表示

10. **TSDF++** — multi-object TSDF formulation  
    论文：<https://arxiv.org/abs/2105.07468>

11. **Panoptic Multi-TSDFs** — long-term dynamic consistency / multi-resolution volumetric mapping  
    论文：<https://arxiv.org/abs/2109.10165>

### 2.3 新一代 Neural / Dense Dynamic SLAM / Dynamic Reconstruction

12. **vMAP** — vectorised object mapping for neural field SLAM  
    论文：<https://openaccess.thecvf.com/content/CVPR2023/html/Kong_vMAP_Vectorised_Object_Mapping_for_Neural_Field_SLAM_CVPR_2023_paper.html>

13. **NID-SLAM** — neural implicit RGB-D SLAM in dynamic environments  
    论文：<https://arxiv.org/abs/2401.01189>

14. **RoDyn-SLAM** — robust dynamic dense RGB-D SLAM with NeRF  
    论文：<https://arxiv.org/abs/2407.01303>

15. **WildGS-SLAM** — dynamic environment Gaussian SLAM（单目，但可用于 modern dynamic dense baseline 对标）  
    论文：<https://openaccess.thecvf.com/content/CVPR2025/html/Zheng_WildGS-SLAM_Monocular_Gaussian_Splatting_SLAM_in_Dynamic_Environments_CVPR_2025_paper.html>

16. **4D Gaussian Splatting SLAM / DyGS-SLAM 系列** — recent dynamic SLAM / reconstruction trend  
    论文：<https://openaccess.thecvf.com/content/ICCV2025/papers/Li_4D_Gaussian_Splatting_SLAM_ICCV_2025_paper.html>  
    论文：<https://www.openaccess.thecvf.com/content/ICCV2025/papers/Hu_DyGS-SLAM_Real-Time_Accurate_Localization_and_Gaussian_Reconstruction_for_Dynamic_Scenes_ICCV_2025_paper.pdf>

### 2.4 对标结论

- 本项目的**高层问题**（dynamic scene 中做静态背景重建 / 动态抑制 / 局部建图）并不新；
- 真正可能形成 paper novelty 的，是：
  - `evidence-gradient + dual-state` 的状态设计；
  - `delayed branch` 作为 conflict-isolated geometry path；
  - `write-time target synthesis` 如果成功，可能是当前 delayed 主线真正值得 claim 的核心点；
- 所以当前论文必须避免把 claim 讲成：
  - “我们也做 dynamic SLAM / background reconstruction”；
- 应讲成：
  - “我们提出一种状态层解耦 + delayed geometry synthesis 的局部建图结构，用于同时兼顾 `Acc / Comp-R / ghost suppression`。”

---

## 3. 当前创新度评估

### 3.1 已有方向

已有大量工作已实现：
- moving object filtering / background-only mapping
- object-level separate maps
- multiple TSDF / multi-surface volumetric representations
- dynamic neural field SLAM / dynamic Gaussian SLAM

### 3.2 当前项目仍可 claim 的创新

当前项目若要维持可投稿创新度，应把创新压缩在以下层面：

1. **状态层解耦**
   - `evidence-gradient` 与 `dual-state` 如何把几何与动态污染分开；

2. **Delayed branch 机制**
   - 不是简单 mask / object map；
   - 而是针对“冲突样本”的 delayed geometry path；

3. **Write-time target synthesis（待完成）**
   - delayed branch 若能拥有 delayed-specific write target，才可能形成真正的新点；

4. **精度与动态抑制的统一优化目标**
   - 若能明确证明比现有方法更好地兼顾 `Acc / Comp-R / ghost`，这会显著增强投稿可行性。

### 3.3 当前创新风险

如果不完成上面的主线收敛，当前稿件最大的创新风险是：
- 看起来像很多已知 dynamic SLAM 机制的组合；
- 但没有一个足够强、足够可复述、足够结果导向的核心命题。

---

## 4. 投稿目标与 Go / No-Go 门槛

### 4.1 最低投稿门槛（不过线不许投稿）

以下门槛必须全部满足：

#### M1. P10 指标门槛
- [ ] TUM oracle: `Acc <= 1.80 cm`
- [ ] TUM oracle: `Comp-R >= 95%`
- [ ] Bonn slam: `Acc <= 2.60 cm`
- [ ] Bonn slam: `Comp-R >= 95%`
- [ ] `ghost_reduction_vs_tsdf >= 35%`

#### M2. 多 seed 稳定性门槛
- [ ] TUM / Bonn 主结论至少 `5-seed`
- [ ] 核心增益在 `mean ± std` 下稳定，不依赖单个 seed
- [ ] 显著性测试完整：`t-test + Wilcoxon`

#### M3. 强基线对标门槛
- [ ] 不仅对 `TSDF`
- [ ] 至少补齐 3 个传统 dynamic RGB-D/dense baseline
- [ ] 至少补齐 1–2 个 modern neural/dynamic baseline（哪怕只在子集）

#### M4. 机制闭环门槛
- [ ] 有从失败链到主线收敛的完整叙事
- [ ] 有组件级 ablation
- [ ] 有负结果裁剪说明
- [ ] 有 failure cases

#### M5. 复现实验门槛
- [ ] 一键脚本
- [ ] 固定表格刷新链
- [ ] 固定 canonical 口径
- [ ] 关键结果可重现

### 4.2 稳中稿顶会/顶刊门槛（建议标准）

在满足最低门槛之外，还应满足：
- [ ] 至少一个核心结果相对强基线呈现**明确且可见的净优势**，不只是 mixed change
- [ ] 论文主贡献可压缩成 **1 个核心结构命题 + 2 个支撑模块**
- [ ] 与 literature 的区别能在 1 段话内讲清楚
- [ ] 所有主图、主表、supp 实验能形成闭环

---

## 5. 最值得保留的主线

### 5.1 论文主命题（必须保留）

**主命题 A：状态层解耦而非后置删点**
- 这是项目区别于大量基于 mask / object-level / post-filter 方法的关键点之一。

**主命题 B：Delayed branch 作为冲突样本隔离路径**
- 这比一般的 outlier removal 更有结构性，也更有 paper 叙事价值。

**主命题 C：Write-time target synthesis（未来必须完成）**
- 这是 delayed 主线继续走下去最有希望变成“可投稿核心创新”的方向。

### 5.2 最值得保留的支撑证据

- `TSDF` 对比（最基础 baseline）
- `precision profile`（说明 Acc 缺口在哪里）
- `negative branch chain`（说明不是瞎调参）
- `canonical 5-seed` 主表与显著性表

---

## 6. 该砍掉的支线

以下内容不应进入主论文主方法章节，只可保留为：
- 附录
- 补充材料
- 负结果链
- 审稿问答备用材料

### 6.1 已应砍掉的旧分支
- [x] `OTV`
- [x] `CSR-XMap`
- [x] `XMem / BECM / RCCM`
- [x] `OBL-3D`
- [x] `CMCT`
- [x] `CGCC`
- [x] `PFVP`
- [x] `PFV-sharp`
- [x] `PFV-bank`

### 6.2 应退出主线叙事的近期分支
- [ ] `tri_map` export 侧小修小补分支，不应继续各自占一节主文
- [ ] `quantile` / `hold` / `residency export` / `local replacement` / `competition replacement` / `dedicated banked readout` / `persistent bank accumulation` 应收束成一条“delayed branch diagnostics chain”

### 6.3 主线压缩原则
- 主文中：只保留**最终保留的 delayed 主线版本**
- 补充材料中：给出其余分支的对比与负结果链

---

## 7. 必补 literature baselines

这是当前最缺的一环。必须补，不补基本不具备稳中稿顶刊/顶会条件。

### 7.1 第一优先级（必须补）

#### B1. `StaticFusion`
- 论文：`StaticFusion: Background Reconstruction for Dense RGB-D SLAM in Dynamic Environments`
- 作用：最直接对标“背景-only dense RGB-D mapping in dynamic scenes”
- 要求：
  - [ ] 复现或适配到当前协议
  - [ ] 至少在 TUM walking 与 Bonn balloon2/all3 子集评估

#### B2. `DynaSLAM / DynaSLAM II`
- 论文：`DynaSLAM`, `DynaSLAM II`
- 作用：最经典动态背景 SLAM 基线
- 要求：
  - [ ] 明确使用哪一种动态检测版本（geometry / DL / both）
  - [ ] 静态背景重建指标对齐

#### B3. `MID-Fusion / MaskFusion / Co-Fusion` 中至少 1–2 个
- 作用：object-aware / object-level dynamic RGB-D SLAM 对标
- 要求：
  - [ ] 选择最可复现、最有代表性的 1–2 个
  - [ ] 强调和本项目的差异：它们依赖 object / segmentation，而我们主张 state-level geometry disentanglement

### 7.2 第二优先级（强烈建议补）

#### B4. `TSDF++` 或 `Panoptic Multi-TSDFs`
- 作用：多表面 / 多对象体表示对标
- 要求：
  - [ ] 至少用论文结果表/复现实验做“representation family”对标
  - [ ] 若无法完整复现，必须在 related work 与 discussion 中充分对齐并说明复现障碍

#### B5. `FlowFusion`
- 作用：传统 dense RGB-D dynamic SLAM 的强对照

### 7.3 第三优先级（现代方法，对冲审稿人）

#### B6. `vMAP`
- 作用：neural field + object-level mapping 对标

#### B7. `NID-SLAM` / `RoDyn-SLAM`
- 作用：modern dynamic dense RGB-D / neural mapping baseline
- 要求：
  - [ ] 即便只做小子集，也要有一个 modern baseline 家族进入论文

### 7.4 基线补充原则
- [ ] 不能只对 `TSDF`
- [ ] 至少做到“经典动态 RGB-D family + modern neural/dynamic family”两条线
- [ ] 对于难复现方法，必须给出“公开代码状态 / 复现障碍 / 采用论文数值还是自复现”的透明说明

---

## 8. 必补实验矩阵

### 8.1 数据集与协议
- [ ] TUM walking 三序列（主）
- [ ] Bonn all3（主）
- [ ] 至少一个 static sanity set（证明不会为了动态抑制破坏静态几何）
- [ ] 必须固定 oracle / slam 双协议口径说明

### 8.2 指标矩阵
- [ ] `Acc`
- [ ] `Chamfer`
- [ ] `Comp-R(5cm)`
- [ ] `F-score`
- [ ] `ghost_ratio`
- [ ] `ghost_tail_ratio`
- [ ] `background_recovery`
- [ ] `runtime / FPS`

### 8.3 组件级 ablation
- [ ] 去掉 delayed branch
- [ ] 去掉 write-time target synthesis
- [ ] 去掉 promotion hold
- [ ] 去掉 export competition
- [ ] 去掉 delayed-specific readout

### 8.4 稳定性与鲁棒性
- [ ] 5-seed
- [ ] 不同 frame budget / stride
- [ ] 不同 scene dynamicity 强度
- [ ] 至少一个 failure case 分析

### 8.5 可视化
- [ ] delayed branch activation maps
- [ ] committed vs delayed export overlays
- [ ] replacement competition visualizations
- [ ] static background / dynamic tail qualitative panels

---

## 9. 当前最可能成稿的 paper 结构

### 主文建议 6 节

1. **Introduction**
   - dynamic local mapping 的核心矛盾：`Acc` 与 dynamic suppression 耦合
2. **Related Work**
   - background-only SLAM
   - object-aware dynamic RGB-D SLAM
   - multi-surface / multi-TSDF representations
   - modern neural dynamic SLAM
3. **Method**
   - evidence-gradient / dual-state disentanglement
   - delayed branch
   - delayed-specific write-time target synthesis（必须做出来）
4. **Experiments**
   - 主结果表
   - 强基线对标
   - ablation
5. **Analysis**
   - failure modes
   - why delayed branch helps / where it fails
6. **Conclusion**

### 主图建议
- 图 1：系统结构图（必须收敛，不要 20 个模块）
- 图 2：delayed branch 工作示意
- 图 3：主结果 qualitative
- 图 4：ablation / profile curve

---

## 10. 需要达成的完整目标任务书（执行清单）

下面是**真正的任务书**。按“宁冗勿缺”原则列全。若全部完成，才可判断具备稳中稿顶刊/顶会水平。

### A. 问题与主线收敛
- [ ] 把论文主命题压缩成一句话，并与 `README.md` / `TASK_LOCAL_TOPTIER.md` / 主文完全一致
- [ ] 明确 delayed 主线最终版本
- [ ] 所有负结果分支从主方法章节移除
- [ ] 形成一条清晰的“从失败链 -> 当前保留主线”的叙事

### B. P10 指标过线
- [ ] TUM oracle 达线
- [ ] Bonn slam 达线
- [ ] ghost_reduction_vs_tsdf 达线
- [ ] 5-seed 下仍达线或非常接近达线

### C. 方法定型
- [ ] delayed-branch write-time target synthesis 完成
- [ ] delayed bank 的写入目标不再是 committed 变体
- [ ] delayed branch 在 export 前即具备独立可解释 surface representation
- [ ] 与 committed branch 的 interaction 规则固定

### D. 基线补齐
- [ ] `StaticFusion`
- [ ] `DynaSLAM` 或 `DynaSLAM II`
- [ ] `MID-Fusion / MaskFusion / Co-Fusion` 至少 1–2 个
- [ ] `TSDF++` 或 `Panoptic Multi-TSDFs`
- [ ] `NID-SLAM / RoDyn-SLAM / vMAP` 至少 1 个现代动态 dense baseline

### E. 实验闭环
- [ ] canonical 主表刷新无漂移
- [ ] significance 完整
- [ ] ablation 完整
- [ ] failure case 完整
- [ ] runtime / complexity / memory 报告完整

### F. 写作与图表
- [ ] 主文结构定稿
- [ ] 主图 4 张以上
- [ ] 相关工作覆盖完整
- [ ] limitations 写明
- [ ] supplemental 完整

### G. 复现与开源
- [ ] 一键运行脚本
- [ ] 环境文件
- [ ] 数据准备说明
- [ ] 结果刷新脚本
- [ ] 表格生成脚本
- [ ] artifact 通过内部自检

### H. 投稿决策门槛
- [ ] 满足全部 A–G
- [ ] 主结果对至少 2 个强基线 family 有明确净优势
- [ ] 创新命题可在摘要中 3 句话讲清楚
- [ ] 论文题目、摘要、方法、结果完全围绕同一主命题展开

---

## 11. Venue 选择决策树

### 11.1 若只完成最低完整任务但创新仍偏系统型
- 首投：`RA-L + ICRA`
- 备选：`IROS`

### 11.2 若 delayed write-time synthesis 形成明确新命题且结果显著提升
- 可冲：`ICCV / CVPR`
- 条件：
  - 结果必须明显更强
  - 强基线必须齐
  - 论文主线必须极其干净

### 11.3 若先发系统稿后扩成长期版本
- 后续扩展：`T-RO`

---

## 12. 当前最现实的判断

若从今天开始继续推进，**最现实的稳中稿路径**是：

1. 立即冻结并收敛主线；
2. 停止继续在 export 末端做小修小补；
3. 把资源集中到：
   - `delayed-branch write-time target synthesis`
   - `Acc` 主线精度改进
   - 强 literature baselines
4. 以 `RA-L + ICRA` 为第一目标；
5. 只有在 write-time synthesis 带来明显净收益后，再考虑 `CVPR / ICCV`。

---

## 13. 结语

这份路线图不是“可选建议”，而是**投稿前最低完备任务书**。  
当前项目已经证明了自己是一个强研究原型，但要达到稳中稿顶刊/顶会水准，必须从“模块试验”转入“主线收敛 + 基线补齐 + 指标过线 + 写作成稿”四线并行阶段。  

如果所有任务都完成，项目将具备：
- 清晰而可辩护的创新点；
- 足够强的 literature 对标；
- 足够硬的结果支撑；
- 足够完整的复现与投稿闭环。  

这才是“稳中稿顶刊/顶会”所需的真正最低条件。
