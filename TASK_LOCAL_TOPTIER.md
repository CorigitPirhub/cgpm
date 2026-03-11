# TASK_LOCAL_TOPTIER / 顶刊顶会冲线总任务书（重写版）

版本：`2026-03-08`
状态：`ACTIVE / SUPERSEDES HISTORY`
定位：本文件从当前项目真实现状出发，定义一份**偏激进、宁冗勿缺、按最高标准写就**的总任务书。  
原则：**只有当本任务书中的全部关键任务严格完成，才可判断项目具备“直接投稿顶刊/顶会且大概率中稿”的最高条件。**

> 历史版本已冻结归档：`archives/taskbooks/TASK_LOCAL_TOPTIER_ARCHIVE_2026-03-08.md`
> 
> 本文件不再承担“实验流水账”职责，而承担“从当前状态出发，迈向高概率中稿顶刊/顶会”的总控任务书职责。

---

## 0. 文档用途与执行原则

### 0.1 用途

本任务书用于统一以下内容：
- 论文主线收敛方向；
- 从当前项目状态起步，到可直接投稿顶刊/顶会的所有阶段任务；
- 每阶段必须达成的客观目标；
- 何种内容应保留、何种内容应降级为附录或负结果链；
- 最终投稿所需的实验、基线、图表、写作、复现与 artifact 条件。

### 0.2 执行原则

1. **主线优先，支线让路**
   - 任何不能稳定推动主线指标或创新命题的分支，都不能继续消耗主线资源。

2. **先过线，再讲故事**
   - 未满足核心指标门槛前，不得进入“包装投稿叙事”阶段。

3. **先补基线，再谈顶会**
   - 对比不足是当前最严重的投稿风险之一；没有强基线闭环，不能视为完成。

4. **先锁 canonical，再写主文**
   - 所有论文数字、报告数字、摘要数字、图表数字必须来自统一 canonical source of truth。

5. **宁冗勿缺**
   - 对顶刊/顶会投稿而言，缺任何一个关键模块、关键基线、关键表格、关键 failure analysis，都可能导致系统性拒稿。

---

## 1. 当前项目现状（重写任务书的起点）

### 1.1 已具备的积极条件

当前项目已具备以下基础，这些基础说明项目不是“从零开始”，而是一个**成熟研究原型**：

- [x] 3D 动态局部建图主线代码完整：
  - `egf_dhmap3d/core/config.py`
  - `egf_dhmap3d/core/types.py`
  - `egf_dhmap3d/core/voxel_hash.py`
  - `egf_dhmap3d/modules/associator.py`
  - `egf_dhmap3d/modules/updater.py`
  - `egf_dhmap3d/modules/pipeline.py`
- [x] 实验链路完整：
  - `scripts/run_benchmark.py`
  - `scripts/run_egf_3d_tum.py`
  - `scripts/run_p10_precision_profile.py`
  - `scripts/update_summary_tables.py`
- [x] 主文档体系完整：
  - `README.md`
  - `BENCHMARK_REPORT.md`
  - `processes/p10/P10_RESEARCH_ASSESSMENT.md`
  - `processes/p10/P10_METHOD_PROPOSALS.md`
  - `processes/governance/PROJECT_AUDIT_CHECKLIST.md`
  - `processes/governance/SUBMISSION_FEASIBILITY_ROADMAP.md`
- [x] 当前项目已拥有清晰的 research bottleneck 定位：
  - 当前最大阻塞项仍是 `P10`
  - 主矛盾仍是 `Acc / ghost / dynamic robustness` 的统一优化
- [x] 当前项目对 `TSDF` 的动态抑制优势明确，且 canonical 表和显著性表已统一刷新。

### 1.2 当前正式 canonical 主表现状

当前正式口径以如下文件为准：
- `output/summary_tables/paper_main_table_local_mapping.csv`
- `output/summary_tables/local_mapping_main_metrics_toptier.csv`
- `output/summary_tables/dual_protocol_multiseed_significance.csv`

当前 `5-seed canonical` 口径：
- TUM dynamic mean：`Acc = 4.1655 cm`, `Comp-R = 100.0%`
- Bonn dynamic mean：`Acc = 6.1481 cm`, `Comp-R = 77.21%`

结论：
- 当前正式主表下，项目还**不具备直接投稿顶刊/顶会主口径**；
- 当前项目最大短板依旧是：
  - `Acc`
  - Bonn `Comp-R`
  - Bonn 动态抑制强度

### 1.3 当前 P10 专项现状

当前 `P10` 硬门槛：
- TUM oracle：`Acc <= 1.80 cm`, `Comp-R >= 95%`
- Bonn slam：`Acc <= 2.60 cm`, `Comp-R >= 95%`
- `ghost_reduction_vs_tsdf >= 35%`

按当前最优专项 profile：
- TUM 最优 `Acc = 2.789 cm`，仍差 `0.989 cm`
- Bonn 最优 `Acc = 3.349 cm`，仍差 `0.749 cm`
- Bonn 最优 `ghost_reduction_vs_tsdf = 15.4%`，仍差约 `19.6` 个百分点

结论：
- 当前离线的核心缺口，不是 `Comp-R`，而是：
  - `Acc`
  - Bonn dynamic suppression 强度

### 1.4 当前主线状态判断

当前最值得保留的结构主线不是所有模块的合集，而是：

> `evidence-gradient + dual-state disentanglement + delayed branch + write-time target synthesis`

当前已经被充分验证、但收益开始衰减的下游子线包括：
- delayed-only routing
- promotion hold / hysteresis
- residency-gated export
- local replacement
- competition replacement
- dedicated delayed banked readout
- persistent delayed bank accumulation

这些分支的共同事实是：
- 机制链路大多被成功打通；
- 但最终指标只呈现极小 mixed change；
- 说明 downstream export-side 微调已经接近边际收益衰减区。

---

## 2. 总体投稿目标与最终判定标准

### 2.1 最终目标不是“可投稿”，而是“高概率中稿顶刊/顶会”

本任务书采用比 `processes/governance/SUBMISSION_FEASIBILITY_ROADMAP.md` 更激进的标准。

最终目标定义为：

- [ ] 论文主线创新命题清晰、简洁、可一段话讲明；
- [ ] 主结果对**经典 dynamic RGB-D baselines + modern neural/dynamic baselines** 都有明确竞争力；
- [ ] `P10` 专项硬门槛通过；
- [ ] canonical 主表显著优于当前状态，且不依赖单个 seed；
- [ ] 具备完整 artifact / reproducibility / supplemental / failure analysis；
- [ ] 达到可直接冲击：
  - `RA-L + ICRA / IROS` 的“高概率中稿标准”；
  - 同时具备向 `CVPR / ICCV / RSS / T-RO` 冲击的最低研究质量。

### 2.2 本任务书的最终通过标准（最高条件版）

只有以下 5 大类条件全部满足，任务书才算完成：

1. **问题与方法主线完成收敛**
2. **核心指标与多 seed 实验过线**
3. **强 literature baselines 补齐**
4. **论文与实验叙事闭环完整**
5. **artifact / reproducibility / release 条件完成**

---

## 3. 最值得保留的论文主线

以下主线是未来论文中最值得保留、且必须重点投入的内容：

### 3.1 主命题 A：状态层解耦而非后置删点

必须保留：
- `evidence-gradient`
- `dual-state`
- “动态污染与静态几何应在状态层而不是后置删点层分离”的核心论点

原因：
- 这是与大量 `mask/object/post-filter` 类方法最大的可辩差异之一；
- 也是解释 `Acc / Comp-R / ghost` 三者耦合问题的理论起点。

### 3.2 主命题 B：delayed branch 作为 conflict-isolated geometry path

必须保留：
- delayed branch 的结构必要性
- delayed branch 与 committed branch 的职责差异
- delayed branch 不是“多一张 map”，而是“冲突几何假设的隔离路径”

### 3.3 主命题 C：write-time target synthesis（未来必须做成）

这是当前最有希望成为**论文核心新增贡献**的方向。

要求：
- delayed branch 的写入期必须拥有 delayed-specific target；
- 不能只是 committed `phi_static / phi_bg / phi_geo` 的弱变体或后处理版本；
- 必须能解释“为什么 delayed branch 最终能改善 `Acc / ghost` 而非只是移动少量 export points”。

### 3.4 主命题 D：统一优化 `Acc / Comp-R / ghost suppression`

最终论文必须强调：
- 不是单指标最优；
- 而是在动态局部建图中，同时兼顾几何精度、完整性和动态抑制。

---

## 4. 必须砍掉或降级的支线

这些内容不允许再占用主论文主方法篇幅，只能：
- 作为附录；
- 作为负结果链；
- 作为 internal diagnostics。

### 4.1 已判负或收益过小的旧支线
- [x] `OTV`
- [x] `CSR-XMap`
- [x] `XMem / BECM / RCCM`
- [x] `OBL-3D`
- [x] `CMCT`
- [x] `CGCC`
- [x] `PFVP`
- [x] `PFV-sharp`
- [x] `PFV-bank`

### 4.2 近期 tri-map/export 微调支线
以下分支不再作为“独立方法点”继续扩写：
- [x] `quantile-calibrated support-gap routing`
- [x] `top-tail delayed-only escalation`
- [x] `promotion hold / hysteresis`
- [x] `residency-gated export`
- [x] `local replacement`
- [x] `competition-scored replacement`
- [x] `dedicated delayed banked readout`
- [x] `persistent delayed bank accumulation`

这些内容在未来稿件中的角色应改为：
- supporting diagnostics chain
- negative results and narrowing-down evidence

---

## 5. 必补 literature baselines（必须完成）

不补齐这一节，项目不具备“高概率中稿顶刊/顶会”的最低条件。

### 5.1 经典 dynamic RGB-D / semantic dynamic baselines

以下 baseline 至少要完成其中的关键组合：

#### B1. `DynaSLAM` / `DynaSLAM II`
- [ ] 至少补一条完整动态背景基线
- [ ] 必须交代检测模式与重建口径对齐方式

#### B2. `4DGS-SLAM`（preferred）/ `D4DGS-SLAM`（fallback）
- [x] 至少冻结 1 条 `2025` recent dynamic dense / 4DGS family 基线家族
- [x] 当前优先选 `4DGS-SLAM`（ICCV 2025）作为更近的对标方案
- [x] 若 `4DGS-SLAM` 本地接入阻塞，再退回 `D4DGS-SLAM`
- [x] `V3D-SLAM` 不再保留为当前 `S1` 任务目标

### 5.2 表达层 / 多表面 family baselines

#### B3. `TSDF++` 或 `Panoptic Multi-TSDFs`
- [ ] 至少补一条“representation family”对标
- [ ] 若无法完整复现，也必须在主文中严肃对齐并给出复现障碍说明

### 5.3 现代 neural / dynamic dense baselines

#### B4. `RoDyn-SLAM`（preferred S1 replacement）
- [x] 作为 `StaticFusion` 的正式替代线进入 `RB-Core`
- [x] 已在 TUM walking 与 Bonn dynamic 上形成 core-dataset protocol-aligned smoke / eval compare
- [x] 已完成本地可运行接入，并固定 source / config / protocol

#### B5. `NID-SLAM`（fallback）
- [ ] 若 `RoDyn-SLAM` 接入阻塞，则以 `NID-SLAM` 作为 recent dynamic dense fallback
- [ ] 必须记录为什么选 fallback 而不是 `RoDyn-SLAM`

#### B6. `vMAP / MUTE-SLAM / NICE-SLAM`（representation/neural family）
- [ ] 至少补 1 条现代 neural RGB-D dense baseline 进入 discussion 或实跑矩阵
- [ ] 若已存在真实输出接入，应优先纳入 canonical compare，而不是只保留在独立报告

### 5.4 明确归档：不再阻塞主线的旧方案

#### Archived-B0. `StaticFusion`
- [x] 由于环境适配 / 构建阻塞明显，不再作为当前项目的正式 blocking baseline
- [x] 可在 related work 中保留论文级对齐，但**不再**作为 `S1` 是否通过的必要接入项
- [x] 当前正式替代目标改为：`RoDyn-SLAM`（preferred）/ `NID-SLAM`（fallback）

### 5.5 可选扩展（2025 watchlist）
- [ ] `WildGS-SLAM` / `DyGS-SLAM` / `DropD-SLAM` 等 2025 方向进入 discussion 与 supplemental 对标
- [ ] 但若输入模态或评测口径与当前 RGB-D local mapping 主设定不一致，不得提前挤占 `S1` 的 blocking baseline 名额

### 5.4 基线完成门槛
以下条件全部满足才算 baseline 补齐：
- [ ] 至少 3 个经典 dynamic RGB-D/dense baseline
- [ ] 至少 1 个 modern neural/dynamic baseline
- [ ] 至少 1 个 representation-level family baseline
- [ ] 主文对比不再只依赖 `TSDF`

---

## 6. 最终可投稿的硬门槛（高标准版）

### 6.1 最低硬门槛（必须满足）

#### P10 专项门槛
- [ ] TUM oracle: `Acc <= 1.80 cm`
- [ ] TUM oracle: `Comp-R >= 95%`
- [ ] Bonn slam: `Acc <= 2.60 cm`
- [ ] Bonn slam: `Comp-R >= 95%`
- [ ] `ghost_reduction_vs_tsdf >= 35%`

#### 多 seed 稳定性门槛
- [ ] 关键主结果至少 `5-seed`
- [ ] `mean ± std` 下结论稳定
- [ ] `t-test + Wilcoxon` 均支持主结论

#### 复现门槛
- [ ] `scripts/update_summary_tables.py` 可稳定刷新主表
- [ ] canonical tables 与主文数字完全一致
- [ ] 工作树冻结前可完成一次完整自检

### 6.2 顶刊/顶会高标准门槛（必须满足）

以下是“高概率中稿”的更强版本，不满足则最多只算“可以投稿”，不算“稳中稿”：

#### H1. 指标层面
- [ ] 至少一条主数据集 family 上，对强基线 family 有**明确净正优势**，不是 mixed change
- [ ] Bonn 侧必须从当前 `Acc / ghost_reduction` 瓶颈中至少解掉一个
- [ ] 不允许主结论建立在单序列特例上

#### H2. 创新层面
- [ ] 主创新能压缩成 `1 个核心结构命题 + 2 个支撑模块`
- [ ] 与 `DynaSLAM / RoDyn-SLAM / NICE-SLAM / 4DGS-SLAM / TSDF++` 的区别能在一段话内说清楚
- [ ] 不是“组合了很多机制”，而是“提出了一个新的结构原则”

#### H3. 实验层面
- [ ] 强基线全补齐
- [ ] 消融闭环完整
- [ ] failure cases 完整
- [ ] runtime / memory / stability 都可交代

#### H4. 写作层面
- [ ] 题目、摘要、方法、主图、主表严格围绕同一主命题
- [ ] related work 没有明显漏引关键家族
- [ ] supplemental 能单独支撑复现和审稿答疑

---

## 7. 分阶段任务书（从当前状态到高概率中稿）

下面的阶段是**严格顺序执行**的；若前一阶段未完成，不得进入后一阶段。

---

## 阶段 S0：项目冻结前治理与现状确认

### 目标
统一当前项目口径，清空主线之外的叙事噪音。

### 必做任务
- [x] 固定 canonical source of truth
- [x] 检查 `README.md` / `BENCHMARK_REPORT.md` / 主文档数字一致性
- [x] 明确当前 active mainline
- [x] 归档所有历史任务书与局部试验链
- [x] 形成“当前状态基线页”

### 阶段产物
- [x] canonical tables 冻结快照
- [x] 当前状态审计结论页
- [x] 主线/支线裁剪表

### 阶段通过标准
- [x] 无主表口径漂移
- [x] 无历史分支混入主线叙事
- [x] 当前状态可被一页纸讲清

### 阶段 S0 完成记录（2026-03-08）

- 已冻结 canonical source of truth，并生成冻结快照：`output/freeze_snapshots/S0_2026-03-08_summary_tables/`
- 已完成主文档口径一致性检查，并修正 `BENCHMARK_REPORT.md` 的 canonical 说明；`README.md` 已补一页式治理页入口。
- 已明确当前 active mainline：`evidence-gradient + dual-state disentanglement + delayed branch + write-time target synthesis`。
- 已归档历史任务书与局部试验链：
  - `archives/taskbooks/TASK_LOCAL_TOPTIER_ARCHIVE_2026-03-08.md`
  - `archives/experiment_chains/EXPERIMENT_ARCHIVE_INDEX_2026-03-08.md`
- 已形成当前状态一页式基线页：`processes/governance/CURRENT_STATE_BASELINE.md`
- 已补充阶段重评估页：`processes/governance/S0_S1_S2_STAGE_REASSESSMENT_2026-03-09.md`
- 已形成主线/支线裁剪表：`processes/governance/MAINLINE_BRANCH_PRUNING_TABLE.md`

### 阶段 S0 结论

结论：**阶段 `S0` 作为治理阶段仍然已完成，但其 `2026-03-08` freeze snapshot 现应视为 historical governance archive，不再直接等同 current-code canonical。**

原因：
- `S0` 的治理动作本身已完成：canonical source 冻结、文档治理、归档、裁剪与基线页都已落实；
- 但经后续 `S2 current-code re-baseline` 复核，`S0` 冻结快照现在更准确地属于 historical archive；
- 因此 `S0` 不回滚，但后续所有文档必须避免把 `output/freeze_snapshots/S0_2026-03-08_summary_tables/` 继续当作 current-code executable truth。

---


## 7A. 阶段执行铁律（S1-S5 强制串行）

以下铁律适用于 `S1-S5` 全部阶段，且**优先级高于各阶段局部策略**：

### 7A.1 串行锁门原则
- `S1-S5` **必须严格串行执行**。
- 任一阶段若存在：
  - 必做任务未全部完成；
  - 阶段产物不完整；
  - 阶段通过标准未全部满足；
  - 未在任务书中记录完成状态与结论；
  则**绝对不允许进入下一阶段**。
- 下一阶段的所有实验、代码改动、文档叙事、主表更新，必须建立在前一阶段已完成且已冻结的产物之上。

### 7A.2 协议与口径红线（绝对禁止）
- [ ] 严禁混用 `oracle` 与 `slam` 结果形成单一主结论。
- [ ] 严禁在 `slam` 口径中引入任何 `GT delta`、未来帧信息或等价泄漏。
- [ ] 严禁使用非 canonical 主表、单次 probe、历史局部实验表直接覆盖主结论。
- [ ] 严禁在不同 frame budget / stride / seed / protocol / dataset split 条件下横向拼接“最好结果”作为阶段结论。
- [ ] 严禁在未记录配置变更的情况下更新主结果或主图表。

### 7A.3 数据泄露与伪增益红线（绝对禁止）
- [ ] 严禁把最终锁定评测集直接当作反复调参开发集使用而不留独立锁箱验证。
- [ ] 严禁基于测试集结果反向改阈值，再把同一测试集作为“最终结论”。
- [ ] 严禁多个模块同时漂移后再宣称某个单模块有效。
- [ ] 严禁没有 ablation / controlled compare / fixed protocol 的“偶然正结果”进入主线。
- [ ] 严禁人工筛选单序列、单 seed、单帧预算特例作为阶段通过依据。

### 7A.4 因果归因原则
- 任何阶段若要宣称“收益来自某方法/模块设计”，必须同时满足：
  - [ ] 固定协议不变
  - [ ] 固定 seed 集不变
  - [ ] 固定 frame/stride 不变
  - [ ] 固定 canonical 刷新链不变
  - [ ] 对照组只改变该模块或其直接依赖项
  - [ ] 有至少一个与当前 active mainline 的 controlled compare 表
- 若不满足上述条件，则该结果最多只能记为：
  - 诊断信息
  - 暂存观察
  - 不得进入阶段通过结论

---


## 7B. S1-S5 统一研发闭环模板（所有阶段必须遵循）

`S1-S5` 的每一个阶段，都不允许采用“先假设方法有效、再进入下一阶段”的思路。  
所有阶段都必须遵循如下统一闭环：

1. **方法设计**
   - 在当前 active mainline 下，只允许提出 `1-3` 个受控候选方法；
   - 每个候选方法必须明确：
     - 试图解决哪个瓶颈；
     - 目标改善哪个指标；
     - 相对当前 best config 只改了哪些模块。

2. **实验验证**
   - 在固定开发协议上跑 controlled experiment；
   - 不允许改变 protocol / seed / frame / stride / source-of-truth；
   - 必须留下完整 config、输出目录、对比表。

3. **强基线对比**
   - 每一阶段都必须接入一套**滚动强基线面板**，不能等到最后才补基线；
   - 每一阶段至少要和：
     - `TSDF`（底线）
     - `DynaSLAM` family（经典 dynamic RGB-D baseline）
     - `RoDyn-SLAM` family（preferred）/ `NID-SLAM` family（fallback）（2024 recent dynamic dense baseline）
     做 protocol-aligned 对比；
   - 对 `S1` 而言，还必须至少冻结：
     - `4DGS-SLAM`（preferred）/ `D4DGS-SLAM`（fallback）作为 `2025` recent dynamic dense 线
     - `NICE-SLAM` 作为 representation / neural dense 线
   - 从 `S2` 开始，还必须至少加入：
     - `4DGS family`（`4DGS-SLAM / D4DGS-SLAM` 中至少一个）
     - 和 `representation family`（如 `TSDF++ / Panoptic Multi-TSDFs / NICE-SLAM / MUTE-SLAM` 中至少一个）
   - 从 `S5` 开始，则必须补齐完整 baseline matrix。

4. **结果分析**
   - 必须明确说明：
     - 改善了什么；
     - 没改善什么；
     - 退化了什么；
     - 这些变化是否能归因于本次方法设计。

5. **方法迭代或淘汰**
   - 每轮候选方法必须被明确标记为：
     - `accept`
     - `iterate`
     - `abandon`
   - 只有被 `accept` 的候选，才允许进入下一个阶段；
   - 未被 `accept` 的候选不得混入后续主线。

6. **阶段冻结**
   - 阶段通过时，必须冻结：
     - 唯一 active config
     - compare table
     - 结果分析结论
     - baseline 对比结论
   - 未冻结前，下一阶段不得开始。

### 7B.1 滚动强基线面板定义

为避免“边做方法边失去 SOTA 参照”，定义如下滚动强基线面板：

#### `RB-Core`（S1 起必须具备）
- `TSDF`
- `DynaSLAM` family
- `RoDyn-SLAM` family（preferred）/ `NID-SLAM` family（fallback）

#### `RB-S1+`（S1 顶刊/顶会实验链补强后必须冻结）
- `RB-Core`
- + `NICE-SLAM`（representation / neural dense）
- + `4DGS-SLAM`（preferred）/ `D4DGS-SLAM`（fallback）

#### `RB-Extended`（S2 起必须具备）
- `RB-S1+`
- + `TSDF++ / Panoptic Multi-TSDFs / NICE-SLAM / MUTE-SLAM` 中至少一个实跑或可信对照

#### `RB-Full`（S5 起必须具备）
- `RB-Extended`
- + `NID-SLAM / RoDyn-SLAM / vMAP` 中至少一个 modern baseline

### 7B.2 阶段结果归因门槛

若某阶段要宣称“方法有效”，必须同时满足：
- [ ] 相对前一阶段冻结配置有明确 improvement；
- [ ] improvement 出现在固定开发协议上；
- [ ] improvement 与滚动强基线面板对比后仍有意义；
- [ ] 锁箱复验不翻转方向；
- [ ] 能写出“为什么有效”的结构解释，而不是只报数字。

---

## 阶段 S1：主命题收敛 + 评价闭环建立阶段

### 前置条件
- [x] `S0` 已完成
- [x] canonical source-of-truth、冻结快照、现状基线页、裁剪表已再次复核无漂移

### 阶段目标
本阶段目标不是“直接得到最终方法”，而是：
1. 固定唯一论文主命题；
2. **完成核心数据集与强基线的本地接入**，建立可持续运行的滚动强基线面板 `RB-Core`；
3. 在固定开发/锁箱协议上完成第一轮**方法设计-实验验证-基线对比-结果分析-方法筛选**闭环；
4. 产出唯一 active candidate，作为后续 `S2` 主创新阶段的起点。

### 本阶段要达成的指标/方面 SOTA 目标
- [x] 至少在一个开发子集上，选出一个 active candidate，使其相对当前 active baseline 满足以下任一条件：
  - TUM `Acc` 改善 `>= 0.10 cm`
  - Bonn `Acc` 改善 `>= 0.10 cm`
  - Bonn `ghost_reduction_vs_tsdf` 提升 `>= 3 pts`
  - TUM 或 Bonn `ghost_tail_ratio` 相对当前 active baseline 降低 `>= 3%`
- [x] 同时 `Comp-R` 不允许下降超过 `1 pt`
- [x] 同时不能被更新后的 `RB-Core` 三个 baseline（`TSDF / DynaSLAM / RoDyn-or-NID`）全面压制

### 必做任务
- [x] 固定论文主命题为：
  - `evidence-gradient + dual-state disentanglement + delayed geometry synthesis`
- [x] 写出 `3` 句话摘要版本
- [x] 写出 `1` 页方法概述
- [x] 明确：
  - active mainline
  - supporting modules
  - appendix-only modules
  - archived negative-result chain
- [x] 固定 `P10` 开发协议与锁箱协议：
  - 开发子集
  - 锁箱子集
  - protocol / seeds / frames / stride / metrics
- [x] 完成核心数据集本地接入与口径确认：
  - TUM walking 三序列
  - Bonn all3（至少 `balloon / balloon2 / crowd2`）
  - static sanity 子集
  - 明确哪些子集是开发集、哪些子集是锁箱集
- [x] 完成更新版 `RB-Core` 的**本地接入**并锁定：
  - `TSDF`
  - `DynaSLAM` family
  - `RoDyn-SLAM` family（preferred）/ `NID-SLAM` family（fallback）
- [x] 明确归档 `StaticFusion`：不再作为 `S1` blocking baseline
- [x] 冻结 1 条 `2025` recent dynamic dense 对比方案：`4DGS-SLAM`（preferred）/ `D4DGS-SLAM`（fallback）
- [x] 在 `TUM walking 三序列 / Bonn all3 / static sanity` 上形成**全核心数据集**而非单 smoke 子集的同口径 compare 计划与表格骨架
- [x] 为每个已接入 baseline 明确：
  - 本地 runner / wrapper 路径
  - 输入输出协议
  - 可重跑状态
  - 不可重跑时的阻塞点与替代计划
- [x] 设计 `1-3` 个第一轮 active candidate
- [x] 完成 controlled experiments
- [x] 完成与更新版 `RB-Core` 的同口径对比
- [x] 形成 `TUM walking 三序列 + Bonn all3 + static sanity` 的 per-sequence compare，而不是只保留 smoke / 单序列 gate
- [x] 完成结果分析，并将候选标记为 `accept / iterate / abandon`
- [x] 仅保留一个 `accept` 候选作为 `S2` 唯一起点

### 阶段产物
- [x] 主命题摘要
- [x] 主方法结构图
- [x] 主线/附录边界表
- [x] 开发/锁箱协议卡
- [x] 数据集本地接入状态表
- [x] `RB-Core` 本地接入状态表
- [x] `RB-Core` 对比表
- [x] 锁箱方向复验表
- [x] recent dynamic dense replacement baseline 决策页（`RoDyn-SLAM` vs `NID-SLAM`）
- [x] 外部基线完成度表（已完成 / 部分完成 / 未开始）
- [x] `RB-S1+` 冻结页（`NICE-SLAM + 4DGS-SLAM`）
- [x] 全核心数据集 per-sequence compare 表
- [x] 第一轮方法筛选记录
- [x] 唯一 active candidate 冻结页

### 阶段通过标准
- [x] 一段话可讲清与现有工作差异
- [x] 主文最多保留 `1` 核心命题 + `2` 支撑模块
- [x] 核心数据集已全部本地接入并完成协议确认
- [x] 更新后的 `RB-Core`（`TSDF + DynaSLAM + RoDyn-or-NID`）已本地接入、可运行且口径对齐
- [x] 第一轮 active candidate 在开发子集上达到本阶段指标目标
- [x] 在更新后的 `RB-Core` 判据下，active candidate 不被全面支配
- [x] `TUM walking 三序列 / Bonn all3 / static sanity` 已全部进入同口径 compare
- [x] `RB-S1+` 已冻结：明确 `4DGS-SLAM`（2025）+ `NICE-SLAM`
- [x] 本阶段所有产物已冻结并记录到任务书

### 进入下一阶段条件
- [x] 上述必做任务、产物与通过标准全部满足
- [x] 核心数据集与更新后的 `RB-Core / RB-S1+` 已冻结
- [x] 仅保留一个 `accept` 候选作为 active mainline
- [x] 当前已满足进入 `S2` 的全部条件

### 阶段 S1 完成记录（2026-03-08）

- 已完成主命题收敛，形成如下产物：
  - `processes/s1/S1_MAIN_THESIS_ABSTRACT.md`
  - `processes/s1/S1_METHOD_OVERVIEW.md`
  - `processes/s1/S1_METHOD_STRUCTURE_DIAGRAM.md`
  - `processes/s1/S1_MAINLINE_APPENDIX_BOUNDARY.md`
- 已固定开发/锁箱协议卡：`processes/s1/S1_P10_DEV_LOCKBOX_PROTOCOL_CARD.md`
- 已完成核心数据集本地接入状态表：
  - `processes/s1/S1_DATASET_LOCAL_INTEGRATION_STATUS.csv`
  - `processes/s1/S1_DATASET_LOCAL_INTEGRATION_STATUS.md`
- 已完成更新版 `RB-Core` 本地接入状态表：
  - `processes/s1/S1_RB_CORE_LOCAL_INTEGRATION_STATUS.csv`
  - `processes/s1/S1_RB_CORE_LOCAL_INTEGRATION_STATUS.md`
- 已完成 `RoDyn-SLAM` 本地接入与 core-dataset protocol check：
  - `scripts/external/run_rodyn_slam_runner.py`
  - `scripts/adapters/run_rodyn_slam_adapter.py`
  - `scripts/external/rodyn_no_dynamo_entry.py`
  - `scripts/run_s1_rodyn_core_protocol.py`
  - `processes/s1/S1_RODYN_CORE_PROTOCOL_CHECK.csv`
  - `processes/s1/S1_RODYN_CORE_PROTOCOL_CHECK.md`
- 已完成更新版 `RB-Core` 开发门槛 / 锁箱 gate：
  - `processes/s1/S1_RB_CORE_COMPARE_TUM_SMOKE.csv`
  - `processes/s1/S1_RB_CORE_COMPARE_TUM_SMOKE.md`
  - `processes/s1/S1_RB_CORE_LOCKBOX_DIRECTION_RECHECK.csv`
  - `processes/s1/S1_RB_CORE_LOCKBOX_DIRECTION_RECHECK_TABLE.md`
  - `processes/s1/S1_RB_CORE_LOCKBOX_DIRECTION_RECHECK.md`
- 已冻结 recent baseline 决策、外部基线完成度与 `RB-S1+`：
  - `processes/s1/S1_RECENT_BASELINE_REPLACEMENT_DECISION.md`
  - `processes/s1/S1_EXTERNAL_BASELINE_COMPLETION_STATUS.md`
  - `processes/s1/S1_RBS1PLUS_FREEZE.md`
  - `processes/s1/S1_EXPERIMENT_CHAIN_SUFFICIENCY.md`
  - `processes/s1/S1_CORE_DATASET_COMPARE_SKELETON.csv`
  - `processes/s1/S1_CORE_DATASET_COMPARE_SKELETON.md`
- 已完成第一轮候选筛选与唯一 active candidate 冻结：
  - `processes/s1/S1_RESULTS_ANALYSIS.md`
  - `processes/s1/S1_ACTIVE_CANDIDATE_FREEZE.md`
- 已完成本轮为 `RoDyn-SLAM` 本地接入所需的兼容修复：
  - `third_party/rodyn-slam/rodynslam.py`
  - `third_party/rodyn-slam/datasets/dataset.py`
  - `third_party/rodyn-slam/external/NumpyMarchingCubes/marching_cubes/src/marching_cubes.cpp`

### 阶段 S1 结论

结论：**阶段 `S1` 仍然已完成，且不需要回滚；但其 handoff candidate 与协议说明必须按 `2026-03-09` 的重评估结果修正表述。**

补充说明：
- `S1` 已完成的内容仍然成立：
  - 主命题收敛
  - `RB-Core` 本地闭环
  - recent baseline 冻结
  - `RB-S1+` 冻结
  - 第一轮方向筛选
- 但 `S1` 文档中所有 `frames=5 / stride=3 / seed=7` 的 gate/protocol 表述，后续都必须显式写出其 `max_points_per_frame`；
- `delayed-branch write-time target synthesis` 仍保留为 `S1 -> S2` 的 historical handoff label；
- 但经 `S2 current-code re-baseline` 复核，不能再把这条 handoff label 直接表述成“current-code superiority 已经保持成立”；
- 因此 `S1` 不回滚，但其对 `S2` 的交接说明必须改写为：方向收敛已完成，而 current-code 下的有效性需要在 `S2` 重新建立。

---

## 阶段 S2：核心方法突破 —— delayed-branch write-time target synthesis 阶段

### 前置条件
- [x] `S1` 已完成并在任务书中记录
- [x] `S1` 的唯一 active candidate 已冻结
- [x] `S1` 中完成的核心数据集与 `RB-Core` 本地接入仍可运行且口径不变
- [x] `RB-Core` 仍保持一致口径

### 阶段目标
围绕 `delayed-branch write-time target synthesis` 做完整闭环：
- 方法设计
- 实验验证
- 强基线对比
- 结果分析
- 迭代或淘汰

最终目标是：把 delayed branch 从“committed 的弱变体”提升为真正独立的 delayed-specific geometry path，并在一个关键维度上达到当前项目内新的子任务 SOTA。

### 本阶段要达成的指标/方面 SOTA 目标
本阶段通过时，冻结主配置必须满足以下至少 `2/4` 项：
- [ ] TUM 开发子集 `Acc <= 2.55 cm`
- [ ] Bonn 开发子集 `Acc <= 3.10 cm`
- [ ] Bonn 开发子集 `ghost_reduction_vs_tsdf >= 22%`
- [ ] 相对 `S1` 冻结配置，在至少一个数据集 family 上对 `RB-Core` 呈现明确净正优势（不是 mixed change）

并且必须同时满足：
- [ ] `Comp-R >= 98%`（开发子集）
- [ ] 锁箱方向不翻转

### 必做任务
- [x] 设计 delayed-specific write target
- [x] 明确其与 `d_static_obs / d_rear_obs / d_bg_obs / phi_geo` 的关系
- [x] 把 synthesized target 接入 delayed branch 写入
- [x] 设计 `1-3` 个 synthesis 候选版本
- [x] 做 controlled ablation：
  - 无 synthesis
  - 弱 synthesis
  - 完整 synthesis
- [x] 在固定开发协议下完成 compare 表
- [ ] 与 `RB-Core` 做同口径对比（当前只完成 `TUM dev quick` 的 partial compare）
- [x] 对结果做 accept/iterate/abandon 判断
- [ ] 对唯一 `accept` 候选跑锁箱复验（当前无 `accept`，仅对唯一继续 `iterate` 候选做了 partial lockbox recheck）

### 阶段产物
- [x] delayed-specific write target 定义页
- [x] controlled ablation compare 表
- [ ] 与 `RB-Core` 对比表（partial only）
- [x] 锁箱复验记录
- [x] `S2` 唯一继续配置冻结页

### 阶段通过标准
- [ ] 达到本阶段 `2/4` 量化目标
- [ ] `Comp-R` 不结构性崩塌
- [ ] 锁箱方向一致
- [x] 能明确说明收益来自 write-time target synthesis，而非伴随模块漂移
- [ ] 本阶段只保留一个 `accept` 候选

### 进入下一阶段条件
- [ ] 上述必做任务、产物与通过标准全部满足
- [ ] `S2` 的唯一 active configuration 已冻结
- [ ] **未满足前，绝对禁止进入 `S3`**

### 阶段 S2 完成记录（2026-03-09）

- 已完成 delayed-specific target 定义页：
  - `processes/s2/S2_DELAYED_WRITE_TARGET_DEFINITION.md`
- 已完成第一轮 + 第二轮 controlled ablation：
  - `processes/s2/S2_CONTROLLED_ABLATION_COMPARE.csv`
  - `processes/s2/S2_CONTROLLED_ABLATION_COMPARE.md`
- 已完成结果分析与淘汰判断：
  - `processes/s2/S2_RESULTS_ANALYSIS.md`
- 已完成 current-code canonical 锁定：
  - `processes/s2/S2_ACTIVE_CONFIGURATION_FREEZE.md`
  - `processes/s2/S2_CURRENT_CODE_CANONICAL_LOCK.md`
  - `processes/s2/S2_CURRENT_CODE_DRIFT_COMPARE.md`
- 已完成 Bonn-side family-specific calibration 的 archive / current-code 双层整理：
  - `processes/s2/S2_BONN_FAMILY_CALIBRATION_COMPARE.md`
- 已完成 geometry-conservative clipping 第三轮验证（historical archive）：
  - `processes/s2/S2_CONTROLLED_ABLATION_COMPARE.csv`（`11/12` 行）
- 已完成 Bonn local clipping refinement 的 current-code re-baseline：
  - `processes/s2/S2_BONN_LOCALCLIP_REFINEMENT_COMPARE.csv`
  - `processes/s2/S2_BONN_LOCALCLIP_REFINEMENT_COMPARE.md`
  - `processes/s2/S2_BONN_LOCALCLIP_REFINEMENT_ANALYSIS.md`
- 已完成 current-code drift 定位所需的 `05_anchor_noroute` 对照重跑：
  - `output/post_cleanup/s2_stage/05_anchor_noroute_recheck600`
- 已完成 `historical 05 -> 14` 失活链条排查与修复尝试报告：
  - `processes/s2/S2_RESTORE_05_TO_14_CHAIN_REPORT.md`
- 已完成 downstream `sync/export` 主线攻击与两轮候选淘汰：
  - `processes/s2/S2_DOWNSTREAM_EXPORT_CHAIN_COMPARE.csv`
  - `processes/s2/S2_DOWNSTREAM_EXPORT_CHAIN_COMPARE.md`
  - `processes/s2/S2_DOWNSTREAM_SOFTBANK_COMPARE.csv`
  - `processes/s2/S2_DOWNSTREAM_SOFTBANK_COMPARE.md`
  - `processes/s2/S2_DOWNSTREAM_CHAIN_ATTACK_REPORT.md`
- 已完成第三轮 `rear/bg state formation before export` 恢复实验：
  - `processes/s2/S2_REAR_BG_STATE_FORMATION_COMPARE.csv`
  - `processes/s2/S2_REAR_BG_STATE_FORMATION_COMPARE.md`
  - `processes/s2/S2_REAR_BG_STATE_FORMATION_ANALYSIS.md`
- 已完成 `rps_commit_score / rps_active / phi_rear / rho_rear` 激活链路实验：
  - `processes/s2/S2_RPS_COMMIT_ACTIVATION_DESIGN.md`
  - `processes/s2/S2_RPS_COMMIT_ACTIVATION_COMPARE.csv`
  - `processes/s2/S2_RPS_COMMIT_ACTIVATION_COMPARE.md`
  - `processes/s2/S2_RPS_COMMIT_ACTIVATION_ANALYSIS.md`
- 已完成 committed rear-bank quality enhancement 受控实验：
  - `processes/s2/S2_RPS_COMMITTED_BANK_QUALITY_DESIGN.md`
  - `processes/s2/S2_RPS_COMMITTED_BANK_QUALITY_COMPARE.csv`
  - `processes/s2/S2_RPS_COMMITTED_BANK_QUALITY_COMPARE.md`
  - `processes/s2/S2_RPS_COMMITTED_BANK_QUALITY_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/30_rps_commit_geom_bg_soft_bank_recheck`
- 已完成 Bonn competition-boundary 诊断与 competition-only 受控实验：
  - `processes/s2/S2_RPS_COMMITTED_BANK_COMPETITION_COMPARE.csv`
  - `processes/s2/S2_RPS_COMMITTED_BANK_COMPETITION_COMPARE.md`
  - `processes/s2/S2_RPS_COMMITTED_BANK_COMPETITION_DIAGNOSIS.md`
  - `processes/s2/S2_RPS_COMMITTED_BANK_COMPETITION_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/30_rps_commit_geom_bg_soft_bank_compete_all3`
  - `output/post_cleanup/s2_stage/34_bonn_compete_bgscore`
  - `output/post_cleanup/s2_stage/35_bonn_compete_softgap`
  - `output/post_cleanup/s2_stage/36_bonn_compete_softgap_support`
- 已完成 committed rear-bank admission / state persistence 受控实验：
  - `processes/s2/S2_RPS_COMMITTED_BANK_ADMISSION_COMPARE.csv`
  - `processes/s2/S2_RPS_COMMITTED_BANK_ADMISSION_COMPARE.md`
  - `processes/s2/S2_RPS_COMMITTED_BANK_ADMISSION_DIAGNOSIS.md`
  - `processes/s2/S2_RPS_COMMITTED_BANK_ADMISSION_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/30_rps_commit_geom_bg_soft_bank_admission_all3`
  - `output/post_cleanup/s2_stage/37_bonn_admission_gate_relax`
  - `output/post_cleanup/s2_stage/38_bonn_state_protect`
  - `output/post_cleanup/s2_stage/39_bonn_admission_gate_plus_protect`
- 已完成 rear-state geometry quality / spatial distribution 受控实验：
  - `processes/s2/S2_RPS_REAR_GEOMETRY_QUALITY_COMPARE.csv`
  - `processes/s2/S2_RPS_REAR_GEOMETRY_QUALITY_COMPARE.md`
  - `processes/s2/S2_RPS_REAR_GEOMETRY_QUALITY_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_REAR_GEOMETRY_QUALITY_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/38_bonn_state_protect_geom_quality_control`
  - `output/post_cleanup/s2_stage/40_bonn_geometry_aligned_admission`
  - `output/post_cleanup/s2_stage/41_bonn_geometry_occlusion_admission`
  - `output/post_cleanup/s2_stage/42_bonn_geometry_density_gate`
- 已完成 rear-point space redirect 受控实验：
  - `processes/s2/S2_RPS_SPACE_REDIRECT_COMPARE.csv`
  - `processes/s2/S2_RPS_SPACE_REDIRECT_COMPARE.md`
  - `processes/s2/S2_RPS_SPACE_REDIRECT_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_SPACE_REDIRECT_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/38_bonn_state_protect_space_redirect_control`
  - `output/post_cleanup/s2_stage/43_history_guided_background_location`
  - `output/post_cleanup/s2_stage/44_history_plus_ghost_suppress`
  - `output/post_cleanup/s2_stage/45_visual_evidence_anchor_strict`
- 已完成 strict history-visible / currently-obstructed 受控实验：
  - `processes/s2/S2_RPS_HISTORY_VISIBLE_OBSTRUCTED_COMPARE.csv`
  - `processes/s2/S2_RPS_HISTORY_VISIBLE_OBSTRUCTED_COMPARE.md`
  - `processes/s2/S2_RPS_HISTORY_VISIBLE_OBSTRUCTED_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_HISTORY_VISIBLE_OBSTRUCTED_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/38_bonn_state_protect_hvo_control`
  - `output/post_cleanup/s2_stage/46_history_background_only_admission`
  - `output/post_cleanup/s2_stage/47_history_visible_obstructed_manifold`
- 已完成 stable background manifold state 受控实验：
  - `processes/s2/S2_RPS_BACKGROUND_MANIFOLD_STATE_COMPARE.csv`
  - `processes/s2/S2_RPS_BACKGROUND_MANIFOLD_STATE_COMPARE.md`
  - `processes/s2/S2_RPS_BACKGROUND_MANIFOLD_STATE_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_BACKGROUND_MANIFOLD_STATE_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/38_bonn_state_protect_bg_manifold_control`
  - `output/post_cleanup/s2_stage/48_stable_background_memory_state`
  - `output/post_cleanup/s2_stage/49_relaxed_manifold_guided_generation`
- 已完成 dense manifold propagation / completion 受控实验：
  - `processes/s2/S2_RPS_DENSE_MANIFOLD_COMPARE.csv`
  - `processes/s2/S2_RPS_DENSE_MANIFOLD_COMPARE.md`
  - `processes/s2/S2_RPS_DENSE_MANIFOLD_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_DENSE_MANIFOLD_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/38_bonn_state_protect_dense_manifold_control`
  - `output/post_cleanup/s2_stage/50_dense_background_propagation`
  - `output/post_cleanup/s2_stage/51_geometry_guided_manifold_completion`
  - `output/post_cleanup/s2_stage/52_dual_scale_manifold_fusion`
- 已完成 surface-constrained manifold expansion 受控实验：
  - `processes/s2/S2_RPS_SURFACE_CONSTRAINED_COMPARE.csv`
  - `processes/s2/S2_RPS_SURFACE_CONSTRAINED_COMPARE.md`
  - `processes/s2/S2_RPS_SURFACE_CONSTRAINED_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_SURFACE_CONSTRAINED_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/38_bonn_state_protect_surface_constrained_control`
  - `output/post_cleanup/s2_stage/53_surface_adjacent_propagation`
  - `output/post_cleanup/s2_stage/54_normal_guided_manifold_extension`
  - `output/post_cleanup/s2_stage/55_surface_constrained_ray_projection`
- 已完成 occlusion-aware gap bridging 受控实验：
  - `processes/s2/S2_RPS_OCCLUSION_BRIDGE_COMPARE.csv`
  - `processes/s2/S2_RPS_OCCLUSION_BRIDGE_COMPARE.md`
  - `processes/s2/S2_RPS_OCCLUSION_BRIDGE_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_OCCLUSION_BRIDGE_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/38_bonn_state_protect_occlusion_bridge_control`
  - `output/post_cleanup/s2_stage/56_temporal_occlusion_tunneling`
  - `output/post_cleanup/s2_stage/57_historical_surface_rear_projection`
  - `output/post_cleanup/s2_stage/58_ghost_aware_surface_inpainting`
- 已完成 high-coverage bridge recovery 受控实验：
  - `processes/s2/S2_RPS_HIGH_COVERAGE_BRIDGE_COMPARE.csv`
  - `processes/s2/S2_RPS_HIGH_COVERAGE_BRIDGE_COMPARE.md`
  - `processes/s2/S2_RPS_HIGH_COVERAGE_BRIDGE_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_HIGH_COVERAGE_BRIDGE_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/38_bonn_state_protect_high_coverage_bridge_control`
  - `output/post_cleanup/s2_stage/59_relaxed_occlusion_tunneling`
  - `output/post_cleanup/s2_stage/60_cone_based_rear_projection`
  - `output/post_cleanup/s2_stage/61_hybrid_confidence_gating`
- 已完成 multi-candidate surface generation 受控实验：
  - `processes/s2/S2_RPS_MULTI_CANDIDATE_GENERATION_COMPARE.csv`
  - `processes/s2/S2_RPS_MULTI_CANDIDATE_GENERATION_COMPARE.md`
  - `processes/s2/S2_RPS_MULTI_CANDIDATE_GENERATION_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_MULTI_CANDIDATE_GENERATION_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/62_dense_patch_projection`
  - `output/post_cleanup/s2_stage/63_multi_hypothesis_depth_sampling`
  - `output/post_cleanup/s2_stage/64_patch_depth_hybrid_generation`
- 已完成 ghost-capped selectivity 受控实验：
  - `processes/s2/S2_RPS_GHOST_CAPPED_SELECTIVITY_COMPARE.csv`
  - `processes/s2/S2_RPS_GHOST_CAPPED_SELECTIVITY_COMPARE.md`
  - `processes/s2/S2_RPS_GHOST_CAPPED_SELECTIVITY_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_GHOST_CAPPED_SELECTIVITY_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/65_ghost_risk_prediction_filter`
  - `output/post_cleanup/s2_stage/66_geometry_constrained_admission`
  - `output/post_cleanup/s2_stage/67_topk_selective_generation`
- 已完成 discriminative feature fusion 受控实验：
  - `processes/s2/S2_RPS_DISCRIMINATIVE_FUSION_COMPARE.csv`
  - `processes/s2/S2_RPS_DISCRIMINATIVE_FUSION_COMPARE.md`
  - `processes/s2/S2_RPS_DISCRIMINATIVE_FUSION_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_DISCRIMINATIVE_FUSION_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/67_topk_selective_generation_df_control`
  - `output/post_cleanup/s2_stage/68_rear_front_score_competition`
  - `output/post_cleanup/s2_stage/69_depth_gap_validation`
  - `output/post_cleanup/s2_stage/70_fused_discriminator_topk`
- 已完成 occlusion-ordered conflict resolution 受控实验：
  - `processes/s2/S2_RPS_SEMANTIC_CLASSIFICATION_COMPARE.csv`
  - `processes/s2/S2_RPS_SEMANTIC_CLASSIFICATION_COMPARE.md`
  - `processes/s2/S2_RPS_SEMANTIC_CLASSIFICATION_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_SEMANTIC_CLASSIFICATION_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/71_occlusion_order_consistency_semantic`
  - `output/post_cleanup/s2_stage/72_local_geometric_conflict_resolution_semantic`
  - `output/post_cleanup/s2_stage/73_front_residual_aware_suppression_semantic`
- 已完成 static-persistence anchored recall 受控实验：
  - `processes/s2/S2_RPS_STATIC_ANCHORED_COMPARE.csv`
  - `processes/s2/S2_RPS_STATIC_ANCHORED_COMPARE.md`
  - `processes/s2/S2_RPS_STATIC_ANCHORED_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_STATIC_ANCHORED_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/72_local_geometric_conflict_resolution_static_anchor_control`
  - `output/post_cleanup/s2_stage/74_static_history_weight_boosting`
  - `output/post_cleanup/s2_stage/75_dynamic_shell_masking`
  - `output/post_cleanup/s2_stage/76_surface_persistent_anchoring`
- 已完成 hybrid optimization 受控实验：
  - `processes/s2/S2_RPS_HYBRID_OPTIMIZATION_COMPARE.csv`
  - `processes/s2/S2_RPS_HYBRID_OPTIMIZATION_COMPARE.md`
  - `processes/s2/S2_RPS_HYBRID_OPTIMIZATION_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_HYBRID_OPTIMIZATION_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/72_local_geometric_conflict_resolution_hybrid_control`
  - `output/post_cleanup/s2_stage/77_hybrid_boost_conflict`
  - `output/post_cleanup/s2_stage/78_conservative_anchoring`
  - `output/post_cleanup/s2_stage/79_feature_weighted_topk`
- 已完成 ray-consistency 受控实验：
  - `processes/s2/S2_RPS_RAY_CONSISTENCY_COMPARE.csv`
  - `processes/s2/S2_RPS_RAY_CONSISTENCY_COMPARE.md`
  - `processes/s2/S2_RPS_RAY_CONSISTENCY_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_RAY_CONSISTENCY_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/72_local_geometric_conflict_resolution_ray_control`
  - `output/post_cleanup/s2_stage/80_ray_penetration_consistency`
  - `output/post_cleanup/s2_stage/81_unobserved_space_veto`
  - `output/post_cleanup/s2_stage/82_static_neighborhood_coherence`
- 已完成 topology-constraint 受控实验：
  - `processes/s2/S2_RPS_TOPOLOGY_CONSTRAINT_COMPARE.csv`
  - `processes/s2/S2_RPS_TOPOLOGY_CONSTRAINT_COMPARE.md`
  - `processes/s2/S2_RPS_TOPOLOGY_CONSTRAINT_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_TOPOLOGY_CONSTRAINT_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/80_ray_penetration_topology_control`
  - `output/post_cleanup/s2_stage/83_minimum_thickness_topology_filter`
  - `output/post_cleanup/s2_stage/84_front_back_normal_consistency`
  - `output/post_cleanup/s2_stage/85_occlusion_ray_convergence_constraint`
- 已完成 plane-attribution 受控实验：
  - `processes/s2/S2_RPS_PLANE_ATTRIBUTION_COMPARE.csv`
  - `processes/s2/S2_RPS_PLANE_ATTRIBUTION_COMPARE.md`
  - `processes/s2/S2_RPS_PLANE_ATTRIBUTION_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_PLANE_ATTRIBUTION_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/86_rear_plane_clustering_snapping`
  - `output/post_cleanup/s2_stage/87_front_mask_guided_back_projection`
  - `output/post_cleanup/s2_stage/88_occlusion_depth_hypothesis_validation`
- 已完成 pairing-evidence 受控实验：
  - `processes/s2/S2_RPS_PAIRING_EVIDENCE_COMPARE.csv`
  - `processes/s2/S2_RPS_PAIRING_EVIDENCE_COMPARE.md`
  - `processes/s2/S2_RPS_PAIRING_EVIDENCE_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_PAIRING_EVIDENCE_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/89_front_back_surface_pairing_guard`
  - `output/post_cleanup/s2_stage/90_background_plane_evidence_accumulation`
  - `output/post_cleanup/s2_stage/91_occlusion_depth_hypothesis_tb_protection`
- 已完成 HSSA 受控实验：
  - `processes/s2/S2_RPS_HSSA_COMPARE.csv`
  - `processes/s2/S2_RPS_HSSA_COMPARE.md`
  - `processes/s2/S2_RPS_HSSA_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_HSSA_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/92_multi_view_ray_support_aggregation`
  - `output/post_cleanup/s2_stage/93_spatial_neighborhood_density_clustering`
  - `output/post_cleanup/s2_stage/94_historical_tsdf_consistency_reactivation`
- 已完成 balloon-cluster 受控实验：
  - `processes/s2/S2_RPS_BALLOON_CLUSTER_COMPARE.csv`
  - `processes/s2/S2_RPS_BALLOON_CLUSTER_COMPARE.md`
  - `processes/s2/S2_RPS_BALLOON_CLUSTER_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_BALLOON_CLUSTER_ANALYSIS.md`
  - `processes/s2/S2_RPS_DEEP_EXPLORE_COMPARE.csv`
  - `processes/s2/S2_RPS_DEEP_EXPLORE_COMPARE.md`
  - `processes/s2/S2_RPS_DEEP_EXPLORE_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_DEEP_EXPLORE_ANALYSIS.md`
  - `processes/s2/S2_RPS_DEEP_EXPLORE_LITERATURE.md`
  - `output/post_cleanup/s2_stage/95_geodesic_balloon_consistency`
  - `output/post_cleanup/s2_stage/96_support_cluster_model_fitting`
  - `output/post_cleanup/s2_stage/97_global_map_anchoring`
  - `output/post_cleanup/s2_stage/98_geodesic_support_diffusion`
  - `output/post_cleanup/s2_stage/99_manhattan_plane_completion`
  - `output/post_cleanup/s2_stage/100_cluster_view_inpainting`
- 已完成 Acc-tightening 受控实验：
  - `processes/s2/S2_RPS_ACC_TIGHTENING_COMPARE.csv`
  - `processes/s2/S2_RPS_ACC_TIGHTENING_COMPARE.md`
  - `processes/s2/S2_RPS_ACC_TIGHTENING_DISTRIBUTION.md`
  - `processes/s2/S2_RPS_ACC_TIGHTENING_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/101_manhattan_plane_projection_hard_snapping`
  - `output/post_cleanup/s2_stage/102_scale_drift_correction`
  - `output/post_cleanup/s2_stage/103_local_cluster_refinement`
- 已完成 upstream-geometry 受控实验：
  - `processes/s2/S2_RPS_UPSTREAM_GEOMETRY_COMPARE.csv`
  - `processes/s2/S2_RPS_UPSTREAM_GEOMETRY_COMPARE.md`
  - `processes/s2/S2_RPS_UPSTREAM_GEOMETRY_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/104_depth_bias_minus1cm`
  - `output/post_cleanup/s2_stage/105_depth_bias_plus1cm`
- 已完成 geometry-chain coupling 受控实验：
  - `processes/s2/S2_RPS_GEOMETRY_CHAIN_COUPLING_COMPARE.csv`
  - `processes/s2/S2_RPS_GEOMETRY_CHAIN_COUPLING_COMPARE.md`
  - `processes/s2/S2_RPS_GEOMETRY_CHAIN_COUPLING_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/108_geometry_chain_coupled_direct`
  - `output/post_cleanup/s2_stage/109_geometry_chain_coupled_projected`
  - `output/post_cleanup/s2_stage/110_geometry_chain_coupled_conservative`
- 已完成 Native Mainline Integration：
  - `processes/s2/S2_RPS_NATIVE_MAINLINE_INTEGRATION_COMPARE.csv`
  - `processes/s2/S2_RPS_NATIVE_MAINLINE_INTEGRATION_REPORT.md`
  - `processes/s2/S2_NATIVE_MAINLINE_GAP_ANALYSIS.md`
  - `output/post_cleanup/s2_stage/111_native_geometry_chain_direct`
  - `output/post_cleanup/s2_stage/112_native_geometry_chain_projected`
- 已完成 lockbox 方向复验记录（partial）：
  - `processes/s2/S2_LOCKBOX_RECHECK.md`
- 已完成 `TUM dev quick` 的 partial `RB-Core` 对比说明：
  - `processes/s2/S2_RB_CORE_COMPARE_TUM_DEV.md`
- 已完成本轮 `S2` 方法实现与脚本：
  - `scripts/run_s2_write_time_synthesis.py`
  - `scripts/run_s2_bonn_localclip_refine.py`
  - `scripts/run_s2_downstream_export_chain.py`
  - `scripts/run_s2_downstream_softbank.py`
  - `scripts/run_s2_rear_bg_state_recovery.py`
  - `scripts/run_s2_rps_commit_activation.py`
  - `scripts/run_s2_rps_commit_bank_quality.py`
  - `scripts/run_s2_rps_commit_competition.py`
  - `scripts/run_s2_rps_commit_admission.py`
  - `scripts/run_s2_rps_rear_geometry_quality.py`
  - `scripts/run_s2_rps_space_redirect.py`
  - `scripts/run_s2_rps_history_visible_obstructed.py`
  - `scripts/run_s2_rps_background_manifold_state.py`
  - `scripts/run_s2_rps_dense_manifold.py`
  - `scripts/run_s2_rps_surface_constrained.py`
  - `scripts/run_s2_rps_occlusion_bridge.py`
  - `scripts/run_s2_rps_high_coverage_bridge.py`
  - `scripts/run_s2_rps_multi_candidate_generation.py`
  - `scripts/run_s2_rps_ghost_capped_selectivity.py`
  - `scripts/run_s2_rps_discriminative_fusion.py`
  - `scripts/run_s2_rps_occlusion_conflict.py`
  - `scripts/run_s2_rps_static_anchored.py`
  - `scripts/run_s2_rps_hybrid_optimization.py`
  - `scripts/run_s2_rps_ray_consistency.py`
  - `scripts/run_s2_rps_topology_constraint.py`
  - `scripts/run_s2_rps_plane_attribution.py`
  - `scripts/run_s2_rps_pairing_evidence.py`
  - `scripts/run_s2_rps_hssa.py`
  - `scripts/run_s2_rps_balloon_cluster.py`
  - `scripts/run_s2_rps_deep_explore.py`
  - `scripts/run_s2_rps_acc_tightening.py`
  - `scripts/run_s2_rps_upstream_geometry.py`
  - `scripts/run_s2_rps_geometry_chain_coupling.py`
  - `scripts/run_s2_rps_native_mainline_integration.py`
  - `egf_dhmap3d/P10_method/bg_manifold.py`
  - `egf_dhmap3d/P10_method/geometry_chain.py`
  - `egf_dhmap3d/P10_method/rps_plane_attribution.py`
  - `egf_dhmap3d/P10_method/rps_selectivity.py`
  - `egf_dhmap3d/P10_method/ptdsf.py`
  - `egf_dhmap3d/P10_method/rps_export_competition.py`
  - `egf_dhmap3d/P10_method/rps_admission.py`
  - `egf_dhmap3d/P10_method/rps.py`
  - `egf_dhmap3d/P10_method/rps_commit.py`
  - `egf_dhmap3d/core/config.py`
  - `scripts/run_egf_3d_tum.py`
  - `scripts/run_benchmark.py`

### 阶段 S2 结论
统一解释“dynamic suppression 为何仍是优势方向、但 current-code canonical 又表现不佳”的说明页：`processes/s2/S2_DYNAMIC_SUPPRESSION_STATUS_EXPLANATION.md`


结论：**本轮已完成 `current-code re-baseline + canonical refresh + drift localization`，但 `S2` 仍未通过，且本阶段当前绝对不能进入 `S3`。**

本轮最重要结论：
- 已确认 historical `S2` dev quick 文档漏写了关键协议项：`max_points_per_frame=600`；
- 在修正协议后，current-code control `14` 的 canonical 结果为：
  - `TUM Acc ≈ 0.9355 cm`, `TUM Comp-R ≈ 68.53%`
  - `Bonn Acc ≈ 2.8864 cm`, `Bonn Comp-R ≈ 83.57%`
  - `Bonn ghost_reduction_vs_tsdf ≈ -8.00%`
- `15/16` 对 `14` 没有任何有效增益，因此当前**不应继续沿 `16` 做 Bonn-only 微调**；
- 额外对照重跑与 `ptdsf_diag` 说明：current-code `14` 与 `05` 在 target synthesis 层仍有显著差异，但这些差异在 downstream `write -> sync -> export` 链条被抹平，因此 historical `05 -> 14` 收益链条尚未恢复；
- 本轮继续围绕 downstream 链设计了 `17-22` 两轮 export family 候选，但全部 zero-delta / abandon，且 `rear_selected / bg_selected / soft_bank_on` 全为 `0`；
- 在第三轮 `23/24/25` 中，`25_rear_bg_coupled_formation` 首次把 `bg_w / bg_cand_w` 拉成全量非零，并把 `TUM Comp-R` 从 `68.53%` 提到 `91.90%`，说明 `rear/bg state formation before export` 确实是有效方向；
- 本轮 `28/29/30` 进一步完成了 `rps_commit_score / rps_active / phi_rear / rho_rear` 激活链路实验，其中 `30_rps_commit_geom_bg_soft_bank` 首次实现 `rear_selected > 0`；
- 本轮继续以 `30_rps_commit_geom_bg_soft_bank` 为唯一工作起点，完成了 `31/32/33` 三个 committed rear-bank quality enhancement 候选；
- `30` 的 current-code recheck 与既有冻结数值完全一致，说明上一轮新增开关没有引入口径漂移；
- 随后本轮又按 `TUM walking all3 + Bonn all3` 的同一低帧协议，完成了 `34/35/36` 三个 Bonn competition-only 变体；
- extract-only 诊断显示：当前 Bonn 真正进入 extract competition 的 rear 候选总数只有 `7`，且这 `7` 个样本已经 `7/7` 全部胜出；
- 因此“当前瓶颈在最终 front-vs-rear competition boundary”这一工作假设已被证伪；
- 本轮进一步按同一 family-mean 口径完成了 `37/38/39` 三个 admission / state persistence 变体，并确认当前控制组的最窄流失点是 extract admission 的 `score_gate`；
- 控制组 `30` 在 Bonn all3 上表现为：`committed_cells_sum = 179`，但 `extract_hard_commit_on_sum = 7`，其中 `extract_fail_score_sum = 172`；
- `37/38/39` 说明 admission 主线是有效的：
  - `37` 把 `bonn_extract_rear_selected_sum` 从 `7` 提到 `193`，`ghost_reduction_vs_tsdf` 提到 `15.00%`；
  - `38` 把 `bonn_extract_rear_selected_sum` 提到 `108`，并取得 admission-phase 最佳 `ghost_reduction_vs_tsdf = 15.47%`；
  - `39` 虽把 `bonn_extract_rear_selected_sum` 提到 `202`，但 `ghost_reduction_vs_tsdf` 反而弱于 `38`；
- 本轮继续围绕 `38` 做 rear-state geometry quality / spatial distribution 诊断与三组几何质量候选 `40/41/42`：
  - `38` 的 rear points 空间分布显示：`108` 个 rear points 中只有 `1` 个落在 true background，`10` 个落在 ghost region，`97` 个落在 hole/noise 区域；
  - `40/41/42` 虽然把 true-background rear points 从 `1` 提到 `2/3/2`，但同时也把 ghost-region rear points提高到 `16/18/18`，因此 Bonn `ghost_reduction_vs_tsdf` 反而从 `15.47%` 退回 `15.22%`；
- 本轮进一步围绕 `38` 做 space-redirect 三组候选 `43/44/45`：
  - `43/45` 把 true-background rear points 从 `1` 提到 `5`，但 ghost-region rear points同步从 `10` 膨胀到 `26`，Bonn `ghost_reduction_vs_tsdf` 反而回落到 `14.86%`；
  - `44` 试图用 ghost-aware suppression 抑制误入，但仍只做到 `true_background = 4` / `ghost = 25`，Bonn `ghost_reduction_vs_tsdf = 15.22%`，仍低于 `38`；
- 本轮进一步按“历史可见且当前被遮挡”硬约束做了 `46/47` 两个变体：
  - `46_history_background_only_admission` 与 `47_history_visible_obstructed_manifold` 都把 rear points 直接压到 `0`；
  - Bonn `ghost_reduction_vs_tsdf` 同时退回到 `13.29%`，说明严格 HVO gate 在当前低帧协议下会过度裁剪 rear generation；
- 本轮进一步引入 stable background manifold state，做了 `48/49` 两个变体：
  - `48_stable_background_memory_state` 把 rear points从 `108` 压到 `29`，但也把 `ghost` 从 `10` 压到 `3`、`hole/noise` 从 `97` 压到 `25`；
  - `49_relaxed_manifold_guided_generation` 把 rear points 恢复到 `39`，但 `true_background` 仍然只有 `1`，Bonn `ghost_reduction_vs_tsdf` 仍只有 `13.50%`；
- 本轮进一步沿“背景支持场稠密化与传播”做了 `50/51/52` 三个变体：
  - `50_dense_background_propagation` 把 rear points 恢复到 `67`，并把 `ghost` 降到 `7`，但 `true_background` 仍停在 `1`，Bonn `ghost_reduction_vs_tsdf = 13.75%`；
  - `51_geometry_guided_manifold_completion` 把 `true_background` 从 `1` 提到 `2`，rear points 恢复到 `88`，但 `ghost` 同时回升到 `12`，Bonn `ghost_reduction_vs_tsdf = 14.41%`；
  - `52_dual_scale_manifold_fusion` 把 rear points 恢复到 `112`，但 `true_background` 仍只有 `1`，`ghost` 升到 `13`，Bonn `ghost_reduction_vs_tsdf = 14.71%`；
- 本轮进一步沿“历史背景表面显式约束传播”做了 `53/54/55` 三个变体：
  - `53_surface_adjacent_propagation` 把 `true_background` 从 `1` 提到 `3`，rear points 保持 `94`，但 `ghost` 升到 `13`，Bonn `ghost_reduction_vs_tsdf = 15.17%`；
  - `54_normal_guided_manifold_extension` 也把 `true_background` 提到 `3`，并把 `ghost` 控制在 `12`，但 Bonn `ghost_reduction_vs_tsdf = 14.77%`；
  - `55_surface_constrained_ray_projection` 保持 `rear_points = 101`，但 `true_background = 3` / `ghost = 14`，Bonn `ghost_reduction_vs_tsdf = 14.51%`；
- 本轮进一步沿“遮挡缝隙桥接”做了 `56/57/58` 三个变体：
  - `56_temporal_occlusion_tunneling` 把 `ghost` 压到 `9`，并把 `true_background` 提到 `2`，但 rear 总量只剩 `57`，Bonn `ghost_reduction_vs_tsdf = 15.02%`；
  - `57_historical_surface_rear_projection` 把 `true_background` 提到 `3`，同时把 `ghost` 控制在 `9`，但 rear 总量仍只有 `58`，Bonn `ghost_reduction_vs_tsdf = 15.02%`；
  - `58_ghost_aware_surface_inpainting` 把 `ghost` 控制在 `9`，但 `true_background` 只到 `2`，Bonn `ghost_reduction_vs_tsdf = 14.87%`；
- 本轮进一步按“桥接覆盖量恢复”做了 `59/60/61` 三个变体：
  - `59_relaxed_occlusion_tunneling` 把 rear 总量恢复到 `65`，并把 `ghost` 保持在 `9`，但 `true_background` 仍只有 `2`，Bonn `ghost_reduction_vs_tsdf = 15.02%`；
  - `60_cone_based_rear_projection` 在当前协议下与 `59` 等价：`rear = 65` / `true_background = 2` / `ghost = 9`；
  - `61_hybrid_confidence_gating` 也只达到 `rear = 65` / `true_background = 2` / `ghost = 9`，未能恢复到 `80+` 覆盖量；
- 本轮进一步按“multi-candidate surface generation”做了 `62/63/64` 三个变体：
  - `62_dense_patch_projection` 把 rear 总量直接推到 `378`、`true_background` 提到 `7`，但 `ghost` 也膨胀到 `53`，Bonn `ghost_reduction_vs_tsdf` 反而回落到 `15.02%`；
  - `63_multi_hypothesis_depth_sampling` 把 `ghost` 控制在 `23`，但 `true_background` 只到 `3`，Bonn `ghost_reduction_vs_tsdf = 15.17%`，仍未超过 `38`；
  - `64_patch_depth_hybrid_generation` 把 rear 总量推到 `442`、`true_background` 提到 `8`，并把 Bonn `ghost_reduction_vs_tsdf` 推到 `16.48%`，但 `ghost` 同步膨胀到 `60`、`Comp-R` 仍停在 `70.80%`；
- 本轮进一步按“ghost-capped selectivity”做了 `65/66/67` 三个变体：
  - `65_ghost_risk_prediction_filter` 没有切掉任何 rear points，结果与 `64` 等价：`rear = 442` / `true_background = 8` / `ghost = 60` / Bonn `ghost_reduction_vs_tsdf = 16.48%`；
  - `66_geometry_constrained_admission` 反而把 rear 总量推到 `555`、`true_background` 提到 `15`，但 `ghost` 同步膨胀到 `84`，Bonn `ghost_reduction_vs_tsdf` 回落到 `15.72%`；
  - `67_topk_selective_generation` 把 rear 总量压到 `231`，并把 Bonn `ghost_reduction_vs_tsdf` 推到 `22.16%`，但 `true_background` 只剩 `3`、`ghost` 仍高达 `38`，没有满足 `ghost <= 15` 和 `TB >= 6` 的双约束；
- 本轮进一步按“discriminative feature fusion”做了 `68/69/70` 三个变体：
  - `68_rear_front_score_competition` 在引入 `rear_score - alpha * front_score` 竞争特征后，把 Bonn `ghost_reduction_vs_tsdf` 进一步推到 `23.52%`，并把 `ghost` 从 `41` 压到 `32`，但 `true_background` 同时从 `3` 掉到 `2`；
  - `69_depth_gap_validation` 与 `70_fused_discriminator_topk` 都退化为 `rear = 0` 的零点解，虽然数值上 Bonn `ghost_reduction_vs_tsdf = 35.21%`，但没有可用几何覆盖，因此不具备方法价值；
- 本轮进一步按“semantic-aware ghost classification”做了 `71/72/73` 三个变体：
  - `71_occlusion_order_consistency` 把 `true_background` 从 `2` 回升到 `4`，但 `ghost` 仍有 `31`，且 Bonn `ghost_reduction_vs_tsdf` 回落到 `22.49%`；
  - `72_local_geometric_conflict_resolution` 在保持 Bonn `ghost_reduction_vs_tsdf = 27.27%` 的同时，把 `ghost` 压到 `24`，并把 `true_background` 从 `2` 回升到 `4`，是当前 semantic-aware 分支的最佳工作基数；
  - `73_front_residual_aware_suppression` 把覆盖拉到 `rear = 257`，但 `ghost` 回升到 `34`、`true_background` 仍只有 `4`；
- 因此当前主瓶颈已进一步修正为：`higher-order semantic/dynamic-context features can improve the current branch from roughly (TB=2, Ghost=24-32) to roughly (TB=4, Ghost=24) while keeping ghost_reduction well above 22%, but they still do not produce enough separability to hit Ghost<=15 and TB>=8 simultaneously`。
- 本轮进一步按“static-persistence anchored recall”做了 `74/75/76` 三个变体：
  - `74_static_history_weight_boosting` 把 `true_background` 从 `3` 提到 `4`，并维持 Bonn `Comp-R = 70.43%`，但 `ghost` 上升到 `33`，且 Bonn `ghost_reduction_vs_tsdf = 21.91%` 没有守住 `22%`；
  - `75_dynamic_shell_masking` 把 `ghost` 压到 `28`，但 `true_background` 仍只有 `3`，且 Bonn `Comp-R` 退回到 `68.82%`；
  - `76_surface_persistent_anchoring` 也只做到 `true_background = 3` / `ghost = 29`，Bonn `Comp-R = 68.81%`；
- 本轮进一步按“hybrid optimization”做了 `77/78/79` 三个变体，并先修复了静态锚定统计链路：
  - `egf_dhmap3d/core/voxel_hash.py` 已补上 `history_anchor / surface_anchor / surface_distance / dynamic_shell` 的 `last_extract_stats` 写回；复跑后 `history_anchor_mean` 已恢复为非零，Bonn `72/77/78/79` 分别为 `0.246 / 0.247 / 0.246 / 0.250`，`surface_anchor_mean` 均为 `1.000`，确认本轮特征统计 Bug 已修复；
  - `77_hybrid_boost_conflict` 把 `true_background` 从 `3` 提到 `4`，并把 `ghost` 从 `30` 降到 `29`，但 Bonn `ghost_reduction_vs_tsdf` 从 `22.10%` 回落到 `21.83%`，没有守住 `22%` 红线；
  - `78_conservative_anchoring` 把 rear 总量推到 `252`，但 `true_background` 仍只有 `3`、`ghost` 反弹到 `32`，Bonn `ghost_reduction_vs_tsdf = 21.37%`；
  - `79_feature_weighted_topk` 守住了 Bonn `ghost_reduction_vs_tsdf = 22.24%`，并把 `ghost` 压到 `29`，但 `true_background` 仍停在 `3`，没有优于 `72` 的召回分布；
- 本轮进一步按“ray consistency / observation veto / static coherence”做了 `80/81/82` 三个变体：
  - 新特征已摆脱全零：控制组 `72` 的 Bonn `penetration_mean = 0.099`、`observation_support_mean = 0.183`、`static_coherence_mean = 0.823`；并且 TB/Noise 均值已出现分离，例如控制组 `penetration(TB/Noise) = 0.156 / 0.105`，`observation(TB/Noise) = 0.246 / 0.183`；
  - `80_ray_penetration_consistency` 把 `ghost` 压到 `15`，并维持 Bonn `ghost_reduction_vs_tsdf = 22.66%`，但 `true_background` 仍只有 `4`，`noise_ratio` 仍高达 `0.840`；
  - `81_unobserved_space_veto` 把 Bonn `ghost_reduction_vs_tsdf` 维持在 `22.18%`，并把 `ghost` 压到 `25`，但 `true_background` 仍只有 `4`，`noise_ratio = 0.874`；
  - `82_static_neighborhood_coherence` 把 `ghost` 压到 `18`，并维持 Bonn `ghost_reduction_vs_tsdf = 22.70%`，但 `true_background` 仍只有 `4`，且 `static_coherence_mean` 再次饱和到 `1.000`；
- 本轮进一步按“front-back topology constraint”做了 `83/84/85` 三个变体：
  - `83_minimum_thickness_topology_filter` 通过最小厚度约束把 Bonn `hole_or_noise_sum` 从 `214` 压到 `106`、`ghost` 压到 `16`，并保持 `ghost_reduction_vs_tsdf = 22.49%`，但 `true_background` 仍停在 `4`；
  - `84_front_back_normal_consistency` 通过法向连贯性把 `hole_or_noise_sum` 压到 `105`，但 `ghost_reduction_vs_tsdf` 回落到 `21.86%`，且 `true_background` 仍只有 `4`；
  - `85_occlusion_ray_convergence_constraint` 通过相邻 rear 候选的拓扑收敛约束把 `hole_or_noise_sum` 压到 `99`、`ghost` 压到 `22`，并保持 `ghost_reduction_vs_tsdf = 22.30%`，但 `true_background` 仍没有突破 `4`；
  - 拓扑统计已形成有效分离：以 `85` 为例，`thickness kept/dropped = 0.383 / 0.055 m`，`normal kept/dropped = 0.940 / 0.694`，`ray_convergence kept/dropped = 0.729 / 0.424`，说明结构级约束明显优于此前的 cell-local 标量特征；
- 本轮进一步按“front-back plane attribution”做了 `86/87/88` 三个变体：
  - `86_rear_plane_clustering_snapping`、`87_front_mask_guided_back_projection`、`88_occlusion_depth_hypothesis_validation` 全部触发了灾难性过清洗：`true_background` 直接跌到 `0`，Bonn `Comp-R` 跌到 `67-69%`，说明“先假设是噪声再做平面吸附”会误杀仅存的 TB；
- 本轮进一步按“pairing evidence”做了 `89/90/91` 三个变体：
  - `89_front_back_surface_pairing_guard` 在保护 `115` 个点、吸附 `96` 个点后，只得到 `TB=4 / Ghost=20 / Noise=95 / Bonn ghost_reduction_vs_tsdf = 22.61% / Comp-R = 70.17%`，避免了 `TB=0`，但没有任何 TB 回升；
  - `90_background_plane_evidence_accumulation` 在保护 `115` 个点、吸附 `69` 个点后，把 `Noise` 降到 `91`、`Comp-R` 微升到 `70.24%`，但 `ghost_reduction_vs_tsdf` 回落到 `21.58%`，未守住动态抑制红线；
  - `91_occlusion_depth_hypothesis_tb_protection` 在保护 `115` 个点、只吸附 `8` 个点的保守条件下守住了 `TB=4 / Ghost=19 / Noise=96 / Bonn ghost_reduction_vs_tsdf = 23.52% / Comp-R = 70.17%`，证明“有证据保留”能避免再次过清洗，但仍没有形成 `Noise -> TB` 转化；
- 本轮进一步按“HSSA / hidden-surface support aggregation”做了 `92/93/94` 三个变体：
  - `92_multi_view_ray_support_aggregation` 证明多视角支持聚合本身是有效信号：在不破坏 Ghost/Comp-R 的前提下，把 `TB` 从 `4` 提到 `5`，并把 `support_score(TB/Noise)` 差值从 `0.064` 拉到 `0.072`，但 `tb_noise_correlation` 仍为 `0.991`；
  - `93_spatial_neighborhood_density_clustering` 首次把 `TB` 从 `4` 直接推到 `13`，同时守住 `ghost_reduction_vs_tsdf = 23.59%` 与 `Ghost = 19`；说明“平坦支持簇 + 非破坏式 patch 扩点”确实能把隐藏表面支持聚合成真实背景假设；
  - `94_historical_tsdf_consistency_reactivation` 在更保守的历史一致性约束下也把 `TB` 提到 `8`，并守住 `ghost_reduction_vs_tsdf = 22.45%` 与 `Ghost = 19`，证明历史稳定性可以作为 patch 召回的有效先验；
  - 但 `92/93/94` 的共同失败点也很明确：`tb_noise_correlation` 仍停在 `0.991` 左右，说明支持增益几乎全部集中在单个序列，family 层面的 `TB-Noise` 解耦尚未形成；
- 本轮进一步按“balloon support-cluster formation”做了 `95/96/97` 三个变体：
  - `95_geodesic_balloon_consistency` 证明仅依靠测地平滑阈值不足以推动分布变化：由于没有任何簇通过 retain 门槛，结果与 `93` 等价；
  - `96_support_cluster_model_fitting` 首次真正打破相关性死锁：通过 `plane-band` 模型保留 crowd2 中的低方差支持簇，把 `TB` 保持在 `12`、把 `Noise` 压到 `23`、把 `Ghost` 压到 `6`，并把 `tb_noise_correlation` 从 `0.991` 拉到 `-0.756`，同时 Bonn `ghost_reduction_vs_tsdf` 提升到 `31.04%`；
  - `97_global_map_anchoring` 在更严格的锚定半径下进一步把 `Noise` 压到 `22`、`Ghost` 压到 `5`，并把 `tb_noise_correlation` 进一步压到 `-0.786`，但 `TB` 回落到 `10`；
  - `95/96/97` 共同说明：相关性死锁确实不是不可打破，问题关键不是“能否恢复 TB”，而是“能否把 crowd2 的伪支撑簇收缩到稳定表面模型上”；cluster-level fitting / anchoring 已经解决了这一点；
- 本轮进一步按“deep explore / purified cluster completion”做了 `98/99/100` 三个变体：
  - `98_geodesic_support_diffusion` 基于 `97` 的 purified cluster，在 cluster plane 上做受限几何传播与 donor-band completion，把 Bonn `Comp-R` 从 `68.90%` 拉回到 `70.08%`，同时保持 `tb_noise_correlation = -0.327`、`ghost_reduction_vs_tsdf = 28.00%` 与 `TB = 40`；
  - `99_manhattan_plane_completion` 用结构化平面补全替代简单膨胀，也把 Bonn `Comp-R` 拉回到 `70.08%`，并保持 `tb_noise_correlation = -0.225`、`ghost_reduction_vs_tsdf = 28.28%` 与 `TB = 39`；这是当前 deep-explore 支链里 `Comp-R / Ghost` 折中最好的配置；
  - `100_cluster_view_inpainting` 用 cluster-conditioned view patch completion 同样把 Bonn `Comp-R` 拉回到 `70.08%`，但 `ghost` 回升到 `24`、相关性仅 `-0.115`，说明 2D 视角补洞更容易把边界噪声一并带回；
  - `98/99/100` 共同说明：文献启发的“purified cluster completion”确实能在不破坏判别力的前提下恢复完整性，且优于简单的无约束扩张，因为它们都严格依赖 `97` 已验证的高置信 cluster plane，而不是重新从全量 Noise 搜索；
- 本轮进一步按“Acc tightening”做了 `101/102/103` 三个变体：
  - `101_manhattan_plane_projection_hard_snapping` 确实把 `mean_distance_to_plane` 从 `0.2208` 压到 `0.2143`，但 Bonn `Acc` 仍停在 `4.347 cm`，且 `TB/Ghost/Noise` 分布反而轻微恶化到 `35/23/32`；
  - `102_scale_drift_correction` 显著恶化：Bonn `Acc` 拉高到 `5.269 cm`、`Comp-R` 跌到 `68.50%`，同时 `mean_distance_to_plane_after` 暴涨到 `0.5763`，说明当前并没有可用的几何尺度漂移校正信号；
  - `103_local_cluster_refinement` 仅把 `mean_distance_to_plane` 从 `0.2166` 微降到 `0.2164`，Bonn `Acc` 仍为 `4.346 cm`，说明局部 cluster tightening 几乎不触及当前误差主因；
  - `101/102/103` 共同说明：当前 `Acc` 的瓶颈已经不再是简单的平面厚度噪声，硬投影与局部紧化都不能把 `4.31 cm` 往 `3.10 cm` 推进，反而暴露出更像是 `front-side bias / geometry distortion` 的系统误差；
- 本轮进一步按“upstream geometry bias validation”做了 `104/105` 两个真实上游变体：
  - `104_depth_bias_minus1cm` 在真实上游重跑中把 Bonn `Acc` 从参考 `99` 的 `4.346 cm` 拉到 `4.238 cm`，同时把同口径平面距离从 `0.0980` 压到 `0.0746`；说明引入负向 `depth_bias` 后，表面确实朝正确方向移动，存在可测的前侧偏置响应；
  - `105_depth_bias_plus1cm` 也把 Bonn `Acc` 拉到 `4.238 cm`，但平面距离回到 `0.0983`，说明当前系统对 `±1 cm` depth bias 都有响应，但符号信息主要体现在平面误差而非 Acc 本身，暂不足以证明单一方向的强偏置；
  - `104/105` 的共同问题同样关键：真实 upstream rerun 没有保住 downstream rear branch（`TB = 0`），因此当前可执行 upstream 前驱与 `99` 的 downstream completion 链仍然断裂；这意味着即使找到了上游偏置响应，也还不能直接升级为可继续主配置；
- 本轮进一步按“geometry chain coupling”做了 `108/109/110` 三个显式集成变体：
  - `108_geometry_chain_coupled_direct` 将 `104` 的 corrected front/surface geometry 与 `99` 的 rear completion donor 直接拼接，恢复了 `TB = 39`，并把 Bonn `Acc` 从原始 `99` 的 `4.346 cm` 压到 `4.233 cm`，同时把 `Comp-R` 提到 `70.86%`；
  - `109_geometry_chain_coupled_projected` 在 direct coupling 基础上，将 donor rear points 重投影到 `104` 的 plane frame，得到本轮最佳几何结果：Bonn `Acc = 4.233 cm`、`Comp-R = 70.84%`、`TB = 39`；
  - `110_geometry_chain_coupled_conservative` 只接入 `97` 的 retained cluster，虽然保住了更强的负相关，但 `TB` 只有 `10`，说明只保留 purified cluster 不足以替代 `99` 的 full completion donor；
  - `108/109/110` 共同说明：结构断裂的根因不是“校正后 cluster 阈值失效”，而是“真实 upstream rerun 根本没有执行 downstream rear completion stage”；一旦把 `104` 的 corrected geometry 显式喂给 `99` 的 rear donor/completion，链路立即恢复且能继承上游几何收益；
- 本轮进一步按“Native Mainline Integration”做了 `111/112` 两个原生集成变体：
  - `111_native_geometry_chain_direct` 在标准 `run_benchmark.py -> run_egf_3d_tum.py` 管道中，原生复现了 `108`：Bonn `Acc = 4.233 cm`，与 `108` 的偏差仅 `7.7e-14 cm`；
  - `112_native_geometry_chain_projected` 在最新原生重跑中得到 Bonn `Acc = 4.235 cm`，与 `109` 的偏差为 `0.00295 cm`，仍满足 `Acc` 偏差 `< 0.01 cm` 的重置标准，但 `TB` 与 `Comp-R` 略弱于 `111`；
  - 这说明耦合逻辑已不再依赖体外 runner，而是已经被标准执行链原生继承；同时按最新原生重跑口径，`111_native_geometry_chain_direct` 是更稳定的 native baseline；
- 同时，本轮输出了 `processes/s2/S2_NATIVE_MAINLINE_GAP_ANALYSIS.md`，明确指出：
  - `Acc` 的剩余 `1.13 cm` 缺口并不是后验厚度噪声，而是上游几何形成畸变与有限 depth bias 的组合；
  - `Comp-R` 的剩余 `27%` 缺口主要集中在 `crowd2` 这类高遮挡动态场景，当前 rear completion 仍无法触达无 support-cluster 的弱证据区域；
- 因此当前主瓶颈已进一步修正为：`there is measurable upstream depth-bias sensitivity, but the executable upstream precursor still cannot preserve the downstream rear completion branch; the bottleneck is now the coupling gap between raw geometry formation and the purified-cluster completion chain`。
- 因此当前主瓶颈已进一步修正为：`support clusters can now be verified and filtered at the cluster level strongly enough to decouple TB and Noise at the family scale, but this comes with a completeness drop (Bonn Comp-R ~68-69%); the next bottleneck is recovering completeness after cluster purification without reintroducing noise`。
- 同时，本轮 deep-explore 已经证明这条 completeness gap 是可被补回的：局部目标 `Comp-R >= 70% + negative correlation + ghost_reduction >= 22% + TB >= 6` 已被 `98/99/100` 同时满足；因此当前剩余问题不再是“能否局部修复”，而是“如何把这条 subchain 升级为全局 S2 通过配置”，包括进一步降低 `Acc` 与把 family-mean `Comp-R` 拉向任务书硬门槛。
 - 但本轮 `Acc tightening` 又进一步证明：当前 `Acc` 缺口不能靠“后验几何紧化”直接补齐；若继续 `S2`，下一步必须从坐标后处理切回到更上游的几何形成链，而不是继续在当前点云上做硬投影。
- 本轮进一步完成了 `111_native_geometry_chain_direct` 的系统级误差分解，输出：
  - `processes/s2/S2_ERROR_DECOMPOSITION_COMPARE.csv`
  - `processes/s2/S2_GEOMETRY_ERROR_DECOMPOSITION_REPORT.md`
  - `processes/s2/S2_COMP_R_GAP_ANALYSIS.md`
- `Acc` 误差分解结论已经收敛：
  - `TUM walking all3` 在 oracle pose 下达到 `Acc = 0.936 cm`，且 `Temporal Drift ≈ 0 cm`；说明当前几何链在无漂移条件下已经可用，`Acc` 主缺口不来自纯传感器噪声底；
  - `Bonn all3` 的诊断量级分解为：`Temporal Drift = 5.205 cm`、`Surface Thickness = 1.860 cm`、`Sensor Noise Floor = 1.577 cm`、`Calibration Bias = 0.010 cm`；
  - 归一化后，Bonn `Acc` 的主贡献者已经明确是 `Temporal Drift (~60.2%)`，其次才是 `Surface Thickness (~21.5%)` 与 `Sensor Noise (~18.2%)`；
  - `104 -> 111` 的 completion 对 Bonn `Acc` 的净影响仅 `-0.005 cm`，方向还是轻微改善；因此当前 `Acc` 绝不是被 downstream completion 拖坏的。
- `Comp-R` 缺口也已被定量锁定：
  - `Bonn all3` 当前缺失参考点约 `2623`，而 rear completion 总点数只有 `90`，覆盖比仅 `3.43%`；
  - `104 -> 111` 的 `Comp-R` 增益约 `0.000 pts`，说明当前 rear completion 对 global completeness 基本无贡献；
  - 在最差序列 `rgbd_bonn_crowd2` 中，缺失点有 `72.4%` 只得到 `<=2` 次稳定观测，`29.2%` 呈现遮挡主导模式，而 `missing_never_in_view_ratio = 0%`；这说明主问题不是“完全盲区”，而是“弱证据累积失败 + 动态遮挡截断”。
- 因此当前 `S2` 的下一轮攻坚方向已经被缩窄为两条：
  1. `Pose / Temporal Drift`：先做位姿与动态边界关联误差分解与约束校正；
  2. `Weak-Evidence Accumulation`：让 one/two-view 弱观测也能被累积成可提交背景，而不是继续扩大局部 rear completion。
- 本轮进一步按“Methodology Innovation: Temporal Drift & Weak-Evidence”做了 `113/114/115` 三个创新原型，产出：
  - `processes/s2/S2_METHODOLOGY_INNOVATION_BRIEF.md`
  - `processes/s2/S2_INNOVATION_ATTACK_COMPARE.csv`
  - `processes/s2/S2_INNOVATION_ATTACK_ANALYSIS.md`
- 三个原型的关键结论已经明确：
  - `113_naive_plane_union`（朴素降阈值）能把 Bonn `Comp-R` 从 `70.86%` 拉到 `76.19%`，但 `Acc` 恶化到 `4.413 cm`，且新增点 `TB/Ghost/Noise = 23/255/3043`，说明单纯放宽 plane-aligned admission 会严重放大动态边界伪支持；
  - `114_papg_plane_union`（Plane-Anchored Pose Graph + weak union）把 Bonn `Acc` 拉回到 `4.304 cm`，同时把 `Comp-R` 推到 `77.06%`，新增点 `TB/Ghost/Noise = 35/224/3091`；这证明 `PAPG` 对弱证据补全有真实几何收益；
  - `115_papg_consensus_activation`（`>=2` 帧共识激活）把 Bonn `Acc` 进一步压到 `4.225 cm`，且新增 `ghost = 26` 已落入安全范围，但 `Comp-R` 只到 `70.96%`；这说明“强共识”足以修 geometry，但不足以补 completeness。
- 因此当前 `S2` 的创新结论并不是“方案无效”，而是：
  - `Plane-Anchored Pose Graph` 已经被验证为有效创新方向；
  - 真正的剩余缺口在于：当前 weak-evidence 仍然是 `union / consensus` 级原型，尚未升级为完整的 `occupancy probability + entropy` 不确定性模型；
  - 下一轮不能回退到阈值微调，而必须继续推进 uncertainty-aware weak-evidence fusion，用来把 `114` 的 coverage 收益保住，同时把新增 `ghost` 压回安全区。
- 本轮进一步完成了 `116_occupancy_entropy_gap_activation` 的概率融合验证，产出：
  - `processes/s2/S2_OCCUPANCY_ENTROPY_INNOVATION_BRIEF.md`
  - `processes/s2/S2_OCCUPANCY_ENTROPY_COMPARE.csv`
  - `processes/s2/S2_OCCUPANCY_ENTROPY_ANALYSIS.md`
- `116` 的关键结果为：
  - Bonn `Acc = 4.120 cm`，优于当前 native baseline `111` 的 `4.233 cm`；
  - Bonn `Comp-R = 76.14%`，显著高于 `111` 的 `70.86%`；
  - `bonn_rear_true_background_sum = 515`，`bonn_rear_ghost_sum = 23`，`bonn_rear_hole_or_noise_sum = 0`；
  - `mean_occupancy_entropy = 0.674`，`gap_activation_ratio = 1.000`，`weak_evidence_coverage_ratio = 0.264`；
  - 这说明：只要激活被限制在真缺口邻域，`occupancy probability + entropy` 的弱证据融合机制确实可以同时保住 `Acc / Comp-R / Ghost`。
- 但本轮必须明确记录一个关键限制：
  - `116` 当前使用的是 **developer-side oracle gap mask**（由 `111` 与 reference 差集得到的 missing-reference neighborhood）；
  - 因此它证明的是“机制成立”，不是“已可直接晋升主线配置”；
  - 在找到 GT-free gap proxy 之前，`116` 仍然不能被视为 S2 fully-pass 配置。
- 本轮进一步完成了 `GT-Free Gap Proxy` 替代实验，产出：
  - `processes/s2/S2_GT_FREE_GAP_PROXY_DESIGN.md`
  - `processes/s2/S2_GT_FREE_GAP_PROXY_COMPARE.csv`
  - `processes/s2/S2_GT_FREE_GAP_PROXY_ANALYSIS.md`
- 结论已经明确：
  - `117_frustum_unobserved_proxy`：Bonn `Acc = 4.136 cm`、`Comp-R = 71.24%`、`Ghost = 152`、`proxy_recall = 0.083`；
  - `118_plane_extrapolation_closure`：Bonn `Acc = 4.199 cm`、`Comp-R = 72.17%`、`Ghost = 92`、`proxy_recall = 0.261`、`proxy_precision = 0.128`；这是当前最接近可用的 GT-free 方案；
  - `119_entropy_guided_proxy`：Bonn `Acc = 4.152 cm`、`Comp-R = 71.17%`、`Ghost = 107`、`proxy_recall = 0.058`；
  - 三个 GT-free proxy 都未达到 `Comp-R >= 75%`、`Ghost <= 50`、`proxy_recall >= 0.6` 的要求，因此当前**还没有找到可替代 Oracle 的可行方案**。
- 本轮进一步尝试了 `120/121` 两个多模态联合 proxy，产出：
  - `processes/s2/S2_JOINT_GAP_PROXY_DESIGN.md`
  - `processes/s2/S2_JOINT_GAP_PROXY_COMPARE.csv`
  - `processes/s2/S2_JOINT_GAP_PROXY_ANALYSIS.md`
- 结果同样明确为负：
  - `120_joint_proxy_geometric_fusion`：Bonn `Acc = 4.184 cm`、`Comp-R = 71.12%`、`Ghost = 167`、`proxy_recall = 0.043`；
  - `121_joint_proxy_score_weighted`：Bonn `Acc = 4.209 cm`、`Comp-R = 71.61%`、`Ghost = 41`、`proxy_recall = 0.132`；
  - 与最佳单点 `118` 相比，`120/121` 都没有在 `Recall / Ghost / Comp-R` 三者中形成净正；说明当前 `Visibility / Plane / Entropy` 三信号虽然概念互补，但在现有估计质量下，简单联合并不能逼近 `116` 的 Oracle 效果。
- 这说明当前真正的难点已经进一步缩窄为：
  - `occupancy + entropy` 机制本身已经成立（由 `116` 证明）；
  - 失败点并不在激活公式，而在 `gap localization` 仍然过弱；
  - `120/121` 又进一步说明：问题不仅是“有没有联合公式”，而是输入信号本身还不够强；下一轮若继续 `S2`，必须先提升 GT-free visibility deficit / plane-hole localization 的质量，再谈融合。
- 本轮进一步完成了 `122/123` 两个 GT-free visibility deficit 信号，产出：
  - `processes/s2/S2_VISIBILITY_DEFICIT_DESIGN.md`
  - `processes/s2/S2_VISIBILITY_DEFICIT_COMPARE.csv`
  - `processes/s2/S2_VISIBILITY_DEFICIT_ANALYSIS.md`
- 关键结果已经明确：
  - `122_evidential_visibility_deficit`：Bonn `Acc = 4.391 cm`、`Comp-R = 74.10%`、`Ghost = 3`、`proxy_recall = 0.519`、`proxy_precision = 0.309`；
  - `123_ray_deficit_accumulation`：Bonn `Acc = 4.398 cm`、`Comp-R = 74.06%`、`Ghost = 9`、`proxy_recall = 0.469`、`proxy_precision = 0.282`；
  - 两者都明显优于 `118` 的 `proxy_recall = 0.261`，并且 `Ghost` 远低于 `118` 的 `92`；
  - 这说明当前已经**成功构建了更强但 GT-free 的 visibility deficit 信号**。
- 但同时也必须如实记录：
  - `122/123` 虽然让 `visibility deficit` 信号过线，但系统级 `Acc` 仍在 `4.39 cm` 左右，显著弱于 `116` 的 `4.120 cm`；
  - 因此它们证明的是“Gap Localization 输入已具备主线资格”，而不是“系统已经完成 S2 通关”；
  - 下一轮若继续 `S2`，重点不再是“能否找到 GT-free gap”，而是“如何把 122/123 的强召回信号与 116 的低 Ghost 激活机制稳定耦合”，在不重新引入噪声的前提下压低 `Acc`。
- 本轮进一步完成了 `124/125` 两个混合集成变体，产出：
  - `processes/s2/S2_HYBRID_INTEGRATION_COMPARE.csv`
  - `processes/s2/S2_HYBRID_INTEGRATION_REPORT.md`
  - `processes/s2/S2_HYBRID_INTEGRATION_ANALYSIS.md`
- 关键结果如下：
  - `124_hybrid_evidential_activation_strict`：Bonn `Acc = 4.238 cm`、`Comp-R = 71.64%`、`Ghost = 5`；
  - `125_hybrid_papg_constrained`：Bonn `Acc = 4.273 cm`、`Comp-R = 72.87%`、`Ghost = 3`；
  - 两者都显著改善了 `122` 的几何质量与 ghost 控制，但仍未满足本轮硬门槛 `Acc <= 4.200 cm` 与 `Comp-R >= 73.5%` 的双赢要求。
- 因此本轮最终判断是：
  - `PAPG + Visibility Deficit + Occupancy Activation` 这条 GT-free 技术路径已经被**原则性打通**；
  - 但其当前形态仍未达到“可交付状态”，因为 `Acc` 还差最后一段收敛；
  - 若继续 `S2`，后续工作不应再扩展新 family，而应集中到“如何消除高召回 Proxy 引入的系统性几何偏差”。 
- 本轮进一步完成了 `126_local_geometry_convergence` 的局部几何收敛实验，产出：
  - `processes/s2/S2_LOCAL_GEOMETRY_CONVERGENCE_DESIGN.md`
  - `processes/s2/S2_LOCAL_GEOMETRY_CONVERGENCE_COMPARE.csv`
  - `processes/s2/S2_FINAL_CLOSING_REPORT_DRAFT.md`
- `126` 的关键结果为：
  - Bonn `Acc = 4.272 cm`；
  - Bonn `Comp-R = 72.94%`；
  - `Ghost = 8`；
  - 局部锚点残差从 `7.129 cm` 压到 `5.827 cm`；
  - 这说明“位置去偏 / 局部几何收敛”方向是有效的，但改进量仍不足以跨过 `Acc <= 4.200 cm` 的最终门槛。
- 因此当前 S2 结项判断必须保持严格：
  - `Comp-R` 与 `Ghost` 已满足最后一轮门槛；
  - 但 `Acc` 仍卡在 `4.27 cm` 左右，未能压到 `4.20 cm` 以下；
  - 所以当前只能形成 `S2_FINAL_CLOSING_REPORT_DRAFT.md` 的草稿版本，**不能正式宣布 S2 fully pass，也不能进入 S3**。
- 本轮进一步完成了 `127/128` 全局刚性/相似校正实验，产出：
  - `processes/s2/S2_GLOBAL_RIGIDITY_ALIGNMENT_DESIGN.md`
  - `processes/s2/S2_GLOBAL_RIGIDITY_ALIGNMENT_COMPARE.csv`
  - `processes/s2/S2_FINAL_CLOSING_REPORT.md`
- 结果明确为负：
  - `127_global_rigidity_alignment`：Bonn `Acc = 4.345 cm`、`Comp-R = 71.88%`、`Ghost = 2`，平均旋转角仅 `0.219 deg`；
  - `128_global_similarity_alignment`：Bonn `Acc = 4.417 cm`、`Comp-R = 71.67%`、`Ghost = 6`，平均旋转角 `0.260 deg`、尺度约 `0.997618`；
  - 这说明当前确实存在可测的全局微小偏差，但其量级太小，无法解释剩余的 `0.07 cm` Acc 缺口；进一步的全局刚性校正反而破坏了原有平衡。
- 因此本轮最终结论必须更新为：
  - `127/128` 证明：最后的误差已不再是简单的“地图整体歪了”问题；
  - 当前 Acc 瓶颈更像是 GT-free 增量点与基线表面之间的系统性局部配准偏差，而非单一全局刚体偏差；
  - `S2_FINAL_CLOSING_REPORT.md` 已生成，但其结论是：**S2 核心技术路径已完整验证，但仍不能正式批准结项；绝对禁止进入 S3。**
- 本轮最终又进一步完成了 `129_local_registration_bias_modeling`，产出：
  - `processes/s2/S2_LOCAL_REGISTRATION_BIAS_MODELING.md`
  - `processes/s2/S2_LOCAL_REGISTRATION_BIAS_MODELING_COMPARE.csv`
  - `processes/s2/S2_LOCAL_REGISTRATION_BIAS_MODELING_ANALYSIS.md`
  - 并同步刷新了 `processes/s2/S2_FINAL_CLOSING_REPORT.md`
- `129` 的结果为：
  - Bonn `Acc = 4.280 cm`、`Comp-R = 72.29%`、`Ghost = 4`；
  - 局部残差从 `12.824 cm` 压到 `5.911 cm`；
  - 这说明局部偏移场本身存在，但其对全局 Acc 的传导极弱，已经无法继续推动 GT-free 主线跨过 `4.200 cm` 的门槛。
- 因此当前最终结项判断保持不变：
  - 当前最佳 GT-free 结果仍是 `126`（`Acc = 4.272 cm`, `Comp-R = 72.94%`, `Ghost = 8`）；
  - `129` 进一步证明：继续沿当前 S2 主线做局部配准补丁，收益已接近天花板；
  - **S2 不通过，禁止进入 S3。**

当前距离 `S2` 通过仍差：
1. 当前所有候选在 `TUM walking all3 + Bonn all3` family-mean 口径下仍未触及 `Comp-R >= 98%`；其中较高动态抑制分支 `67/68` 仅有 `Bonn Comp-R = 70.37% / 70.33%`；
2. Bonn 动态抑制门槛已经被 `67/68` 跨过：`ghost_reduction_vs_tsdf = 22.43% / 23.52%`，因此当前主问题不再是“能否破 22%”，而是“如何在保持 `22%+` 的同时把 `ghost` 压回 `<=15` 并恢复 TB”；
3. `39` 证明“更多 admission”不自动等于“更好的最终几何质量”，`40-45` 证明软重定向仍强耦合动态区，`46/47` 证明硬 HVO 约束会直接剪没 rear points，`48/49` 证明稳定背景流形状态过稀，`50-55` 证明稠密化/表面约束传播仍不能形成足够的 true-background 覆盖，`56-61` 证明保守桥接虽然能压 Ghost，但 rear 覆盖量只能恢复到 `65` 左右，`62-64` 证明单纯 multi-candidate 扩张会把 `ghost` 放大到 `23-60`，`65-67` 证明导出端粗筛选仍无法守住 TB，`68-70` 证明简单的 `rear/front/gap` 判别会走向过裁剪，`71-73` 证明语义/动态上下文只能把分布改善到 `TB=4 / Ghost=24`，`74-79` 证明静态历史锚定与局部冲突组合仍不足，`80-82` 证明物理特征能开始分离 TB/Noise，`83-85` 进一步证明结构级拓扑约束确实能把 `hole_or_noise_sum` 压到 `99-106`、把 `ghost` 压到 `16-22`，`86-91` 证明“避免过清洗”还不等于“恢复 TB”，`92-94` 证明支持聚合已经能在局部序列上把 `TB` 从 `4` 推到 `8-13`，`95-97` 证明簇级验证已经足以把 `tb_noise_correlation` 从 `0.991` 直接压到 `-0.756 / -0.786`，`98-100` 进一步证明：在 `97` 的 purified cluster plane 上做受控 completion，可以把 Bonn `Comp-R` 重新拉回 `70.08%` 且保持负相关，`101-103` 又进一步证明：当前 `Acc` 缺口并不能靠后验几何紧化补齐，`104/105` 则第一次证明：真实 upstream depth bias 的确会让 Bonn `Acc` 与平面误差同步响应，而 `108-110` 最终又证明：`104` 与 `99` 之间的结构断裂并不是不可修复，只需显式把 corrected geometry 喂给 downstream completion chain，就能恢复 `TB` 并继承上游几何收益；因此当前 S2 的剩余问题已从“能否耦合”切换成“如何把这条已修复的 one-way chain 内化回可执行主线”；
4. `RB-Core` 同口径 compare 与 lockbox 仍只有 partial closure；
5. 因此当前**不能诚实地把 `S2` 记为 fully pass**。

下一轮若继续 `S2`，唯一合理主线是：
- 固定 current-code canonical：`frames=5 / stride=3 / seed=7 / max_points_per_frame=600`；
- 固定当前 native baseline：`111_native_geometry_chain_direct`；
- 把 `114_papg_plane_union` 作为“coverage-leaning innovation baseline”，把 `115_papg_consensus_activation` 作为“geometry-safe baseline”；
- 把 `116_occupancy_entropy_gap_activation` 作为**方法机制验证上界**：后续一切 weak-evidence 研究必须以不引入 oracle gap mask 的前提，逼近 `116` 的 `Acc / Comp-R / Ghost` 组合；
- 下一轮不再允许回退到“简单降阈值 / 扩大 union”路线；`113` 已经证明朴素 admission 只会放大 dynamic-boundary ghost；
- 下一轮必须围绕两条方法论主线推进：
  1. `Plane-Anchored Pose Graph` 继续深化：把平面约束从当前离线原型推进到更稳定的 pose / map 联合校正；
  2. `Uncertainty-aware Weak-Evidence Fusion` 正式实现：把当前 `116` 的 oracle gap-only activation 替换为 GT-free gap proxy，目标是在保住 `Comp-R` 的同时继续把 `ghost` 维持在 `<=35`；
- 在此之前，严禁继续把资源投入到后验点云紧化、简单 plane snapping 或导出端 Top-K 微调，因为 `101-103` 与 `113` 都已证明这不是当前主矛盾。
- 已被证伪的 competition-only family（`34/35/36`）不得继续扩展。

---

## 阶段 S3：组合主线阶段 —— delayed synthesis + Acc line 联合优化

### 前置条件
- [ ] `S2` 已完成并在任务书中记录
- [ ] `S2` 唯一 active delayed 主线已冻结
- [ ] `S1` 完成的数据集本地接入与 `RB-Core` 本地接入仍保持可运行状态
- [ ] `RB-Extended` 已接入：
  - `RB-Core`
  - + `object-aware family` 至少一个
  - + `representation family` 至少一个

### 阶段目标
把 delayed 主线与 `Acc` 主线真正组合起来，并通过完整闭环验证“组合版本”是否达到当前项目内新的联合 SOTA。

### 本阶段要达成的指标/方面 SOTA 目标
冻结主配置必须同时满足：
- [ ] TUM 开发子集 `Acc <= 2.30 cm`
- [ ] Bonn 开发子集 `Acc <= 2.95 cm`
- [ ] Bonn 开发子集 `ghost_reduction_vs_tsdf >= 28%`
- [ ] `Comp-R >= 98%`（两边）
- [ ] 相对 `S2` 配置，在至少一个数据集 family 上对 `RB-Extended` 呈现明确净正优势

### 必做任务
- [ ] 审查 `lzcd / stcg / dualch / geo debias` 家族中最接近过线的配置
- [ ] 形成 `Acc` 主线候选池（`1-3` 个）
- [ ] 将其与 `S2` delayed 主线组合
- [ ] 跑 controlled combine experiments
- [ ] 量化：
  - delayed-only 收益
  - Acc-line-only 收益
  - combine 收益
- [ ] 与 `RB-Extended` 做同口径对比
- [ ] 进行锁箱复验
- [ ] 最终只保留一个组合主配置

### 阶段产物
- [ ] `Acc` 专项 profile 表
- [ ] combine compare 表
- [ ] `RB-Extended` 对比表
- [ ] 锁箱复验记录
- [ ] `S3` 唯一主配置冻结页

### 阶段通过标准
- [ ] 所有量化目标全部满足
- [ ] 组合收益方向稳定
- [ ] 收益可归因于 delayed 主线与 `Acc` 主线的互补，而非口径漂移
- [ ] 本阶段只保留一个组合主配置

### 进入下一阶段条件
- [ ] 上述必做任务、产物与通过标准全部满足
- [ ] `S3` 唯一主配置已冻结
- [ ] **未满足前，绝对禁止进入 `S4`**

---

## 阶段 S4：P10 全量过线阶段

### 前置条件
- [ ] `S3` 已完成并在任务书中记录
- [ ] `S3` 唯一主配置已冻结
- [ ] `RB-Extended` 保持不变

### 阶段目标
用 `S3` 的唯一主配置完成 `P10` 全量硬过线，并且在 `5-seed + significance + robustness` 下仍然成立。

### 本阶段要达成的指标/方面 SOTA 目标
冻结主配置必须同时满足全部：
- [ ] TUM oracle: `Acc <= 1.80 cm`
- [ ] TUM oracle: `Comp-R >= 95%`
- [ ] Bonn slam: `Acc <= 2.60 cm`
- [ ] Bonn slam: `Comp-R >= 95%`
- [ ] `ghost_reduction_vs_tsdf >= 35%`
- [ ] `5-seed` 下仍成立或极接近成立
- [ ] 相对 `RB-Extended`，在至少一个主数据集 family 上形成可见净优势

### 必做任务
- [ ] 固定 `S3` 主配置为唯一评测配置
- [ ] 跑完整 `P10` suite
- [ ] 跑 `5-seed`
- [ ] 跑显著性
- [ ] 跑 failure cases / robustness
- [ ] 与 `RB-Extended` 做 full compare
- [ ] 形成 locked `P10 pass/fail` 判定页

### 阶段产物
- [ ] `P10` 全量结果表
- [ ] `5-seed` 聚合表
- [ ] 显著性结果表
- [ ] `RB-Extended` 全量 compare 表
- [ ] failure cases / robustness 报告
- [ ] `P10 pass/fail` 判定页

### 阶段通过标准
- [ ] 上述量化目标全部满足
- [ ] 5-seed 与显著性支持主结论
- [ ] failure cases 不推翻主结论
- [ ] `P10` 已在任务书中标记为“过线”

### 进入下一阶段条件
- [ ] 上述必做任务、产物与通过标准全部满足
- [ ] `P10` 过线已被任务书显式记录
- [ ] **未满足前，绝对禁止进入 `S5`**

---

## 阶段 S5：强基线补齐与投稿竞争力阶段

### 前置条件
- [ ] `S4` 已完成并确认 `P10` 过线
- [ ] 当前主配置已冻结，不允许再做主方法漂移
- [ ] `S1` 完成的数据集与强基线本地接入链路仍保持可运行
- [ ] 只允许补基线、补实验、补写作闭环，不允许重写主方法

### 阶段目标
在 `P10` 已过线的基础上，把论文从“专项过线原型”升级成“对强 literature baselines 有竞争力、具备高概率中稿条件”的系统性工作。

### 本阶段要达成的指标/方面 SOTA 目标
本阶段通过时必须同时满足：
- [ ] 至少 `3` 个经典 baseline + `1` 个 modern baseline + `1` 个 representation baseline 完成对齐
- [ ] 主方法对至少 `2` 个 baseline family 呈现明确净优势
- [ ] canonical 主表相对当前状态出现论文级改进：
  - [ ] TUM canonical `Acc <= 3.30 cm`
  - [ ] Bonn canonical `Acc <= 4.90 cm`
  - [ ] Bonn canonical `Comp-R >= 85%`
- [ ] 不出现“只在单序列/单 seed/单 budget 下好看”的伪 SOTA

### 必做任务
- [x] 完成 `RoDyn-SLAM` 对齐
- [ ] 完成 `DynaSLAM / DynaSLAM II` 对齐
- [ ] 完成 `MID-Fusion / MaskFusion / Co-Fusion` 中至少 `1–2` 个
- [ ] 完成 `TSDF++ / Panoptic Multi-TSDFs` 对齐
- [ ] 完成 `NICE-SLAM / 4DGS-SLAM / NID-SLAM / vMAP` 中至少 `2` 条 modern baseline line
- [ ] 明确所有 baseline 的：
  - 数据集
  - 协议
  - 复现状态
  - 数值来源
- [ ] 与全部 baseline family 形成统一 compare matrix
- [ ] 形成 family-level discussion
- [ ] 形成投稿竞争力判断页

### 阶段产物
- [ ] 强基线总对比表
- [ ] baseline 复现状态页
- [ ] baseline family 分组讨论页
- [ ] canonical 主表强化版
- [ ] 投稿竞争力总结页

### 阶段通过标准
- [ ] 所有量化目标全部满足
- [ ] 对至少 `2` 个 baseline family 有稳定净优势
- [ ] 所有对比均在统一协议与统一口径下成立
- [ ] 不存在基线数值来源不透明、口径不可比、数据泄露或 cherry-pick 风险

### 进入下一阶段条件
- [ ] 上述必做任务、产物与通过标准全部满足
- [ ] `S5` 已完成并在任务书中记录
- [ ] **未满足前，绝对禁止进入 `S6`**

---
## 阶段 S6：完整实验与补充材料阶段

### 目标
形成完整的审稿闭环。

### 必做任务
- [ ] 主结果表
- [ ] 消融表
- [ ] 复杂度 / 速度 / 内存表
- [ ] failure cases
- [ ] qualitative figures
- [ ] temporal trend / robustness
- [ ] supplemental 全文档

### 阶段通过标准
- [ ] 审稿人能针对“创新/结果/复杂度/失败案例/复现”提出的主要问题，都有现成回答材料

---

## 阶段 S7：论文成稿阶段

### 目标
把研究原型变成审稿可读的 paper。

### 必做任务
- [ ] 标题
- [ ] 摘要
- [ ] Introduction
- [ ] Related Work
- [ ] Method
- [ ] Experiments
- [ ] Discussion / Limitations
- [ ] Conclusion
- [ ] Supplemental

### 阶段通过标准
- [ ] 题目、摘要、主图、主表、主方法完全围绕同一命题
- [ ] 论文长度、图表密度、supp 完整度达到顶刊/顶会常规要求

---

## 阶段 S8：Artifact / Release / Submission 阶段

### 目标
满足可复现投稿要求。

### 必做任务
- [ ] 环境文件
- [ ] 数据准备说明
- [ ] 一键实验脚本
- [ ] 一键主表刷新脚本
- [ ] artifact checklist
- [ ] release 自检
- [ ] 工作树冻结

### 阶段通过标准
- [ ] 新环境中按说明可复现实验主结果
- [ ] 主表与文中数字完全一致
- [ ] artifact 可通过内部审查

---

## 8. 直接投稿顶刊/顶会的最高通过条件

只有当以下全部满足，才算达到本任务书定义的“高概率中稿最高条件”：

### C1. 指标
- [ ] `P10` 全部硬门槛通过
- [ ] 至少一个主数据集 family 上出现明确净优势而非 mixed change
- [ ] 5-seed 稳定

### C2. 创新
- [ ] 主命题不再是“很多机制组合”，而是“一个明确的新结构原则”
- [ ] delayed write-time synthesis 成为真正核心贡献

### C3. 对比
- [ ] 强基线补齐
- [ ] 现代 baseline 至少 1 条
- [ ] representation family baseline 至少 1 条

### C4. 论文
- [ ] 主文结构定稿
- [ ] 主图和主表完整
- [ ] supplemental 完整

### C5. 复现
- [ ] artifact 可复现
- [ ] release 可发布
- [ ] canonical 数字统一无漂移

---

## 9. 当前最现实的 venue 选择策略

### 如果只完成最低完整条件
- 首选：`RA-L + ICRA`
- 备选：`IROS`

### 如果 delayed write-time synthesis 做成且结果显著推进
- 可冲：`CVPR / ICCV`

### 如果先发系统稿再扩展长期版本
- 长线：`T-RO`

---

## 10. 当前执行建议（从今天开始）

从当前状态看，最合理的执行顺序是：

1. **停止继续扩写 export 末端小修小补分支**；
2. **把 delayed 主线资源集中到 write-time target synthesis**；
3. **并行重启 `Acc` 主线强化**；
4. **尽早补 `RoDyn-SLAM / DynaSLAM / NICE-SLAM / 4DGS-SLAM family`**；
5. **直到 `P10` 过线前，不进入正式主文写作阶段**；
6. **一旦 `P10` 过线，立即切到强基线闭环与成稿阶段**。

---

## 11. 当前重写任务书结论

本任务书已经把“从当前项目现状出发，冲击高概率中稿顶刊/顶会”的所有关键阶段重新定义完毕。  
从此刻起，`TASK_LOCAL_TOPTIER.md` 的职责不再是历史实验日志，而是**总控任务书**。  

后续所有工作，都应以本文件为最高调度依据。
