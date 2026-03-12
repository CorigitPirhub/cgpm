# TASK_LOCAL_TOPTIER / 顶刊顶会冲线总任务书（重写版）

版本：`2026-03-08`
状态：`ACTIVE / SUPERSEDES HISTORY`
定位：本文件从当前项目真实现状出发，定义一份**偏激进、宁冗勿缺、按最高标准写就**的总任务书。  
原则：**只有当本任务书中的全部关键任务严格完成，才可判断项目具备“直接投稿顶刊/顶会且大概率中稿”的最高条件。**

> 历史版本已冻结归档：`archives/taskbooks/TASK_LOCAL_TOPTIER_ARCHIVE_2026-03-08.md`
> 
> 本文件不再承担“实验流水账”职责，而承担“从当前状态出发，迈向高概率中稿顶刊/顶会”的总控任务书职责。

> 阶段结果总览统一放在 `output/<stage>/OVERVIEW.md`；每个尝试在对应阶段目录下保留一个结果汇总 `.csv` 与一个结论/未来计划 `.md`。
>
> 自本版起，后续所有新增、迁移、归档文件都必须按 `PROJECT_STRUCTURE_GUIDE.md` 组织；不再接受脱离该规范的临时落盘方式。

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

6. **文件组织必须遵守 `PROJECT_STRUCTURE_GUIDE.md`**
   - 主线稳定代码放在 `egf_dhmap3d/` 与 `scripts/`。
   - 实验脚本与实验性方法放在 `experiments/<stage>/`。
   - 每个尝试在 `output/<stage>/` 下保留两个文件：一个结果汇总 `.csv`，一个结论/未来计划 `.md`。
   - 阶段总览放在 `output/<stage>/OVERVIEW.md`，中间产物放在 `output/tmp/`。

7. **主线实验必须可从原始数据独立重建**
   - 进入阶段结论、主基线比较或主线判断的实验，必须能从原始数据目录出发独立重建。
   - 不得把依赖隐式中间产物、历史 donor/control 目录或手工残留缓存的链路，直接当作当前主线结论来源。
   - 依赖上游产物的脚本只能作为诊断/分析工具，除非整条流水线已被显式定义并可一键从原始数据完整重建。

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
  - `experiments/p10/run_p10_precision_profile.py`
  - `scripts/update_summary_tables.py`
- [x] 主文档体系完整：
  - `README.md`
  - `BENCHMARK_REPORT.md`
  - `output/<stage>/OVERVIEW.md`
  - `PROJECT_STRUCTURE_GUIDE.md`
  - `TASK_LOCAL_TOPTIER.md`
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

本任务书采用比阶段结果归档更激进的标准。

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
- 已形成当前状态一页式基线页：`output/<stage>/OVERVIEW.md`
- 已补充阶段重评估页：`output/<stage>/OVERVIEW.md`
- 已形成主线/支线裁剪表：`output/<stage>/OVERVIEW.md`

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
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
- 已固定开发/锁箱协议卡：`output/<stage>/OVERVIEW.md`
- 已完成核心数据集本地接入状态表：
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
- 已完成更新版 `RB-Core` 本地接入状态表：
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
- 已完成 `RoDyn-SLAM` 本地接入与 core-dataset protocol check：
  - `scripts/external/run_rodyn_slam_runner.py`
  - `scripts/adapters/run_rodyn_slam_adapter.py`
  - `scripts/external/rodyn_no_dynamo_entry.py`
  - `experiments/s1/run_s1_rodyn_core_protocol.py`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
- 已完成更新版 `RB-Core` 开发门槛 / 锁箱 gate：
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
- 已冻结 recent baseline 决策、外部基线完成度与 `RB-S1+`：
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
- 已完成第一轮候选筛选与唯一 active candidate 冻结：
  - `output/<stage>/OVERVIEW.md`
  - `output/<stage>/OVERVIEW.md`
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

### 阶段 S2 完成记录（2026-03-11 refresh）

- 已完成 `S2` 关键链路的路径与运行修复：
  - `output/s2/S2_PIPELINE_REPAIR_REPORT.md`
  - `PROCESS.md`
- 已恢复当前可运行 native baseline：
  - `experiments/s2/run_s2_native_geometry_chain.py`
  - `output/s2/S2_NATIVE_BASELINE_111.csv`
  - `output/s2/S2_NATIVE_BASELINE_111.md`
- 已重新生成当前有效的 `S2` 对比结果：
  - `output/s2/S2_OCCUPANCY_ENTROPY_COMPARE.csv`
  - `output/s2/S2_VISIBILITY_DEFICIT_COMPARE.csv`
  - `output/s2/S2_HYBRID_INTEGRATION_COMPARE.csv`
  - `output/s2/S2_LOCAL_GEOMETRY_CONVERGENCE_COMPARE.csv`
  - `output/s2/S2_GLOBAL_RIGIDITY_ALIGNMENT_COMPARE.csv`
  - `output/s2/S2_LOCAL_REGISTRATION_BIAS_MODELING_COMPARE.csv`
- 已确认一批历史 `RPS` 链在 clean rerun 下失效，仅保留为诊断材料：
  - `output/s2/S2_RPS_RAY_CONSISTENCY_COMPARE.csv`
  - `output/s2/S2_RPS_HSSA_COMPARE.csv`
  - `output/s2/S2_RPS_BALLOON_CLUSTER_COMPARE.csv`
  - `output/s2/S2_RPS_DEEP_EXPLORE_COMPARE.csv`
  - `experiments/s2/deprecated/README.md`

### 阶段 S2 结论

统一解释当前 `S2` 主线状态与链路裁剪的汇总页：`output/s2/S2_PIPELINE_REPAIR_REPORT.md`

结论：**`S2` 已完成 current-code repair 与关键链路复跑，但仍未通过，且绝对不能进入 `S3`。**

本阶段历史叙述正式压缩为三层结构：

#### 一、当前有效

当前只有以下链条可作为 `S2` 的有效主线与同口径比较对象：

- `111_native_geometry_chain_direct`
  - 当前可运行 native baseline
  - 当前复跑结果：Bonn `Acc = 4.452 cm`, `Comp-R = 66.87%`, `Ghost = 21`
  - 结果：`output/s2/S2_NATIVE_BASELINE_111.csv`
- `116_occupancy_entropy_gap_activation`
  - 当前可运行 oracle 机制上界
  - 当前复跑结果：Bonn `Acc = 4.308 cm`, `Comp-R = 73.57%`, `Ghost = 25`
  - 结果：`output/s2/S2_OCCUPANCY_ENTROPY_COMPARE.csv`
- `122_evidential_visibility_deficit`
  - 当前 GT-free visibility deficit 主信号
  - 当前复跑结果：Bonn `Acc = 4.706 cm`, `Comp-R = 70.83%`, `Ghost = 6`, `proxy_recall = 0.490`
  - 结果：`output/s2/S2_VISIBILITY_DEFICIT_COMPARE.csv`
- `123_ray_deficit_accumulation`
  - 当前 GT-free visibility deficit 轻量对照信号
  - 当前复跑结果：Bonn `Acc = 4.538 cm`, `Comp-R = 71.20%`, `Ghost = 10`, `proxy_recall = 0.502`
  - 结果：`output/s2/S2_VISIBILITY_DEFICIT_COMPARE.csv`
- `125_hybrid_papg_constrained`
  - 当前 GT-free 系统折中最优参考
  - 当前复跑结果：Bonn `Acc = 4.450 cm`, `Comp-R = 70.26%`, `Ghost = 12`
  - 结果：`output/s2/S2_HYBRID_INTEGRATION_COMPARE.csv`

#### 二、历史诊断

以下链条仍保留诊断价值，但只用于 failure analysis，不再作为当前主结论：

- `114_papg_plane_union`
  - 仍保留为 coverage-leaning 中间验证节点
  - 当前复跑结果：Bonn `Acc = 4.512 cm`, `Comp-R = 73.96%`, `Ghost = 189`
- `115_papg_consensus_activation`
  - 仍保留为 geometry-safe 中间验证节点
  - 当前复跑结果：Bonn `Acc = 4.450 cm`, `Comp-R = 66.91%`, `Ghost = 9`
- `80/93/97/99`
  - 在 clean rerun 下已整体退化为 `Acc = inf`, `Comp-R = 0`, `TB = 0`
  - 仅保留为历史 RPS 断链诊断材料
  - 结果：
    - `output/s2/S2_RPS_RAY_CONSISTENCY_COMPARE.csv`
    - `output/s2/S2_RPS_HSSA_COMPARE.csv`
    - `output/s2/S2_RPS_BALLOON_CLUSTER_COMPARE.csv`
    - `output/s2/S2_RPS_DEEP_EXPLORE_COMPARE.csv`
    - `experiments/s2/deprecated/README.md`

#### 三、已废弃

以下变体不再作为当前 `S2` 主线候选：

- `113`：朴素 union 路线，已被证伪为只会扩大动态边界噪声；
- `117/118/119/120/121`：旧 GT-free proxy 与联合 proxy 路线，已被 `122/123` 取代；
- `124`：严格过滤虽能压 `Ghost`，但 `Comp-R` 明显回退；
- `126/127/128/129`：局部收敛、全局刚体、局部偏移场等后验补丁均未能恢复到当前重跑目标线，也未优于当前有效主线：
  - `126_local_geometry_convergence`：Bonn `Acc = 4.469 cm`、`Comp-R = 69.27%`、`Ghost = 6`、局部残差 `7.159 cm -> 6.148 cm`；
  - `127_global_rigidity_alignment`：Bonn `Acc = 4.536 cm`、`Comp-R = 69.14%`、`Ghost = 1`、平均旋转角 `0.182 deg`；
  - `128_global_similarity_alignment`：Bonn `Acc = 4.473 cm`、`Comp-R = 69.30%`、`Ghost = 4`、平均旋转角 `0.200 deg`、尺度 `1.001547`；
  - `129_local_registration_bias_modeling`：Bonn `Acc = 4.449 cm`、`Comp-R = 69.17%`、`Ghost = 8`、局部残差 `13.275 cm -> 6.226 cm`。

当前 `S2` 的真实状态已经收敛为：

1. `111` 已恢复为当前可运行 native baseline；
2. `116` 仍证明 occupancy+entropy 机制本身成立；
3. `122/123` 仍证明 GT-free visibility deficit 信号已经可用；
4. `125` 仍是当前 GT-free 主线的最好系统折中；
5. `80/93/97/99` 等旧 RPS 深链已不再是当前主线的一部分；
6. 因此 `S2` 当前的唯一合理推进方向，只剩：
   - `111 -> 125`：恢复 GT-free 主线的 `Acc / Comp-R / Ghost` 折中；
   - `125 -> 116`：缩小 GT-free 主线与 oracle 上界之间的剩余系统差距。

需明确记录：
- 历史 `S2` 探索未能形成稳定闭环；
- 当前 `S2` 必须基于 `125` 及其上游 `111/116/122/123` 重新构建核心技术突破路径；
- `80/93/97/99` 仅保留为诊断链，不再作为当前主线修复对象。

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
