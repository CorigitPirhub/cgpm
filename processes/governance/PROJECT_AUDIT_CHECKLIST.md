# 当前项目现状 / 风险 / 下一步 审计清单

审计日期：`2026-03-07`
审计范围：仓库代码、顶层设计文档、`output/summary_tables`、关键实验结果目录、刷新脚本链路。

## 1. 本轮审计结论

- [x] 已完成仓库主线代码、主文档、主实验表的系统复核。
- [x] 已确认项目主线仍是 `EGF-DHMap` 的 3D 动态局部建图方案，而不是 legacy/2D 分支。
- [x] 已确认当前真正的研究阻塞项仍是 `P10`，不是 `P0-P9` 的基础闭环能力缺失。
- [x] 已确认 `summary_tables` 存在“主聚合表已更新，但论文主表/顶刊主表滞后”的口径漂移问题。
- [x] 已在本轮修复刷新链路：`scripts/update_summary_tables.py` 现在会重建双协议多 seed 汇总和论文主表。
- [x] 已完成本轮主表刷新，当前主表时间戳已统一到 `2026-03-07 19:40` 左右。

## 2. 当前项目现状

### 2.1 代码与工程状态

- [x] 核心实现存在且结构清晰：
  - `egf_dhmap3d/core/config.py`
  - `egf_dhmap3d/core/types.py`
  - `egf_dhmap3d/core/voxel_hash.py`
  - `egf_dhmap3d/modules/associator.py`
  - `egf_dhmap3d/modules/updater.py`
  - `egf_dhmap3d/modules/pipeline.py`
- [x] 统一实验入口存在：
  - `scripts/run_benchmark.py`
  - `scripts/run_egf_3d_tum.py`
  - `scripts/run_local_top_tier_suite.py`
  - `scripts/run_p10_precision_profile.py`
  - `scripts/update_summary_tables.py`
- [x] 数据目录就绪：
  - `data/tum/` 含静态与 walking 序列
  - `data/bonn/` 含 `balloon/balloon2/crowd2`
- [x] 第三方基线目录存在：`third_party/DynaSLAM/`、`third_party/nice-slam/`
- [ ] 工作树尚未冻结；当前仍有未提交改动，主要集中在 `P10` 主线与刷新链路。

### 2.2 任务状态

- [x] `P0` 协议锁定：已有固定报告与复验结果。
- [x] `P1` 静态-动态平衡：已有锁定报告。
- [x] `P2` 机制证据：已有锁定报告。
- [x] `P3` 外部真实输出接入：链路存在且可追踪。
- [x] `P6-P9` 指标、效率、外部基线、压力测试：已有汇总产物。
- [ ] `P10` 仍未过线，且仍是当前最关键的研究与投稿阻塞项。

### 2.3 当前正式口径主表（本轮刷新后）

当前应以以下文件作为正式口径来源：

- 论文动态主表：`output/summary_tables/paper_main_table_local_mapping.csv`
- 顶刊主口径表：`output/summary_tables/local_mapping_main_metrics_toptier.csv`
- 双协议显著性：`output/summary_tables/dual_protocol_multiseed_significance.csv`
- 双协议聚合：
  - `output/summary_tables/dual_protocol_multiseed_reconstruction_agg.csv`
  - `output/summary_tables/dual_protocol_multiseed_dynamic_agg.csv`
- 分数据集多 seed 聚合：
  - `output/summary_tables/tum_reconstruction_metrics_multiseed_agg.csv`
  - `output/summary_tables/bonn_reconstruction_metrics_multiseed_agg.csv`

本轮刷新后，按当前正式口径统计：

#### 动态主表均值（EGF）

- [x] TUM / oracle / dynamic：
  - `F-score = 0.7903`
  - `Chamfer = 0.05317`
  - `ghost_ratio = 0.2566`
- [x] Bonn / slam / dynamic：
  - `F-score = 0.6463`
  - `Chamfer = 0.10116`
  - `ghost_ratio = 0.08613`

#### 顶刊主口径均值（EGF）

- [x] TUM / oracle / dynamic：
  - `Acc = 4.1655 cm`
  - `Comp = 1.1517 cm`
  - `Comp-R(5cm) = 100.0%`
- [x] Bonn / slam / dynamic：
  - `Acc = 6.1481 cm`
  - `Comp = 3.9675 cm`
  - `Comp-R(5cm) = 77.21%`

#### 显著性（EGF vs TSDF，当前正式口径）

- [x] TUM / oracle / dynamic：
  - `F-score` 改善显著，`p = 4.81e-10`
  - `Chamfer` 改善显著，`p = 1.96e-08`
  - `ghost_ratio` 改善显著，`p = 1.72e-24`
- [x] Bonn / slam / dynamic：
  - `F-score` 改善显著，`p = 1.06e-18`
  - `Chamfer` 改善显著，`p = 8.21e-12`
  - `ghost_ratio` 改善显著，`p = 5.35e-05`

### 2.4 P10 当前状态

- [ ] `P10` 未通过当前硬门槛：
  - TUM walking mean：`Acc <= 1.80 cm` 未满足
  - Bonn all3：`Acc <= 2.60 cm` 未满足
  - Bonn `Comp-R(5cm) >= 95%` 未满足
- [x] 当前研究主线已从单图层 veto/记忆链，转向 `dual_map + PFV`。
- [x] 多个结构分支已被项目内部实验明确判为负结果，不宜回退反复试错：
  - `OTV`
  - `CSR-XMap`
  - `XMem / BECM / RCCM`
  - `OBL-3D`
  - `CMCT`
  - `CGCC`
  - `PFVP`
  - `PFV-sharp`
  - `PFV-bank`

## 3. 当前风险清单

### R1. 主表口径漂移风险

- [x] 已发现：`paper_main_table_local_mapping.*` 与 `local_mapping_main_metrics_toptier.*` 曾晚于主聚合表刷新，存在陈旧口径。
- [x] 已修复刷新链路。
- [ ] 尚未同步修正文档叙事；`README.md`、`BENCHMARK_REPORT.md`、`TASK_LOCAL_TOPTIER.md` 中仍可能保留旧数字或旧结论。

### R2. 投稿叙事与当前正式口径不一致

- [ ] 当前刷新后的正式口径数值，明显弱于旧版论文主表/报告中的部分数字。
- [ ] 这意味着：如果继续沿用旧叙事，存在“文档与当前 canonical table 不一致”的投稿风险。
- [ ] 本轮虽然已统一表格，但尚未统一所有对外文字表述。

### R3. P10 仍是核心阻塞项

- [ ] 当前正式口径下，`P10` 的 `Acc / Comp-R / ghost` 三重门槛依旧未通过。
- [ ] 即使动态优势与显著性仍在，顶刊/顶会主口径中的 `Acc` 与 Bonn `Comp-R` 仍明显不足。

### R4. 发布链路不完整

- [ ] `README.md` 与 `scripts/package_release.sh` 仍引用缺失文件：
  - `MERGED_DOCS.md`
  - `LICENSE`
- [ ] 如直接执行打包脚本，发布产物链路并不完整。

### R5. 文档时间线风险

- [ ] 仓库文档中已经出现 `2026-03-08` 的记录，但当前审计日期为 `2026-03-07`。
- [ ] 后续对“最新结果”进行引用时，必须明确说明是“文档记录日期”还是“当前仓库实际审计日期”。

### R6. 仓库冻结风险

- [ ] 当前工作树仍有未提交改动：`P10` 主线与核心模块尚处于活跃开发态。
- [ ] 在未冻结前，不宜把当前仓库直接视为最终可投稿归档版本。

## 4. 本轮已执行动作

- [x] 修复 `scripts/update_summary_tables.py`：
  - 增加双协议多 seed 汇总重建
  - 增加论文主表自动重建
- [x] 执行：
  - `python scripts/update_summary_tables.py --prefer_p4_final --no_refresh_p6_p9 --verbose`
- [x] 执行：
  - `python scripts/build_local_mapping_main_toptier.py --tum_multiseed output/summary_tables/tum_reconstruction_metrics_multiseed.csv --bonn_multiseed output/summary_tables/bonn_reconstruction_metrics_multiseed.csv --out_csv output/summary_tables/local_mapping_main_metrics_toptier.csv --out_md output/summary_tables/local_mapping_main_metrics_toptier.md`
- [x] 当前主表时间戳已统一刷新至 `2026-03-07 19:40` 左右：
  - `output/summary_tables/paper_main_table_local_mapping.csv`
  - `output/summary_tables/local_mapping_main_metrics_toptier.csv`
  - `output/summary_tables/dual_protocol_multiseed_reconstruction_agg.csv`
  - `output/summary_tables/dual_protocol_multiseed_dynamic_agg.csv`
  - `output/summary_tables/manifest.json`

## 5. 下一步建议（按优先级）

### N1. 先统一“文档叙事”到当前正式口径

- [ ] 用本轮刷新后的 canonical tables 回写：
  - `README.md`
  - `BENCHMARK_REPORT.md`
  - `TASK_LOCAL_TOPTIER.md`
- [ ] 明确哪些数字属于“历史轮次”，哪些数字属于“当前正式口径”。

### N2. 锁定 canonical source of truth

- [ ] 以后所有论文主表、报告主表、顶刊主表，只允许从以下文件派生：
  - `tum_reconstruction_metrics_multiseed_agg.csv`
  - `bonn_reconstruction_metrics_multiseed_agg.csv`
  - `dual_protocol_multiseed_significance.csv`
  - `local_mapping_main_metrics_toptier.csv`
  - `paper_main_table_local_mapping.csv`
- [ ] 禁止再把旧版 `summary_tables` 或历史局部实验表直接写回主结论。

### N3. P10 主线只保留 `dual_map + PFV`

- [ ] 后续 P10 研究优先级建议固定为：
  1. `dual_map + PFV` 主线增强
  2. 背景图 / 前景图内的更持久自由空间表示
  3. 不再回到已证伪的单图层 veto/记忆路线

### N4. 发布前补齐打包缺口

- [ ] 处理 `MERGED_DOCS.md` 缺失问题。
- [ ] 处理 `LICENSE` 缺失问题。
- [ ] 在冻结版本前，重新跑一次发布打包链路自检。

### N5. 版本冻结前做一次“结果治理审计”

- [ ] 在最终投稿前，追加一次只针对以下内容的冻结审计：
  - 文档数字是否与 canonical tables 一致
  - `summary_tables/manifest.json` 是否无缺项
  - 论文主表 / 顶刊主表 / 显著性是否同源
  - 工作树是否干净

## 6. 当前建议结论

- 当前仓库**不是不可用**，但也**不是可直接投稿归档的最终冻结态**。
- 当前最重要的积极进展，是：
  - 主表刷新链路已补齐；
  - `summary_tables` 的正式口径已经重新统一。
- 当前最重要的负面事实，是：
  - 按本轮统一后的正式口径，`P10` 仍未过线；
  - 且部分历史文档叙事已落后于当前 canonical tables。

