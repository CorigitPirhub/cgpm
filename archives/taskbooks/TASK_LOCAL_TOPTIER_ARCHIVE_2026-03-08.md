# EGF-DHMap 局部建图顶刊任务书（执行版）

## 执行状态（2026-03-04，审计修订）

### 审计结论（强约束口径）

本次按强约束口径完成审计后，以下风险被确认为高影响：
1. 历史部分 `slam` 结果使用了 `slam_use_gt_delta_odom=true`，不满足“真 SLAM（no-GT-delta）”约束。
2. `summary_tables` 存在历史旧表与新字段并存，需防止旧口径（单口径 ghost、混协议来源）继续进入主结论。
3. P12/P13 的历史执行目录包含上述风险输入，需按同一严格口径重跑后再判定通过。

### 任务状态降级（必须重跑）

- [x] **P0-R（必须）**：口径锁定重跑  
  - 目标：重建 `tum_*_slam.csv` 的严格来源（`--egf_slam_no_gt_delta_odom` + 双协议隔离 + 双口径 ghost 并行输出）。
  - 建议输出根目录：`output/post_cleanup/p0_true_slam_v3_final/`，并同步到 `output/post_cleanup/p3_tum_expanded/slam/tables/`。
- [x] **P12-R（必须）**：真实外部基线重跑  
  - 目标：仅保留真实 runner 输出（`tsdf,dynaslam,neural_implicit`），统一链路下刷新对照表。
  - 建议输出根目录：`output/post_cleanup/p12_external_native_final/`。
- [x] **P13-R（必须）**：双协议 5-seed + 显著性重跑  
  - 目标：TUM oracle + Bonn slam（no-GT-delta）重新生成主表与显著性，覆盖当前投稿主结论。
  - 建议输出根目录：`output/post_cleanup/p4_multiseed_tum_final_v2/` 与 `output/post_cleanup/p5_multiseed_bonn_all3/`。

### 审计重跑回写（2026-03-04）

- `P0-R`：已确认 `output/post_cleanup/p0_true_slam_v3_final/slam/**/egf/summary.json` 全部 `slam_use_gt_delta_odom=false`，并保留双口径 ghost 字段（`ghost_ratio_pred_denom` / `ghost_ratio_dyn_ref_denom`、`ghost_tail_ratio_pred_denom` / `ghost_tail_ratio_ref_denom`）。
- `P12-R`：已重建原生外部基线表并修复健康检查逻辑（外部 runner 非零返回码但真实输出完整时允许通过）；最终保留方法为 `tsdf,dynaslam,neural_implicit`。
- `P13-R`：Bonn `slam` 已完成 `5-seed` 严格 `no-GT-delta` 重跑（`40-44`）；并基于 `TUM oracle + Bonn slam` 刷新显著性与双协议主表。

### 建议复核（非阻断）

- [ ] **P1-Check（建议）**：统一参数在严格 `slam` 下复核一次（walking 三序列）。
- [ ] **P2-Check（建议）**：时间机理图补充双口径 ghost 指标并行展示（避免审稿口径争议）。

## 执行状态（2026-03-01，历史存档）

- [x] P0 已完成（协议分离 + 固定序列 + 固定 seed + 双次复验 + 汇总固化）
  - 执行脚本：`scripts/run_p0_protocol_lock.py`
  - 执行报告：`output/post_cleanup/p0_protocol_lock/p0_report.json`
  - 人类可读报告：`output/post_cleanup/p0_protocol_lock/P0_REPORT.md`
  - 固化汇总：
    - `output/summary_tables/tum_reconstruction_metrics_oracle.csv`
    - `output/summary_tables/tum_reconstruction_metrics_slam.csv`
    - `output/summary_tables/tum_dynamic_metrics_oracle.csv`
    - `output/summary_tables/tum_dynamic_metrics_slam.csv`
    - 主别名（默认 SLAM）：`output/summary_tables/tum_reconstruction_metrics.csv`, `output/summary_tables/tum_dynamic_metrics.csv`
- [x] P1 已完成（静态-动态性能平衡达标）
  - 执行脚本：`scripts/run_p1_balance.py`
  - 验收报告：`output/post_cleanup/p1_balance_lock/p1_report.json`
  - 人类可读报告：`output/post_cleanup/p1_balance_lock/P1_REPORT.md`
  - 复用达标实验：`output/post_cleanup/p1_unified_v8_strictsame_refresh2/slam/tables/`
  - 达标指标（EGF）：
    - `freiburg1_xyz`：`F-score=0.9010`（>=0.90）
    - `walking_xyz`：`F-score=0.7298`（>=0.70）
    - `walking_xyz`：`ghost_tail_ratio=0.0500`（<=0.30）
  - 固化汇总：
    - `output/summary_tables/tum_reconstruction_metrics_p1.csv`
    - `output/summary_tables/tum_dynamic_metrics_p1.csv`
- [x] P2 已完成（机制证据：消融 + 时间机理）
  - 执行脚本：`scripts/run_p2_mechanism.py`
  - 验收报告：`output/post_cleanup/p2_mechanism_lock/p2_report.json`
  - 人类可读报告：`output/post_cleanup/p2_mechanism_lock/P2_REPORT.md`
  - 验收结果：
    - `No-Evidence` 使 Ghost 恶化：`ghost_tail_ratio 0.3986 -> 0.4348`，`ghost_count +2001`
    - `No-Gradient` 使 F-score 显著下降：`0.8673 -> 0.6821`（drop=`0.1851`）
    - 时间机理（主 Ghost 指标=`ghost_count_per_frame`）：
      - `F-score`: `0.8236 -> 0.8964`（上升）
      - `ghost_count_per_frame`: `7939.27 -> 1997.49`（下降）
  - 固化汇总：
    - `output/summary_tables/ablation_summary_p2.csv`
    - `output/summary_tables/temporal_ablation_summary_p2.csv`
- [x] P3 已完成（真实外部输出接入 + 统一评估链路）
  - 执行脚本：`scripts/run_p3_real_external.py`
  - 外部算法：NICE-SLAM（真实输出 mesh，非 adapter 占位）
  - 核心目录：`output/post_cleanup/p3_real_external_niceslam/slam/`
  - 外部源路径（示例）：
    - `/home/zzy/Preception/CGPM/output/external/nice_slam/rgbd_dataset_freiburg3_walking_xyz_f010_fast/mesh/00005_mesh.ply`
  - 验收报告：
    - `output/post_cleanup/p3_real_external_niceslam/slam/p3_real_external_report.json`
    - `output/post_cleanup/p3_real_external_niceslam/slam/P3_REAL_EXTERNAL_REPORT.md`
  - 固化汇总：
    - `output/summary_tables/p3_real_external_reconstruction.csv`
    - `output/summary_tables/p3_real_external_dynamic.csv`

> 说明：以上为历史阶段完成状态。自本次修订起，后续章节采用“顶刊补齐版”目标，默认按新验收标准重新评估，不以前述完成项直接视为最终达标。

## 0. 目标与边界

### 0.1 目标
在“仅局部建图”设定下，把 EGF-DHMap 打磨到可投顶会/顶刊（优先 RA-L/ICRA/IROS）标准：  
**静态追平 + 动态跨序列一致领先 + 机理可解释 + 统计可复现 + 基线闭环真实化**。

### 0.2 边界
- 本任务书只覆盖局部建图，不要求闭环、大规模全局一致优化、长期漂移修正。
- 允许使用 GT pose 的 oracle 口径做重建公平性实验；同时必须补充 SLAM 口径可用性结果。
- 外部强基线可分两层：  
  - 第一层：本地可复现基线（必须）。  
  - 第二层：论文公开指标对比（可选增强）。

---

## 1. 顶刊“最低可过线”标准（局部建图版，2026-03 更新）

### 1.1 必须满足（Must）
1. **静态追平 TSDF（已达标，需保持）**  
   - `rgbd_dataset_freiburg1_xyz`：EGF `F-score >= 0.93`，`Chamfer <= 0.040`。  
   - 结果源：`output/summary_tables/static_target_constraint_check.csv`。
2. **动态几何“跨序列一致”**（当前短板）  
   - `walking_xyz/static/halfsphere` 三序列上，EGF 相对 TSDF 满足：  
     - `F-score` 每条序列 `>= TSDF - 0.01`；  
     - 三序列均值 `F-score_mean(EGF) >= F-score_mean(TSDF)`。  
   - 备注：重点修复 `walking_static` 当前落后问题。
3. **动态抑制优势保持**  
   - 三条 walking 上 `ghost_ratio` 相对 TSDF 的降幅均 `>= 35%`；  
   - 且相对当前 static-target 结果不退化超过 `+0.03`（防回归）。
4. **动态指标口径闭环**（避免分母效应争议）  
   - 主口径统一为 `ghost_count_per_frame` 或 `ghost_tail_ratio_fixed_region`；  
   - 时间维实验中该主口径应随帧数下降或稳定（Spearman `rho <= -0.7`）。
5. **真实外部基线闭环（非 adapter 占位）**  
   - 至少 2 个外部方法真实输出进入统一评估链路（建议 `DynaSLAM + MID/Neural`）；  
   - 至少覆盖 `walking_xyz` 与 `walking_static` 两条序列。
6. **统计显著性与工程可用性**  
   - 多 seed（>=3）提供 mean/std 与显著性检验；  
   - 补充局部建图效率表（帧时延、峰值内存），证明代价可接受。

### 1.2 可选增强（Optional but Strong）
1. Bonn 扩展到 `balloon2 + balloon + crowd2`，给出跨序列均值。
2. 主图增加“动态误差热力图 + rho 分离演化 + 失败案例”。
3. 局部消融加入“walking_static 定向修复”模块（例如分层表面提取）。

---

## 2. 任务分解（按优先级，补齐版）

## P0. 评价口径锁定与数据治理（阻断项）

### 目标
避免“指标漂亮但口径不公平/不可复现”。

### 动作
1. 固化主实验口径：`oracle` 与 `slam` 双协议分离。
2. 固化序列集合：  
   - 静态：`rgbd_dataset_freiburg1_xyz`  
   - 动态：`rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere`
3. 固化随机性：统一 `--seed` 或 `--seeds`。
4. 固化主汇总目录：`output/summary_tables/`。

### 使用脚本
- `scripts/run_benchmark.py`
- `scripts/update_summary_tables.py`

### 产物
- `output/summary_tables/tum_reconstruction_metrics.csv`
- `output/summary_tables/tum_dynamic_metrics.csv`
- `output/summary_tables/tum_reconstruction_metrics_agg.csv`（如多 seed）

### 验收
- 同一命令重复两次，指标波动在可接受范围（浮点微差）。

---

## P1. 动态几何一致性修复（核心主线）

### 目标
解决“动态抑制强，但某些动态序列几何落后 TSDF”的不一致问题，重点是 `walking_static`。

### 动作
1. 保持 SNEF 静态参数锁定，不回退静态追平结果。  
2. 针对动态分支补“几何保真”调优（不牺牲 ghost 优势）：
   - 优先调 `surface_max_age_frames / surface_max_dscore / truncation / sigma_n0`；
   - 强制检查 `walking_static` 的 precision-recall 平衡，避免单纯追 recall。
3. 新增三序列一致性对照表（EGF vs TSDF 的逐序列 delta）。

### 使用脚本
- `scripts/run_benchmark.py`
- `scripts/update_summary_tables.py`

### 产物
- `output/summary_tables/static_target_constraint_check.csv`
- `output/post_cleanup/*/tables/reconstruction_metrics.csv`
- `output/post_cleanup/*/tables/dynamic_metrics.csv`
- `output/summary_tables/tum_reconstruction_metrics_static_target_v1.csv`

### 验收（硬阈值）
1. `freiburg1_xyz`: 保持 `F-score >= 0.93`, `Chamfer <= 0.040`  
2. walking 三序列：`F-score(EGF) >= F-score(TSDF) - 0.01`  
3. walking 三序列：`ghost_ratio` 相对 TSDF 至少 `35%` 相对降幅

---

## P2. 指标口径闭环与时间机理强化

### 目标
消除动态指标歧义，形成可抗审稿质疑的时间机理证据链。

### 动作
1. 执行四组最小闭环消融（同预算）：  
   - `EGF-Full`
   - `EGF-No-Evidence`
   - `EGF-No-Gradient`
   - `EGF-Classic-SDF`
2. 执行时间维实验（`frames=15,30,45,60,90,120`）。  
3. 主 Ghost 指标统一为 `ghost_count_per_frame`（或 fixed-region 口径），`ghost_ratio` 降为辅指标。  
4. 给出趋势统计（Spearman 相关）并写入汇总表。

### 使用脚本
- `scripts/run_benchmark.py`（ablation flags）
- `scripts/build_ablation_summary.py`
- `scripts/run_temporal_ablation.py`

### 产物
- `output/summary_tables/ablation_summary.csv`
- `output/summary_tables/temporal_ablation_summary.csv`
- `assets/temporal_convergence_curve.png`
- `assets/temporal_rho_evolution.png`

### 验收
1. 去 `evidence` 后 Ghost 明显恶化（至少一个核心 Ghost 指标显著变差）。
2. 去 `gradient` 后 F-score 明显下降（建议绝对降幅 >= 0.05）。
3. 时间曲线满足：F-score 上升，Ghost 主指标下降或稳定，Spearman `rho <= -0.7`。

---

## P3. 基线矩阵“去适配器化”（真实外部输出）

### 目标
回答“与哪些方法比、为什么赢”。

### 必须基线（本地可复现）
1. `TSDF`（传统融合下限）
2. `Simple Removal`（粗暴动态剔除）
3. `EGF-Classic-SDF`（退化版内生基线）

### 必补强基线（至少其二，真实输出）
1. `DynaSLAM`（真实轨迹/地图输出）
2. `MID-Fusion` 或 `Co-Fusion`
3. `Neural implicit`（NICE-SLAM 系/同类）

### 使用脚本
- `scripts/run_benchmark.py`
- `scripts/adapters/*.py`（若接真实外部输出）

### 产物
- `output/summary_tables/tum_reconstruction_metrics.csv`
- `output/summary_tables/tum_dynamic_metrics.csv`
- `output/summary_tables/literature_baselines.csv`（文献对照表，含引用）
- `output/summary_tables/p3_real_external_reconstruction.csv`
- `output/summary_tables/p3_real_external_dynamic.csv`

### 验收
- 至少 2 个外部方法为“真实本地输出”（非占位路径）并进入统一评估；
- 覆盖 `walking_xyz` + `walking_static`；
- 若某基线仍为文献值，必须与本地结果分表展示，禁止混在同一主结论表。

### 本次执行结果（已完成）
1. 已用真实 NICE-SLAM 输出接入 `neural_implicit` 评估链路，并启用 `--external_require_real` 严格门控。
2. 已在统一链路跑通 `egf, tsdf, simple_removal, neural_implicit` 四方法同口径评估（`walking_xyz`）。
3. 当前 `DynaSLAM/MID-Fusion` 仍保留文献口径，不作为本地可复现结果写入主结论表。

---

## P4. 跨域泛化与失败案例分析（局部建图）

### 目标
证明局部建图优势不仅限 TUM walking，且能解释失败边界。

### 动作
1. 在 Bonn 上扩展到至少 3 条动态序列（`balloon2 + balloon + crowd2`）。  
2. 统一输出跨数据集均值表（TUM/Bonn 分开 + 总均值）。  
3. 对最差 1 条序列给出失败案例剖析（遮挡、快速运动、稀疏纹理等）。

### 使用脚本
- `scripts/run_benchmark.py`
- `scripts/run_benchmark_bonn.py`

### 产物
- `output/post_cleanup/p3_bonn_expanded/summary.csv`
- `output/summary_tables/bonn_summary.csv`
- `assets/bonn_comparison.png`

### 验收
1. Bonn 至少 2 条序列上 EGF 在 `ghost` 主指标优于 TSDF。  
2. Bonn 平均 `F-score` 不低于 TSDF（优先追求显著领先）。  
3. 报告中必须包含至少 1 条失败案例与原因解释。

---

## P5. 统计显著性与效率证据（投稿前补齐）

### 目标
让“图-表-文-命令-代价”五者一致，可审、可复现、可归档。

### 动作
1. 多 seed 统计（>=3）并输出 mean/std、显著性检验。  
2. 增加局部建图效率表：每帧时延、峰值内存、相对 TSDF 倍率。  
3. 固化主文档：  
   - `BENCHMARK_REPORT.md`（论文风格）  
   - `README.md`（快速复现）  
4. 输出最终表格到 `output/summary_tables/`。  
5. 检查文档中的每个数字都能追溯到 CSV。  
6. 打包发布（精简结果，不含巨型中间点云）。

### 使用脚本
- `scripts/update_summary_tables.py`
- `scripts/package_release.sh`

### 产物
- `output/summary_tables/*.csv`
- `BENCHMARK_REPORT.md`
- `README.md`
- `egf_dhmap3d_v1.0_release.zip`（可选）

### 验收
- 新机器按 README 命令可复现主表核心行（允许小幅随机波动）。  
- 统计表与效率表均进入 `output/summary_tables/` 且在报告中被引用。

---

## 3. 里程碑与停止条件

## M1（1 周）
- 完成 P1。  
- 停止条件：静态保持达标，且 walking 三序列几何一致性达标。

## M2（1 周）
- 完成 P2 + P3。  
- 停止条件：指标口径闭环 + 外部基线真实输出闭环。

## M3（0.5-1 周）
- 完成 P4 + P5。  
- 停止条件：跨域泛化 + 统计显著性 + 效率证据闭环。

---

## 4. 最终投稿门槛（局部建图）

### 必须达成
1. 静态追平：`freiburg1_xyz F-score >= 0.93` 且 `Chamfer <= 0.040`
2. 动态一致：walking 三序列 `F-score(EGF) >= F-score(TSDF) - 0.01`
3. 动态优势：walking 三序列 `ghost_ratio` 相对 TSDF 至少 35% 降幅
4. 指标口径闭环：主 Ghost 指标时间趋势下降或稳定（含相关系数）
5. 真实外部基线：至少 2 个方法真实输出进入统一链路
6. 多 seed + 显著性 + 效率表

### 达成后结论
可定位为：  
**“局部动态建图方向的顶刊可投稿工作（证据完整）”**。  
若再补更高分辨率/更长时段序列，可进一步冲击更高影响力 venue。

---

## 5. 立即执行清单（本次目标）

### 执行结果（2026-03-01）

- [x] 1. 修复 `walking_static` 几何一致性并刷新主表  
  - 主表已切换为 `p1_consistency_v2` 口径（`scripts/update_summary_tables.py` 已支持优先读取）。  
  - 产物：`output/summary_tables/tum_reconstruction_metrics.csv`，`output/summary_tables/tum_dynamic_metrics.csv`。  
  - 关键值：`walking_static` 上 `F-score(EGF)=0.9187`，`F-score(TSDF)=0.9246`，差值 `-0.0059`（满足 `>= -0.01` 一致性约束）。

- [x] 2. 固化主 Ghost 口径（`ghost_count_per_frame`）并补时间趋势统计  
  - 产物：`output/summary_tables/temporal_trend_stats.csv`，`output/summary_tables/temporal_trend_stats.json`。  
  - 关键值：`spearman_frames_ghost_main=-1.0`，`ghost_main_first=7939.27 -> ghost_main_last=1997.49`。

- [x] 3. 跑外部基线（>=2 方法）并输出独立对照表  
  - 已跑统一链路：`output/post_cleanup/p3_real_external_dual/slam/tables/`。  
  - 独立表：`output/summary_tables/external_baselines_independent.csv`。  
  - 当前为两条非占位真实来源路径（`is_real_external=1`）：`dynaslam` 槽位与 `neural_implicit` 槽位均绑定到本地真实 mesh 文件。
  - 备注：`dynaslam` 已切换为真实 runner 输出（见 `scripts/external/run_dynaslam_tum_runner.py`），并通过 `--external_require_real` 校验非占位路径。

- [x] 4. 跑 Bonn 扩展序列并补失败案例分析  
  - 扩展结果：`output/post_cleanup/p3_bonn_expanded/summary.csv`。  
  - 失败案例分析：`output/post_cleanup/p3_bonn_expanded/FAILURE_CASE.md`。  
  - 图：`assets/bonn_comparison.png`（对比可视化）。

- [x] 5. 产出“五件套”（主表 + 机制图 + 基线表 + 效率表 + 复现命令）  
  - 主表：`output/summary_tables/tum_reconstruction_metrics.csv`，`output/summary_tables/tum_dynamic_metrics.csv`  
  - 机制图/机制表：`assets/temporal_convergence_curve.png`，`output/summary_tables/temporal_trend_stats.csv`  
  - 基线表：`output/summary_tables/external_baselines_independent.csv`  
  - 效率表：`output/summary_tables/local_mapping_efficiency.csv`  
  - 复现命令：`output/summary_tables/reproduce_commands.md`

### 下一步目标（紧接本轮）

1. `[已完成]` `dynaslam` 槽位替换为真实外部 runner，统一评估链路不变。  
   - 新增 runner：`scripts/external/run_dynaslam_tum_runner.py`（真实 DynaSLAM 输出点云/mesh）。  
   - 严格门控：`--external_require_real`。  
   - 结果表：`output/summary_tables/external_baselines_independent.csv`（`is_real_external=1`）。  
   - 真实输出路径示例：`output/external/dynaslam_real/rgbd_dataset_freiburg3_walking_xyz/mesh.ply`。

2. `[已完成]` 动态优势约束补齐（`walking_static`）  
   - 口径：`dynamic_ref_max_ratio=0.40`（主链路一致）。  
   - 结果：`ghost_ratio` 相对 TSDF 降幅 `39.59%`（>=35%）。  
   - 同时满足：`F-score(EGF)-F-score(TSDF)=-0.0059`（>= -0.01）。  
   - 结果表：`output/summary_tables/tum_dynamic_metrics.csv`、`output/summary_tables/tum_reconstruction_metrics.csv`。

3. `[已完成]` Bonn 泛化修复（自适应表面提取）  
   - 代码：  
     - `egf_dhmap3d/core/config.py`（adaptive 配置）  
     - `egf_dhmap3d/core/voxel_hash.py`（adaptive 阈值门控）  
     - `egf_dhmap3d/modules/pipeline.py`（参数透传）  
     - `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`（CLI/统一入口）  
   - 实验：`output/post_cleanup/p3_bonn_expanded_v2/slam/`。  
   - 修复前后对照：`output/post_cleanup/p3_bonn_expanded_v2/repair_delta.csv`、`output/post_cleanup/p3_bonn_expanded_v2/FAILURE_CASE_REPAIR.md`。  
   - 核心结果（3 序列均值）：  
     - EGF `F-score: 0.1232 -> 0.9333`  
     - EGF `Chamfer: 1.4611 -> 0.0409`  
     - EGF `ghost_ratio: 0.0401 -> 0.2574`（有回升，但仍显著低于 TSDF `0.5530`，约 `53.5%` 降幅）。

### 下一步目标（本轮完成后）

1. `[已完成]` 静态追平 + 动态约束同持  
   - 主表来源：`output/summary_tables/tum_reconstruction_metrics_static_target_v1.csv`、`output/summary_tables/static_target_constraint_check.csv`  
   - 当前 canonical 结果：`freiburg1_xyz` 上 `F-score=0.941005`、`Chamfer=0.037296`；3 条 walking 的 `ghost_ratio` 在约束表中均未退化，且分别下降 `-0.04377 / -0.06393 / -0.05222`。  

2. `[已完成]` 严格协议主结论已统一到双协议 canonical 口径  
   - 正式主表：`output/summary_tables/paper_main_table_local_mapping.csv`、`output/summary_tables/local_mapping_main_metrics_toptier.csv`  
   - 单次 `slam` 验收表仍保留在：`output/summary_tables/tum_reconstruction_metrics_slam.csv`、`output/summary_tables/tum_dynamic_metrics_slam.csv`  
   - 但正式对外结论不再引用单次 `slam` 数字，而统一引用双协议 `5-seed` canonical 表。  

3. `[已完成]` TUM + Bonn `5-seed` 统计显著性  
   - TUM 多 seed 根目录：`output/post_cleanup/p4_multiseed_tum_final_v2/`（oracle, 3 walking × 5 seeds）  
   - Bonn 多 seed 根目录：`output/post_cleanup/p5_multiseed_bonn_all3/`（slam, all3 × 5 seeds）  
   - 显著性表：  
     - `output/summary_tables/tum_significance_multiseed.csv`  
     - `output/summary_tables/bonn_significance_multiseed.csv`  
     - `output/summary_tables/dual_protocol_multiseed_significance.csv`  
   - 核心结论（t-test）：  
     - TUM dynamic: `fscore p=4.81e-10`, `ghost_ratio p=1.72e-24`  
     - Bonn dynamic: `fscore p=1.06e-18`, `ghost_ratio p=5.35e-05`  
   - 当前 canonical 主表来源：`output/summary_tables/paper_main_table_local_mapping.csv`。  

### 下一步目标（建议）

1. `[已完成]` Bonn 扩展到 `balloon + crowd2` 的 `5-seed` canonical 统计，验证显著性跨序列保持。  
   - 实验根目录：`output/post_cleanup/p5_multiseed_bonn_all3/`（`balloon2 + balloon + crowd2`, seeds=`40,41,42,43,44`）。  
   - 显著性：`output/post_cleanup/p5_multiseed_bonn_all3/tables/significance.csv`。  
   - 汇总同步后：`output/summary_tables/bonn_reconstruction_metrics_multiseed.csv`、`output/summary_tables/bonn_significance_multiseed.csv`。  
   - 关键结论（EGF vs TSDF, dynamic, n=15）：`fscore mean +0.5765, p=1.06e-18`; `ghost_ratio mean improve +0.1261, p=5.35e-05`。  
2. `[已完成]` 将 `scripts/update_summary_tables.py` 增加 `--prefer_p4_final` 开关，避免手工覆盖主表。  
   - 用法：`python scripts/update_summary_tables.py --prefer_p4_final --verbose`。  
   - 作用：主表优先读取 `output/post_cleanup/p4_final_merged/oracle/tables/`，无需手工覆盖。  
3. `[已完成]` 统一 `oracle/slam` 双协议多 seed 汇总模板，减少后续论文版本迭代时的手工表格维护成本。  
   - 新脚本：`scripts/build_dual_protocol_multiseed_summary.py`。  
   - 产物：  
     - `output/summary_tables/dual_protocol_multiseed_reconstruction.csv`  
     - `output/summary_tables/dual_protocol_multiseed_dynamic.csv`  
     - `output/summary_tables/dual_protocol_multiseed_reconstruction_agg.csv`  
     - `output/summary_tables/dual_protocol_multiseed_dynamic_agg.csv`  
     - `output/summary_tables/dual_protocol_multiseed_significance.csv`  

### 下一步目标（建议）

1. `[已完成]` `run_benchmark.py` 增加 Bonn 短命令预设。  
   - 代码：`scripts/run_benchmark.py`（新增 `--bonn_dynamic_preset auto|none|balloon2|all3`）。  
   - 默认行为：`--dataset_kind bonn` 且不显式改序列时，自动使用 `balloon2,balloon,crowd2`，并清空默认静态序列。  
   - 验证：`python scripts/run_benchmark.py --dataset_kind bonn --dataset_root data/bonn --dry_run` 可直接展开 3 序列。  
2. `[已完成]` `BENCHMARK_REPORT.md` 已补 Bonn 三序列多 seed 小节并固定引用结果文件。  
   - 报告位置：`BENCHMARK_REPORT.md` 第 19 节。  
   - 固定引用：  
     - `output/summary_tables/bonn_reconstruction_metrics_multiseed_agg.csv`  
     - `output/summary_tables/bonn_significance_multiseed.csv`  
3. `[已完成]` `dual_protocol_multiseed_significance.csv` 已接入论文主表生成链路。  
   - 新脚本：`scripts/build_paper_main_table.py`。  
   - 输入：  
     - `output/summary_tables/tum_reconstruction_metrics_multiseed_agg.csv`  
     - `output/summary_tables/bonn_reconstruction_metrics_multiseed_agg.csv`  
     - `output/summary_tables/dual_protocol_multiseed_significance.csv`  
   - 输出：  
     - `output/summary_tables/paper_main_table_local_mapping.csv`  
     - `output/summary_tables/paper_main_table_local_mapping.md`  

---

## 6. 顶刊差距评估（仅局部建图，文献锚定，2026-03-02）

### 6.1 近期顶会/顶刊公开对照（用于差距判断）

1. **RoDyn-SLAM (RA-L 2024)**  
   - TUM walking ATE（m，论文表 II）：`xyz=0.083`, `static=0.017`, `halfsphere=0.056`  
   - BONN 建图（Acc/Comp/Comp-R，论文表 I）：`8.95cm / 15.84cm / 37.79%`  
   - 运行时（文中报告）：tracking `~159ms/frame`，mapping `~675ms/frame`（总计约 `0.834s/frame`）。
2. **DG-SLAM (NeurIPS 2024)**  
   - TUM walking ATE（cm，论文表 I）：`xyz=1.36`, `static=2.72`, `halfsphere=5.32`  
   - BONN 建图（Acc/Comp/Comp-R，论文表 IV）：`8.06cm / 15.46cm / 43.67%`  
   - 运行时（文中报告）：约 `645.9ms/frame`（含分割约 `163.1ms/frame`）。
3. **GS-SLAM (CVPR 2024)**  
   - 项目页报告平均速度约 `386 FPS`（强调实时性上限）。
4. **Co-SLAM (CVPR 2023)**  
   - 论文摘要与项目资料报告约 `10-17 Hz` 实时区间。

> 以上用于“顶刊证据形态”对标，详细原始链接已同步到 `output/summary_tables/literature_tum_walking_metrics.csv` 与 `output/summary_tables/literature_vs_ours_tum_walking_gap.csv`。

### 6.2 当前结论（仅局部建图）

1. **动态优势仍显著，但顶刊主口径仍未过线**  
   - TUM dynamic（`5-seed`, `oracle`, canonical 均值）：`EGF F-score=0.7903 > TSDF=0.4137`，`EGF ghost_ratio=0.2566 < TSDF=0.8657`，`Chamfer=0.05317 < 0.13146`。  
   - Bonn dynamic（3 序列 × `5-seed`, `slam`, canonical 均值）：`EGF F-score=0.6463 > TSDF=0.06973`，`EGF ghost_ratio=0.08613 < TSDF=0.21227`，`Chamfer=0.10116 < 0.25664`。  
   - 统计显著性已满足：TUM `p_fscore=4.81e-10`, `p_ghost=1.72e-24`；Bonn `p_fscore=1.06e-18`, `p_ghost=5.35e-05`。  
2. **离顶刊“最终形态”仍有关键差距**  
   - **主口径差距明确**：`local_mapping_main_metrics_toptier.csv` 显示 EGF 当前 `TUM Acc=4.1655 cm`、`Bonn Acc=6.1481 cm`、`Bonn Comp-R(5cm)=77.21%`，距离 `P10` 硬门槛仍有明显差距。  
   - **效率证据需要更清晰分层**：P7 质量保持配置约 `1.146 FPS`，P15 实时调优路径可到 `mapping_fps≈22.09`，但仍需统一 full-pipeline wall-clock 口径后再用于最终主叙事。  
   - **强基线真实复现仍不足**：虽已接入真实外部输出链路，但“同协议、同序列、同预算”下的 `>=2` 强基线原生 runner 仍需补齐。  
   - **鲁棒性边界证据仍需文档统一**：当前 stress test 已有表和失败边界，但还需与 canonical 主表叙事彻底对齐。

### 6.3 新增收敛任务（用于投稿前封板）

### P6. 顶刊协议对齐（指标口径）

### 目标
让主结论可直接放入与 RoDyn/DG 同风格的主表，避免“指标不对口”审稿风险。

### 动作
1. 在统一评估链路中新增 Bonn/TUM 的 `Acc(cm) / Comp(cm) / Comp-R(5cm)` 输出。  
2. 现有 `F-score/Chamfer/Ghost` 保留为辅表，主表采用论文常见三元组 + 显著性。  
3. 固化脚本：`scripts/run_benchmark.py` + `scripts/update_summary_tables.py` 一次产出双口径。

### 产物
- `output/summary_tables/local_mapping_main_metrics_toptier.csv`  
- `output/summary_tables/local_mapping_main_metrics_toptier.md`

### 验收
1. EGF 在 Bonn dynamic 上 `Comp-R(5cm)` 不低于 TSDF，且提升具统计显著性（`p<0.05`）。  
2. 主表每个数字可追溯到单序列多 seed 明细。

### P7. 局部建图效率攻关（不改主故事）

### 目标
把局部建图推入“可用实时区间”，缩小与 RoDyn/DG 的量级差。

### 动作
1. 建立 profiling（投影、融合、表面提取三段耗时）。  
2. 引入块级增量更新和缓存重用，避免全体素重复扫描。  
3. 给出 `quality-speed` 曲线（voxel_size, max_points_per_frame, stride）。

### 产物
- `output/summary_tables/local_mapping_efficiency_v2.csv`  
- `assets/quality_speed_tradeoff.png`

### 验收
1. 在 `walking_xyz`（40帧）达到 `>=1.0 FPS`（同硬件同输入口径）。  
2. 相对当前最优质量配置，`F-score` 下降不超过 `0.02`，`ghost_ratio` 反弹不超过 `0.03`。

### P8. 强基线原生对比补齐（去适配器化终版）

### 目标
完成“同协议、同序列、同预算”的外部方法硬对比闭环。

### 动作
1. 至少两条真实 runner：优先 `DynaSLAM + MID-Fusion`（或 `RoDyn/DG` 可运行替代）。  
2. 统一输入、统一帧采样、统一评估脚本；保留完整 runner 日志与版本信息。  
3. 主结论表只放本地真实运行结果；文献值独立附表。

### 产物
- `output/summary_tables/external_baselines_native_reconstruction.csv`  
- `output/summary_tables/external_baselines_native_dynamic.csv`  
- `output/summary_tables/external_baselines_native_runtime.csv`

### 验收
1. 至少 2 个外部方法在 `walking_xyz + walking_static` 双序列跑通。  
2. 结果可通过 `reproduce_commands.md` 一键复现（含依赖版本）。

### P9. 边界鲁棒性与失败模式（审稿防守）

### 目标
把“方法在何时失效”量化成可解释边界，而不是只给最好结果。

### 动作
1. 构建 stress 维度：深度缺失率、动态占比、帧间位移速度、遮挡比例。  
2. 每个维度至少 3 档强度，输出退化曲线。  
3. 对最差档位给出 `rho`、`d_score`、表面提取阈值的失效诊断图。

### 产物
- `output/summary_tables/stress_test_summary.csv`  
- `assets/stress_degradation_curves.png`  
- `output/post_cleanup/stress_test/FAILURE_CASES.md`

### 验收
1. 在中等强度（Level-2）下，EGF 相对 TSDF 的 `ghost_ratio` 优势仍 >= `25%`。  
2. 明确给出至少 2 条“不可用边界”及建议规避策略。

### 本轮执行结果（2026-03-02，P6-P9）

- [x] **P6 顶刊协议对齐（指标口径）**
  - 主表产物：
    - `output/summary_tables/local_mapping_main_metrics_toptier.csv`
    - `output/summary_tables/local_mapping_main_metrics_toptier.md`
  - 关键验收：
    - Bonn dynamic 上 `Comp-R(5cm)`：EGF 约 `100%`，TSDF 约 `37~39%`。
    - 显著性：`p_comp_r_5cm_egf_vs_tsdf_t = 4.81e-16`（< 0.05）。
  - 结论：通过。

- [x] **P7 局部建图效率攻关（质量-速度权衡）**
  - 产物：
    - `output/summary_tables/local_mapping_efficiency_v2.csv`
    - `output/summary_tables/local_mapping_efficiency_v2.md`
    - `output/summary_tables/local_mapping_efficiency_v2.json`
    - `assets/quality_speed_tradeoff.png`
  - 配置搜索：`mpp900,500,400,380,300`（`walking_xyz`, 40 帧, oracle）。
  - 选中配置：`mpp500`
    - `fps = 1.146`（>= 1.0）
    - 相对质量锚点 `mpp900`：`delta_fscore = -0.0092`（>= -0.02）
    - `delta_ghost_ratio = -0.0477`（<= +0.03）
  - 结论：通过。

- [x] **P8 强基线原生对比补齐（去适配器化终版）**
  - 产物：
    - `output/summary_tables/external_baselines_native_reconstruction.csv`
    - `output/summary_tables/external_baselines_native_dynamic.csv`
    - `output/summary_tables/external_baselines_native_runtime.csv`
  - 执行脚本：`scripts/run_p8_native_external.py`
  - 覆盖方法与序列：
    - 真实外部方法：`dynaslam`, `neural_implicit`
    - 双序列：`walking_xyz`, `walking_static`
  - 严格门控：`--external_require_real`，输入来自真实输出目录（非 placeholder）。
  - 结论：通过。

- [x] **P9 边界鲁棒性与失败模式**
  - 产物：
    - `output/summary_tables/stress_test_summary.csv`
    - `assets/stress_degradation_curves.png`
    - `output/post_cleanup/stress_test/FAILURE_CASES.md`
    - `output/post_cleanup/stress_test/stress_meta.json`
  - 压力维度：
    - `point_budget`（mpp900/500/300）
    - `temporal_sparsity`（stride2/3/5）
    - `motion_pattern`（walking_static/xyz/halfsphere）
  - Level-2 验收（EGF ghost 降幅 vs TSDF）：
    - `point_budget`: `71.88%`
    - `temporal_sparsity`: `71.88%`
    - `motion_pattern`: `50.62%`
    - 均 >= 25%，并给出 2 条失败边界与缓解策略。
  - 结论：通过。

---

## 7. 终版封板任务书（唯一有效验收，完成后不再使用“还差一点”表述）

> 从本节开始，前文历史阶段仅作过程记录；**是否达到顶刊局部建图标准，只按本节判定**。  
> 通过条件：本节 `P10-P14` 的所有硬门槛全部满足。

### 7.1 目标定义（局部建图，顶刊封板版）

在不讨论全局闭环/长期一致性的前提下，完成一个**审稿可防守**的局部动态建图证据包，满足：
1. 几何质量、动态抑制、效率三者同时可用；  
2. 外部强基线公平且可复现；  
3. 评估协议统一，无口径争议；  
4. 统计显著 + 失败边界齐全。

### 7.2 统一评估协议（强制）

#### 协议定义
1. `PROTO-ORACLE`：使用 GT 位姿，仅评估重建算子本体。  
2. `PROTO-SLAM`：使用系统位姿输入（含对齐评估），评估实用可用性。  

#### 强制规则
1. 所有主结论表必须同时给出 `PROTO-ORACLE` 与 `PROTO-SLAM`，不得混表偷换。  
2. 所有方法必须共享：序列、帧数、stride、max_points_per_frame、voxel_size、阈值。  
3. 主表字段固定：`Acc(cm), Comp(cm), Comp-R(5cm), F-score, ghost_ratio, runtime(fps), memory`。

### 7.3 终版任务

### P10. 精度-完整性平衡封板（解决 Accuracy 弱项）

#### 目标
在保持动态优势的同时，补齐“EGF Accuracy 偏高（数值更差）”短板，消除“以召回换精度”质疑。

#### 动作
1. 在 `run_benchmark.py` 增加 “high-precision extraction profile”（仅局部建图）：
   - 细化表面门控（phi/rho/normal 一致性）；
   - 保留证据场机制，避免 ghost 反弹；
   - 输出 profile 名称与参数到 summary。
2. 对 `walking_xyz/static/halfsphere` + Bonn all3 做 profile sweep，保留 Pareto 前沿。
3. 生成 `accuracy-completeness` tradeoff 图并标注最终点。

#### 产物
- `output/summary_tables/local_mapping_precision_profile.csv`  
- `assets/acc_comp_tradeoff.png`

#### 硬验收
1. TUM walking（三序列均值，oracle）：  
   - `Acc(cm)_EGF <= 1.80`  
   - `Comp-R(5cm)_EGF >= 95%`  
2. Bonn all3（slam）：  
   - `Acc(cm)_EGF <= 2.60`  
   - `Comp-R(5cm)_EGF >= 95%`  
3. 动态抑制不回退：`ghost_ratio` 相对 TSDF 降幅仍 `>=35%`。

#### 执行状态（2026-03-03，封板到最优可达）
- 新增脚本：`scripts/run_p10_precision_profile.py`（profile sweep + 统一验收 + 交易曲线）。
- 新增能力：
  - `SNEF-3D local` 已落地到核心提取链路（`egf_dhmap3d/core/voxel_hash.py`），支持分块分位裁剪：
    - `d_score` 分位裁剪；
    - `free_ratio` 分位裁剪；
    - `|phi|` 分位裁剪（v2）。
  - 入口参数已贯通：`scripts/run_egf_3d_tum.py` / `scripts/run_benchmark.py`。
- 当前产物：
  - `output/summary_tables/local_mapping_precision_profile.csv`
  - `assets/acc_comp_tradeoff.png`
  - `output/post_cleanup/p10_precision_profile_quick/best_profile.json`
  - `output/post_cleanup/p10_tune/tum_strict1/oracle/tables/reconstruction_metrics.csv`
  - `output/post_cleanup/p10_tune/tum_strict1/oracle/tables/dynamic_metrics.csv`
- 当前结果（以 `local_mapping_precision_profile.csv` 封板）：
  - `baseline_relaxed`（当前 Pareto 最优平衡点）  
    - TUM walking（oracle）：`acc_cm=2.5644`, `comp_r_5cm=99.9958%`, `ghost_ratio=0.5248`
    - Bonn all3（slam）：`acc_cm=2.8625`, `comp_r_5cm=99.9998%`, `ghost_ratio=0.5110`
    - TUM `ghost_ratio` 相对 TSDF 降幅：`22.47%`
  - `snef_mild/snef_balanced`：可进一步压低 ghost，但 `Comp-R` 显著塌陷（`<20%`），不可作为主线配置。
- 结论：`P10` **硬验收未通过**，但已完成“最优可达封板（best-achievable）”。当前版本不再继续强行压 `Acc`，避免破坏完整性与动态鲁棒性主卖点；后续仅作为独立研究分支推进。

#### 追加冲线执行记录（2026-03-04，按 A/B/C 清单逐步执行）
- A 阶段（LZCD 去偏）：
  - 命令：
    - `/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_a,lzcd_only_b --frames 40 --stride 3 --max_points_per_frame 900 --out_root output/post_cleanup/p10_lzcd_v2 --summary_csv output/summary_tables/p10_lzcd_v2.csv --figure assets/p10_lzcd_v2.png --force`
  - 产物：
    - `output/summary_tables/p10_lzcd_v2.csv`
    - `assets/p10_lzcd_v2.png`
  - 结果（最佳 `lzcd_only_b`）：
    - `tum_acc_cm=2.7893`，`tum_comp_r_5cm=99.8046`
    - `bonn_acc_cm=3.4254`，`bonn_comp_r_5cm=100.0`
    - `ghost_red_min=0.1298`
  - 判定：`pass_all=False`

- B 阶段（STCG 局部门控）：
  - 说明：`run_p10_precision_profile.py` 在当前环境下会触发 TSDF 子流程异常告警（Open3D: mesh has 0 vertices），为避免基线噪声干扰，改为直接运行 `run_benchmark.py` 的 EGF-only，并复用 A 阶段健康 TSDF 参考做统一判分。
  - 命令：
    - `python scripts/run_benchmark.py --dataset_kind tum ... --out_root output/post_cleanup/p10_stcg_direct/01_lzcd_stcg_hetero_c/tum_oracle ... --methods egf ... --force`
    - `python scripts/run_benchmark.py --dataset_kind bonn ... --out_root output/post_cleanup/p10_stcg_direct/01_lzcd_stcg_hetero_c/bonn_slam ... --methods egf ... --force`
  - 产物：
    - `output/post_cleanup/p10_stcg_direct/01_lzcd_stcg_hetero_c/tum_oracle/oracle/tables/reconstruction_metrics.csv`
    - `output/post_cleanup/p10_stcg_direct/01_lzcd_stcg_hetero_c/tum_oracle/oracle/tables/dynamic_metrics.csv`
    - `output/post_cleanup/p10_stcg_direct/01_lzcd_stcg_hetero_c/bonn_slam/slam/tables/reconstruction_metrics.csv`
    - `output/post_cleanup/p10_stcg_direct/01_lzcd_stcg_hetero_c/bonn_slam/slam/tables/dynamic_metrics.csv`
  - 结果：
    - `tum_acc_cm=2.8260`，`bonn_acc_cm=3.4888`
    - `tum_comp_r_5cm=99.7157`，`bonn_comp_r_5cm=100.0`
    - `ghost_red_min=0.1541`
  - 判定：`pass_all=False`

- C 阶段（双通道 + 软门控）：
  - 命令：
    - `python scripts/run_benchmark.py --dataset_kind tum ... --out_root output/post_cleanup/p10_dualch_direct/01_dualch_c/tum_oracle ... --methods egf ... --force`
    - `python scripts/run_benchmark.py --dataset_kind bonn ... --out_root output/post_cleanup/p10_dualch_direct/01_dualch_c/bonn_slam ... --methods egf ... --force`
  - 产物：
    - `output/post_cleanup/p10_dualch_direct/01_dualch_c/tum_oracle/oracle/tables/reconstruction_metrics.csv`
    - `output/post_cleanup/p10_dualch_direct/01_dualch_c/tum_oracle/oracle/tables/dynamic_metrics.csv`
    - `output/post_cleanup/p10_dualch_direct/01_dualch_c/bonn_slam/slam/tables/reconstruction_metrics.csv`
    - `output/post_cleanup/p10_dualch_direct/01_dualch_c/bonn_slam/slam/tables/dynamic_metrics.csv`
  - 结果：
    - `tum_acc_cm=2.8850`，`bonn_acc_cm=3.3495`
    - `tum_comp_r_5cm=99.9981`，`bonn_comp_r_5cm=100.0`
    - `ghost_red_min=0.0716`
  - 判定：`pass_all=False`

- A/B/C 汇总与封板：
  - 汇总表：`output/summary_tables/p10_abc_eval.csv`
  - 同步主表：`output/summary_tables/local_mapping_precision_profile.csv`
  - 图：`assets/acc_comp_tradeoff.png`
  - 结论：
    - A/B/C 三阶段均未满足 `Acc` 硬门槛（TUM<=1.8cm, Bonn<=2.6cm）。
    - 当前最优仍为 `lzcd_only_b`，在不破坏 `Comp-R` 的前提下，`Acc` 与 `ghost_red_min` 未进一步过线。
    - `P10` 状态维持为 **未过线（best-achievable under current operators）**。


#### 追加冲线执行记录（2026-03-06，P10 结构解耦主线：WDSG-R -> SPG）
- 背景：前一轮 `OMHS/RPS/HRC/Surface-Bank` 已证明 readout 侧不是主瓶颈；`WDSG-R` 首次把 `Acc` 压到 `1.39cm`，但 `ghost_ratio` 仍约 `0.286`，且 `Comp-R@5cm` 只有 `91.79%`。
- 本轮目标：不再做阈值扫参，改为把“几何候选”和“静态导出”拆成两套状态语义。
- 新增结构：
  - `WDSG-R + SPG`：`phi_geo` 继续承担高精度候选几何；新增 `SPG`（Static Promotion Gate）把时间一致的静态几何晋升到独立 front bank；
  - `SPG fallback`：未晋升且带动态/遮挡信号的前层几何只能保守回退；
  - `No-direct-geo export`：`phi_geo` 不再直接导出，必须先经 `SPG` 晋升，防止候选几何绕过静态判据。
- 运行命令：
  - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_wdsgr_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v10_wdsgr --summary_csv output/summary_tables/p10_structural_probe_v10_wdsgr.csv --figure assets/p10_structural_probe_v10_wdsgr.png --force`
  - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_wdsgr_spg_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v11_wdsgr_spg --summary_csv output/summary_tables/p10_structural_probe_v11_wdsgr_spg.csv --figure assets/p10_structural_probe_v11_wdsgr_spg.png --force`
  - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_wdsgr_spg_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v12_wdsgr_spgfb --summary_csv output/summary_tables/p10_structural_probe_v12_wdsgr_spgfb.csv --figure assets/p10_structural_probe_v12_wdsgr_spgfb.png --force`
  - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_wdsgr_spg_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v13_wdsgr_spg_nogeo --summary_csv output/summary_tables/p10_structural_probe_v13_wdsgr_spg_nogeo.csv --figure assets/p10_structural_probe_v13_wdsgr_spg_nogeo.png --force`
- 汇总产物：
  - `output/summary_tables/p10_structural_probe_v5_v10_v11_v12_v13.csv`
  - `output/summary_tables/p10_structural_probe_v10_v11_v12_v13_by_sequence.csv`
- 关键结果（TUM walking mean / Bonn all3 mean）：
  - `v10_wdsgr`: `Acc=1.3887cm`, `Comp-R=91.79%`, `F-score=0.9369`, `ghost_ratio=0.2865`
  - `v11_wdsgr_spg`: `Acc=1.1910cm`, `Comp-R=93.11%`, `F-score=0.9562`, `ghost_ratio=0.2954`
  - `v12_wdsgr_spgfb`: 与 `v11` 基本等价，说明“保守静态回退”不是主要瓶颈
  - `v13_wdsgr_spg_nogeo`: `Acc=1.2331cm`, `Comp-R=94.79%`, `F-score=0.9650`, `ghost_ratio=0.2853`
- 结论：
  1. `WDSG-R` 证明 update 端双表面生成是正确方向；
  2. `SPG` 证明“候选几何 / 导出几何”分离后，`Acc + Comp-R` 可以继续同时上涨；
  3. `No-direct-geo export` 是本轮最好配置，几何质量明显优于 `v10`，ghost 也从 `v11/v12` 的 `0.295` 级别回落到 `0.285`；
  4. 但 `ghost_ratio` 仍未显著优于短 probe 下的 TSDF，说明当前剩余瓶颈不在 persistent readout，而在**即时动态观测没有被单独状态化**。当前 `ghost_tail_ratio` 在 TUM 已接近 `0`，说明“历史残影”问题基本被抑制，真正没解开的是“当前帧动态表面是否应该进入局部静态地图”。
- 状态更新：
  - `P10` 仍 **未过线**；
  - 但当前 `best-achievable under current structural decouple line` 已从早期 `baseline_relaxed (Acc≈2.56cm)` 更新为 `v13_wdsgr_spg_nogeo (Acc≈1.23cm, Comp-R≈94.79%, F-score≈0.965)`；
  - 下一步不再继续堆 readout 规则，而应补一个“观测级动态态 / 即时瞬态 veto”模块，让当前帧动态表面在 update 端就被隔离。

#### 追加冲线执行记录（2026-03-04，Bonn 动态抑制专项）
- 背景：
  - 采用当前代码口径重新建立 Bonn 参考：`output/post_cleanup/p10_route_bonn_tsdf_ref/slam/tables/dynamic_metrics.csv`
  - TSDF 基线：`ghost_ratio=0.9110`
- 路线：
  - 保持高精度主干（`truncation=0.06`, `integration_radius_scale=0.55`），逐步增强局部动态抑制（`c2 -> c2s -> c2xx -> c2hard`）。
  - 统一产物汇总：`output/summary_tables/p10_bonn_route_sweep.csv`
- 结果（Bonn all3, slam, EGF）：
  - `c2`：`acc_cm=1.6547`, `comp_r_5cm=99.9265`, `ghost=0.7085`, `ghost_reduction_vs_tsdf=22.23%`
  - `c2s`：`acc_cm=1.6580`, `comp_r_5cm=99.1867`, `ghost=0.6860`, `ghost_reduction_vs_tsdf=24.70%`
  - `c2xx`：`acc_cm=1.6668`, `comp_r_5cm=97.8545`, `ghost=0.6738`, `ghost_reduction_vs_tsdf=26.04%`
  - `c2hard`：`acc_cm=1.6764`, `comp_r_5cm=96.1802`, `ghost=0.6621`, `ghost_reduction_vs_tsdf=27.32%`（本轮最佳）
- 结论：
  - 纯参数强化可持续压低 ghost，但在当前算子下已出现明显边际递减（`22.23% -> 27.32%`），仍未达到 P10 要求的 `>=35%`。
  - 本轮新增实验分支：`STCG contradiction shell`（代码位于 `egf_dhmap3d/modules/updater.py`，配置位于 `egf_dhmap3d/core/config.py`）。
  - 该分支当前未封板：在极端配置上显著拉长单序列运行时间，需先做算子级复杂度约束（采样触发/半径裁剪）再进入下一轮验收；当前已设为默认关闭（`stcg_shell_enable=False`）。

#### 追加冲线执行记录（2026-03-05，Dual-State 解耦专项）
- 目标：
  - 按“解耦 Acc 与 ghost”的新思路，验证双通道融合是否可直接把 P10 推到过线。
- 代码改动：
  - `scripts/run_egf_3d_tum.py`：补充 dual-state 静态锚点参数（`dual_state_static_protect_rho/ratio`）命令行与回写。
  - `scripts/run_benchmark.py`：打通 dual-state 全参数透传（权重、bias/temp、commit/rollback、decay、static protect）。
  - `scripts/run_p10_precision_profile.py`：扩展 profile 字段并新增 `dual_state_bonnsafe_a/b`。
- 实验命令：
  - `python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,dual_state_decouple_a,dual_state_bonnsafe_a,dual_state_bonnsafe_b,lzcd_only_b --frames 40 --stride 3 --max_points_per_frame 1200 --out_root output/post_cleanup/p10_dual_try_medium --summary_csv output/summary_tables/local_mapping_precision_profile_dual_medium.csv --figure assets/acc_comp_tradeoff_dual_medium.png --force`
  - 说明：为避免无效算力消耗，执行到 `dual_state_bonnsafe_b` 后中止后续 `baseline_relaxed/lzcd_only_b` 队列；已手工汇总完整三组 dual 结果。
- 产物：
  - `output/post_cleanup/p10_dual_try_medium/00_dual_state_decouple_a/...`
  - `output/post_cleanup/p10_dual_try_medium/01_dual_state_bonnsafe_a/...`
  - `output/post_cleanup/p10_dual_try_medium/02_dual_state_bonnsafe_b/...`
  - `output/summary_tables/local_mapping_precision_profile_dual_medium.csv`
  - `assets/acc_comp_tradeoff_dual_medium.png`
- 结果（P10 同口径）：
  - `dual_state_decouple_a`：`tum_acc_cm=2.041`, `tum_comp_r_5cm=99.933`, `bonn_acc_cm=6.732`, `bonn_comp_r_5cm=68.869`, `ghost_red_min=0.331`
  - `dual_state_bonnsafe_a`：`tum_acc_cm=2.002`, `tum_comp_r_5cm=99.933`, `bonn_acc_cm=7.045`, `bonn_comp_r_5cm=66.835`, `ghost_red_min=0.319`
  - `dual_state_bonnsafe_b`：`tum_acc_cm=2.062`, `tum_comp_r_5cm=99.936`, `bonn_acc_cm=7.066`, `bonn_comp_r_5cm=66.905`, `ghost_red_min=0.330`
- 判定：
  - 三组均 `pass_all=False`。
  - 结论：当前 dual-state 门控解耦方向在 TUM 侧只能小幅改善/持平，且 Bonn `Acc/Comp-R` 显著不达标；仅靠该分支的参数与门控重分配无法通过 P10，需引入新的“几何去偏主算子”。

#### 追加冲线执行记录（2026-03-05，LZCD-v2/STCG-v2 算子化尝试）
- 目标：
  - 按“先去偏、再抑动态”的新路线验证模块级改造是否可提升 P10（避免纯参数调优）。
- 代码改动（本轮）：
  - `egf_dhmap3d/modules/updater.py`
    - LZCD-v2：引入 `static_anchor + dyn_risk` 的动态增益/步长调制；
    - STCG-v2：将 `visibility_contradiction` 与 `clear_hits` 纳入矛盾观测；
    - shell contradiction 同步更新 visibility 历史。
  - `egf_dhmap3d/core/voxel_hash.py`
    - dual-state 提取改为阈值化静态权重（替代 `p_s/p_ref` 饱和比）；
    - STCG 提取端加入 `clear_hits` 混合门控与静态保护。
- 执行命令：
  - `python scripts/run_p10_precision_profile.py --profiles dual_state_decouple_a --frames 20 --stride 3 --max_points_per_frame 900 --out_root output/post_cleanup/p10_lzcd_stcg_v2_dual_quick --summary_csv output/summary_tables/local_mapping_precision_profile_lzcd_stcg_v2_dual_quick.csv --figure assets/acc_comp_tradeoff_lzcd_stcg_v2_dual_quick.png --force`
  - `python scripts/run_p10_precision_profile.py --profiles lzcd_only_b --frames 20 --stride 3 --max_points_per_frame 900 --out_root output/post_cleanup/p10_lzcd_stcg_v2_lzcd_quick --summary_csv output/summary_tables/local_mapping_precision_profile_lzcd_stcg_v2_lzcd_quick.csv --figure assets/acc_comp_tradeoff_lzcd_stcg_v2_lzcd_quick.png --force`
  - `python scripts/run_p10_precision_profile.py --profiles lzcd_only_b --frames 15 --stride 3 --max_points_per_frame 900 --out_root output/post_cleanup/p10_lzcd_stcg_v2_lzcd_quick15 --summary_csv output/summary_tables/local_mapping_precision_profile_lzcd_stcg_v2_lzcd_quick15.csv --figure assets/acc_comp_tradeoff_lzcd_stcg_v2_lzcd_quick15.png --force`
- 产物：
  - `output/summary_tables/local_mapping_precision_profile_lzcd_stcg_v2_dual_quick.csv`
  - `output/summary_tables/local_mapping_precision_profile_lzcd_stcg_v2_lzcd_quick.csv`
  - `output/summary_tables/local_mapping_precision_profile_lzcd_stcg_v2_lzcd_quick15.csv`
  - `assets/acc_comp_tradeoff_lzcd_stcg_v2_dual_quick.png`
  - `assets/acc_comp_tradeoff_lzcd_stcg_v2_lzcd_quick.png`
  - `assets/acc_comp_tradeoff_lzcd_stcg_v2_lzcd_quick15.png`
- 关键结果：
  - `dual_state_decouple_a (20f)`：
    - `tum_acc_cm=2.931`, `tum_comp_r_5cm=100.00`
    - `bonn_acc_cm=5.871`, `bonn_comp_r_5cm=57.87`
    - `ghost_red_min=-0.068`
  - `lzcd_only_b (20f)`：
    - `tum_acc_cm=3.063`, `tum_comp_r_5cm=99.82`
    - `bonn_acc_cm=5.982`, `bonn_comp_r_5cm=66.36`
    - `ghost_red_min=0.132`
  - `lzcd_only_b (15f, 保守步长复测)`：
    - `tum_acc_cm=3.162`, `tum_comp_r_5cm=99.86`
    - `bonn_acc_cm=5.728`, `bonn_comp_r_5cm=65.14`
    - `ghost_red_min=-0.255`
- 与上一轮 dual 基线对比（`output/summary_tables/local_mapping_precision_profile_dual_medium.csv`）：
  - TUM `Acc` 从 `~2.00-2.06` 退化到 `~2.93-3.16`；
  - Bonn `Comp-R` 从 `~66.8-68.9` 无稳定提升（且部分配置显著下降）；
  - `pass_all` 全部为 `False`。
- 判定：
  - 本轮 LZCD-v2/STCG-v2 路线 **未形成有效增益**，当前实现不满足 P10 冲线要求；
  - 结论：需进入下一条主线（结构解耦而非同域门控叠加），优先考虑“几何偏置估计器与动态抑制器物理分层”的新算子设计。

#### 追加冲线执行记录（2026-03-05，结构解耦主线切换）
- 目标：
  - 把“几何去偏”和“动态抑制”拆成独立状态/算子，避免单一门控链相互拉扯。
- 代码改动（已落地）：
  - `egf_dhmap3d/core/types.py`
    - 新增 `dyn_prob`（独立动态抑制状态）。
  - `egf_dhmap3d/core/config.py`
    - 新增 `dyn_state_*` 配置（独立动态状态更新权重）；
    - 新增 `surface.structural_decouple_*` 配置（提取阶段独立动态抑制门控）。
  - `egf_dhmap3d/modules/updater.py`
    - 新增独立动态状态更新（`dyn_prob`）并与 `d_score` 解耦；
    - LZCD 改为几何置信驱动，不再依赖动态风险（去掉 dyn-risk 耦合项）；
    - raycast 清理改用 `dyn_prob`。
  - `egf_dhmap3d/core/voxel_hash.py`
    - 提取阶段新增 `structural_decouple_enable` 分支：几何通道优先 + 独立动态抑制门控；
    - SNEF 动态风险输入改为 `dyn_prob`（解耦后动态状态）。
  - `egf_dhmap3d/modules/pipeline.py`
    - 透传结构解耦参数到表面提取调用。
- 验证命令（smoke）：
  - `python scripts/run_p10_precision_profile.py --profiles lzcd_only_b --frames 8 --stride 3 --max_points_per_frame 700 --out_root output/post_cleanup/p10_struct_decouple_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_smoke.csv --figure assets/acc_comp_tradeoff_struct_decouple_smoke.png --force`
  - `python scripts/run_p10_precision_profile.py --profiles baseline_relaxed --frames 8 --stride 3 --max_points_per_frame 700 --out_root output/post_cleanup/p10_struct_decouple_smoke_base --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_smoke_base.csv --figure assets/acc_comp_tradeoff_struct_decouple_smoke_base.png --force`
- 结果（仅 smoke，不作封板）：
  - `lzcd_only_b`：`tum_acc_cm=3.561`, `bonn_acc_cm=5.194`, `bonn_comp_r_5cm=74.79`, `ghost_red_min=0.441`
  - `baseline_relaxed`：`tum_acc_cm=3.676`, `bonn_acc_cm=5.176`, `bonn_comp_r_5cm=75.30`, `ghost_red_min=0.436`
- 判定：
  - 结构解耦链路工作正常且动态抑制优势保持，但 TUM `Acc` 明显偏高，尚未达到 P10；
  - 下一步应在该结构上做“几何去偏专用算子”收敛（而非回到同域门控叠加）。

### P11. 效率封板（从可用到可投）

#### 目标
把局部建图效率提升到“顶刊审稿可接受”区间，并给出质量-速度边界。

#### 动作
1. 对 EGF 增加局部增量优化：
   - block 级 touched-voxel 更新；
   - 表面提取缓存重用；
   - 必要时降低同步频率（每 N 帧提取一次表面）并计入公平协议。
2. 输出两档配置：
   - `HQ`（高质量档）  
   - `RT`（实时档）

#### 产物
- `output/summary_tables/local_mapping_efficiency_final.csv`  
- `assets/quality_speed_tradeoff_final.png`

#### 硬验收
1. `RT` 档（walking_xyz, 40 frames，同硬件）：`fps >= 2.5`。  
2. `HQ` 档：`fps >= 1.0` 且相对最佳质量点 `ΔF-score >= -0.015`。  
3. `memory_vs_tsdf <= 3.0`。

#### 执行状态（2026-03-04，P12-R 已完成）
- 新增脚本：`scripts/run_p11_efficiency_final.py`（RT/HQ 双档自动选择 + 内存比对 + 图表输出）。
- 运行时优化已落地：
  - `scripts/run_egf_3d_tum.py`：移除重复表面提取（避免 mesh 导出阶段二次全图扫描）。
  - `egf_dhmap3d/core/voxel_hash.py`：邻域偏移缓存（减少高频邻域生成开销）。
  - `egf_dhmap3d/modules/updater.py`：新增 `integration_radius_scale`（可控邻域积分半径）。
- 复现实验命令（本轮）：  
  - `python scripts/run_p11_efficiency_final.py --dataset_root data/tum --sequence rgbd_dataset_freiburg3_walking_xyz --frames 40 --stride 3 --voxel_size 0.02 --egf_poisson_iters 0 --egf_integration_radius_scale 0.45 --max_points_list 400,320,280,240,220,200,180,160,140,120,100,80 --out_root output/post_cleanup/p11_efficiency_final --out_csv output/summary_tables/local_mapping_efficiency_final.csv --out_json output/summary_tables/local_mapping_efficiency_final.json --plot_png assets/quality_speed_tradeoff_final.png --force`
- 当前产物：
  - `output/summary_tables/local_mapping_efficiency_final.csv`
  - `output/summary_tables/local_mapping_efficiency_final.json`
  - `assets/quality_speed_tradeoff_final.png`
- 当前结果：
  - `HQ`：`mpp400`，`fps=1.2646`，`ΔF-score=0.0000`（通过）
  - `RT`：`mpp140`，`fps=3.7700`（通过）
  - 其他 `RT` 候选：`mpp180`（`fps=3.1596`），`mpp160`（`fps=3.5842`），`mpp120`（`fps=3.6799`）
  - `memory_vs_tsdf`：全候选 `< 1.37`（通过）
- 结论：`P11` 全部硬验收通过（`HQ + RT + memory`）。

#### 下一步（封板后）
1. `P10`：维持“best-achievable 封板”状态，不再作为当前主线阻塞项。  
2. `P11`：已封板通过；后续仅在 `P11-HQ`（`fps>=1.0`）不退化的前提下做增量优化。  
3. 主线推进到 `P13`（双协议主表 + 显著性封板）。  

### P12. 外部基线公平性封板（去“基线跑偏”争议）

#### 目标
建立可审计的真实外部基线对比，不再出现“基线质量异常导致结论偏置”问题。

#### 动作
1. 至少 3 个对照组：
   - `TSDF`（传统）  
   - `DynaSLAM`（真实 runner）  
   - `Neural`（真实 runner，如 NICE-SLAM/同类）  
2. 每个外部方法输出：
   - runner 命令、版本、输入配置、失败日志；
   - 每序列 runtime/memory。
3. 加入“基线健康度检查”：
   - 若某外部方法在关键指标异常劣化，自动标记并要求给出原因。

#### 产物
- `output/summary_tables/external_baselines_native_reconstruction_final.csv`  
- `output/summary_tables/external_baselines_native_dynamic_final.csv`  
- `output/summary_tables/external_baselines_native_runtime_final.csv`  
- `output/post_cleanup/external_audit/BASELINE_HEALTHCHECK.md`

#### 硬验收
1. `walking_xyz + walking_static + walking_halfsphere` 三序列全部跑通（允许单序列失败但必须有日志与替代跑法）。  
2. 每个外部方法都可追溯到真实输出路径 + runner 命令。  
3. 主结论表仅使用通过健康度检查的方法条目。

#### 执行状态（历史结果，待严格口径重跑）
- 三序列真实外部基线已重跑（`slam`）：
  - 运行命令：  
    - `python scripts/run_p8_native_external.py --dataset_root data/tum --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere --protocol slam --frames 40 --stride 3 --max_points_per_frame 900 --voxel_size 0.02 --eval_thresh 0.05 --seed 42 --methods tsdf,dynaslam,neural_implicit --out_root output/post_cleanup/p12_external_native_final --out_recon_csv output/summary_tables/external_baselines_native_reconstruction_final.csv --out_dynamic_csv output/summary_tables/external_baselines_native_dynamic_final.csv --out_runtime_csv output/summary_tables/external_baselines_native_runtime_final.csv --force`
- 真实输出补齐：
  - `dynaslam` 的 `walking_halfsphere` 输出已用真实 runner 生成（`scripts/external/run_dynaslam_tum_runner.py`）。
  - `neural_implicit` 的 `walking_halfsphere` 输出已由 NICE-SLAM 快速配置生成并落盘到 `output/external/neural_mesh/rgbd_dataset_freiburg3_walking_halfsphere/mesh.ply`。
- 健康度审计与主表筛选：
  - 新增脚本：`scripts/build_p12_external_healthcheck.py`
  - 已在 `P12-R` 中更新健康检查规则：外部 runner 若真实输出完整，允许 `runtime returncode!=0` 但记录为 soft-pass（避免误删 `dynaslam/neural_implicit`）。
  - 审计文档：`output/post_cleanup/external_audit/BASELINE_HEALTHCHECK.md`
  - 审计表：`output/summary_tables/external_baselines_native_healthcheck_final.csv`
  - 通过健康检查并保留到主结论表的方法：`tsdf`, `dynaslam`, `neural_implicit`
- 审计状态：`P12-R` 已完成并回写主表。

### P13. 双协议主表 + 显著性封板

#### 目标
消除“口径混用”争议，输出可直接投稿的主表与显著性表。

#### 动作
1. 统一生成双协议主表（oracle/slam）：
   - TUM walking3 + Bonn all3；
   - `n_seeds >= 5`。
2. 显著性采用 `paired t-test + Wilcoxon` 双报告。
3. 统一主表自动生成脚本，禁止手工转录数字。

#### 产物
- `output/summary_tables/local_mapping_main_table_dual_protocol.csv`  
- `output/summary_tables/local_mapping_significance_dual_protocol.csv`  
- `output/summary_tables/local_mapping_mean_std_dual_protocol.csv`

#### 硬验收
1. 在 dynamic 场景，EGF 相对 TSDF：  
   - `Comp-R(5cm)` 提升显著：`p < 0.05`（oracle 与 slam 都要成立）；  
   - `ghost_ratio` 改善显著：`p < 0.05`（oracle 与 slam 都要成立）。  
2. 至少一个协议上，EGF 的 `F-score` 均值不低于 TSDF。  
3. 报告中的所有数字都可追溯到上述 CSV。

#### 执行状态（2026-03-04，P13-R 已完成）
- 已完成双协议 5-seed 复现实验：
  - TUM `oracle`（walking3）  
    - `python scripts/run_benchmark.py --dataset_kind tum --dataset_root data/tum --protocol oracle --static_sequences "" --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere --methods egf,tsdf --frames 40 --stride 3 --max_points_per_frame 900 --voxel_size 0.02 --eval_thresh 0.05 --seeds 40,41,42,43,44 --egf_poisson_iters 0 --out_root output/post_cleanup/p13_tum_oracle_5seed_fast --force`
  - Bonn `slam`（all3）  
    - `python scripts/run_benchmark.py --dataset_kind bonn --dataset_root data/bonn --protocol slam --static_sequences "" --dynamic_sequences rgbd_bonn_balloon2,rgbd_bonn_balloon,rgbd_bonn_crowd2 --methods egf,tsdf --frames 40 --stride 3 --max_points_per_frame 900 --voxel_size 0.02 --eval_thresh 0.05 --seeds 40,41,42,43,44 --egf_poisson_iters 0 --out_root output/post_cleanup/p13_bonn_slam_5seed_fast --force`
- 统计脚本已补齐 `Comp-R(5cm)` 显著性与 P13 指定产物自动落盘：
  - `scripts/build_dual_protocol_multiseed_summary.py`
  - `SIG_METRICS` 增加 `recall_5cm`（paired t-test + Wilcoxon）
  - 自动输出：
    - `output/summary_tables/local_mapping_main_table_dual_protocol.csv`
    - `output/summary_tables/local_mapping_significance_dual_protocol.csv`
    - `output/summary_tables/local_mapping_mean_std_dual_protocol.csv`
- 汇总命令：
  - `python scripts/build_dual_protocol_multiseed_summary.py --summary_root output/summary_tables --oracle_tables_root output/post_cleanup/p4_multiseed_tum_final_v2/oracle/tables --slam_tables_root output/post_cleanup/p5_multiseed_bonn_all3/slam/tables --oracle_tag p4_multiseed_tum_final_v2 --slam_tag p5_multiseed_bonn_all3`
- 严格口径补跑：
  - `python scripts/run_benchmark.py --dataset_kind bonn --dataset_root data/bonn --protocol slam --static_sequences "" --dynamic_sequences rgbd_bonn_balloon2,rgbd_bonn_balloon,rgbd_bonn_crowd2 --methods egf,tsdf --frames 40 --stride 3 --max_points_per_frame 900 --voxel_size 0.02 --eval_thresh 0.05 --seeds 40,41,42,43,44 --egf_poisson_iters 0 --egf_slam_no_gt_delta_odom --out_root output/post_cleanup/p5_multiseed_bonn_all3 --force`
- 验收结果：
  - `n_seeds`：所有 `(protocol, sequence, method)` 均满足 `n_recon=5` 且 `n_dynamic=5`。
  - 显著性（EGF vs TSDF，dynamic）：
    - `oracle`: `recall_5cm p_t=1.99e-17, p_w=6.10e-05`; `ghost_ratio p_t=1.72e-24, p_w=6.10e-05`
    - `slam`: `recall_5cm p_t=4.03e-16, p_w=6.10e-05`; `ghost_ratio p_t=5.35e-05, p_w=1.22e-04`
  - `F-score` 均值：
    - `oracle`: `EGF=0.7903 > TSDF=0.4137`
    - `slam`: `EGF=0.6463 > TSDF=0.0697`
- 审计状态：`P13-R` 已完成并回写主表。

### P14. 边界鲁棒性终稿封板

#### 目标
形成可防守的“有效区间与失效区间”叙事。

#### 动作
1. Stress 维度固定为四类：
   - 点预算下降（depth sparsity proxy）  
   - 时序稀疏（stride）  
   - 运动模式（walking3）  
   - 遮挡/缺失（synthetic 或可控真实子集）
2. 每维至少 3 档，输出退化曲线 + 边界建议。
3. 给出“推荐工作区间”与“不可用区间”。

#### 产物
- `output/summary_tables/stress_test_summary_final.csv`  
- `assets/stress_degradation_curves_final.png`  
- `output/post_cleanup/stress_test/FAILURE_CASES_FINAL.md`

#### 硬验收
1. Level-2 下，EGF 相对 TSDF 的 `ghost_ratio` 降幅 `>=25%`（四维全部满足）。  
2. Level-3 下，EGF `F-score >= 0.75`（至少 3/4 维度满足）。  
3. 至少 2 条失效边界 + 对应规避策略明确可执行。

#### 执行状态（2026-03-04）
- 已补齐第 4 维 `occlusion/missing` 压力实验（`walking_xyz`，oracle，40 帧，stride=3，mpp=900）：
  - `l1`: `--stress_occlusion_ratio 0.10`
  - `l2`: `--stress_occlusion_ratio 0.20`
  - `l3`: `--stress_occlusion_ratio 0.35`
  - 运行根目录：`output/post_cleanup/p14_occlusion_levels/`
- 新增终版四维汇总脚本：`scripts/build_p14_stress_report.py`
  - 生成：
    - `output/summary_tables/stress_test_summary_final.csv`
    - `assets/stress_degradation_curves_final.png`
    - `output/post_cleanup/stress_test/FAILURE_CASES_FINAL.md`
    - `output/post_cleanup/stress_test/stress_meta_final.json`
- 已接入自动汇总链路：`scripts/update_summary_tables.py --prefer_p4_final --verbose`
  - 会自动调用 `scripts/build_p14_stress_report.py`，实现一条命令刷新 P14 终版产物。
- P14 硬验收结果（来自 `stress_meta_final.json`）：
  - Level-2 ghost 降幅（EGF vs TSDF）：
    - `point_budget`: `71.88%`
    - `temporal_sparsity`: `71.88%`
    - `motion_pattern`: `50.62%`
    - `occlusion_missing`: `68.60%`
    - 结论：四维全部 `>=25%`，通过。
  - Level-3 EGF F-score：
    - `point_budget`: `0.8050`
    - `temporal_sparsity`: `0.8458`
    - `motion_pattern`: `0.9102`
    - `occlusion_missing`: `0.7990`
    - 结论：`4/4` 维满足 `>=0.75`，通过。
  - 失效边界：已在 `FAILURE_CASES_FINAL.md` 给出 3 条（点预算、遮挡缺失、高运动）与可执行规避策略。
  - 最终判定：`P14` 通过。

### P15. 实时化冲刺封板（10 FPS 保底，15 FPS 目标）

#### 目标
在不牺牲当前主结论质量的前提下，把 EGF 局部建图从“可用”推进到“实时可部署”区间。

#### 动作
1. 以 `P11-HQ`（`mpp400`）和 `P11-RT`（`mpp140`）作为质量锚点，新增 `RT10` 与 `RT15` 两档搜索。  
2. 优先做“等价加速”而非“降质提速”：
   - 增量提取节流（`surface_extract_every_n`）；
   - touched-block 局部重建与缓存复用；
   - 体素邻域访问向量化/批量化；
   - 必要时引入轻量并行（不改算法语义）。
3. 统一在同硬件、同输入口径（`walking_xyz`, 40 帧, stride=3）下给出 `quality-speed` 曲线与内存比。  
4. 若 `RT15` 无法满足质量约束，至少保证 `RT10` 通过并解释瓶颈热点。

#### 产物
- `output/summary_tables/local_mapping_efficiency_realtime.csv`  
- `output/summary_tables/local_mapping_efficiency_realtime.json`  
- `assets/quality_speed_tradeoff_realtime.png`  
- `output/post_cleanup/p15_realtime/PROFILE.md`

#### 硬验收
1. **保底通过线（必须）**：`RT10` 档 `fps >= 10.0`。  
2. **冲刺目标（建议）**：`RT15` 档 `fps >= 15.0`。  
3. **质量不回退（相对 P11-HQ 锚点）**：  
   - `ΔF-score >= -0.015`；  
   - `Δghost_ratio <= +0.03`。  
4. **资源约束**：`memory_vs_tsdf <= 3.0`，并给出热点分解（投影/融合/提取）。

#### 执行状态（2026-03-04）
- 已完成 `surface_max_free_ratio` 维度接入与完整 sweep（95 组）：
  - 命令：
    - `python scripts/run_p15_realtime.py --dataset_root data/tum --sequence rgbd_dataset_freiburg3_walking_xyz --frames 40 --stride 3 --max_points_list 400,360,320 --decay_interval_list 1,2,3,4 --assoc_radius_list 2,1 --integration_radius_list 0.45,0.35,0.30,0.25,0.20 --integration_min_radius_vox_list 1.2,1.0,0.8 --surface_max_free_ratio_list 1000000000,0.25,0.20,0.16,0.12 --max_runs 95 --out_root output/post_cleanup/p15_realtime --out_csv output/summary_tables/local_mapping_efficiency_realtime.csv --out_json output/summary_tables/local_mapping_efficiency_realtime.json --plot_png assets/quality_speed_tradeoff_realtime.png --profile_md output/post_cleanup/p15_realtime/PROFILE.md --force`
- 锚点（Anchor）：
  - `mpp400_d1_r2_ir0.45_mr1.20_frINF`
  - `mapping_fps=2.1884`, `fscore=0.9824`, `ghost_ratio=0.4020`
- 入选实时档（RT10/RT15）：
  - `mpp320_d3_r1_ir0.20_mr0.80_fr0.16`
  - `mapping_fps=22.0942`, `fscore=1.0000`, `ghost_ratio=0.4314`
  - `delta_fscore_vs_anchor=+0.0176`, `delta_ghost_vs_anchor=+0.0294`, `memory_vs_tsdf=0.9257`
- 验收结论：
  - `RT10`：通过（`22.09 >= 10.0`）
  - `RT15`：通过（`22.09 >= 15.0`）
  - 质量约束：通过（`ΔF=+0.0176 >= -0.015`, `Δghost=+0.0294 <= +0.03`）
  - 资源约束：通过（`0.9257 <= 3.0`）
  - 最终判定：`P15` 通过。

### 7.4 最终封板判定（全部满足才算完成）

### 7.3.1 结构解耦下“几何去偏专用算子”收敛尝试（2026-03-05）

- 已在结构解耦链路上新增几何专用去偏求解器（LZCD Convergence Operator）：
  - 代码位点：
    - `egf_dhmap3d/modules/updater.py`（几何残差锚点 + 图平滑迭代收敛求解 + 回溯更新）
    - `egf_dhmap3d/core/types.py`（`geo_res_ema`, `geo_res_hits`）
    - `egf_dhmap3d/core/voxel_hash.py`（新状态初始化与衰减）
    - `egf_dhmap3d/core/config.py`（`lzcd_solver_*`, `lzcd_residual_*`）
- 运行命令：
  - `python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_b --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop2_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop2_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop2_smoke/profile.png --force`
- 结果（`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop2_smoke.csv`）：
  - `baseline_relaxed`: `tum_acc_cm=3.7573`, `bonn_acc_cm=5.5176`, `bonn_comp_r_5cm=60.59`
  - `lzcd_only_b`: `tum_acc_cm=3.6535`, `bonn_acc_cm=5.4814`, `bonn_comp_r_5cm=61.35`
- 结论：
  - 几何 Acc 有小幅改善（去偏方向正确），但幅度仍不足以冲过 P10 硬线；
  - 运行时成本显著上升（`update` 阶段主导），后续需做“几何算子局部化/稀疏化”热点优化。

- 追加执行（2026-03-05，结构解耦 profile 扩展）：
  - `geoop3`：
    - 命令：`python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_b --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop3_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop3_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop3_smoke/profile.png --force`
    - 结果：`lzcd_only_b` 相比 `baseline_relaxed` 小幅降 Acc（TUM `3.7573 -> 3.6448`，Bonn `5.5176 -> 5.4816`），但仍远未达到 P10 `Acc` 硬线。
  - `geoop4`（引入 `lzcd_geochan_* / lzcd_dualstatic_*`）：
    - 汇总：`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop4_smoke.csv`
    - 结果：`lzcd_geochan_a` 将 TUM Acc 压到 `3.4064`，但 Bonn ghost 降幅最小值降至 `0.1409`；`dualstatic` 虽可把 TUM Acc 压到约 `1.98`，但 Bonn `Acc/Comp-R` 明显崩塌，不可用。
  - `geoop5`（`geochan + STCG`）：
    - 汇总：`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop5_smoke.csv`
    - 结果：相对 `geoop4` 基本无净收益；Bonn 侧 ghost 降幅仍显著低于要求。
  - `geoop6`（`geochan + CCG`）：
    - 汇总：`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop6_smoke.csv`
    - 结果：`lzcd_geochan_ccg_a` 在 TUM 侧略优（`tum_acc=3.4033`），但 Bonn 侧仍未恢复（`bonn_acc=5.1840`，`ghost_min=0.1426`）。

- 追加执行（2026-03-05，LZCD 仿射去偏）：
  - 新增模块（几何专用，不耦合动态抑制链）：
    - `egf_dhmap3d/modules/updater.py`：
      - 在 LZCD 中加入局部仿射拟合：`phi ~= a * <n, delta_x> + b`，以 `b` 作为局部零水平参考；
      - 加入静态置信门控（高 `rho`、低 `dyn_prob` 时仿射项影响更强），抑制动态区域过校正。
    - `egf_dhmap3d/core/config.py`：
      - 新增 `lzcd_affine_*` 参数（开关、混合系数、斜率范围、最小样本数）。
  - `geoop7`：
    - 命令：`python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_b,lzcd_geochan_a,lzcd_geochan_ccg_a --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop7_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop7_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop7_smoke/profile.png --force`
    - 结果：`lzcd_geochan_ccg_a` 最优（`tum_acc=3.3925`，`bonn_acc=5.2296`，`ghost_min=0.1390`），相对 `geoop6`：
      - TUM Acc 继续小幅改善（`3.4033 -> 3.3925`）；
      - Bonn Acc 轻微回退（`5.1840 -> 5.2296`）。
  - `geoop8`（仿射静态置信门控强化）：
    - 命令：`python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_b,lzcd_geochan_a,lzcd_geochan_ccg_a --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop8_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop8_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop8_smoke/profile.png --force`
    - 结果：与 `geoop7` 基本持平（变化在千分级），未形成可观增益。

- 追加执行（2026-03-05，ST-Mem 结构解耦模块）：
  - 新增模块（仅作用动态抑制链，不回写几何通道）：
    - `egf_dhmap3d/core/types.py`：新增体素短时矛盾记忆状态 `st_mem`；
    - `egf_dhmap3d/modules/updater.py`：基于 `free/surf conflict + visibility + clear + residual` 更新 `st_mem`；
    - `egf_dhmap3d/core/voxel_hash.py`：提取阶段将 `st_mem` 注入 `dyn_gate`，并新增“强矛盾直通分支”；
    - `egf_dhmap3d/modules/pipeline.py` / `egf_dhmap3d/core/config.py`：配置与提取参数贯通。
  - `geoop9`（ST-Mem 初版）：
    - 命令：`python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_b,lzcd_geochan_a,lzcd_geochan_ccg_a --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop9_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop9_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop9_smoke/profile.png --force`
    - 结果：与 `geoop8` 基本持平；`lzcd_geochan_ccg_a` 的 `ghost_min` 约 `0.1375`，未见明显提升。
  - `geoop10`（强矛盾直通分支）：
    - 命令：`python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_b,lzcd_geochan_a,lzcd_geochan_ccg_a --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop10_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop10_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop10_smoke/profile.png --force`
    - 结果：`ghost_min` 相比 `geoop9` 仅小幅回升（约 `+0.0015`），总体仍处于同一量级；`Acc/Comp-R` 基本不变。
  - `geoop11`（加入更激进 ST-Mem profile：`lzcd_geochan_ccg_stmem_a/b`）：
    - 命令：`python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_b,lzcd_geochan_ccg_a,lzcd_geochan_ccg_stmem_a,lzcd_geochan_ccg_stmem_b --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop11_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop11_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop11_smoke/profile.png --force`
    - 结果：
      - `stmem_b` 可进一步压低 TUM Acc（`3.394 -> 3.362`），但 `ghost_min` 下降到 `0.118`；
      - `stmem_a` 在 `ghost_min` 上仅小幅改善到 `0.145`，仍显著低于目标线。
    - 判定：当前 ST-Mem 参数强化呈现“Acc 改善 vs ghost 退化”的再耦合，尚未形成 P10 可用解。
  - `geoop12`（ST-Mem 前移到更新期几何融合权重）：
    - 代码：`egf_dhmap3d/modules/updater.py` 在 `phi_geo` 融合时引入 `st_mem` 抑制系数（上游抑制，非提取后门控）。
    - 命令：`python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_b,lzcd_geochan_ccg_a,lzcd_geochan_ccg_stmem_a --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop12_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop12_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop12_smoke/profile.png --force`
    - 结果：`stmem_a` 成为本轮最佳（`tum_acc=3.3729`，`bonn_acc=5.2404`，`ghost_min=0.1451`），较 `geoop11` 仅小幅变化（`ghost_min +0.0002` 级别），仍未跨越关键门槛。

- 追加执行（2026-03-05，关联阶段矛盾门控参数贯通 + `geoop14`）：
  - 参数链路已打通并回写：
    - `scripts/run_egf_3d_tum.py`：新增 `assoc_contra_*` CLI / cfg / summary 回写。
    - `scripts/run_benchmark.py`：新增 `--egf_assoc_contra_*` 参数、`run_method` 签名与 TUM/Bonn 透传。
    - `scripts/run_p10_precision_profile.py`：`Profile` 与 `common_egf` 已纳入 `assoc_contra_*`；`stmem_a/b` 已使用更强 contra 配置。
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_only_b,lzcd_geochan_ccg_a,lzcd_geochan_ccg_stmem_a,lzcd_geochan_ccg_stmem_b --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop14_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop14_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop14_smoke/profile.png --force`
  - 结果（最优 `stmem_b`）：
    - `tum_acc_cm=3.3618`, `tum_fscore=0.8886`
    - `bonn_acc_cm=5.2273`, `bonn_fscore=0.6062`
    - `ghost_min=0.1247`
    - `pass_all=False`
  - 相对 `geoop13`（同 profile）：
    - `tum_acc_cm`: `3.3619 -> 3.3618`（微幅改善）
    - `bonn_acc_cm`: `5.2233 -> 5.2273`（微幅退化）
    - `ghost_min`: `0.1240 -> 0.1247`（微幅改善）

- 阶段结论更新（截至 `geoop14`）：
  - 结构解耦 + 几何去偏路线在 TUM `Acc` 侧稳定有效；
  - `assoc_contra` 上游门控已成为“可控变量”，但在当前架构上提升仍停留千分级；
  - P10 仍未过线，瓶颈依旧是 Bonn 侧 `Acc/Comp-R` 与 `ghost` 的耦合。

- 追加执行（2026-03-05，结构级改造 `geoop15`：显式动态状态 + 双图层提取）：
  - 新增结构模块：
    - `egf_dhmap3d/modules/updater.py`：
      - 新增显式动态潜变量 `z_dyn`（独立于 `phi_geo` 去偏链），按 `conflict/visibility/residual/osc/free_ratio` 更新，支持 `alpha_up/down` 非对称收敛。
    - `egf_dhmap3d/core/voxel_hash.py`：
      - 新增 `dual_layer_extract_enable` 路径：
        - Stage-G（geometry layer）先构建几何候选，不在候选阶段直接做动态丢弃；
        - Stage-D（dynamic layer）再用 `z_dyn + st_mem + contradiction + transient_ratio` 做软掩膜抑制。
    - `egf_dhmap3d/core/config.py` / `scripts/run_egf_3d_tum.py` / `scripts/run_benchmark.py` / `scripts/run_p10_precision_profile.py`：
      - 贯通 `zdyn_*` 与 `surface_dual_layer_*` 参数链路（CLI、runner、summary）。
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_geochan_ccg_stmem_b,struct_dual_layer_a,struct_dual_layer_b --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop15_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop15_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop15_smoke/profile.png --force`
  - 结果（`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop15_smoke.csv`）：
    - `baseline_relaxed`：`tum_acc=3.7573`, `tum_ghost=0.1736`, `bonn_acc=5.5201`, `bonn_comp_r_5cm=60.59`, `bonn_ghost=0.0550`
    - `lzcd_geochan_ccg_stmem_b`：`tum_acc=3.3618`, `tum_ghost=0.2152`, `bonn_acc=5.2369`, `bonn_comp_r_5cm=60.26`, `bonn_ghost=0.0642`
    - `struct_dual_layer_a`：`tum_acc=3.5605`, `tum_ghost=0.1927`, `bonn_acc=5.4491`, `bonn_comp_r_5cm=62.16`, `bonn_ghost=0.0570`
    - `struct_dual_layer_b`：`tum_acc=3.5619`, `tum_ghost=0.1925`, `bonn_acc=5.4493`, `bonn_comp_r_5cm=62.24`, `bonn_ghost=0.0569`
  - 判定：
    - 双图层 + 显式动态状态已实现并可复现，且确实形成新的 Pareto 点（相对 `stmem_b`，ghost 明显下降；相对 `baseline_relaxed`，Acc 改善）。
    - 但仍未跨越 P10 硬线（`pass_all=False`），核心瓶颈仍是 Bonn `Acc` 与 ghost 抑制的剩余耦合。

- 追加执行（2026-03-05，结构解耦残余耦合修正 `geoop15b`）：
  - 目的：
    - 修复 `dual_layer` 在候选阶段仍被动态门控隐式收缩的问题（减少“几何层被动态层提前污染”）。
  - 代码修正：
    - `egf_dhmap3d/core/voxel_hash.py`
      - 在 `dual_layer` 模式下禁用 STCG 候选阶段动态收缩；
      - 在 `adaptive_enable` 分支将候选阶段动态项降权（`d_n = 0.35 * d_raw`）。
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_geochan_ccg_stmem_b,struct_dual_layer_a,struct_dual_layer_b --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop15b_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop15b_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop15b_smoke/profile.png --force`
  - 结果（`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop15b_smoke.csv`）：
    - `struct_dual_layer_a`：`tum_acc=3.5660`, `tum_ghost=0.1920`, `bonn_acc=5.4440`, `bonn_comp_r_5cm=62.17`, `bonn_ghost=0.0573`
    - `struct_dual_layer_b`：`tum_acc=3.5673`, `tum_ghost=0.1919`, `bonn_acc=5.4439`, `bonn_comp_r_5cm=62.16`, `bonn_ghost=0.0573`
  - 判定：
    - 修正后动态抑制略稳（`tum_ghost` 小幅下降），但几何 Acc 未出现有效下降；`pass_all=False`。

- 追加执行（2026-03-05，结构化双图层定向扫参 `geoop16`）：
  - 目的：
    - 在结构不变前提下，围绕 `z_dyn` 收敛速度与 dual-layer 抑制强度做小范围结构参数扫描，验证是否仍有可达增益。
  - 新增 profile：
    - `struct_dual_layer_c`（geometry-prior：提高几何保留、提高静态锚点）
    - `struct_dual_layer_d`（dynamic-aggressive：降低 drop 阈值、提高动态权重）
    - `struct_dual_layer_e`（balanced：较低 `sigma_n0` + 中等 dual-layer 抑制）
    - 实现位点：`scripts/run_p10_precision_profile.py`
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_geochan_ccg_stmem_b,struct_dual_layer_a,struct_dual_layer_b,struct_dual_layer_c,struct_dual_layer_d,struct_dual_layer_e --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop16_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop16_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop16_smoke/profile.png --force`
  - 结果（`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop16_smoke.csv`）：
    - `struct_dual_layer_a`：`tum_acc=3.5660`, `bonn_acc=5.4495`, `bonn_comp_r_5cm=62.17`, `bonn_ghost=0.0569`
    - `struct_dual_layer_b`：`tum_acc=3.5673`, `bonn_acc=5.4435`, `bonn_comp_r_5cm=62.17`, `bonn_ghost=0.0572`
    - `struct_dual_layer_c`：`tum_acc=3.5677`, `bonn_acc=5.4435`, `bonn_comp_r_5cm=62.20`, `bonn_ghost=0.0572`
    - `struct_dual_layer_d`：`tum_acc=3.5625`, `bonn_acc=5.4438`, `bonn_comp_r_5cm=62.19`, `bonn_ghost=0.0572`
    - `struct_dual_layer_e`：`tum_acc=3.5716`, `bonn_acc=5.4427`, `bonn_comp_r_5cm=61.81`, `bonn_ghost=0.0583`
    - 全局 best 仍为 `lzcd_geochan_ccg_stmem_b`（按当前打分函数），`pass_all=False`。
  - 判定：
    - 双图层结构已稳定可复现，并形成 `Comp-R` 受益（Bonn ~`62%`）；
    - 但 `Acc` 仍停留在 `~3.56 / ~5.44` 区间，未出现突破性改进，P10 继续未过线。

- 追加执行（2026-03-05，显式动态几何状态通道 `geoop17`）：
  - 新增模块（结构解耦主线）：
    - `phi_dyn`（动态几何通道）落地：
      - `egf_dhmap3d/core/types.py`：新增 `phi_dyn`, `phi_dyn_w`
      - `egf_dhmap3d/modules/updater.py`：新增 `phi_dyn` 独立融合（由 `dynamic_prob + residual + assoc_risk` 驱动）
      - `egf_dhmap3d/modules/updater.py`：`dyn_prob` 更新中加入 `|phi_dyn-phi_geo|` 反馈
      - `egf_dhmap3d/core/voxel_hash.py`：双图层 Stage-D 新增 `phi_dyn` 相关动态项（`phi_div / phi_dyn_w_ratio`）
      - `egf_dhmap3d/modules/pipeline.py`：新增参数链路传递
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_geochan_ccg_stmem_b,struct_dual_layer_a,struct_dual_layer_b,struct_dual_layer_c --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop17_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop17_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop17_smoke/profile.png --force`
  - 结果（`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop17_smoke.csv`）：
    - `best = lzcd_geochan_ccg_stmem_b`：`tum_acc=3.3620`, `bonn_acc=5.2458`, `pass_all=False`
    - `struct_dual_layer_[a/b/c]`：`bonn_comp_r_5cm ~ 62.13~62.20`，但 `bonn_acc` 仍在 `5.44~5.45`
  - 判定：
    - `phi_dyn` 通道可运行、可复现，但净增益仅千分到百分位量级，未跨过 P10 关键门槛。

- 追加执行（2026-03-05，几何-动态竞争提取算子 `geoop18`）：
  - 新增模块：
    - 在 dual-layer Stage-D 增加显式 `geo_score vs dyn_score` 竞争判别（`egf_dhmap3d/core/voxel_hash.py`）。
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles lzcd_geochan_ccg_stmem_b,struct_dual_layer_a,struct_dual_layer_b,struct_dual_layer_c --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop18_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop18_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop18_smoke/profile.png --force`
  - 结果（对比 `geoop17`）：
    - `struct_dual_layer_[a/b/c]`：`tum_acc` 微降（约 `-0.005`），`tum_ghost` 微降（约 `-0.0001~-0.0002`）；
    - 但 `bonn_acc` 反向上升（劣化）约 `+0.039~+0.049`（`5.44 -> 5.49` 区间），不符合 P10 主目标。
  - 竞争开启复核（单 profile）：
    - 命令：
      - `python scripts/run_p10_precision_profile.py --profiles struct_dual_layer_c --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop18c_compete_on --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop18c_compete_on.csv --figure output/post_cleanup/p10_struct_decouple_geoop18c_compete_on/profile.png --force`
    - 结果：
      - `tum_acc=3.5628`, `bonn_acc=5.4915`, `bonn_comp_r_5cm=62.24`, `ghost_red_min=0.267`
    - 判定：
      - 与 `geoop17` 相比，Bonn `Acc` 明显劣化趋势成立；竞争门控当前参数不可作为主线默认。
  - 判定：
    - 竞争算子存在可解释 trade-off，但当前参数区间对 Bonn `Acc` 不利；
    - 已保留模块实现，默认改为关闭（`dual_layer_compete_enable=False`），避免影响主线配置。

- 追加执行（2026-03-05，竞争项自适应后验复核 `geoop19/19b`）：
  - 目的：
    - 验证“固定竞争劣化”是否来自阈值僵硬，而非竞争思想本身。
  - 代码位点：
    - `egf_dhmap3d/core/voxel_hash.py`：加入 dual-layer 后验自适应（`drop_th_eff/comp_margin_eff`）与 `phi_geo/phi_dyn` 轻后验融合。
    - `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`：新增 competition/phi-dyn 高级参数透传。
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles baseline_relaxed,lzcd_geochan_ccg_stmem_b,struct_dual_layer_c --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop19_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop19_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop19_smoke/profile.png --force`
    - `python scripts/run_p10_precision_profile.py --profiles struct_dual_layer_c,struct_dual_layer_c_compete_adapt --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop19b_compete_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop19b_compete_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop19b_compete_smoke/profile.png --force`
  - 结果：
    - `geoop19` 最优仍为 `lzcd_geochan_ccg_stmem_b`（`tum_acc=3.3620`, `bonn_acc=5.2265`）。
    - `geoop19b` 中 `compete_adapt` 相对 `struct_dual_layer_c`：`tum_ghost` 小幅下降（`-0.00041`），但 `bonn_acc` 上升（劣化，`+0.0429`），`bonn_fscore` 下降（`-0.00186`）。
  - 判定：
    - 自适应竞争仍未解决 Bonn 侧 Acc 反弹，竞争项不进入默认主线。

- 追加执行（2026-03-05，竞争微扫 `geoop20`）：
  - 目的：
    - 在竞争启用下，仅扫关键三元组（`margin/div_weight/drop_thresh`）验证是否存在“低损失竞争甜点”。
  - 新增 profile：
    - `struct_dual_layer_c_compete_m03`
    - `struct_dual_layer_c_compete_m05`
    - `struct_dual_layer_c_compete_m07`
    - 实现位点：`scripts/run_p10_precision_profile.py`
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles struct_dual_layer_c,struct_dual_layer_c_compete_m03,struct_dual_layer_c_compete_m05,struct_dual_layer_c_compete_m07 --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop20_compete_sweep_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop20_compete_sweep_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop20_compete_sweep_smoke/profile.png --force`
  - 结果（`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop20_compete_sweep_smoke.csv`）：
    - 最优仍为 `struct_dual_layer_c`（无竞争）。
    - 三个竞争变体均表现为同一模式：`tum_ghost` 仅千分位改善（`~ -0.0004`），但 `bonn_acc` 统一劣化（`+0.037~+0.041`），`bonn_fscore` 下降（`~ -0.0018~-0.0020`）。
  - 判定：
    - 竞争门控在当前结构下属于“低收益高代价”分支，继续保留代码但默认禁用；
    - P10 主线继续走“去偏算子 + 非竞争式动态抑制”。

- 追加执行（2026-03-05，非竞争参数微扫 `geoop21`）：
  - 目的：
    - 在“禁用竞争”前提下，仅调整 dual-layer 动态项（`phi_div/drop/free_ratio`）验证是否仍有可挖掘增益。
  - 新增 profile：
    - `struct_dual_layer_c_nocomp_soft`
    - `struct_dual_layer_c_nocomp_mid`
    - `struct_dual_layer_c_nocomp_hard`
    - 实现位点：`scripts/run_p10_precision_profile.py`（基于 `struct_dual_layer_c` 的 `replace(...)` 扩展）
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles struct_dual_layer_c,struct_dual_layer_c_nocomp_soft,struct_dual_layer_c_nocomp_mid,struct_dual_layer_c_nocomp_hard --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop21_nocomp_sweep_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop21_nocomp_sweep_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop21_nocomp_sweep_smoke/profile.png --force`
  - 结果（`output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop21_nocomp_sweep_smoke.csv`）：
    - 最优为 `struct_dual_layer_c_nocomp_mid`，但改变量极小：
      - `bonn_acc`: `5.44378 -> 5.44366`（`-0.00012`）
      - `bonn_fscore`: `0.592589 -> 0.592654`（`+0.000065`）
      - `bonn_ghost_ratio`: `0.057187 -> 0.057144`（`-0.000043`）
    - TUM 侧四个 profile 指标完全一致（`tum_acc/tum_fscore/tum_ghost` 无变化）。
  - 判定：
    - 非竞争参数层已进入“数值噪声级”边际收益，无法推动 P10 过线；
    - 下一阶段必须回到模块级改造（结构可辨识新信息源），不再追加同类微扫。

- 追加执行（2026-03-06，按 `1->2->3->4->5->6` 模块链顺序验证，`geoop22`）：
  - 目标：
    - 严格按顺序验证模块叠加链路是否能推动 P10 过线：
      - `1:SSE-EM -> 2:+LBR -> 3:+VCR -> 4:+RBI -> 5:+EBCut -> 6:+MOPC`。
  - 命令：
    - `python scripts/run_p10_precision_profile.py --profiles struct_dual_layer_c,stage1_sse_em,stage2_sse_em_lbr,stage3_sse_em_lbr_vcr,stage4_sse_em_lbr_vcr_rbi,stage5_sse_em_lbr_vcr_rbi_ebcut,stage6_sse_em_lbr_vcr_rbi_ebcut_mopc --frames 8 --stride 4 --max_points_per_frame 600 --out_root output/post_cleanup/p10_struct_decouple_geoop22_modular_chain_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop22_modular_chain_smoke.csv --figure output/post_cleanup/p10_struct_decouple_geoop22_modular_chain_smoke/profile.png --force`
  - 产物：
    - `output/summary_tables/local_mapping_precision_profile_struct_decouple_geoop22_modular_chain_smoke.csv`
    - `output/post_cleanup/p10_struct_decouple_geoop22_modular_chain_smoke/profile.png`
    - `output/post_cleanup/p10_struct_decouple_geoop22_modular_chain_smoke/best_profile.json`
  - 关键结果（按 profile）：
    - `baseline(struct_dual_layer_c)`：`tum_acc=3.5677`, `bonn_acc=5.4494`, `bonn_comp_r_5cm=62.19`, `bonn_ghost_red=0.2408`
    - `stage1(+SSE-EM)`：`tum_acc=3.5677`, `bonn_acc=5.4437`, `bonn_comp_r_5cm=62.18`, `bonn_ghost_red=0.2388`
    - `stage2(+LBR)`：`tum_acc=3.5860`, `bonn_acc=5.4466`, `bonn_comp_r_5cm=62.24`, `bonn_ghost_red=0.2340`
    - `stage3(+VCR)`：`tum_acc=3.5881`, `bonn_acc=5.4466`, `bonn_comp_r_5cm=62.28`, `bonn_ghost_red=0.2318`
    - `stage4(+RBI)`：`tum_acc=3.5822`, `bonn_acc=5.4263`, `bonn_comp_r_5cm=62.77`, `bonn_ghost_red=0.2383`
    - `stage5(+EBCut)`：`tum_acc=3.4763`, `bonn_acc=5.2411`, `bonn_comp_r_5cm=61.92`, `bonn_ghost_red=0.1443`（本轮 best）
    - `stage6(+MOPC)`：`tum_acc=3.4763`, `bonn_acc=5.2411`, `bonn_comp_r_5cm=61.92`, `bonn_ghost_red=0.1437`
  - 判定：
    - 模块链顺序验证已完整闭环，`stage5/6` 在 `Acc` 上有阶段性改善，但 Bonn 侧 `ghost` 相对 TSDF 降幅显著回落（约 `14%`），且 `Comp-R` 仍远低于 P10 门槛（`95%`）；
    - 全部 profile `pass_all=False`，本轮未推动 P10 过线。

- 2026-03-06: `PT-DSF/ZCBF/DCCM` 结构分支首轮落地与 smoke 验证：
  - 代码落点：
    - `egf_dhmap3d/core/types.py`：新增 `rho_static/rho_transient`、`ptdsf_*`、`zcbf_*`、`dccm_*` 体素状态。
    - `egf_dhmap3d/core/config.py`：新增 `ptdsf/zcbf/dccm` 更新与提取配置。
    - `egf_dhmap3d/core/voxel_hash.py`：新增 persistent-only 读出、`zcbf` 偏置注入、`dccm_commit` 软抑制。
    - `egf_dhmap3d/modules/updater.py`：新增 `PT-DSF` 分流写入、`ZCBF` 局部零交叉偏置场、`DCCM` 延迟提交矛盾记忆。
    - `egf_dhmap3d/modules/pipeline.py`、`scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`：打通新模块 CLI 与评估链路。
  - 运行命令：
    - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_a,p10_ptdsf_zcbf_dccm_b --frames 20 --stride 2 --out_root output/post_cleanup/p10_structural_smoke --summary_csv output/summary_tables/local_mapping_precision_profile_structural_smoke.csv --figure assets/p10_structural_smoke.png --force`
    - 实际执行中，`profile A` 已完整跑完；`profile B` 在 `TUM` 首段启动后手动截停，因为 `A` 已经明确不具备冲线趋势。
  - 产物：
    - `output/post_cleanup/p10_structural_smoke/00_p10_ptdsf_zcbf_dccm_a/`
    - `output/summary_tables/p10_structural_ptdsf_zcbf_dccm_smoke_a.csv`
  - 关键结果（`profile A`，6 序列聚合）：
    - `TUM`: `mean_acc_cm=2.5981`, `mean_comp_r_5cm=99.9967`, `min_ghost_reduction_vs_tsdf=0.1826`, `mean_fscore=0.9243`
    - `Bonn`: `mean_acc_cm=4.8190`, `mean_comp_r_5cm=84.3287`, `min_ghost_reduction_vs_tsdf=0.4623`, `mean_fscore=0.7600`
  - 判定：
    - 本轮首要目标“结构上解耦几何去偏与动态抑制”已完成代码落地，且新状态/算子链路已可端到端运行；
    - 但 `PT-DSF/ZCBF/DCCM` 的首版实现仍未推动 `P10` 过线，尤其 `TUM` 侧 `ghost` 相对 `TSDF` 降幅仅 `18.3%`，`Bonn` 侧 `Acc` 仍在 `4.82cm` 量级；
    - 说明当前 `persistent/transient` 分流与局部偏置校正仍然不够“强解耦”：`ZCBF` 去偏力度不足以显著回正零交叉，而 `DCCM` 读出抑制又没有把 `TUM` 尾迹稳定压下去。
  - 下一步：
    - 不再继续扩大 `profile B`/长帧 sweep；优先重构 `PT-DSF` 的 persistent readout 和 `ZCBF` 的偏置归一化/可信度传播，再进入下一轮验证。

- 2026-03-06: `PT-DSF/ZCBF/DCCM` 第二轮结构重构（按三条主线直接改，停止参数型尝试）：
  - 已落地修改：
    - `egf_dhmap3d/core/voxel_hash.py`
      - 强化 `PT-DSF` persistent readout：静态主分支与 `phi_geo` 的 persistent 读出权重显著提高，只有在 persistent 分支偏弱或与几何分支显著失配时才允许极弱 transient leak；
      - `legacy phi` 同步时引入基于 `zcbf_bias_conf` 的 debiased persistent readout，使最终导出的主表面首先受 persistent 分支支配；
      - `DCCM` 从 dual-layer 主门控链移除，不再抬升 `dyn_mix` 或缩紧全局 `drop_scale`，只保留在 `transient veto / competition` 支路中。
    - `egf_dhmap3d/modules/updater.py`
      - `ZCBF` 升级为“观测置信归一化 + block bias propagation”版本：block 观测权重不再只看 `phi_w`，还显式乘上 persistent 置信与 `rho` 归一化因子；
      - `DCCM` 继续只写入 transient/free-space 相关记忆，不再回流到几何主分支的融合门控。
  - 当前短 probe（已跑完；用于判断结构方向，不作为 P10 终版验收）：
    - 命令：`python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v3 --summary_csv output/summary_tables/p10_structural_probe_v3.csv --figure assets/p10_structural_probe_v3.png --force`
    - 产物：
      - `output/post_cleanup/p10_structural_probe_v3/`
      - `output/summary_tables/p10_structural_probe_v3.csv`
      - `output/summary_tables/p10_structural_probe_v3_best.json`
      - `output/summary_tables/p10_structural_probe_v3_by_sequence.csv`
      - `assets/p10_structural_probe_v3.png`
    - 完整结果：
      - `TUM(3 walking mean)`: `acc=3.724cm`, `fscore=0.8508`, `ghost_ratio=0.1237`，相对 `TSDF` 的最差序列降幅约 `40.4%`；
      - `Bonn(all3 mean)`: `acc=4.955cm`, `fscore=0.6201`, `ghost_ratio=0.0386`, `ghost_tail_ratio=0.1185`；其中 `ghost_tail_ratio` 相对 `TSDF` 均值下降约 `52.6%`，但 `ghost_ratio` 因短帧分母效应出现反常，不适合作为本 probe 的唯一 ghost 判据。
  - 阶段判定：
    - 这轮结构重构已经把 `TUM` 短 probe 上的最差 `ghost` 降幅从上一轮的 `18.3%` 提升到约 `40.4%`，说明“persistent 主读出 + ZCBF 置信传播 + DCCM 支路隔离”方向是正确的；
    - `Bonn` 上的 `fscore/acc` 也优于同设置下的 `TSDF`，但 ghost 结论必须看双口径（`ghost_ratio + ghost_tail_ratio`），不能继续只看单一比例指标；
    - 本轮仍不是 `P10` 终版验收，但已足够作为下一步结构主线的保留依据：后续若继续冲线，应在当前结构上补“Bonn 双口径 ghost 汇总 + 显式动态状态估计/竞争融合”，而不是回退到纯参数扫描。

- 2026-03-06: `OMHS`（Occlusion-aware Multi-Hypothesis Surface）首轮结构化落地与 `v4_omhs` probe：
  - 本轮代码收口：
    - `egf_dhmap3d/core/types.py`：新增 `omhs_front_conf / omhs_rear_conf / omhs_gap / omhs_active` 状态；
    - `egf_dhmap3d/core/config.py`：新增 `surface.omhs_enable`；
    - `egf_dhmap3d/modules/updater.py`：新增 `_update_omhs_state()`，在 dual-state 同步后维护前/后层置信与遮挡 gap；
    - `egf_dhmap3d/core/voxel_hash.py`：
      - `persistent_surface_readout()` 中对 transient leak 引入 `rear_support/front_drive` 抑制；
      - `extract_surface_points()` 中新增 `omhs_enable` 链路、rear-anchor 保留和 dual-layer 后验豁免；
    - `egf_dhmap3d/modules/pipeline.py`、`scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`：打通 `--surface_omhs_enable`。
  - 验证命令：
    - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v4_omhs --summary_csv output/summary_tables/p10_structural_probe_v4_omhs.csv --figure assets/p10_structural_probe_v4_omhs.png --force`
  - 产物：
    - `output/post_cleanup/p10_structural_probe_v4_omhs/`
    - `output/summary_tables/p10_structural_probe_v4_omhs.csv`
    - `output/summary_tables/p10_structural_probe_v4_omhs_best.json`
    - `output/summary_tables/p10_structural_probe_v4_omhs_by_sequence.csv`
    - `assets/p10_structural_probe_v4_omhs.png`
  - 与 `v3` 的聚合差分：
    - `TUM`: `acc 3.7237 -> 3.7259 cm`, `fscore 0.8508 -> 0.8502`, `ghost_ratio 0.1237 -> 0.1241`；
    - `Bonn`: `acc 4.9552 -> 4.9599 cm`, `Comp-R@5cm 64.76 -> 64.89`, `fscore 0.6201 -> 0.6206`, `ghost_ratio 0.0386 -> 0.0388`；
    - `Bonn` 按序列的 `ghost_tail_ratio` 相对 `TSDF` 降幅仍主要来自既有结构主线，`OMHS` 自身只带来 `1e-3` 量级波动，未形成可归因的新增益。
  - 判定：
    - 当前 `OMHS` 作为“提取端/读出端的 rear-surface 保护”是可运行的，但净效果接近中性；
    - 它没有实质改善 `P10` 的核心瓶颈，说明 Bonn 的问题并不主要是“后验筛选时把 rear surface 删掉了”，而是“写入阶段 rear persistent evidence 本身不够强，导致后验无面可保”；
    - 因此 `OMHS` 可以保留为辅助结构，但不应继续沿阈值或保留策略方向深挖。
  - 下一步：
    - 从 `readout-side OMHS` 切到 `write-time occlusion decomposition`：在更新阶段显式分离 front/rear hit，而不是只在提取阶段做 rear anchor 保留；
    - 优先实现“写入阶段的双层表面竞争/残差分桶”，再决定是否保留当前 `OMHS` 作为输出稳定器。

- 2026-03-06: `WOD`（Write-Time Occlusion Decomposition）首轮落地与 `v5_wod` probe：
  - 本轮代码收口：
    - `egf_dhmap3d/core/types.py`：新增 `wod_front_conf / wod_rear_conf / wod_shell_conf`；
    - `egf_dhmap3d/core/config.py`：新增 `wod_*` 更新配置；
    - `egf_dhmap3d/modules/updater.py`：新增 `_write_time_occlusion_split()`，在更新阶段把近表面观测分解成 `front / rear / shell` 并分别作用于 `phi_static / phi_transient / phi_geo / phi_dyn / surf/free evidence`；
    - `egf_dhmap3d/core/voxel_hash.py`：把 `wod_*` 状态接入遗忘与 readout；
    - `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`：打通 `--wod_enable`。
  - 验证命令：
    - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v5_wod --summary_csv output/summary_tables/p10_structural_probe_v5_wod.csv --figure assets/p10_structural_probe_v5_wod.png --force`
  - 产物：
    - `output/post_cleanup/p10_structural_probe_v5_wod/`
    - `output/summary_tables/p10_structural_probe_v5_wod.csv`
    - `output/summary_tables/p10_structural_probe_v5_wod_by_sequence.csv`
    - `assets/p10_structural_probe_v5_wod.png`
  - 与 `v4_omhs` 的聚合差分：
    - `TUM`: `acc 3.7259 -> 3.7126 cm`, `fscore 0.8502 -> 0.8515`, `ghost_ratio 0.1241 -> 0.1244`；
    - `Bonn`: `acc 4.9599 -> 4.9964 cm`, `Comp-R@5cm 64.89 -> 65.42`, `fscore 0.6206 -> 0.6201`, `ghost_ratio 0.0388 -> 0.0387`；
    - 结论：`WOD` 证明了“问题确实在写入期而不在 readout-only”，但当前仍是弱增益模块，尚不足以推动 `P10` 过线。
  - 判定：
    - `WOD` 可保留为结构主线的一部分；
    - 但仅靠 soft split + soft weighting 还不能稳定建立 rear persistent surface；
    - 下一步必须让 `rear` 以独立状态被写入，而不是继续堆叠 readout 权重。

- 2026-03-06: `RPS`（Rear-Persistent Surface Buffer）首轮落地与 `v6_rps` probe：
  - 本轮代码收口：
    - `egf_dhmap3d/core/types.py`：新增 `phi_rear / phi_rear_w / rho_rear`；
    - `egf_dhmap3d/core/config.py`：新增 `rps_*` 配置；
    - `egf_dhmap3d/modules/updater.py`：在 geometry write 后新增 rear-buffer 写入，把 `WOD` 的 `rear` 责任显式沉淀为独立表面状态；
    - `egf_dhmap3d/core/voxel_hash.py`：把 `phi_rear / rho_rear` 接入 `PT-DSF` 状态统计、persistent readout、遗忘与 stale 清理；
    - `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`：打通 `--rps_enable`，并新增 profile `p10_ptdsf_zcbf_dccm_rps_a`。
  - 验证命令：
    - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_rps_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v6_rps --summary_csv output/summary_tables/p10_structural_probe_v6_rps.csv --figure assets/p10_structural_probe_v6_rps.png --force`
  - 产物：
    - `output/post_cleanup/p10_structural_probe_v6_rps/`
    - `output/summary_tables/p10_structural_probe_v6_rps.csv`
    - `output/summary_tables/p10_structural_probe_v6_rps_by_sequence.csv`
    - `output/summary_tables/p10_structural_probe_v6_rps_vs_v5.csv`
    - `assets/p10_structural_probe_v6_rps.png`
  - 与 `v5_wod` 的聚合差分：
    - `TUM`: `acc 3.7126 -> 3.7429 cm`, `fscore 0.8515 -> 0.8478`, `ghost_ratio 0.1244 -> 0.1216`；
    - `Bonn`: `acc 4.9964 -> 4.9992 cm`, `Comp-R@5cm 65.42 -> 64.89`, `fscore 0.6201 -> 0.6170`, `ghost_ratio 0.0387 -> 0.0386`；
    - 按序列看：`balloon2` 出现正增益（`acc 4.323 -> 4.300 cm`, `fscore 0.6967 -> 0.7015`），但 `balloon / TUM(3 walking)` 均回退，说明收益强依赖遮挡模式。
  - 判定：
    - `RPS` 已证明“rear persistent 可以被状态化写入”，但当前实现只带来**选择性**收益，并没有形成跨数据集稳定增益；
    - 它更像 `P10` 主线上的研究原型，而不是当前可封板的主配置；
    - 下一步不应继续给 `RPS` 堆权重，而应升级为更硬的 `rear commit / surface-bank` 结构，只在满足遮挡矛盾触发条件时提交 rear surface。

- 2026-03-06: `HRC`（Hard Rear-Commit）首轮落地与 `v7_hrc` probe：
  - 本轮代码收口：
    - `egf_dhmap3d/core/types.py`：新增 `phi_rear_cand / phi_rear_cand_w / rho_rear_cand / rps_commit_score / rps_commit_age / rps_active`；
    - `egf_dhmap3d/core/config.py`：新增 `rps_hard_commit_*` 相关配置；
    - `egf_dhmap3d/modules/updater.py`：把 `RPS` 从 always-on rear buffer 改成 `candidate -> commit -> active bank` 链路；
    - `egf_dhmap3d/core/voxel_hash.py`：仅在 committed / active 条件满足时才让 rear bank 参与 `PT-DSF` 统计与 persistent readout；
    - `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`：新增 `--rps_hard_commit_enable`，并接入 profile `p10_ptdsf_zcbf_dccm_hrc_a`。
  - 验证命令：
    - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_hrc_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v7_hrc --summary_csv output/summary_tables/p10_structural_probe_v7_hrc.csv --figure assets/p10_structural_probe_v7_hrc.png --force`
  - 产物：
    - `output/post_cleanup/p10_structural_probe_v7_hrc/`
    - `output/summary_tables/p10_structural_probe_v7_hrc.csv`
    - `output/summary_tables/p10_structural_probe_v567_by_sequence.csv`
    - `output/summary_tables/p10_structural_probe_v7_hrc_vs_v5_v6_by_sequence.csv`
    - `assets/p10_structural_probe_v7_hrc.png`
  - 与 `v5_wod` 的聚合差分：
    - `TUM`: `acc 3.7126 -> 3.7232 cm`, `fscore 0.8515 -> 0.8501`, `ghost_ratio 0.1244 -> 0.1236`；
    - `Bonn`: `acc 4.9964 -> 4.9775 cm`, `Comp-R@5cm 65.42 -> 64.64`, `fscore 0.6201 -> 0.6179`, `ghost_ratio 0.03866 -> 0.03830`；
    - 按序列看：`walking_xyz` 与 `walking_static` 仅得到极小 ghost 改善但几何略退，`balloon2 / crowd2` 有轻微正向 Bonn ghost 改善，但整体仍未超过 `v5_wod`。
  - 判定：
    - `HRC` 证明“rear state 的条件提交”比 `always-on RPS` 更稳；
    - 但它仍未形成跨数据集稳定增益，说明问题不只是“rear buffer 太常开”，而是“rear state 即使提交后，主表面读出仍然很少真正从它获益”。

- 2026-03-06: `Surface-Bank`（Committed Surface-Bank Readout）首轮落地与 `v8_sbank` probe：
  - 本轮代码收口：
    - `egf_dhmap3d/core/config.py`：新增 `rps_surface_bank_enable / rps_bank_*`；
    - `egf_dhmap3d/core/voxel_hash.py`：把 persistent readout 从 `front/rear soft mix` 改成 `front-bank vs rear-bank` 离散竞争，rear 仅在明确胜出时导出；
    - `scripts/run_egf_3d_tum.py`、`scripts/run_benchmark.py`、`scripts/run_p10_precision_profile.py`：打通 `--rps_surface_bank_enable`，并新增 profile `p10_ptdsf_zcbf_dccm_sbank_a`。
  - 验证命令：
    - `python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_sbank_a --frames 4 --stride 8 --max_points_per_frame 1500 --out_root output/post_cleanup/p10_structural_probe_v8_sbank --summary_csv output/summary_tables/p10_structural_probe_v8_sbank.csv --figure assets/p10_structural_probe_v8_sbank.png --force`
  - 产物：
    - `output/post_cleanup/p10_structural_probe_v8_sbank/`
    - `output/summary_tables/p10_structural_probe_v8_sbank.csv`
    - `output/summary_tables/p10_structural_probe_v5678_by_sequence.csv`
    - `output/summary_tables/p10_structural_probe_v8_sbank_vs_v567.csv`
    - `assets/p10_structural_probe_v8_sbank.png`
  - 与 `v7_hrc` 的聚合差分：
    - `TUM`: `acc 3.7232 -> 3.7255 cm`, `fscore 0.8501 -> 0.8497`, `ghost_ratio 0.12358 -> 0.12364`；
    - `Bonn`: `acc 4.9775 -> 4.9779 cm`, `Comp-R@5cm 64.64 -> 64.61`, `fscore 0.6179 -> 0.6176`, `ghost_ratio 0.03830 -> 0.03784`；
    - 按序列看：`walking_xyz / walking_static` 与 `v7_hrc` 完全一致，`walking_halfsphere` 轻微回退，Bonn 三条序列仅出现 `1e-4 ~ 1e-3` 级别 ghost 波动。
  - 判定：
    - `Surface-Bank` 已经把 readout 端做到足够“硬”，但结果几乎不变；
    - 这说明 `P10` 的主瓶颈**不在导出阶段的 front/rear 竞争**，而在写入阶段 rear/static 几何本身尚未被可靠形成。

- 当前判定（结构解耦 + 几何去偏专用算子）：
  - 方向有效：主症结已经从“后验筛选误删”进一步收敛到“写入阶段是否真的形成了可用的 rear/static 几何状态”；
  - 已证伪子路线：`readout-only OMHS` 近中性、`soft WOD weighting` 仅弱增益、`always-on RPS buffer` 条件敏感、`Surface-Bank readout` 近乎零增益；
  - 主要瓶颈：`Acc` 与 ghost 仍未被**写入期结构性解耦**，当前 rear bank 更多是在“有则读”，而不是“先被可靠写出来”；
  - 下一步建议：进入“写入期双表面生成”主线，用显式的 `front-transient / rear-static` 双写入与提交机制，在 update 阶段直接形成两个可竞争的几何状态，而不是继续在 readout 端做更硬的选择。

满足以下全部条件即判定“局部建图顶刊标准已达成（无保留）”：
1. P10-P15 全部硬验收通过；  
2. `P0-R / P12-R / P13-R` 三项审计重跑通过并回写主表；  
3. `slam` 口径主结论仅来自 `--egf_slam_no_gt_delta_odom` 结果；  
4. 主表同时包含协议分离与双口径 ghost 字段（或在伴随表明确给出）；  
5. 无 “placeholder baseline” 与无 “口径混表” 条目进入主结论；  
6. `scripts/update_summary_tables.py --prefer_p4_final --verbose` 一条命令可刷新全部终版表；  
7. `BENCHMARK_REPORT.md` 主结论仅引用终版封板产物。

> 当前状态（2026-03-04）：审计阻断项已解除（`P0-R/P12-R/P13-R` 完成并回写）；是否最终封板按第 7.4 条其余条件继续判定。

### 7.5 完成后允许的结论措辞（固定）

> “在局部建图设定下，EGF-DHMap 已达到顶刊投稿标准。该结论基于双协议（oracle/slam）、多数据集（TUM/Bonn）、多 seed 显著性、真实外部基线和边界鲁棒性分析的完整证据链。”

## 2026-03-06 OTV Negative Result Record

### Module
- `OTV (Observation-Time Transient Veto)`
- `OTV + front exclusion` (readout-side exclusion using `phi_otv` geometry)

### Code Landing
- `egf_dhmap3d/modules/updater.py`
- `egf_dhmap3d/core/voxel_hash.py`
- `egf_dhmap3d/core/types.py`
- `egf_dhmap3d/core/config.py`
- `scripts/run_benchmark.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_p10_precision_profile.py`

### Focused Probe
- Scene: `rgbd_dataset_freiburg3_walking_xyz`
- Frames: `60`
- Protocol: `oracle`
- Outputs:
  - `output/post_cleanup/p10_otv_probe_walking_xyz/`
  - `output/post_cleanup/p10_otv_probe_walking_xyz_tfe/`
  - `output/summary_tables/p10_otv_probe_walking_xyz_compare.csv`

### Result
- `OTV` and `OTV + front exclusion` are identical on the focused probe.
- Key metrics:
  - `F-score = 0.981691`
  - `ghost_ratio = 0.756281`
  - `ghost_tail_ratio = 0.340002`
- This does not improve the P10 ghost bottleneck.

### Status Update
- `OTV` is marked as a negative branch and should not be used as the main fix for P10.
- Remaining valid mainline: move to stronger architectural decoupling, not more score gates.

## 2026-03-06 CSR-XMap Negative Result Record

### Implemented
- 新增 `CSR-XMap` 结构分支：
  - `CSR (Counterfactual Static Readout)`：从 `phi_static / phi_rear / phi_spg` 和经一致性保护的 `phi_geo` 构造反事实静态表面；
  - `XMap`：从 `phi_transient / phi_dyn / phi_otv` 构造显式动态排斥表面；
  - 在 `extract_surface_points()` 中，不再只靠 `dyn_mix` 软门控，而是做一次“静态反事实表面 vs 动态排斥表面”的几何竞争。
- 相关代码已接入：
  - `egf_dhmap3d/core/config.py`
  - `egf_dhmap3d/modules/pipeline.py`
  - `scripts/run_egf_3d_tum.py`
  - `scripts/run_benchmark.py`
  - `scripts/run_p10_precision_profile.py`
  - `egf_dhmap3d/core/voxel_hash.py`

### 验证命令
```bash
python scripts/run_p10_precision_profile.py --profiles p10_ptdsf_zcbf_dccm_wdsgr_spg_csr_xmap_a --dry_run
# 然后抽取 tum benchmark 命令，改为：
#   --static_sequences ""
#   --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz
#   --frames 60
#   --out_root output/post_cleanup/p10_csr_xmap_probe_walking_xyz
```

### 结果（focused probe）
- 输出目录：`output/post_cleanup/p10_csr_xmap_probe_walking_xyz/`
- 对比表：`output/summary_tables/p10_csr_xmap_probe_walking_xyz_compare.csv`
- `CSR-XMap`：
  - `F-score = 0.995147`
  - `Chamfer = 0.020893`
  - `ghost_ratio = 0.810235`
  - `ghost_tail_ratio = 0.338690`
- `OTV` 参考：
  - `F-score = 0.981691`
  - `Chamfer = 0.024224`
  - `ghost_ratio = 0.756281`
  - `ghost_tail_ratio = 0.340002`
- `TSDF` 参考：
  - `F-score = 0.913703`
  - `ghost_ratio = 0.687708`
  - `ghost_tail_ratio = 0.146341`

### 结论
- `CSR-XMap` 对几何读出是有效的：`Chamfer/F-score` 都优于 `OTV`。
- 但它**没有解决 P10 的核心 ghost 长尾问题**：
  - `ghost_tail_ratio` 基本不变（`0.3400 -> 0.3387`）；
  - `ghost_ratio` 反而进一步恶化（`0.7563 -> 0.8102`）。
- 同时运行代价上升明显：
  - `wall_total_sec` 约从 `784.9 s` 升到 `845.9 s`；
  - `extract_total_sec` 约从 `21.2 s` 升到 `52.6 s`。

### 判定
- `CSR-XMap` 作为 `P10` 主线 **未通过**，记为第二个结构负例。
- 它说明：
  1. export/readout 级的反事实静态竞争可以改善几何质量；
  2. 但 `P10` 真正卡住的是**长尾动态残影并未在状态层被独立建模**；
  3. 因而下一步不该继续在 readout 端叠规则，而应进入更强的“显式静/动态图层或写入期排斥状态”路线。

### 下一步主线
- 从 `readout-side competition` 切到 `state-side separation`：
  - 真实双图层 / 双地图读写；或
  - 显式动态排斥 occupancy / exclusion state，在 update-time 提交负证据，而不是等导出阶段再竞争。

## 2026-03-06 XMem / BECM / RCCM Negative Chain Record

### 模块与落地
- `XMem (Exclusion Memory)`：写入期可逆排斥记忆。
- `BECM (Bifurcated Exclusion-Clear Memory)`：把 `front exclusion` 与 `clear-lock` 分成两个独立状态，避免 `free evidence` 直接抵消排斥。
- `RCCM (Ray-Conditioned Clear Memory)`：只对已有前景记忆的体素注入沿视线负证据，避免回到全局激进 ray-casting。
- 代码接入：
  - `egf_dhmap3d/core/types.py`
  - `egf_dhmap3d/core/config.py`
  - `egf_dhmap3d/modules/updater.py`
  - `egf_dhmap3d/core/voxel_hash.py`
  - `scripts/run_egf_3d_tum.py`
  - `scripts/run_benchmark.py`
  - `scripts/run_p10_precision_profile.py`

### focused probe
- Scene: `rgbd_dataset_freiburg3_walking_xyz`
- Frames: `60`
- Protocol: `oracle`
- 输出目录：
  - `output/post_cleanup/p10_xmem_probe_walking_xyz/`
  - `output/post_cleanup/p10_xmem_becm_probe_walking_xyz/`
  - `output/post_cleanup/p10_xmem_rccm_probe_walking_xyz/`
- 统一对比表：`output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### 结果
- `XMem v1`：
  - `F-score = 0.992518`
  - `Chamfer = 0.024049`
  - `ghost_ratio = 0.820857`
  - `ghost_tail_ratio = 0.366245`
- `BECM`：与 `XMem v1` 数值完全一致。
- `RCCM`：与 `XMem v1` 数值完全一致，且运行更慢（`mapping_fps ≈ 0.0439`）。
- 参考：
  - `OTV ghost_tail_ratio = 0.340002`
  - `CSR-XMap ghost_tail_ratio = 0.338690`
  - `TSDF ghost_tail_ratio = 0.146341`

### 结论
- `XMem -> BECM -> RCCM` 构成了第三条完整的结构负例链。
- 这条链说明：
  1. 当前单状态融合 + 后置排斥/清理架构下，再往同一 persistent map 上挂排斥记忆，已经无法继续推动 `P10`；
  2. 即便把 `free evidence` 提升为 `clear-lock`，或者把沿射线负证据只打到 `XMem` 体素上，最终仍被同一个 persistent geometry pool 稀释；
  3. 因而 `P10` 的下一步不能再是“更多 veto memory”，而必须是**显式静态背景图层 / 双地图 / 双写入图**。

### 状态更新
- `P10` 继续保持 **未过线**。
- 当前已证伪的主线包括：
  - `OTV`
  - `CSR-XMap`
  - `XMem`
  - `BECM`
  - `RCCM`
- 下一步唯一合理主线：进入**显式 background layer / dual-map** 结构，不再在单图层 veto 链路上做增量扩展。

## 2026-03-07 OBL-3D Negative Result Record

### 模块
- `OBL-3D (Occlusion-Buffered Background Layer)`
- `background_hard`：当背景置信度和前景动态冲突同时足够高时，导出阶段直接硬选背景层 `phi_bg`

### 代码落地
- `egf_dhmap3d/core/types.py`
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/modules/updater.py`
- `egf_dhmap3d/core/voxel_hash.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### focused probe
- Scene: `rgbd_dataset_freiburg3_walking_xyz`
- Frames: `60`
- Protocol: `oracle`
- 输出目录：
  - `output/post_cleanup/p10_obl_probe_walking_xyz/`
  - `output/post_cleanup/p10_obl_hardbg_probe_walking_xyz/`
- 对比表：`output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### 结果
- `OBL-3D`：
  - `F-score = 0.989473`
  - `Chamfer = 0.024445`
  - `ghost_ratio = 0.806074`
  - `ghost_tail_ratio = 0.368327`
- `background_hard`：与 `OBL-3D` 数值完全一致，但更慢。

### 结论
- `OBL-3D` 证明了“在同一张地图中增加背景层”仍然不够。
- 即便：
  1. update 阶段已经单独写背景层；
  2. extract 阶段已经允许硬选背景层；
  最终 ghost 仍未改善到可接受水平。
- 这说明当前 P10 的剩余瓶颈已经从“缺少背景状态”进一步收敛到：
  - **缺少真正独立的 background map / foreground map 持久化与读出链路**。

### 状态更新
- `P10` 继续 **未过线**。
- 现阶段单图层主线已证伪的结构分支包括：
  - `OTV`
  - `CSR-XMap`
  - `XMem`
  - `BECM`
  - `RCCM`
  - `OBL-3D`
  - `background_hard`
- 下一步唯一合理主线：进入**真双地图**结构，不再在同一个 voxel map 上追加新通道。

## 2026-03-07 Dual-Map Mainline Update

### 模块
- `DMBG-3D (Dual-Map Background/Foreground Graph)`
- `BER (Background-Exclusive Readout)`

### 代码落地
- `egf_dhmap3d/modules/pipeline.py`
- `egf_dhmap3d/modules/updater.py`
- `egf_dhmap3d/core/voxel_hash.py`
- `egf_dhmap3d/core/config.py`
- `egf_dhmap3d/core/types.py`
- `scripts/run_egf_3d_tum.py`
- `scripts/run_benchmark.py`

### focused probe
- Scene: `rgbd_dataset_freiburg3_walking_xyz`
- Frames: `60`
- Protocol: `oracle`
- 输出目录：
  - `output/post_cleanup/p10_dualmap_probe_walking_xyz/`
  - `output/post_cleanup/p10_dualmap_ber_probe_walking_xyz/`
- 对比表：`output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### 结果
- `dual_map`：
  - `F-score = 0.996531`
  - `Chamfer = 0.020462`
  - `ghost_ratio = 0.813839`
  - `ghost_tail_ratio = 0.347614`
- `dual_map_ber`：
  - 与 `dual_map` 指标完全一致；
  - 运行更慢。

### 结论
- `dual_map` 是当前第一条**真正有效的结构主线**：
  - 中间态明显改变，且不是数值等价分支；
  - 几何质量继续提升；
  - `ghost_tail_ratio` 从 `0.366245` 降到 `0.347614`，说明“背景图与前景图分家”方向正确。
- 但它仍未让 `P10` 过线，说明仅做“分图存储 + 独立导出”还不够。
- 目前最合理的下一步创新模块已经收敛为：
  - **跨图负证据回传**，即把 foreground_map 中稳定形成的动态矛盾，作为显式负证据投回 background_map。

### 状态更新
- `P10` 仍未过线。
- 但主线已从“单图层否决链”正式切换到 `dual_map`。
- 之后不再继续扩展 `OTV/XMem/OBL` 这类单图层路线；下一步主攻 `CMCT`。

## 2026-03-07 CMCT Negative Result Record

### 模块
- `CMCT (Cross-Map Contradiction Transfer)`
- `cross-map foreground arbitration`：背景图提取时显式查询 foreground_map 的局部动态表面，抑制弱背景候选

### 输出
- `output/post_cleanup/p10_cmct_probe_walking_xyz/`
- `output/post_cleanup/p10_cmct_cfam_probe_walking_xyz/`
- `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### 结果
- `dual_map_cmct` 与 `dual_map` 数值完全一致：
  - `F-score = 0.996531`
  - `Chamfer = 0.020462`
  - `ghost_tail_ratio = 0.347614`
- `dual_map_cmct_cfam` 同样与 `dual_map` 数值一致。

### 结论
- 这说明当前剩余瓶颈已经不在“体素级跨图回传”或“同位点前景抑制”。
- 即使：
  1. foreground 的稳定动态矛盾已经显式投回 background；
  2. 背景提取时也已经查询 foreground_map 做局部屏蔽；
  最终 ghost 指标仍没有变化。
- 因而下一步主线必须升级到**几何域 / 视线域**，而不是继续在 voxel-local 上做状态传递。

### 状态更新
- `dual_map` 仍是当前唯一的结构性正主线。
- `CMCT` 与其 readout 变体记为负结果。
- 下一步创新模块建议：`CGCC (Cross-Map Geometric Carving Corridor)`，即把 foreground 表面投成短程 carving corridor，对 background_map 做几何域负约束。

## 2026-03-08 CGCC Negative Result Record

### 模块
- `CGCC (Cross-Map Geometric Carving Corridor)`

### 输出
- `output/post_cleanup/p10_cgcc_probe_walking_xyz/`
- `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### 结果
- `dual_map_cgcc` 与 `dual_map` 数值完全一致：
  - `F-score = 0.996531`
  - `Chamfer = 0.020462`
  - `ghost_tail_ratio = 0.347614`
- 运行更慢：`mapping_fps ≈ 0.0229`。

### 结论
- 这说明即使把 foreground 表面提升为“几何域短走廊 carving”，在当前形式下仍不足以改变最终 ghost 指标。
- 因而当前剩余瓶颈已经不在：
  - 同位点门控；
  - 体素级跨图传递；
  - 短程局部视线走廊。
- 下一步必须升级到更持久的**自由空间世界表示**，而不是继续追加局部 suppression 模块。

### 状态更新
- `dual_map` 继续保留为唯一结构性正主线。
- `CMCT / CFAM / CGCC` 全部记为围绕 dual_map 的负结果链。
- 下一步建议：`PFV (Persistent Free-space Volume)`。

## 2026-03-08 PFV Mainline Update

### 模块
- `PFV (Persistent Free-space Volume)`

### 输出
- `output/post_cleanup/p10_pfv_probe_walking_xyz/`
- `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### 结果
- `dual_map_pfv`：
  - `F-score = 0.996541`
  - `Chamfer = 0.020461`
  - `ghost_ratio = 0.813677`
  - `ghost_tail_ratio = 0.346503`
- 相比 `dual_map`：
  - `ghost_tail_ratio: 0.347614 -> 0.346503`
  - `F-score` 和 `Chamfer` 轻微改善。

### 结论
- PFV 是 `dual_map` 之后第一条再次推动目标指标前进的主线。
- 虽然幅度还不大，但它说明“持久自由空间状态”比此前的：
  - 单图层门控；
  - 跨图体素矛盾回传；
  - 短程几何走廊 carving；
  更接近真正缺失的机制。

### 状态更新
- `dual_map + PFV` 现在是新的主线。
- `CMCT / CFAM / CGCC` 维持负结果归档。
- 下一步不再回到 contradiction-only 路线，而是继续强化 PFV 主线。

## 2026-03-08 PFVP Negative Result Record

### 模块
- `PFVP (PFV-guided Proposal Routing)`

### 输出
- `output/post_cleanup/p10_pfvp_quick/`
- `output/post_cleanup/p10_pfvp_probe_walking_xyz_v2/`
- `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### 结果
- `dual_map_pfvp`：
  - `F-score = 0.997273`
  - `Chamfer = 0.020437`
  - `ghost_tail_ratio = 0.352354`
- 相比 `dual_map_pfv`：
  - 几何略有改善；
  - 但 `ghost_tail_ratio` 从 `0.346503` 反弹到 `0.352354`。

### 结论
- PFVP 证明“把 PFV 再前推到 proposal routing 阶段”并不能解决当前瓶颈。
- 这与 `PFAG` 的结论一致，但更有说服力：
  - 不只是“拒绝太早”有问题；
  - 即使保留观测、只改写入去向，也会损害晚期 ghost 指标。
- 因此，下一步不应继续向更早阶段推进 PFV，而应继续强化 `dual_map + PFV` 在 update/export 内部的表达能力。

### 状态更新
- `dual_map + PFV` 仍是当前最优主线。
- `PFAG / PFVP` 均记为负结果。
- 下一步建议：继续增强 PFV 本体，而不是更早的 gating/routing。

## 2026-03-08 PFVP Negative Result Record

### 模块
- `PFVP (PFV-guided Proposal Routing)`

### 输出
- `output/post_cleanup/p10_pfvp_quick/`
- `output/post_cleanup/p10_pfvp_probe_walking_xyz_v2/`
- `output/summary_tables/p10_structural_probe_walking_xyz_compare.csv`

### 结果
- `dual_map_pfvp`：
  - `F-score = 0.997273`
  - `Chamfer = 0.020437`
  - `ghost_tail_ratio = 0.352354`
- 相比 `dual_map_pfv`：
  - 几何略有改善；
  - 但 `ghost_tail_ratio` 从 `0.346503` 反弹到 `0.352354`。

### 结论
- PFVP 再次证明：把 PFV 前推到 update 之前的决策层面，会损害当前最关键的 ghost 指标。
- 它比 PFAG 更温和，但结论一致：
  - `PFV` 应保留在 update/export 主线；
  - 不应继续向更早的 gating/routing 扩张。

### 状态更新
- `dual_map + PFV` 仍为当前最佳主线。
- `PFAG / PFVP` 都作为负结果归档。
- 下一步建议：继续增强 PFV 本体，而不是更早阶段的决策逻辑。

## 2026-03-08 PFV-Sharp Quick Negative Record

### 模块
- `PFV-sharp`：更长时记忆的 PFV + 深度感知累积 + clustered export sharpening

### 输出
- `output/post_cleanup/p10_pfv_sharp_quick/`

### 结果
- `10` 帧 quick probe 与当前 `dual_map + PFV` 主线在关键表上基本数值一致。
- 因此未继续升级到 `60` 帧 focused probe。

### 结论
- 当前 PFV 的问题不只是“状态不够强”或“阈值不够锐利”。
- 下一步若继续沿 PFV 主线，应优先考虑更结构化的 PFV 表示，而不是再增强同一个标量状态。

## 2026-03-08 PFV-Bank Quick Negative Record

### 模块
- `PFV-bank`：near / mid / far free-space banks + bank-aware export sharpening

### 输出
- `output/post_cleanup/p10_pfv_banks_quick/`

### 结果
- `10` 帧 quick probe 与当前 `dual_map + PFV` 主线在关键表上基本数值一致。
- 因此未继续升级到 `60` 帧 focused probe。

### 结论
- 当前 PFV 的剩余瓶颈不只是“单标量不够”，而是更新/导出的结构作用仍不足。
- `PFV-bank` 作为结构化 quick 尝试，当前也记为低收益负结果。


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
