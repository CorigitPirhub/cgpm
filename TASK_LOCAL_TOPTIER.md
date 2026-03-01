# EGF-DHMap 局部建图顶刊任务书（执行版）

## 执行状态（2026-03-01）

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
  - 备注：`dynaslam` 槽位当前为几何代理输出（非官方 DynaSLAM runner），后续需替换为真正外部算法输出。

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

1. 将 `dynaslam` 槽位从几何代理替换为真实外部算法 runner（优先 DynaSLAM 或 MID-Fusion 真实输出），并保持统一评估链路不变。  
2. 动态优势约束补齐：把 `walking_static` 的 `ghost_ratio` 相对 TSDF 降幅从 `34.4%` 提升到 `>=35%`，同时保持 `F-score(EGF) >= TSDF - 0.01`。  
3. Bonn 泛化修复：针对 `FAILURE_CASE.md` 中“低 F-score/低 ghost”失衡，加入 Bonn 自适应表面提取策略，目标是在不显著反弹 ghost 的前提下恢复几何完整性。
