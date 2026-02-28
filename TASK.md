# EGF-DHMap 3D 顶刊冲刺可执行任务单

## 0. 目标与通过线

### 0.1 总目标
将当前 EGF-DHMap 3D 从“结果可用”升级到“顶刊可审稿”级别：
- 协议公平（Oracle/SLAM 分离）
- 基线充分（强基线 + 本地可复现）
- 统计完备（mean±std + 显著性）
- 报告自动化（数据到文档一致）

### 0.2 顶刊通过线（硬约束）
1. 至少 3 个数据域，>=20 条序列，总帧数 >=20k。
2. 至少 5 个可复现实验基线（不含本方法消融）。
3. 核心结论提供 mean±std，且有显著性检验（p < 0.05）。
4. 同时报告几何、动态抑制、跟踪、效率四类指标。
5. 提供一键复现实验脚本与完整失败案例。

### 0.3 当前执行状态
- [x] P0 协议修复（已完成）
- [x] P1 指标体系补齐（已完成）
- [x] P2 强基线补齐（已完成）
- [x] P3 数据集扩展（已完成）
- [ ] P4 深度消融
- [ ] P5 一键全流程与自动报告

---

## 1. P0 协议修复（必须先完成）

### 1.1 修复 GT pose 开关（阻断性）
- 文件：`scripts/run_egf_3d_tum.py`
- 当前问题：`--use_gt_pose` 为 `store_true + default=True`，导致默认总是 GT。
- 改造：改为互斥参数组
  - `--use_gt_pose`
  - `--no_gt_pose`
  - 默认设为 `False`（SLAM 协议）。
- 验收：`summary.json` 内 `use_gt_pose` 能正确出现 `true/false` 两种值。

### 1.2 在 benchmark 引入协议维度
- 文件：`scripts/run_benchmark.py`
- 改造：新增 `--protocol oracle|slam`，并向 `run_egf_3d_tum.py` 传递对应开关。
- 输出目录规范：
  - `output/journal_suite/oracle/...`
  - `output/journal_suite/slam/...`
- 验收：同一序列可生成两套独立结果，且互不覆盖。

### 1.3 固化随机性
- 文件：
  - `scripts/run_benchmark.py`
  - `scripts/run_egf_3d_tum.py`
  - `scripts/run_tsdf_baseline.py`
  - `scripts/run_simple_removal_baseline.py`
- 改造：统一增加 `--seed`，并控制采样/下采样随机源。
- 验收：同 seed 重跑指标稳定（浮点微差可接受）。

---

## 2. P1 指标体系补齐

### 2.1 几何指标补齐
- 文件：`egf_dhmap3d/eval/metrics.py`
- 新增指标：
  - `normal_consistency`
  - `fscore@0.02`
  - `fscore@0.05`
  - `fscore@0.10`
- 验收：各方法 `summary.json` 均含上述字段。

### 2.2 动态指标口径修正
- 文件：`scripts/run_benchmark.py`
- 新增指标：
  - `roi_ghost_ratio`（固定 ROI 分母）
  - `roi_background_recovery`
- 验收：`dynamic_metrics.csv` 同时保留 legacy + ROI 口径。

### 2.3 显著性检验脚本
- 新文件：`scripts/stats_significance.py`
- 输入：多序列、多 seed 结果表
- 输出：`output/journal_suite/tables/significance.csv`
- 最低内容：
  - paired t-test 或 Wilcoxon
  - p-value
  - effect size（Cohen's d 或 Cliff's delta）

---

## 3. P2 强基线补齐

### 3.1 本地基线统一多 seed
- 现有脚本：
  - `scripts/run_tsdf_baseline.py`
  - `scripts/run_simple_removal_baseline.py`
- 改造：支持批量 seed，统一输出结构与字段。

### 3.2 外部强基线适配（推荐适配器方案）
- 新文件：
  - `scripts/adapters/run_dynaslam_adapter.py`
  - `scripts/adapters/run_midfusion_adapter.py`
- 目标：统一导出点云/轨迹后接入同一评测脚本。
- 注意：文献抄数只可作为补充，不可替代主表。

### 3.3 最低基线集合（建议）
- TSDF
- Simple Removal
- DynaSLAM（适配器）
- MID-Fusion（适配器）
- iSDF / NICE-SLAM / Point-SLAM 任选其一（至少一个神经隐式对照）

### 3.4 P2 完成产物（已落地）
- 新增适配器脚本：
  - `scripts/adapters/run_dynaslam_adapter.py`
  - `scripts/adapters/run_midfusion_adapter.py`
  - `scripts/adapters/run_neural_implicit_adapter.py`
  - `scripts/adapters/_baseline_adapter_core.py`
- `scripts/run_benchmark.py` 已支持：
  - `--methods` 中加入 `dynaslam,midfusion,neural_implicit`
  - `--seeds` 多 seed 批跑（`seed_XXXX` 目录结构）
  - 多 seed 聚合表：`reconstruction_metrics_agg.csv`, `dynamic_metrics_agg.csv`
  - 适配器模板参数：`--dynaslam_*_template`, `--midfusion_*_template`
- 已验证产物目录：
  - `output/post_cleanup/p2_smoke/`（DynaSLAM + MID-Fusion 适配链路）
  - `output/post_cleanup/p2_neural_smoke/`（Neural 适配链路）
  - `output/post_cleanup/p2_multiseed/`（多 seed + 聚合表）
  - `output/post_cleanup/p2_local_multiseed/`（TSDF/Simple 多 seed 本地基线）
  - `output/post_cleanup/p2_verify_fullset/`（最低基线集合 6 方法联调）
  - `output/post_cleanup/p2_verify_multiseed/`（多 seed 聚合 + 显著性脚本复验）
  - `output/post_cleanup/p2_verify_skipcheck/`（`--external_allow_missing` 跳过路径复验）

---

## 4. P3 数据集扩展

### 4.1 TUM 扩展
- 动态序列扩到 >= 6 条。
- 保留静态序列 `rgbd_dataset_freiburg1_xyz` 用作静态性能守门。

### 4.2 Bonn 扩展
- 文件：`scripts/run_benchmark_bonn.py`, `scripts/data/bonn_rgbd.py`
- 至少 3 条 Bonn 动态序列（不止 `balloon2`）。

### 4.3 合成压力测试
- 新文件：`scripts/run_stress_synth.py`（或复用已有生成器）
- 扫描：动态比例 / 速度 / 遮挡强度。
- 输出：可控变量曲线与表格。

### 4.4 P3 完成产物（已落地）
- TUM 扩展结果（1 静态 + 6 动态）：
  - `output/post_cleanup/p3_tum_expanded/slam/tables/reconstruction_metrics.csv`
  - `output/post_cleanup/p3_tum_expanded/slam/tables/dynamic_metrics.csv`
  - `output/post_cleanup/p3_tum_expanded/slam/tables/benchmark_summary.json`
- Bonn 扩展结果（3 动态序列）：
  - `output/post_cleanup/p3_bonn_expanded/summary.csv`
  - `output/post_cleanup/p3_bonn_expanded/summary_agg.csv`
  - `output/post_cleanup/p3_bonn_expanded/figures/`
- 合成压力测试（扫描 3 维变量）：
  - `scripts/run_stress_synth.py`
  - `output/post_cleanup/p3_stress_synth/stress_summary.csv`
  - `output/post_cleanup/p3_stress_synth/stress_summary_agg.csv`
  - `output/post_cleanup/p3_stress_synth/stress_curves.png`

---

## 5. P4 深度消融（论文核心证据）

### 5.1 消融矩阵
- Full
- No-Evidence
- No-Gradient
- No-Uncertainty
- No-Local-Dynamic-Score
- No-Raycast
- Aggressive-Raycast
- No-Frontier/No-SeedFallback

### 5.2 多 seed 消融
- 每组消融 >= 5 seeds。
- 输出：`output/journal_suite/ablation/summary.csv`
- 每行包含 mean±std。

### 5.3 时间机制复验
- 文件：`scripts/run_temporal_ablation.py`
- 对每个帧数点跑多 seed，绘制误差棒。
- 输出图：
  - `temporal_convergence_curve.png`
  - `temporal_rho_evolution.png`

---

## 6. P5 一键全流程与自动报告

### 6.1 全流程脚本
- 新文件：`scripts/run_full_journal_suite.sh`
- 功能：一键跑 `oracle + slam`、`TUM + Bonn`、`baselines + ablations + temporal`。

### 6.2 汇总脚本升级
- 文件：`scripts/update_summary_tables.py`
- 输出（必须）：
  - `output/journal_suite/tables/main_recon.csv`
  - `output/journal_suite/tables/main_dynamic.csv`
  - `output/journal_suite/tables/ablation.csv`
  - `output/journal_suite/tables/significance.csv`

### 6.3 报告自动渲染（建议）
- 新文件：`scripts/render_benchmark_report.py`
- 目标：由 CSV 自动生成 `BENCHMARK_REPORT.md` 主表与关键结论。

---

## 7. 执行命令模板

### 7.1 快速冒烟（单序列）
```bash
python scripts/run_benchmark.py \
  --dataset_root data/tum \
  --static_sequences rgbd_dataset_freiburg1_xyz \
  --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz \
  --methods egf,tsdf,simple_removal \
  --protocol slam \
  --frames 40 \
  --stride 3 \
  --seed 7 \
  --out_root output/journal_suite_smoke/slam \
  --force
```

### 7.2 全量主实验
```bash
bash scripts/run_full_journal_suite.sh
```

### 7.3 汇总与统计
```bash
python scripts/update_summary_tables.py --root output/journal_suite
python scripts/stats_significance.py --root output/journal_suite
```

---

## 8. 里程碑与工时

### M1（2-3 天）
- 完成 P0 + P1
- 产物：协议分离结果 + 新指标字段 + 显著性脚本初版

### M2（7-14 天）
- 完成 P2 + P3
- 产物：强基线对照和扩展数据集结果

### M3（4-6 天）
- 完成 P4 + P5
- 产物：完整消融、全流程脚本、自动汇总报告

### 总工期
- 单人：约 2-4 周（取决于强基线适配难度）

---

## 9. 最终验收清单（勾选）
- [ ] Oracle/SLAM 双协议结果均完整。
- [ ] 强基线 >= 5 并统一评测口径。
- [ ] 多 seed + 显著性检验齐全。
- [ ] TUM + Bonn + 合成压力测试齐全。
- [ ] 关键结论有主表、消融、时间机制三重证据。
- [ ] 一键脚本可从空目录生成主结果与报告。
