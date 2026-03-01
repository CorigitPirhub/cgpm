# EGF-DHMap: Evidence-Gradient Field Dynamic Mapping

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Mapping](https://img.shields.io/badge/Mapping-3D-success)
![Dynamic](https://img.shields.io/badge/Dynamic-Scene%20Robustness-orange)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey)

EGF-DHMap 是一个面向动态场景的 3D 隐式建图原型：
- 使用证据场 `rho` 的时序累积进行动静分离（Time-Adaptive）；
- 使用梯度场约束保持几何一致性；
- 在不启用激进清理（`dyn_forget_gain=0`）时仍具备较强 Ghost 抑制能力。

合并文档入口：`MERGED_DOCS.md`（包含 README、设计、实现、基准与数据说明）。

> 当前仓库已完成结构精简：早期 2D/CGPM legacy 代码目录已移除，仅保留 3D 主线实现与复现实验脚本。

## 核心结果（置顶）

### Bonn 泛化结果（高动态）
在 `rgbd_bonn_balloon2` 上：
- EGF: `F-score = 0.9452`
- TSDF: `F-score = 0.5612`
- 相对提升：`(0.9452 - 0.5612) / 0.5612 = 68.4%`（约 69%）

![Bonn Comparison](assets/bonn_comparison.png)

### TUM 论文级对比图
![Final Comparison](assets/final_comparison_paper_ready.png)

### Static 场景修复（freiburg1_xyz）
- 问题：EGF 在纯静态场景精度低于 TSDF。
- 方法（SNEF）：`Static Narrow-Band Evidence Fusion`，仅在静态分支启用窄截断带和时间持久性门控（不改动态分支）。
- 结果：EGF `F-score 0.8416 -> 0.9410`，`Chamfer 0.0464 -> 0.0373`，达到目标 `F>=0.93, Chamfer<=0.040`。
- 动态约束：3 条 walking 的 `ghost_ratio` 仅小幅变化（`+0.00310/+0.00284/+0.00202`），均满足 `<= +0.03`。
- 结果目录：`output/post_cleanup/static_target_v1/oracle/`
- 约束汇总：`output/summary_tables/static_target_constraint_check.csv`

## Quick Start

### 0) 一键完整任务脚本（本地顶会标准套跑）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_local_top_tier_suite.py \
  --dataset_root_tum data/tum \
  --dataset_root_bonn data/bonn \
  --out_root output/post_cleanup/local_top_tier \
  --force
```
默认会顺序执行：主 benchmark、多 seed 显著性、消融、时间维分析、Bonn 泛化，并把关键表汇总到 `output/post_cleanup/local_top_tier/tables/`。

### 0.1) P0 协议锁定与可复验（oracle/slam + 双次复验）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_p0_protocol_lock.py \
  --dataset_root data/tum \
  --out_root output/post_cleanup/p0_protocol_lock \
  --summary_root output/summary_tables \
  --frames 20 --stride 5 --max_points_per_frame 1500 \
  --seed 7 --force
```
完成后查看：`output/post_cleanup/p0_protocol_lock/p0_report.json`。

### 0.2) P1 静态-动态平衡验收（自动停在达标配置）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_p1_balance.py \
  --use_existing_dir output/post_cleanup/p1_unified_v8_strictsame_refresh2 \
  --out_root output/post_cleanup/p1_balance_lock \
  --summary_root output/summary_tables \
  --target_static_fscore 0.90 \
  --target_walking_fscore 0.70 \
  --target_walking_ghost_tail_ratio 0.30
```
完成后查看：`output/post_cleanup/p1_balance_lock/p1_report.json`。

### 0.3) P2 机制验收（消融 + 时间机理）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_p2_mechanism.py \
  --ablation_csv output/summary_tables/ablation_summary.csv \
  --temporal_csv output/summary_tables/temporal_ablation_summary.csv \
  --out_root output/post_cleanup/p2_mechanism_lock \
  --summary_root output/summary_tables
```
完成后查看：`output/post_cleanup/p2_mechanism_lock/p2_report.json`。

### 1) 一行命令跑 TUM 基准（EGF vs TSDF）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark.py --dataset_kind tum --dataset_root data/tum --out_root output/post_cleanup/benchmark_tum --frames 80 --stride 3 --max_points_per_frame 3000 --voxel_size 0.02 --eval_thresh 0.05 --ghost_thresh 0.08 --bg_thresh 0.05 --methods egf,tsdf,simple_removal --force
```

### 1.1) SNEF 静态追平实验（含动态不退化约束）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark.py \
  --dataset_kind tum \
  --dataset_root data/tum \
  --out_root output/post_cleanup/static_target_v1 \
  --protocol oracle \
  --static_sequences rgbd_dataset_freiburg1_xyz \
  --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere \
  --frames 80 --stride 3 --max_points_per_frame 3000 \
  --voxel_size 0.02 --eval_thresh 0.05 --ghost_thresh 0.08 --bg_thresh 0.05 \
  --seed 42 --methods egf,tsdf,simple_removal \
  --egf_static_truncation 0.05 \
  --egf_static_surface_max_age_frames 60 \
  --egf_static_surface_no_zero_crossing \
  --force
```
核验表：`output/summary_tables/static_target_constraint_check.csv`（先执行 `scripts/update_summary_tables.py`）。

### 2) 一行命令跑时间维机理实验
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_temporal_ablation.py --dataset_root data/tum --sequence rgbd_dataset_freiburg3_walking_xyz --frames_list 15,30,45,60,90,120 --stride 3 --max_points_per_frame 3000 --voxel_size 0.02 --eval_thresh 0.05 --bg_thresh 0.10 --out_root output/post_cleanup/temporal_ablation --curve_png assets/temporal_convergence_curve.png --rho_png assets/temporal_rho_evolution.png --force
```

### 3) 一行命令跑 Bonn 泛化实验
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark_bonn.py --dataset_root data/bonn --sequence rgbd_bonn_balloon2 --frames 80 --stride 3 --max_points_per_frame 3000 --voxel_size 0.02 --eval_thresh 0.05 --bg_thresh 0.10 --out_root output/post_cleanup/benchmark_bonn --compare_png assets/bonn_comparison.png --force
```

### 4) 一键刷新 `output/summary_tables`
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/update_summary_tables.py --verbose
```

### 5) 多 seed + 强基线适配（DynaSLAM/MID-Fusion/Neural）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark.py \
  --dataset_root data/tum \
  --static_sequences rgbd_dataset_freiburg1_xyz \
  --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz \
  --methods egf,tsdf,simple_removal,dynaslam,midfusion,neural_implicit \
  --seeds 1,2,3 \
  --protocol slam \
  --out_root output/post_cleanup/benchmark_multiseed \
  --dynaslam_pred_points_template \"{project_root}/external_preds/dynaslam/{sequence}/seed_{seed}/surface_points.ply\" \
  --midfusion_pred_points_template \"{project_root}/external_preds/midfusion/{sequence}/seed_{seed}/surface_points.ply\" \
  --neural_pred_points_template \"{project_root}/external_preds/neural/{sequence}/seed_{seed}/surface_points.ply\"
```
若外部基线结果尚未准备，可加 `--external_allow_missing` 让适配器跳过并继续跑其余方法。

### 6) P3: TUM 扩展（1 静态 + 6 动态）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark.py \
  --dataset_kind tum \
  --dataset_root data/tum \
  --static_sequences rgbd_dataset_freiburg1_xyz \
  --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere,rgbd_dataset_freiburg2_desk_with_person,rgbd_dataset_freiburg3_sitting_xyz,rgbd_dataset_freiburg3_sitting_static \
  --methods egf,tsdf,simple_removal \
  --frames 60 --stride 4 --max_points_per_frame 2200 \
  --voxel_size 0.02 --eval_thresh 0.05 --ghost_thresh 0.08 --bg_thresh 0.05 \
  --protocol slam --seed 7 \
  --out_root output/post_cleanup/p3_tum_expanded
```

### 7) P3: Bonn 扩展（3 动态序列）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark_bonn.py \
  --dataset_root data/bonn \
  --sequences rgbd_bonn_balloon2,rgbd_bonn_balloon,rgbd_bonn_crowd2 \
  --download \
  --frames 80 --stride 3 --max_points_per_frame 3000 \
  --voxel_size 0.02 --eval_thresh 0.05 --ghost_thresh 0.08 --bg_thresh 0.10 \
  --out_root output/post_cleanup/p3_bonn_expanded
```

### 8) P3: 合成压力测试（动态比例/速度/遮挡）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_stress_synth.py \
  --dataset_root data/synth_stress \
  --out_root output/post_cleanup/p3_stress_synth \
  --frames 36 --stride 1 --max_points_per_frame 2200 \
  --voxel_size 0.03 --eval_thresh 0.05 --seed 7 --force
```

### 9) P0: 真 SLAM 口径验收（不使用 GT 增量）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark.py \
  --dataset_kind tum \
  --dataset_root data/tum \
  --protocol slam \
  --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere \
  --methods egf \
  --frames 120 --stride 1 --max_points_per_frame 3000 \
  --voxel_size 0.02 --eval_thresh 0.05 --ghost_thresh 0.08 --bg_thresh 0.05 \
  --seed 7 --egf_sigma_n0 0.26 --egf_slam_no_gt_delta_odom \
  --egf_rho_decay 0.97 --egf_phi_w_decay 0.97 \
  --egf_forget_mode global --egf_dyn_forget_gain 0.35 \
  --egf_raycast_clear_gain 0.20 \
  --egf_surface_max_age_frames 12 --egf_surface_max_dscore 0.75 \
  --egf_surface_max_free_ratio 0.7 --egf_surface_prune_free_min 1.0 \
  --egf_surface_prune_residual_min 0.2 --egf_surface_max_clear_hits 6 \
  --out_root output/post_cleanup/p0_true_slam_v3_final --force
```
验收结果见 `output/post_cleanup/p0_true_slam_v3_final/slam/tables/reconstruction_metrics.csv`：
- `walking_xyz`: EGF `F-score=0.7033`, `ghost_tail_ratio=0.0171`
- `walking_static`: EGF `F-score=0.7954`, `ghost_tail_ratio=0.0372`
- `walking_halfsphere`: EGF `F-score=0.6582`, `ghost_tail_ratio=0.0180`
- 轨迹稳定性：`traj_finite_ratio=1.0`，并输出 `ATE/RPE` 字段（同表）。

### 10) P1: 静态-动态同参自适应（同一参数集）
- 参数集：`sigma_n0=0.26, surface_max_age_frames=16, surface_max_dscore=0.60`（并显式将 `egf_static_*` 设为相同值）。
- 结果目录：`output/post_cleanup/p1_unified_v8_strictsame_refresh2/slam/tables/reconstruction_metrics.csv`
- 验收：
  - `freiburg1_xyz`: `F-score=0.9010`（>=0.90）
  - `walking_xyz`: `F-score=0.7298`（>=0.70）, `ghost_tail_ratio=0.0500`（<=0.05）

### 11) P3: 外部基线去适配器化（严格真实输出）
当前 `dynaslam/midfusion/neural_implicit` 支持统一评估链路对接；占位路径在严格模式会被拒绝。
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark.py \
  --dataset_kind tum --dataset_root data/tum --protocol slam \
  --static_sequences rgbd_dataset_freiburg1_xyz --dynamic_sequences '' \
  --methods egf,tsdf,simple_removal,dynaslam \
  --frames 20 --stride 3 --max_points_per_frame 1200 \
  --voxel_size 0.03 --eval_thresh 0.05 --ghost_thresh 0.08 --bg_thresh 0.05 \
  --seed 7 --out_root output/post_cleanup/p2_strict_gate_check \
  --dynaslam_pred_points_template "{out_dir}/../simple_removal/surface_points.ply" \
  --external_require_real --force
```
该命令会报错：`dynaslam source is placeholder/non-real`，用于阻止占位结果进入主表。要完成 P3，需要替换为真实 DynaSLAM/MID-Fusion/Neural 输出路径或各自 runner。

### 12) P3: 一键接入真实外部输出（NICE-SLAM）
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_p3_real_external.py \
  --dataset_root data/tum \
  --sequence rgbd_dataset_freiburg3_walking_xyz \
  --subset_frames 10 \
  --frames 10 --stride 1 \
  --protocol slam \
  --out_root output/post_cleanup/p3_real_external_niceslam \
  --summary_root output/summary_tables \
  --ensure_rtree
```
完成后查看：
- `output/post_cleanup/p3_real_external_niceslam/slam/P3_REAL_EXTERNAL_REPORT.md`
- `output/summary_tables/p3_real_external_reconstruction.csv`
- `output/summary_tables/p3_real_external_dynamic.csv`

## 项目结构（精简后）
- `egf_dhmap3d/`: 3D 核心实现（数据结构、预测、关联、更新、评估）。
- `scripts/`: 实验与绘图入口（benchmark/temporal/bonn/ablation）。
- `scripts/adapters/`: 外部强基线适配器（DynaSLAM/MID-Fusion/Neural）。
- `data/`: TUM 与 Bonn 数据目录。
- `assets/`: 论文级图表。
- `output/post_cleanup/`: 删除 legacy 后的全量复现结果。
- `output/summary_tables/`: 最终汇总 CSV（推荐对外分发）。

## 当前推荐入口脚本
- `scripts/run_benchmark.py`: 通用基准入口（支持 `--dataset_kind tum|bonn|auto`）
  - 支持 `--seeds` 多 seed 批跑与聚合表输出（`*_agg.csv`）
  - 支持 `dynaslam/midfusion/neural_implicit` 适配方法
- `scripts/run_temporal_ablation.py`: 时间维机理分析（收敛曲线 + rho 演化）
- `scripts/run_benchmark_bonn.py`: Bonn 动态序列专项对比

## 结果文件
- 主报告：`BENCHMARK_REPORT.md`
- 合并总文档：`MERGED_DOCS.md`
- TUM 汇总：`output/summary_tables/tum_reconstruction_metrics.csv`, `output/summary_tables/tum_dynamic_metrics.csv`
- TUM 静态修复专项：`output/summary_tables/tum_reconstruction_metrics_static_fix.csv`, `output/summary_tables/tum_dynamic_metrics_static_fix.csv`
- 时间维 CSV：`output/summary_tables/temporal_ablation_summary.csv`
- Bonn CSV：`output/summary_tables/bonn_summary.csv`
- 消融 CSV：`output/summary_tables/ablation_summary.csv`
- P3 TUM 扩展：`output/post_cleanup/p3_tum_expanded/slam/tables/reconstruction_metrics.csv`, `output/post_cleanup/p3_tum_expanded/slam/tables/dynamic_metrics.csv`
- P3 Bonn 扩展：`output/post_cleanup/p3_bonn_expanded/summary.csv`, `output/post_cleanup/p3_bonn_expanded/summary_agg.csv`
- P3 合成压力：`output/post_cleanup/p3_stress_synth/stress_summary.csv`, `output/post_cleanup/p3_stress_synth/stress_summary_agg.csv`, `output/post_cleanup/p3_stress_synth/stress_curves.png`
- P3 真实外部输出：`output/post_cleanup/p3_real_external_niceslam/slam/tables/reconstruction_metrics.csv`, `output/post_cleanup/p3_real_external_niceslam/slam/tables/dynamic_metrics.csv`
- 时间收敛图：`assets/temporal_convergence_curve.png`
- rho 演化图：`assets/temporal_rho_evolution.png`
- Bonn 对比图：`assets/bonn_comparison.png`

## 视频产物说明
当前默认实验脚本不生成视频文件（无 `.mp4/.avi/.gif` 输出）；默认产物是 `tables/*.csv`、`summary.json`、`*.ply` 与 `assets/*.png`。
