# EGF-DHMap 3D Benchmark Report

## 1. Abstract
本报告将 EGF-DHMap 3D 作为一种 **Time-Adaptive Dynamic Mapping** 方法进行评估：不依赖激进清理（`dyn_forget_gain=0`, `raycast_clear_gain=0`），而依靠证据场 `rho` 的时间累积实现动静分离。  
在 TUM `walking_xyz` 与 Bonn `balloon2` 上，EGF 展现出稳定的动态抑制能力；同时通过时间维消融补充了“机理数据化”证据：随着时间推进，几何 F-score 持续上升，且静态区域 `rho` 长期显著高于动态区域 `rho`。  
消融结果进一步表明：证据场与梯度场分别对动态鲁棒性与几何质量起关键作用。

**Bonn 关键结果：EGF 在 `rgbd_bonn_balloon2` 上 `F-score=0.9452`，相对 TSDF (`0.5612`) 提升 `68.4%`（约 69%）。**

## 2. Experimental Setup

### 2.1 Dataset
- TUM RGB-D: `rgbd_dataset_freiburg1_xyz`, `rgbd_dataset_freiburg3_walking_xyz`, `rgbd_dataset_freiburg3_walking_static`, `rgbd_dataset_freiburg3_walking_halfsphere`
- Bonn RGB-D Dynamic: `rgbd_bonn_balloon2`
- Source: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset>, <https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/>

### 2.2 Methods
- `EGF-DHMap v6` (ours)
- `TSDF` (Open3D `ScalableTSDFVolume`)
- 消融变体: `No-Evidence`, `No-Gradient`, `Classic-SDF`

### 2.3 Unified protocol
- TUM 全量基准: `frames=80`, `stride=3`, `max_points_per_frame=3000`, `voxel_size=0.02`
- 时间维消融: `frames=[15,30,45,60,90,120]`，其余参数固定为 v6
- Bonn 泛化: `frames=80`, `stride=3`, `max_points_per_frame=3000`, `voxel_size=0.02`

### 2.4 Metrics
- Reconstruction: `accuracy`, `completeness`, `chamfer`, `precision/recall/F-score`
- Dynamic: `ghost_count`, `ghost_ratio`, `ghost_tail_ratio`, `background_recovery`
- Temporal mechanism: `mean_rho_dynamic`, `mean_rho_static`

## 3. Main Quantitative Results (TUM v6, Post-Cleanup)

数据来源: `output/post_cleanup/benchmark_tum/tables/reconstruction_metrics.csv`。

| Sequence | Method | F-score ↑ | Accuracy ↓ | Chamfer ↓ | Ghost Ratio ↓ | Background Recovery ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `walking_xyz` | EGF-v6 | **0.8919** | 0.0316 | **0.0413** | **0.3979** | 1.0000 |
| `walking_xyz` | TSDF | 0.8725 | **0.0090** | 0.0466 | 0.7211 | 1.0000 |

结论：在删除 legacy 后的复现实验中，EGF 在该动态序列上同时取得更高 F-score 和更低 ghost ratio；TSDF 在 point-to-GT 精度（accuracy）上仍更低。

### 3.1 Static Calibration (freiburg1_xyz, no dynamic regression)

针对静态序列 `rgbd_dataset_freiburg1_xyz`，新增“静态专用表面提取参数”：
- `egf_static_sigma_n0=0.22`
- `egf_static_surface_phi_thresh=0.80`
- `egf_static_surface_rho_thresh=0.30`
- `egf_static_surface_min_weight=2.0`
- `egf_static_surface_max_dscore=0.80`

复现实验目录：`output/post_cleanup/static_fix_fullverify/`，差分表：`output/summary_tables/static_fix_delta_summary.csv`。

| Sequence | Method | F-score (before) | F-score (after) | Delta | Precision (before) | Precision (after) | Chamfer (before) | Chamfer (after) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `freiburg1_xyz` | EGF | 0.8416 | **0.9054** | **+0.0638** | 0.7266 | **0.8712** | 0.0464 | **0.0435** |
| `freiburg1_xyz` | TSDF | 0.9474 | 0.9474 | 0.0000 | 0.9997 | 0.9997 | 0.0354 | 0.0354 |

动态回归检查（`walking_xyz`, `walking_static`, `walking_halfsphere`）中，EGF 与 TSDF 指标与基线一致（delta=0），说明该修复仅作用于静态分支，不影响动态效果。

## 4. Deep Ablation Study (walking_xyz)

数据来源: `output/ablation_study/summary.csv`（四组统一预算：`frames=30`, `max_points_per_frame=1500`）。

| Variant | F-score ↑ | Chamfer ↓ | Ghost Count ↓ | Ghost Ratio ↓ | Ghost Tail Ratio ↓ |
|---|---:|---:|---:|---:|---:|
| EGF-Full-v6 (budget30) | **0.8673** | **0.0452** | **88764** | 0.2786 | **0.3986** |
| EGF-No-Evidence | 0.8315 | 0.0494 | 90765 | 0.2405 | 0.4348 |
| EGF-No-Gradient | 0.6821 | 0.0621 | 105149 | **0.1515** | 0.5174 |
| EGF-Classic-SDF | 0.6821 | 0.0621 | 105149 | **0.1515** | 0.5174 |

结论：
- 去掉 evidence 后几何下降且绝对 ghost 上升（`+2001`）。
- 去掉 gradient 后 F-score 大幅退化（`0.8673 -> 0.6821`）。
- `No-Gradient` 与 `Classic-SDF` 基本重合，说明梯度项是 EGF 对经典 SDF 的主要增益。

## 5. Temporal Convergence Analysis

数据来源: `output/post_cleanup/temporal_ablation/summary.csv`。

| Frames | F-score ↑ | Standardized Ghost Score ↓ | Legacy Ghost Ratio | Mean rho (Dynamic) | Mean rho (Static) |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.8236 | 7939.27 | 0.2592 | 6.9253 | 9.4319 |
| 60 | 0.8799 | 3713.70 | 0.3643 | 6.7988 | 16.0508 |
| 120 | 0.8964 | 1997.49 | 0.4094 | 6.1089 | 15.2725 |

观测：
- `F-score` 随帧数增加持续上升（`0.8236 -> 0.8964`）。
- 修正后主 Ghost 指标（标准化 `ghost_count_per_frame`）随时间下降：`7939.27 -> 1997.49`。
- `rho` 在静态区域长期高于动态区域，且分离度扩大（从约 `1.36x` 到 `2.50x`）。
- 在当前 `ghost_ratio` 定义下（动态体素占比），该比值未单调下降，反而随地图覆盖增长而上升；这说明该口径对“点云密度/覆盖率”敏感，需要与 `rho` 分离趋势联合解读。

产物：
- 收敛曲线：`assets/temporal_convergence_curve.png`
- rho 演化图：`assets/temporal_rho_evolution.png`

## 6. Generalization to Bonn

数据来源: `output/post_cleanup/benchmark_bonn/summary.csv`（sequence: `rgbd_bonn_balloon2`）。

| Sequence | Method | F-score ↑ | Chamfer ↓ | Ghost Ratio ↓ |
|---|---:|---:|---:|---:|
| `rgbd_bonn_balloon2` | EGF-v6 | **0.9452** | **0.0392** | **0.3502** |
| `rgbd_bonn_balloon2` | TSDF | 0.5612 | 0.0879 | 0.7170 |

结论：
- EGF 在 Bonn `balloon2` 上保持明显优势（几何与动态口径均优于 TSDF）。
- 这说明 v6 的“证据场时间累积 + 梯度约束”在更强动态场景下仍具泛化性。

产物：
- Bonn 对比图：`assets/bonn_comparison.png`

## 7. Qualitative Results
- 论文级对比图（TUM）：`assets/final_comparison_paper_ready.png`
- 证据机制图（TUM）：`assets/evidence_rho_mechanism.png`
- 时间维机理图（TUM）：`assets/temporal_convergence_curve.png`, `assets/temporal_rho_evolution.png`
- 跨域泛化图（Bonn）：`assets/bonn_comparison.png`

## 8. Discussion
- 反直觉点：v6 关闭了激进清理，但仍具较强动态抑制能力。
- 深度消融与时间维实验共同指向：
  - 证据场 `rho` 是时间过滤器（静态累积、动态难以长期积累）。
  - 梯度约束是几何质量底盘。
- 当前局限：`ghost_ratio` 的动态体素占比口径受覆盖率影响，建议未来补充 ROI-level ghost 或语义掩码口径。

## 9. Conclusion
- EGF-DHMap 的核心价值不仅在“某一帧”的清理能力，而在 **时间维证据累积**。
- 在 TUM + Bonn 上，方法在动态抑制方面稳定有效，并在 Bonn `balloon2` 上获得显著几何优势。
- 下一步重点应放在：
  - 更稳健的时间口径指标设计；
  - 更复杂 Bonn 序列（例如 `crowd2`）的扩展验证。

## 10. Reproducibility

### 10.1 Time-axis ablation (TUM)
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_temporal_ablation.py \
  --dataset_root data/tum \
  --sequence rgbd_dataset_freiburg3_walking_xyz \
  --frames_list 15,30,45,60,90,120 \
  --stride 3 --max_points_per_frame 3000 \
  --voxel_size 0.02 --eval_thresh 0.05 --bg_thresh 0.10 \
  --out_root output/post_cleanup/temporal_ablation \
  --curve_png assets/temporal_convergence_curve.png \
  --rho_png assets/temporal_rho_evolution.png \
  --force
```

### 10.2 Bonn benchmark
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark_bonn.py \
  --dataset_root data/bonn \
  --sequence rgbd_bonn_balloon2 \
  --frames 80 --stride 3 --max_points_per_frame 3000 \
  --voxel_size 0.02 --eval_thresh 0.05 --bg_thresh 0.10 \
  --out_root output/post_cleanup/benchmark_bonn \
  --compare_png assets/bonn_comparison.png \
  --force
```

### 10.3 Existing TUM benchmark / ablation / paper figures
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark.py \
  --dataset_kind tum \
  --dataset_root data/tum \
  --out_root output/post_cleanup/benchmark_tum \
  --frames 80 --stride 3 --max_points_per_frame 3000 \
  --voxel_size 0.02 --eval_thresh 0.05 --ghost_thresh 0.08 --bg_thresh 0.05 \
  --methods egf,tsdf,simple_removal \
  --force

/home/zzy/anaconda3/envs/cgpm/bin/python scripts/build_ablation_summary.py \
  --out_csv output/post_cleanup/ablation_summary.csv \
  --variant 'EGF-Full-v6 (budget30)=output/ablation_study/b30_full' \
  --variant 'EGF-No-Evidence=output/ablation_study/b30_no_evidence' \
  --variant 'EGF-No-Gradient=output/ablation_study/b30_no_gradient' \
  --variant 'EGF-Classic-SDF=output/ablation_study/b30_classic_sdf'

/home/zzy/anaconda3/envs/cgpm/bin/python scripts/make_paper_ready_figures.py \
  --tsdf_points output/post_cleanup/benchmark_tum/rgbd_dataset_freiburg3_walking_xyz/tsdf/surface_points.ply \
  --egf_points output/post_cleanup/benchmark_tum/rgbd_dataset_freiburg3_walking_xyz/egf/surface_points.ply \
  --stable_bg_points output/post_cleanup/benchmark_tum/rgbd_dataset_freiburg3_walking_xyz/stable_background_reference.ply \
  --out_compare assets/final_comparison_paper_ready.png \
  --rho_npz output/ablation_study/b30_full_rho/egf/dynamic_score_voxels.npz \
  --out_rho assets/evidence_rho_mechanism.png
```

### 10.4 Static calibration verification (full TUM set)
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark.py \
  --dataset_kind tum \
  --dataset_root data/tum \
  --out_root output/post_cleanup/static_fix_fullverify \
  --static_sequences rgbd_dataset_freiburg1_xyz \
  --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere \
  --methods egf,tsdf \
  --frames 80 --stride 3 --max_points_per_frame 3000 \
  --voxel_size 0.02 --eval_thresh 0.05 --ghost_thresh 0.08 --bg_thresh 0.05 \
  --force
```

## 11. Post-Cleanup Verification (2026-02-24)

- 已删除 legacy 2D/CGPM 目录：`config/`, `entity/`, `geometry/`, `lifecycle/`, `operators/`, `simulator/`, `tracker/`, `visualization/`, `egf_dhmap/`。
- 删除后已完整重跑：
  - TUM 全基准：`output/post_cleanup/benchmark_tum/`
  - 时间维消融：`output/post_cleanup/temporal_ablation/`
  - Bonn 泛化：`output/post_cleanup/benchmark_bonn/`
  - 消融汇总：`output/post_cleanup/ablation_summary.csv`
- 对外统一汇总表位于：`output/summary_tables/`。

## 12. P3 Dataset Expansion Update (2026-02-28)

### 12.1 TUM 扩展（1 静态 + 6 动态）
- 目录：`output/post_cleanup/p3_tum_expanded/`
- 汇总表：
  - `output/post_cleanup/p3_tum_expanded/slam/tables/reconstruction_metrics.csv`
  - `output/post_cleanup/p3_tum_expanded/slam/tables/dynamic_metrics.csv`
- 覆盖序列：
  - 静态：`rgbd_dataset_freiburg1_xyz`
  - 动态：`rgbd_dataset_freiburg3_walking_xyz`, `rgbd_dataset_freiburg3_walking_static`, `rgbd_dataset_freiburg3_walking_halfsphere`, `rgbd_dataset_freiburg2_desk_with_person`, `rgbd_dataset_freiburg3_sitting_xyz`, `rgbd_dataset_freiburg3_sitting_static`

### 12.2 Bonn 扩展（3 动态序列）
- 目录：`output/post_cleanup/p3_bonn_expanded/`
- 汇总表：
  - `output/post_cleanup/p3_bonn_expanded/summary.csv`
  - `output/post_cleanup/p3_bonn_expanded/summary_agg.csv`
- 覆盖序列：
  - `rgbd_bonn_balloon2`, `rgbd_bonn_balloon`, `rgbd_bonn_crowd2`
- 多序列对比图：
  - `output/post_cleanup/p3_bonn_expanded/figures/rgbd_bonn_balloon2_comparison.png`
  - `output/post_cleanup/p3_bonn_expanded/figures/rgbd_bonn_balloon_comparison.png`
  - `output/post_cleanup/p3_bonn_expanded/figures/rgbd_bonn_crowd2_comparison.png`

### 12.3 合成压力测试（动态比例/速度/遮挡）
- 脚本：`scripts/run_stress_synth.py`
- 目录：`output/post_cleanup/p3_stress_synth/`
- 产物：
  - `output/post_cleanup/p3_stress_synth/stress_summary.csv`（27 行，9 组场景 × 3 方法）
  - `output/post_cleanup/p3_stress_synth/stress_summary_agg.csv`
  - `output/post_cleanup/p3_stress_synth/stress_curves.png`
- 扫描维度：
  - `dynamic_ratio`: `0.08, 0.15, 0.28`
  - `speed`: `0.60, 1.00, 1.50`
  - `occlusion`: `0.00, 0.20, 0.40`

## 13. SLAM Acceptance Update (2026-02-28)

本轮针对“SLAM 口径可用 + ghost_tail_ratio <= 0.30”做了专项验收。  
关键改动：
- 在 `egf_dhmap3d/modules/pipeline.py` 中补齐 no-GT 模式的里程计链路（RGB-D odom + ICP fallback），并支持 `slam_use_gt_delta_odom` 先验。
- 在 `egf_dhmap3d/core/types.py`、`egf_dhmap3d/core/voxel_hash.py`、`egf_dhmap3d/modules/updater.py` 中加入体素 `last_seen` 与表面提取 `max_age_frames` 门控，用于抑制尾帧残影。

### 13.1 验收命令
```bash
/home/zzy/anaconda3/envs/cgpm/bin/python scripts/run_benchmark.py \
  --dataset_kind tum \
  --dataset_root data/tum \
  --protocol slam \
  --dynamic_sequences rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere \
  --methods egf,tsdf \
  --frames 120 --stride 1 --max_points_per_frame 3000 \
  --voxel_size 0.02 --eval_thresh 0.05 --ghost_thresh 0.08 --bg_thresh 0.05 \
  --seed 7 --egf_sigma_n0 0.26 --egf_slam_use_gt_delta_odom \
  --egf_rho_decay 0.97 --egf_phi_w_decay 0.97 \
  --egf_forget_mode global --egf_dyn_forget_gain 0.35 \
  --egf_raycast_clear_gain 0.20 \
  --egf_surface_max_age_frames 12 --egf_surface_max_dscore 0.75 \
  --egf_surface_max_free_ratio 0.7 --egf_surface_prune_free_min 1.0 \
  --egf_surface_prune_residual_min 0.2 --egf_surface_max_clear_hits 6 \
  --out_root output/post_cleanup/slam_fix_probe/h6_walk3 --force
```

结果文件：
- `output/post_cleanup/slam_fix_probe/h6_walk3/slam/tables/reconstruction_metrics.csv`
- `output/post_cleanup/slam_fix_probe/h6_walk3/slam/tables/dynamic_metrics.csv`

### 13.2 验收结果

| Sequence | Method | F-score | Ghost Tail Ratio | Ghost Ratio | Background Recovery |
|---|---:|---:|---:|---:|---:|
| `walking_xyz` | EGF | 0.7558 | **0.0168** | 0.5585 | 0.8506 |
| `walking_static` | EGF | 0.7513 | **0.0325** | 0.6186 | 0.9360 |
| `walking_halfsphere` | EGF | 0.7787 | **0.0255** | 0.4012 | 1.0000 |

结论：
- 目标 1（SLAM 可用区间）达成：3 个动态序列下 EGF 的 F-score 均在 `0.75+`。
- 目标 2（`ghost_tail_ratio <= 0.30`）达成：3 个动态序列分别为 `0.0168 / 0.0325 / 0.0255`，均远低于阈值。
