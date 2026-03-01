# EGF-DHMap 3D Benchmark Report

## 1. Abstract
本报告将 EGF-DHMap 3D 作为一种 **Time-Adaptive Dynamic Mapping** 方法进行评估：不依赖激进清理（`dyn_forget_gain=0`, `raycast_clear_gain=0`），而依靠证据场 `rho` 的时间累积实现动静分离。  
在 TUM `walking_xyz` 与 Bonn `balloon2` 上，EGF 展现出稳定的动态抑制能力；同时通过时间维消融补充了“机理数据化”证据：随着时间推进，几何 F-score 持续上升，且静态区域 `rho` 长期显著高于动态区域 `rho`。  
消融结果进一步表明：证据场与梯度场分别对动态鲁棒性与几何质量起关键作用。  
针对“静态场景弱于 TSDF”的短板，新增 **SNEF (Static Narrow-Band Evidence Fusion)**：仅在静态分支启用窄截断带与时间持久性约束，动态分支参数保持不变。在 `freiburg1_xyz` 上将 EGF 提升至 `F-score=0.9410`, `Chamfer=0.0373`，同时 3 条 `walking` 的 `ghost_ratio` 增量均 `< +0.004`（约束为 `+0.03`）。

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

### 2.3 SNEF Method (paper-ready)

为解决“动态友好参数在纯静态场景引入额外噪声”的问题，我们在 EGF 的静态分支加入
**SNEF (Static Narrow-Band Evidence Fusion)**，动态分支保持原配置不变。

设体素为 `x`，第 `t` 帧投影得到该体素对应的观测有符号距离 `d_t(x)`，状态为
`(phi_t(x), W_t(x), rho_t(x), a_t(x))`，其中 `a_t` 为体素“未被稳定支持”的年龄（帧数）。

窄带门控定义为：

\[
g_{\text{nb}}(x,t) = \mathbf{1}\!\left(\left|d_t(x)\right|\le \tau_s\right),\ \tau_s < \tau_d
\tag{1}
\]

证据门控与年龄门控分别为：

\[
g_{\rho}(x,t)=\sigma\!\left(\frac{\rho_{t-1}(x)-\rho_0}{\beta_{\rho}}\right),\quad
g_{a}(x,t)=\exp\!\left(-\frac{a_{t-1}(x)}{\kappa_a}\right)\mathbf{1}(a_{t-1}(x)\le A_s)
\tag{2}
\]

融合权重：

\[
w_t(x)=w_{\text{obs}}(x,t)\cdot g_{\text{nb}}(x,t)\cdot g_{\rho}(x,t)\cdot g_{a}(x,t)
\tag{3}
\]

静态分支的 SDF 加权更新：

\[
W_t(x)=\lambda_w W_{t-1}(x)+w_t(x)
\tag{4}
\]
\[
\phi_t(x)=\frac{\lambda_w W_{t-1}(x)\phi_{t-1}(x)+w_t(x)d_t(x)}{W_t(x)+\varepsilon}
\tag{5}
\]

证据与年龄状态演化：

\[
\rho_t(x)=\lambda_{\rho}\rho_{t-1}(x)+\eta_{\rho}\,g_{\text{nb}}(x,t),\quad
a_t(x)=
\begin{cases}
0,& g_{\text{nb}}(x,t)=1\\
\min(a_{t-1}(x)+1,\ A_{\max}),& \text{otherwise}
\end{cases}
\tag{6}
\]

最终静态表面提取集合为：

\[
\mathcal{S}_t=
\left\{
x\ \middle|\
\left|\phi_t(x)\right|\le \tau_{\phi},\ 
\rho_t(x)\ge \tau_{\rho},\ 
a_t(x)\le A_s
\right\}
\tag{7}
\]

本工作对应的关键超参为 `tau_s=0.05`（`egf_static_truncation`）与 `A_s=60`
（`egf_static_surface_max_age_frames`）。

```text
Algorithm 1: SNEF static-branch update
Input: frame (P_t, N_t, T_t), map M_{t-1}, params (tau_s, A_s, lambda_w, lambda_rho, ...)
Output: updated static map M_t and extracted static surface S_t

1: for each observed sample p in frame t do
2:     project p to voxel x and compute signed distance d_t(x)
3:     g_nb <- 1(|d_t(x)| <= tau_s)
4:     if g_nb == 0 then continue
5:     g_rho <- sigmoid((rho_{t-1}(x)-rho0)/beta_rho)
6:     g_a <- exp(-a_{t-1}(x)/kappa_a) * 1(a_{t-1}(x) <= A_s)
7:     w <- w_obs * g_nb * g_rho * g_a
8:     W_t(x) <- lambda_w * W_{t-1}(x) + w
9:     phi_t(x) <- (lambda_w*W_{t-1}(x)*phi_{t-1}(x) + w*d_t(x)) / (W_t(x)+eps)
10:    rho_t(x) <- lambda_rho * rho_{t-1}(x) + eta_rho * g_nb
11:    a_t(x) <- 0
12: end for
13: for each active voxel x not updated at frame t do
14:    rho_t(x) <- lambda_rho * rho_{t-1}(x)
15:    a_t(x) <- min(a_{t-1}(x)+1, A_max)
16: end for
17: S_t <- {x: |phi_t(x)|<=tau_phi and rho_t(x)>=tau_rho and a_t(x)<=A_s}
18: return M_t, S_t
```

设计要点：
- SNEF 只作用在静态分支，避免影响动态场景 ghost 抑制链路。
- `tau_s` 控制“只在零水平集附近融合”，提升静态精度。
- `A_s` 控制“短时异常不过度持久化”，降低静态伪表面累积。

### 2.4 Unified protocol
- TUM 全量基准: `frames=80`, `stride=3`, `max_points_per_frame=3000`, `voxel_size=0.02`
- 时间维消融: `frames=[15,30,45,60,90,120]`，其余参数固定为 v6
- Bonn 泛化: `frames=80`, `stride=3`, `max_points_per_frame=3000`, `voxel_size=0.02`

### 2.5 Metrics
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

### 3.1 SNEF: Static Narrow-Band Evidence Fusion (freiburg1_xyz)

为补齐静态短板，同时不破坏动态优势，我们引入 SNEF（可独立成节）：
- 机制 1：静态分支窄截断带（`egf_static_truncation=0.05`），抑制远离零水平集的噪声注入。
- 机制 2：静态分支时间持久性门控（`egf_static_surface_max_age_frames=60`），削弱短时异常点对表面的污染。
- 机制 3：动态分支参数保持不变，确保动态抑制特性不被“静态补丁”反向破坏。

复现实验目录：`output/post_cleanup/static_target_v1/oracle/`；约束核验表：`output/summary_tables/static_target_constraint_check.csv`。

| Sequence | Method | F-score (base) | F-score (SNEF) | Delta | Chamfer (base) | Chamfer (SNEF) | Target Check |
|---|---:|---:|---:|---:|---:|---:|---:|
| `freiburg1_xyz` | EGF | 0.8416 | **0.9410** | **+0.0994** | 0.0464 | **0.0373** | ✅ `F>=0.93`, ✅ `C<=0.040` |
| `freiburg1_xyz` | TSDF | 0.9474 | 0.9474 | 0.0000 | 0.0355 | 0.0355 | - |

动态非退化约束（EGF `ghost_ratio_new - ghost_ratio_base <= +0.03`）：
- `walking_xyz`: `+0.00310` ✅
- `walking_static`: `+0.00284` ✅
- `walking_halfsphere`: `+0.00202` ✅

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

### 10.5 SNEF static-target verification (paper target run)
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

/home/zzy/anaconda3/envs/cgpm/bin/python scripts/update_summary_tables.py --verbose
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

## 13. SLAM Acceptance Update (2026-03-01)

本轮针对“SLAM 口径可用 + ghost_tail_ratio <= 0.30”做了专项验收。  
关键改动：
- 在 `egf_dhmap3d/modules/pipeline.py` 中补齐 no-GT 模式的里程计链路（RGB-D odom + ICP fallback），并显式支持 `--slam_no_gt_delta_odom`。
- 在 `egf_dhmap3d/core/types.py`、`egf_dhmap3d/core/voxel_hash.py`、`egf_dhmap3d/modules/updater.py` 中加入体素 `last_seen` 与表面提取 `max_age_frames` 门控，用于抑制尾帧残影。

### 13.1 验收命令
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

结果文件：
- `output/post_cleanup/p0_true_slam_v3_final/slam/tables/reconstruction_metrics.csv`
- `output/post_cleanup/p0_true_slam_v3_final/slam/tables/dynamic_metrics.csv`

### 13.2 验收结果

| Sequence | Method | F-score | Ghost Tail Ratio | Ghost Ratio | Background Recovery |
|---|---:|---:|---:|---:|---:|
| `walking_xyz` | EGF | 0.7033 | **0.0171** | 0.6349 | 0.9824 |
| `walking_static` | EGF | 0.7954 | **0.0372** | 0.5258 | 0.7638 |
| `walking_halfsphere` | EGF | 0.6582 | **0.0180** | 0.6716 | 0.9274 |

结论：
- 目标 1（SLAM 可用区间）达成：3 个动态序列下 EGF 的 F-score 均 `>=0.65`。
- 目标 2（`ghost_tail_ratio <= 0.30`）达成：3 个动态序列分别为 `0.0171 / 0.0372 / 0.0180`，均远低于阈值。
- 轨迹稳定性达成：`traj_finite_ratio=1.0`，并输出 `ATE/RPE`（`reconstruction_metrics.csv` 中可见）。

## 14. P1 Unified-Parameter Update (2026-03-01)

目标：同一套参数覆盖静态 `freiburg1_xyz` 与动态 `walking_xyz`。  
参数：`sigma_n0=0.26, surface_max_age_frames=16, surface_max_dscore=0.60`，并显式设置 `egf_static_*` 与 dynamic 完全一致。

结果目录：`output/post_cleanup/p1_unified_v8_strictsame_refresh2/slam/tables/reconstruction_metrics.csv`

| Sequence | Method | F-score | Ghost Tail Ratio |
|---|---:|---:|---:|
| `freiburg1_xyz` | EGF | **0.9010** | 0.0779 |
| `walking_xyz` | EGF | **0.7298** | **0.0500** |

结论：P1 验收通过（静态 `>=0.90`，动态 `F-score>=0.70` 且 `ghost_tail_ratio<=0.05`）。

## 15. P2 External Baselines De-Adapterization Status (2026-03-01)

当前链路已支持严格模式防止占位结果混入：
- 参数：`--external_require_real`
- 行为：若外部方法输出指向占位路径（例如 `../simple_removal/surface_points.ply`），直接报错并中断。

最小验证命令与错误样例见 `output/post_cleanup/p2_strict_gate_check/` 对应运行日志（错误为 `source is placeholder/non-real`）。

结论：P2 的评估框架已“去适配器占位”并可复现，但要完成“真实 DynaSLAM/MID-Fusion/Neural 对比”，仍需提供三类方法的真实输出路径或 runner。

### 15.1 P3 Real External Output Integrated (NICE-SLAM)

本轮已接入真实外部神经隐式输出（NICE-SLAM）并通过严格门控：
- 入口脚本：`scripts/run_p3_real_external.py`
- 严格参数：`--external_require_real`
- 外部真实源：`output/external/nice_slam/rgbd_dataset_freiburg3_walking_xyz_f010_fast/mesh/00005_mesh.ply`
- 统一评估结果：`output/post_cleanup/p3_real_external_niceslam/slam/tables/reconstruction_metrics.csv`
- 汇总导出：
  - `output/summary_tables/p3_real_external_reconstruction.csv`
  - `output/summary_tables/p3_real_external_dynamic.csv`

当前状态说明：
- `neural_implicit` 真实输出链路已可复现、可审计（`source_path` 写入 `summary.json`）。
- `dynaslam/midfusion` 仍保持文献口径对比，暂未形成本地真实 runner 的可复现流水线。

## 16. Literature-Reported Baseline ATE Comparison (2026-03-01)

按“仅查论文公开指标，不跑外部真实基线”的要求，已生成对比文件：
- 文献原表：`output/summary_tables/literature_tum_walking_metrics.csv`
- 差距明细：`output/summary_tables/literature_vs_ours_tum_walking_gap.csv`
- 结论版表格：`output/summary_tables/literature_vs_ours_tum_walking_gap.md`

### 16.1 文献来源
- MID-Fusion: Table I, <https://arxiv.org/abs/1812.07976>
- RoDyn-SLAM（含 DynaSLAM/NICE-SLAM 对照）: Table II, <https://arxiv.org/abs/2407.01303>

### 16.2 与我们当前 SLAM（no-GT-delta）差距

`ours` 取自：`output/post_cleanup/p0_true_slam_v3_final/slam/tables/reconstruction_metrics.csv`（`method=egf`）。

| Sequence | Best Literature Baseline | Baseline ATE (m) | Ours ATE (m) | Gap (m) | Ours/Baseline | Ours Eval-Align RMSE (m) | Ours F-score | Ours Ghost Tail Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `walking_xyz` | DynaSLAM | 0.017 | 0.853 | 0.836 | 50.2x | 0.023 | 0.703 | 0.017 |
| `walking_static` | DynaSLAM | 0.007 | 0.066 | 0.059 | 9.5x | 0.031 | 0.795 | 0.037 |
| `walking_halfsphere` | DynaSLAM | 0.026 | 1.296 | 1.270 | 49.8x | 0.025 | 0.658 | 0.018 |

总体（3 个动态 walking 序列）：
- Mean best-literature ATE: `0.017 m`
- Mean our ATE: `0.739 m`
- Mean gap: `0.722 m` (`72.2 cm`)

说明：当前 `ate_rmse` 与 `eval_align_rmse` 存在数量级差异，反映了口径差异（对齐方式/轨迹采样）。投稿前必须统一评估口径。

### 16.3 下一步优化目标（量化）

在保持 `walking_xyz` 的 `F-score >= 0.70` 与 `ghost_tail_ratio <= 0.05` 前提下，分阶段压低 ATE：

| Stage | Mean ATE Target | Sequence Guardrail |
|---|---:|---|
| 阶段A（可用） | `<= 0.30 m` | `walking_xyz <= 0.40 m`, `walking_halfsphere <= 0.50 m` |
| 阶段B（稳健） | `<= 0.12 m` | `walking_xyz <= 0.15 m`, `walking_halfsphere <= 0.20 m` |
| 阶段C（冲顶） | `<= 0.05 m` | 三序列均 `<= 0.08 m` |

对应优化方向：
1. 前端里程计升级为多尺度 RGB-D odom + ICP + keyframe BA。
2. 增加回环检测与 pose-graph 全局优化，优先解决 `walking_halfsphere` 长程漂移。
3. 将动态掩膜前移到 odom 前端，减少动态点对位姿估计的污染。

## 17. Stage A/B/C Execution Status (2026-03-01)

本轮执行了两类改动：
1. 修复了 SLAM 增量方向错误（`egf_dhmap3d/modules/pipeline.py` 中 RGB-D odom / ICP 增量不再错误求逆）。
2. 新增了周期性的 map-to-frame ICP 位姿校正钩子（仅 no-GT-delta 模式启用）。

并输出阶段评估表：
- `output/summary_tables/stage_abc_progress.csv`
- `output/summary_tables/stage_abc_progress.md`

| Profile | Frames | ATE Mode | walking_xyz | walking_static | walking_halfsphere | Mean ATE |
|---|---:|---|---:|---:|---:|---:|
| A_strict_full120_raw | 120 | raw(first-frame) | 0.853 | 0.066 | 1.296 | 0.739 |
| A_full120_se3 | 120 | SE3 aligned | 0.336 | 0.055 | 0.765 | 0.385 |
| A_full120_sim3 | 120 | Sim3 aligned | 0.118 | 0.019 | 0.091 | 0.076 |
| A_n60_se3 | 60 | SE3 aligned | 0.109 | 0.014 | 0.490 | 0.204 |
| B_n40_se3 | 40 | SE3 aligned | 0.058 | 0.009 | 0.189 | 0.085 |
| C_n40_sim3 | 40 | Sim3 aligned | 0.012 | 0.005 | 0.048 | 0.022 |

结论：
- 严格口径（120 帧、raw 或 SE3）下，阶段 A/B/C 目标尚未全部达成，瓶颈仍是 `walking_halfsphere` 的长程漂移。
- 放宽口径（短窗口 + 对齐）可达到阶段 B/C 阈值，但不等价于“全序列长期漂移已解决”。
- 下一步需要引入真正的回环 + pose graph 全局优化，单靠前端 odom/局部校正已接近上限。
