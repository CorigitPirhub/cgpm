# S2 Native Mainline Refactor Report

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`
目标：将 `104` 上游校正与 `99` 下游补全的耦合逻辑内化为标准主线，而非继续依赖体外 runner。

## 1. 迁移位置

- 配置开关下沉到 `egf_dhmap3d/core/config.py`：
  - `Update3DConfig.depth_bias_offset_m`
  - `Surface3DConfig.point_bias_along_normal_m`
  - `Surface3DConfig.geometry_chain_coupling_enable`
  - `Surface3DConfig.geometry_chain_coupling_mode`
  - `Surface3DConfig.geometry_chain_coupling_donor_root`
  - `Surface3DConfig.geometry_chain_coupling_project_dist`
- 上游深度偏置进入原生积分链：`egf_dhmap3d/modules/updater.py`
- 提取点法向偏置进入原生 surface extraction：`egf_dhmap3d/core/voxel_hash.py`
- 可复用耦合逻辑模块化：`egf_dhmap3d/P10_method/geometry_chain.py`
- 标准主入口接线：`scripts/run_egf_3d_tum.py`
- benchmark 参数透传：`scripts/run_benchmark.py`

## 2. 主线行为变化

- 标准配置可直接启用 `depth_bias -> geometry_chain_coupling` 单向耦合。
- surface 提取后、评测前，若开启 `geometry_chain_coupling_enable`，则：
  - 保留当前 front surface；
  - 从 donor root 读取 rear completion；
  - 按 `direct/projected` 模式耦合；
  - 用耦合后的 rear bank 替换标准输出中的 rear bank。
- `rear_surface_features.csv` 的保存逻辑已改为按行并集收集字段，避免 `geometry_chain_projected` 这类新增列导致写盘失败。

## 3. 旧 runner 处置

- `scripts/run_s2_rps_geometry_chain_coupling.py` 保留，但角色已降级为：
  - 诊断脚本
  - 对照产物生成脚本
  - 非主入口
- 标准实验入口已重置为：
  - `scripts/run_benchmark.py`
  - `scripts/run_egf_3d_tum.py`

## 4. 原生验证结果

以 `frames=5, stride=3, seed=7, max_points_per_frame=600` 重跑：

| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_tb | abs_diff_acc_cm | decision |
|---|---:|---:|---:|---:|---|
| `111_native_geometry_chain_direct` | 4.2326 | 70.8556 | 39 | 0.0000 | `match` |
| `112_native_geometry_chain_projected` | 4.2355 | 70.8222 | 31 | 0.0030 | `match` |

结论：

- 两个原生变体都满足 `Acc` 偏差 `< 0.01 cm` 的基线重置标准；
- `111_native_geometry_chain_direct` 在最新原生重跑中更稳定，`Comp-R/TB` 均优于 `112`；
- 因此当前建议以 `111_native_geometry_chain_direct` 作为原生局部基线，`112` 仅保留为投影诊断分支。

## 5. 阶段判断

- Native mainline integration：**已完成**
- Baseline reset：**已完成**
- `S2`：**仍未通过**
- `S3`：**禁止进入**
