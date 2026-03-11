# S2 Geometry Error Decomposition Report

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`
基线：`111_native_geometry_chain_direct`
协议：`frames=5, stride=3, seed=7, max_points_per_frame=600`

> 说明：以下“误差分量占比”是诊断占比，不是严格可加的因果分解。它用于排序主要矛盾，而非宣称各分量线性相加后恰好等于最终 `Acc`。

## 1. 家族级分解

| family | acc_cm | acc_gap_cm | noise_cm | calib_bias_cm | drift_cm | thickness_cm | completion_delta_cm | primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tum_all3 | 0.936 | 0.000 | 1.241 | 0.027 | 0.000 | 1.720 | 0.000 | surface_thickness_cm |
| bonn_all3 | 4.233 | 1.133 | 1.577 | 0.010 | 5.205 | 1.860 | -0.005 | temporal_drift_cm |

## 2. Bonn 缺口的主贡献者

- `Temporal Drift`：5.205 cm，占诊断量级的 `60.2%`
- `Surface Thickness`：1.860 cm，占 `21.5%`
- `Sensor Noise Floor`：1.577 cm，占 `18.2%`
- `Calibration Bias`：0.010 cm，占 `0.1%`
- `Completion Artifacts`：-0.005 cm；为负值表示 completion 对 `Acc` 是轻微净改善，而不是主要误差源

## 3. 关键判断

- `TUM` 在 oracle pose 下达到 `Acc=0.936 cm`，且 `Temporal Drift≈0.000 cm`，说明当前几何链在无漂移条件下已经足够接近目标。
- `Bonn` 的家族均值 `Temporal Drift=5.205 cm`，已经高于剩余 `Acc` 缺口 `1.133 cm` 的同量级；这直接说明剩余瓶颈不是传感器噪声极限，而是位姿/动态边界相关的系统误差。
- `Calibration Bias` 只有 `0.010 cm` 量级，说明上一轮 `depth_bias` 修正已基本触顶，不值得继续作为主线。
- `Surface Thickness` 仍在 `1.860 cm` 量级，说明即使修复了 drift，front-side distortion / fusion thickness 仍然需要二阶段校正。
- `104 -> 111` 的 completion 仅带来 `-0.005 cm` 的 `Acc` 变化，且方向为改善；因此当前 `Acc` 不是被下游补全拖坏的。

## 4. 优先级排序

1. `Temporal Drift / Pose Error`：先做位姿与动态边界关联误差分解，这是 Bonn `Acc` 最大贡献者。
2. `Upstream Geometry Distortion`：在 drift 受控后，继续拆 front-side bias 与 fusion thickness。
3. `Weak-Evidence Geometry Admission`：不是为了压 `Acc`，而是为后续 `Comp-R` 恢复做准备。
4. `Calibration Bias`：当前量级过小，降为低优先级检查项。

## 5. 结论

- 当前 `4.233 cm` 的 Bonn `Acc` 不是传感器噪声极限。
- 主要矛盾已经锁定为：`SLAM drift + dynamic-boundary geometry distortion`。
- 下一轮若只做几何小修或 rear completion 微调，无法覆盖当前缺口。
