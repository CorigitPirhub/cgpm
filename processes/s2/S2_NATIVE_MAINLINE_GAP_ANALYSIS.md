# S2 Native Mainline Gap Analysis

日期：`2026-03-11`
阶段：`S2 / not-pass / no-S3`
原生基线：`111_native_geometry_chain_direct`

## 1. Native Mainline 集成状态

已完成原生化位置：
- 配置开关：`egf_dhmap3d/core/config.py`
- 可复用耦合模块：`egf_dhmap3d/P10_method/geometry_chain.py`
- 标准启动脚本接线：`scripts/run_egf_3d_tum.py`
- benchmark 透传：`scripts/run_benchmark.py`
- 原生验证 runner：`scripts/run_s2_rps_native_mainline_integration.py`

验证结果：
- `111_native_geometry_chain_direct` 与 `108_geometry_chain_coupled_direct` 的 Bonn `Acc` 误差仅 `7.7e-14 cm`
- `112_native_geometry_chain_projected` 与 `109_geometry_chain_coupled_projected` 的 Bonn `Acc` 偏差为 `0.00295 cm`

结论：
- `108/109` 的耦合逻辑已经不再依赖体外循环脚本；
- 原生主线已经成功继承了耦合效果。

## 2. 当前原生基线

以 `111_native_geometry_chain_direct` 为当前原生局部最优参考：
- Bonn `Acc = 4.233 cm`
- Bonn `Comp-R = 70.86%`
- Bonn `ghost_reduction_vs_tsdf = 14.58%`
- Bonn `TB = 39`

相对 S2 硬门槛仍差：
- `Acc`：`4.233 cm -> 3.10 cm`，仍差 `1.133 cm`
- `Comp-R`：`70.86% -> 98%`，仍差 `27.14` 个百分点

## 3. Acc Gap 来源

### 3.1 不是后验平面厚度噪声
- `101/102/103` 已证明：硬投影、尺度校正、局部紧化都无法实质拉低 `Acc`
- `101` 虽降低平面厚度，但 `Acc` 几乎不变
- `102` 显著恶化，说明没有可直接利用的 geometry-only scale drift 结构

### 3.2 存在可测的上游 depth bias，但收益有限
- `104_depth_bias_minus1cm` 把 Bonn `Acc` 从 `4.346 cm` 拉到 `4.238 cm`
- 说明确有 `front-side bias` / `depth bias` 响应
- 但该响应只有约 `0.11 cm`，远不足以覆盖剩余的 `1.13 cm`

### 3.3 更可能的主因
- 上游融合阶段的系统性几何畸变仍存在；
- 其中最可疑的是：
  - `front-side bias` 在不同序列中的幅度不一致；
  - pose / association 误差将动态边界附近的表面推到错误侧；
  - 提取阶段仅做零交叉与局部偏置补偿，不足以纠正更大尺度的几何失真。

判断：
- 当前 `Acc` 瓶颈不是“纯传感器噪声极限”；
- 更像是“有限 depth bias + 上游几何形成畸变”的组合问题。

## 4. Comp-R Gap 来源

### 4.1 缺失区域并不均匀
`111_native_geometry_chain_direct` 的 Bonn 分序列表现：
- `rgbd_bonn_balloon2`: `Acc = 2.886 cm`, `Comp-R = 83.57%`
- `rgbd_bonn_balloon`: `Acc = 4.154 cm`, `Comp-R = 70.23%`
- `rgbd_bonn_crowd2`: `Acc = 5.658 cm`, `Comp-R = 58.77%`

结论：
- 缺失的 27% 完整性主要集中在 `crowd2` 这类重遮挡动态场景；
- 不是统一的传感器盲区问题。

### 4.2 当前 Rear Completion 触达范围有限
- `99/108/109/111/112` 的 completion 依赖高置信 support cluster plane；
- 这使它对 `crowd2` 的局部平面有效，但对：
  - `balloon2 / balloon` 中无高置信 cluster 的区域，
  - 更大面积的动态遮挡后方空洞，
  - 非平面或弱平面结构，
  仍然无法触达。

### 4.3 因此缺失的 27% 地图主要来自
- 动态物体大面积遮挡的静态背景；
- 当前 support-cluster 不能启动的弱证据区域；
- 传感器视角稀疏下未形成 donor plane 的空洞。

## 5. 结论

当前已经明确：
- **结构断裂已解决**
- **原生主线已重置**
- **局部正链已原生化**

但 `S2` 仍未通过，原因也已清楚：
- `Acc` 仍受上游几何形成畸变主导；
- `Comp-R` 仍受 support-cluster 可触达范围限制；
- 这两者都不是再靠后验点云修补能解决的问题。

## 6. 剩余攻坚方向

若继续 `S2`，唯一合理方向是：
- 以 `111_native_geometry_chain_direct` 为原生局部基线；
- 向上游继续推进：
  1. 修复 geometry formation 中的系统性 front-side bias / fusion distortion
  2. 扩展 support-cluster completion 的可触达范围，尤其是 `crowd2` 的大面积空洞
  3. 最终与 `RB-Core` 做完整净正闭环

阶段判断：
- `S2` **仍未通过**
- **绝对禁止进入 `S3`**
