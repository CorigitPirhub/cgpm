# CGPM → 梯度场构建框架探索

> 检索与分析时间：2026-02-22（UTC+8）  
> 结论先行：**有启发**。CGPM 的“证据场 + 假设实体 + 贝叶斯状态演化”可以迁移到梯度场构建，并可形成一个可工程化的新框架。

## 1. CGPM 模型核心要点回顾

### 1.1 参数化流形与证据密度

CGPM 在形式化层面把局部几何写成参数化曲线 $\gamma(s)$（README 的定义 2.1）并围绕参数域维护证据集合 $\mathcal{E}$ 与核密度 $\mu_E(s)$（README 定义 2.2/4.1）。`Entity` 中也直接对应了该结构：`model`（曲线模型）、`evidence`（证据集）、`pose/covariance`（位姿与协方差）[C1][C3]。  

代码里证据密度与导数有明确实现：`get_density_at` 和 `get_density_derivative_at`，且导数被用于端点/开放边界检测（`compute_endpoints`）[C2][C4]。这说明 CGPM 已经具备“标量场（密度）+ 梯度（一阶导）”的雏形。

### 1.2 假设实体与状态演化算子

CGPM 在代码中形成了可运行的 `Predict -> Associate -> Update` 链路：

- Predict：`Predictor` 基于 STATIC/CV/CTRV 预测状态与协方差 [C5]。  
- Associate：投影算子做“观测到模型”的最近点投影 + 牛顿 refine，再做马氏门限关联 [C6][C7]。  
- Update：`LocalUpdater` 采用“位姿估计（EKF）与几何融合解耦”的两阶段更新 [C8][C9]。  
- 证据累积：关联成功后写回 `EvidenceSet`，并有 out-of-domain cache/prefit cache 支持边界外延 [C2][C7][C9]。

### 1.3 几何闭合与系统性质

`Entity.is_geometrically_closed` 实现了覆盖率 + 位置闭合 + 切向闭合三条判据，对应 README 的几何闭合条件 [C1][C3]。  

需要注意：README 有 `Merge` 的形式化定义，但当前代码中 `operators/combiner/` 为空，`Merge` 尚未工程化落地，属于“理论有定义、实现缺位”[C1][C10]。

---

## 2. 梯度场相关工作调研

## 2.1 隐式场 + 梯度场表示

### UN3-Mapping（2025, RA-L）[P1]

UN3-Mapping 的核心是**隐式神经距离场 + 显式梯度场**混合表示：用神经距离场建模几何，同时从法向信息优化显式梯度场，再把原始测距转成 non-projective SDF 标签去训练隐式地图。它还引入在线不确定性学习，并利用高不确定区域做动态障碍抑制。  

这里梯度场不仅是“辅助正则”，而是标签构造与动态区域判别的关键中介。

### Gradient-SDF（CVPR 2022）[P2]

Gradient-SDF 在每个体素同时存储 SDF 和梯度向量场（semi-implicit）：保留体素隐式表达，同时把法向/梯度信息显式化。  

该设计直接带来两点：  
- 可在体素上直接做 SDF tracking（不必先转点云/网格）。  
- 可在体素表示中做 photometric bundle adjustment，联合优化几何与位姿。  

因此梯度场在这里同时承担“几何表达增强 + 优化变量桥接”角色。

### MS²IS（2025, RA-L）[P3]

MS²IS 把 semi-implicit 思路推进到多尺度体素结构：通过多尺度与自适应分辨率平衡内存和精度，并在多尺度体素中维护梯度场。  

其直接收益是：隐式体素模型获得部分显式能力（可直接提取顶点/法向），从而增强位姿跟踪与重建效率。换言之，梯度场在 MS²IS 中是“跨尺度几何可读性”的关键载体。

### 小结（2.1）

这三类工作共同结论：  
- 梯度场并非只做正则项，而是可成为几何表达的“一等状态”。  
- 梯度构建来源主要是法向/深度监督 + 体素或神经场一致性优化。  
- 相比纯 TSDF/SDF，优势在于法向一致性、细节锐度、位姿优化耦合能力；难点是梯度噪声与内存/计算开销。

## 2.2 梯度场融合（多视图 / 多传感器）

### Remote Sensing Image Fusion on Gradient Field（ICPR 2006）[P4]

该工作给出典型“梯度域融合”路线：  
- 目标函数优先匹配高分辨图像梯度。  
- 通过 Poisson 方程（带 Dirichlet 边界）重建融合结果。  
- 还给出“梯度差 + 色彩差”联合优化版本。  

这基本是“梯度估计 -> 梯度融合 -> 场重建”的标准范式。

### Stochastic Fusion of Multi-View Gradient Fields（ICIP 2008）[P5]

该文把多视图梯度视为带噪观测，建模为可利用结构噪声形式的线性估计问题，得到稳健融合梯度场，再用于场景重建。  

关键价值是：把“梯度融合”显式做成统计估计（而非简单平均），更适合多视角不一致和空间变噪声场景。

### Gradient field multi-exposure fusion for HDR（JVCIR 2012）[P6]

该文在 HDR 场景下融合多曝光梯度：通过多曝光梯度与 LDR 梯度估计 HDR 梯度场，再结合亮度均值进行重建。  

它说明了一个跨模态共性：在高动态范围/饱和区域，梯度域往往比强度域更稳健，更能保结构细节。

### 小结（2.2）

与“直接在强度/深度域融合”相比，梯度场融合的优势在于边缘结构保持和跨传感器对齐鲁棒性；难点在于积分重建（Poisson/变分）时的边界条件与低频漂移。

## 2.3 梯度场用于动态点云与场景重建

### Dynamic Point Cloud Denoising via Gradient Fields（TOMM 2025）[P7]

该工作把梯度场定义为噪声点云对数概率的梯度（score-like quantity），并通过梯度上升把点收敛到干净表面，同时利用时间对应关系建模动态。  

这与“概率密度场 -> 梯度场 -> 几何更新”路径高度一致，直接启发可把 CGPM 的证据密度扩展为空间概率/置信度场，再由其梯度引导数据关联与更新。

---

## 3. CGPM 对梯度场构建的启发评估

## 3.1 表示层面

### 可迁移点

- **证据密度是天然标量场**：CGPM 已有 $\mu_E(s)$ 与 $\partial\mu_E/\partial s$ 实现 [C2]。  
- **曲线导数已显式可用**：`model.derivative(s)` 在关联和闭合中持续被调用 [C3][C7]。  
- **开放边界由密度导数触发**：`compute_endpoints` 用 `dmu/ds < delta` 判边界 [C4]。

### 对梯度场构建的意义

把 1D 参数域标量场 $\mu_E(s)$ 推广为 2D/3D 空间置信度场 $\rho(\mathbf{x})$ 后，$\nabla\rho$ 就可作为“证据驱动梯度”。这与 UN3/MS²IS 中“显式梯度场参与几何构建”思路是同向的 [P1][P3]。

## 3.2 数据关联与证据累积

CGPM 当前关联是“投影 + 马氏门限 + 证据写回” [C6][C7]。这可平移为梯度场关联：

- 投影对象从 $\gamma(s)$ 改为零水平集 $\phi(\mathbf{x})=0$。  
- 关联判据从几何残差扩展为“距离残差 + 法向一致性（梯度方向）+ 不确定度”。  
- 证据仍按带权样本累积，只是样本从 `s` 变为体素/局部场坐标。

因此这一块启发强，且改造成本可控。

## 3.3 状态演化与优化结构

CGPM 的解耦更新（先位姿 EKF，再几何融合）在代码已稳定存在 [C8][C9]。  
这与 Gradient-SDF 的“几何与位姿联合优化”并不冲突，反而可形成“先滤波稳定，再局部联合优化精修”的两层架构 [P2]。

结论：CGPM 提供的是**流程模板与工程组织方式**，而梯度场文献提供的是**几何状态定义与优化对象**，两者互补。

## 3.4 几何闭合与拓扑约束

CGPM 已有覆盖率/位置/切向闭合判据 [C3]。在梯度场中可改写为：

- 覆盖率：高置信区域的零水平集覆盖率。  
- 一致性：$\|\nabla\phi\|\approx 1$（Eikonal）与法向一致性。  
- 可积性：$\nabla\times\mathbf{g}\approx 0$（梯度场应近似保守场）。  

另外，README 给出 Merge 定义但代码缺位 [C1][C10]，这反而提示新框架应把 merge/split 做成一等公民。

## 3.5 总体评估结论（含量化打分）

评分规则：1=几乎不可迁移，3=可迁移但需较大改造，5=可直接迁移。

| 维度 | 评分（5分制） | 可迁移思想 | 现实现缺位 |
|---|---:|---|---|
| 表示层面 | **4.0** | `μ_E(s)` 与 `dμ/ds` 已形成“标量场+梯度”雏形，可外推到 3D 置信度场 [C2][C4] | 当前是 1D 参数场，未直接维护 3D 空间梯度场 |
| 关联与证据 | **4.5** | 投影+马氏门限+证据累积流程可直接平移到 SDF/梯度场关联 [C6][C7] | 尚无“梯度方向一致性+场不确定度”联合门限 |
| 状态演化 | **4.0** | Predict/Associate/Update 解耦结构稳定，可承载梯度场状态 [C5][C8][C9] | 位姿-几何联合优化与梯度后验尚未建模 |
| 拓扑闭合 | **2.5** | 闭合判据思想可迁移到 Eikonal/可积性约束 [C3] | `Merge` 仅在 README 定义，代码未落地（`operators/combiner` 为空）[C1][C10] |

综合得分：**3.75 / 5**。  
**结论：有启发（中到强）**。核心可迁移价值在“证据驱动 + 假设实体管理 + 贝叶斯时序流程”，核心缺口在“拓扑操作工程化（Merge/Split）与梯度场原生状态建模”。

---

## 4. 梯度场构建框架设计（有启发前提下）

## 4.1 应用场景与问题定义

### 场景

**动态 RGB-D / LiDAR 室内建图（可含行人/移动物体）**。  

选择理由：

- 深度/点云可直接提供法向与距离约束，适合梯度场构建 [P1][P2][P3]。  
- 动态环境对“不确定性 + 假设管理”要求高，CGPM 思路可直接发挥 [C3][C8]。  
- 多视图融合天然需要梯度融合与重建环节 [P4][P5][P6]。

## 4.2 梯度场状态定义

定义局部假设实体（3D 版本）：

\[
H_i = (\phi_i,\mathbf{g}_i,\rho_i,\Sigma_i^{g},\hat{T}_i,\Sigma_i^{T},\mathcal{E}_i,\Omega_i,t_i)
\]

- $\phi_i(\mathbf{x})$：局部 SDF（隐式标量场）。  
- $\mathbf{g}_i(\mathbf{x})=\nabla\phi_i(\mathbf{x})$：显式梯度场。  
- $\rho_i(\mathbf{x})$：证据置信度场（由历史观测核密度累积）。  
- $\Sigma_i^{g}(\mathbf{x})$：梯度不确定度（每体素 3×3 或对角近似）。  
- $\hat{T}_i,\Sigma_i^T$：实体位姿与协方差（SE(3)）。  
- $\mathcal{E}_i$：证据集合（点、法向、时间、权重）。  
- $\Omega_i$：局部活跃体素哈希块（多尺度）。  

证据场定义：

\[
\rho_i(\mathbf{x})=\sum_{e_k\in\mathcal{E}_i} w_k\,K_h(\|\mathbf{x}-\mathbf{x}_k\|), \quad
\mathbf{c}_i(\mathbf{x})=\nabla \rho_i(\mathbf{x})
\]

其中 $\mathbf{c}_i$ 是“证据梯度场”，用于置信加权与边界外延决策（CGPM 的 $\mu_E \to \rho$ 推广）。

离散表示采用**多尺度哈希体素 + 可选神经残差头**：

- 体素层：存 $(\phi,\mu_g,\Sigma_g,\rho)$，用于实时更新与关联。  
- 神经层：对体素残差建模（可选），用于高频补偿。

## 4.3 状态演化流程设计

### Predict

沿用 CGPM 的运动预测思想：对 $\hat{T}_i,\Sigma_i^T$ 做 CV/CTRV 传播；对 $\rho_i$ 做时间衰减，对 $\Sigma_i^g$ 做过程噪声扩散 [C5]。

### Associate

对观测点 $\mathbf{x}_k$，在候选实体内做零水平集投影：

\[
\mathbf{x}_k^\*=\mathbf{x}_k-\frac{\phi_i(\mathbf{x}_k)}{\|\mathbf{g}_i(\mathbf{x}_k)\|^2+\epsilon}\mathbf{g}_i(\mathbf{x}_k)
\]

残差项：

\[
r_d=\phi_i(\mathbf{x}_k), \quad
r_n = 1-\mathbf{n}_k^\top\hat{\mathbf{n}}_i(\mathbf{x}_k),\quad
\hat{\mathbf{n}}_i=\frac{\mathbf{g}_i}{\|\mathbf{g}_i\|}
\]

门限采用马氏距离并由不确定度/证据梯度调权：

\[
d^2 = \mathbf{r}^\top \Sigma_r^{-1}\mathbf{r}, \quad
\Sigma_r \leftarrow \Sigma_r + \beta\,f(\|\mathbf{c}_i\|)
\]

直观上，证据梯度越弱（边界/空洞区域），关联越保守。

### Update

分三步：

1. **位姿滤波更新**：EKF/IEKF 更新 $\hat{T}_i,\Sigma_i^T$（延续 CGPM 结构）[C8]。  
2. **梯度场贝叶斯融合**：每体素维护高斯后验  
\[
\mathbf{g}_v \sim \mathcal{N}(\mu_v,\Sigma_v)
\]
\[
K_v=\Sigma_v^-(\Sigma_v^-+R_k)^{-1},\;
\mu_v^+=\mu_v^-+K_v(\tilde{\mathbf{g}}_k-\mu_v^-),\;
\Sigma_v^+=(I-K_v)\Sigma_v^-
\]
其中 $\tilde{\mathbf{g}}_k$ 可由法向观测与证据梯度混合构造。  
3. **SDF 重建/回归**：解
\[
\min_{\phi_i}
\sum_{v\in\Omega_i}\|\nabla\phi_i(v)-\mu_v\|^2
+\lambda_{eik}\sum_v(\|\nabla\phi_i(v)\|-1)^2
+\lambda_{np}\sum_k w_k\big(\phi_i(\mathbf{x}_k)-d^{np}_k\big)^2
\]
其中 $d^{np}_k$ 是 non-projective 距离标签（借鉴 UN3）[P1]。  

这一步可用局部 Poisson + Eikonal 迭代近似实现。

### Merge / Split（拓扑管理）

- Merge 条件：邻域重叠 + 位姿统计一致 + 法向兼容。  
\[
\xi_{ab}^\top\Sigma_{ab}^{-1}\xi_{ab} < \tau_{pose},\quad
\delta_n=\frac1{|\Gamma|}\sum_{\mathbf{x}\in\Gamma}\|\hat{\mathbf n}_a-\hat{\mathbf n}_b\|<\tau_n
\]
- Split 条件：单实体内残差/运动出现多峰，或 $\rho$ 形成稳定断裂带。  

这对应 CGPM 的“假设实体管理思想”向梯度场扩展。

## 4.4 与 CGPM 的对比与创新点

### 借鉴自 CGPM

1. 证据驱动：从 $\mu_E(s)$ 出发，保持“观测不是一次性拟合而是可追溯证据”的思想 [C2]。  
2. 时序流程：保留 Predict/Associate/Update 的工程骨架 [C5][C7][C8]。  
3. 局部实体：继续使用 Hypothesis Entity 作为生命周期单元 [C3]。  
4. 边界缓存：借鉴 out-of-domain cache/prefit cache 做前沿生长触发 [C2][C7][C9]。

### 新增创新

1. **双场耦合**：几何梯度场 $\nabla\phi$ 与证据梯度场 $\nabla\rho$ 同时建模。  
2. **梯度不确定度后验**：每体素维护梯度协方差，关联与融合统一在贝叶斯框架。  
3. **隐式重建显式约束化**：通过 Poisson/Eikonal 把融合梯度转回一致 SDF。  
4. **可落地的 Merge/Split**：补齐 CGPM 现实现缺位的拓扑管理。

与 UN3/Gradient-SDF/MS²IS 相比，本方案的创新不是再造一种底层表示，而是把**“证据场+贝叶斯假设管理”系统化注入梯度场建图**，重点强化动态场景下的稳健性与可解释性。

## 4.5 可行性与实现路线简述

### 理论可行性

- 梯度融合 + 标量场重建（Poisson/变分）是成熟路线 [P4][P5]。  
- 梯度作为几何主状态在 3D 重建中已有成功先例 [P1][P2][P3]。  
- 概率梯度（log-probability gradient）在动态点云去噪中已验证有效 [P7]。

### 计算可行性

- 采用多尺度哈希体素可把复杂度控制在活跃体素规模。  
- 在线阶段只做局部块更新，支持实时近似。  
- 可先不加神经残差头，先做纯体素版本降低工程风险。

### 实现复杂度与风险

主要风险：

1. 法向噪声会放大梯度估计误差。  
2. 双场（$\nabla\phi,\nabla\rho$）耦合权重不当会造成过度平滑或误关联。  
3. Merge/Split 阈值需数据驱动调参。  

建议分阶段实施：

1. 阶段 A：复现 Gradient-SDF 式体素 SDF+梯度基线。  
2. 阶段 B：加入证据场 $\rho$ 与基于 $\nabla\rho$ 的关联调权。  
3. 阶段 C：加入实体级 Merge/Split 与动态不确定度抑制。  

---

## 5. 参考文献

### 5.1 梯度场相关论文

[P1] Song, S., Zhao, J., Veas, E., et al. **UN3-Mapping: Uncertainty-Aware Neural Non-Projective Signed Distance Fields for 3D Mapping**. IEEE RA-L, 2025.  
DOI: https://doi.org/10.1109/LRA.2025.3588410  
IEEE 页面: https://ieeexplore.ieee.org/document/11078897

[P2] Sommer, C., Sang, L., Schubert, D., Cremers, D. **Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction**. CVPR, 2022.  
DOI: https://doi.org/10.1109/CVPR52688.2022.00618  
CVF 页面/PDF: https://openaccess.thecvf.com/content/CVPR2022/html/Sommer_Gradient-SDF_A_Semi-Implicit_Surface_Representation_for_3D_Reconstruction_CVPR_2022_paper.html

[P3] Deng, Z., Wu, X., Wang, M. Y. **MS²IS: A Multi-Scale Semi-Implicit Surface Representation for 3D Reconstruction**. IEEE RA-L, 2025.  
DOI: https://doi.org/10.1109/LRA.2025.3597899  
IEEE 页面: https://ieeexplore.ieee.org/document/11122617

[P4] Wen, J., Li, Y., Gong, H. **Remote Sensing Image Fusion on Gradient Field**. ICPR, 2006.  
DOI: https://doi.org/10.1109/ICPR.2006.995  
IEEE 页面: https://ieeexplore.ieee.org/document/1699748

[P5] Sankaranarayanan, A. C., Chellappa, R. **Stochastic fusion of multi-view gradient fields**. ICIP, 2008.  
DOI: https://doi.org/10.1109/ICIP.2008.4712007  
IEEE 页面: https://ieeexplore.ieee.org/document/4712007

[P6] Gu, B., Li, W., Wong, J., et al. **Gradient field multi-exposure images fusion for high dynamic range image visualization**. JVCIR, 2012.  
DOI: https://doi.org/10.1016/J.JVCIR.2012.02.009  
ScienceDirect 页面: https://www.sciencedirect.com/science/article/pii/S1047320312000438

[P7] Hu, Q., Hu, W. **Dynamic Point Cloud Denoising via Gradient Fields**. TOMM, 2025.  
DOI: https://doi.org/10.1145/3721431  
ACM 页面: https://dl.acm.org/doi/10.1145/3721431

[P8] Song, S., Zhao, J., Veas, E., et al. **N3-Mapping: Normal Guided Neural Non-Projective Signed Distance Fields for 3D Mapping**. arXiv, 2024.  
arXiv: https://arxiv.org/abs/2401.03412

### 5.2 CGPM 代码与文档证据

[C1] 形式化定义与算子：`README.md:23`, `README.md:39`, `README.md:76`, `README.md:90`, `README.md:116`, `README.md:129`, `README.md:150`, `README.md:164`  
[C2] 证据集与密度/导数：`entity/evidence.py:24`, `entity/evidence.py:284`, `entity/evidence.py:310`  
[C3] 假设实体与闭合：`entity/entity.py:523`, `entity/entity.py:569`  
[C4] 端点与密度导数边界：`entity/endpoint.py:7`, `entity/endpoint.py:47`  
[C5] 预测模型：`operators/predictor/predictor.py:7`, `operators/predictor/predictor.py:147`, `operators/predictor/predictor.py:162`  
[C6] 投影算子：`operators/associator/projection.py:11`, `operators/associator/projection.py:50`, `operators/associator/projection.py:75`  
[C7] 关联与证据写回：`operators/associator/associator.py:33`, `operators/associator/associator.py:73`, `operators/associator/associator.py:109`, `operators/associator/associator.py:220`, `operators/associator/associator.py:278`, `operators/associator/associator.py:396`  
[C8] 位姿估计（EKF）：`operators/updater/state_estimator.py:7`, `operators/updater/state_estimator.py:28`  
[C9] 局部更新与几何融合/边界生长：`operators/updater/updater_local.py:9`, `operators/updater/updater_local.py:20`, `operators/updater/geometry_fuser.py:9`, `operators/updater/geometry_fuser.py:211`, `operators/updater/geometry_fuser.py:254`  
[C10] Merge 实现缺位证据：`operators/combiner`, `README.md:129`
