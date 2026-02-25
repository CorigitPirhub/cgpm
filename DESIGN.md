# DESIGN: EGF-DHMap（Evidence-coupled Gradient Field Dynamic Hypothesis Mapping）

> 版本：v1.0  
> 设计日期：2026-02-22  
> 设计目标：在动态场景中构建“可解释、可维护、可实时演化”的梯度场地图，深度融合 CGPM 的证据驱动与假设实体思想。

## 1. 框架概述与核心创新

### 1.1 框架命名

**EGF-DHMap**（Evidence-coupled Gradient Field Dynamic Hypothesis Mapping）

### 1.2 一句话总结

EGF-DHMap 将 CGPM 的“证据场 + 假设实体 + 贝叶斯时序”迁移到 3D 梯度场建图，用“双场耦合（几何梯度场 $\nabla\phi$ 与证据梯度场 $\nabla\rho$）+ 不确定度后验 + 拓扑级 Merge/Split”实现动态环境下的稳健构图。

### 1.3 核心创新点（3-5 个）

1. **双场耦合状态（Geometry-Gradient + Evidence-Gradient）**  
解决“纯几何梯度易受噪声放大、缺乏观测密度语义”的问题。以 $\mathbf g=\nabla\phi$ 表示几何，以 $\mathbf c=\nabla\rho$ 表示证据前沿，联合驱动关联与更新。  
依据：CGPM 的证据密度与密度导数 [C2][C4]，梯度场显式表达范式 [P1][P2][P3]。

2. **证据-不确定度联合贝叶斯融合（EU-Bayes Fusion）**  
在每体素维护梯度均值与协方差 $(\mu_g,\Sigma_g)$，并用证据梯度自适应调节测量噪声，实现“证据足→快收敛，证据弱/不确定高→保守更新”。  
依据：多视图梯度随机融合 [P5]，CGPM EKF 与马氏门限 [C7][C8]。

3. **可积性约束的局部重建（Poisson+Eikonal）**  
将融合后的梯度场反演为一致 SDF，显式抑制非保守梯度引起的拓扑伪影。  
依据：梯度域 Poisson 重建思想 [P4][P6]，半隐式场约束 [P2][P3]。

4. **前沿缓存驱动的实体生长（Frontier Cache Growth）**  
继承 CGPM 的开放边界与缓存机制，把“边界外观测”转为前沿证据缓存，达到触发阈值后进行局部生长/扩展。  
依据：CGPM out-of-domain/prefit cache 与边界处理 [C2][C7][C9]。

5. **拓扑管理创新：Gradient-aware Merge/Split**  
将 README 中仅定义未实现的 Merge 机制落地，并补充 Split：基于梯度相似性、法向兼容性、证据重叠与位姿统计一致性。  
依据：CGPM Merge 定义与实现缺位 [C1][C10]。

---

## 2. 数学模型定义

## 2.1 观测与符号

给定时刻 $t$ 的观测集合：
\[
\mathcal Z_t=\{z_k^t\}_{k=1}^{N_t},\quad
z_k^t=(\mathbf x_k,\mathbf n_k,\sigma_{d,k}^2,\sigma_{n,k}^2,t) \tag{1}
\]

- $\mathbf x_k\in\mathbb R^3$：观测点坐标（LiDAR/RGB-D）  
- $\mathbf n_k\in\mathbb S^2$：观测法向（可由局部平面估计）  
- $\sigma_{d,k}^2,\sigma_{n,k}^2$：距离和法向噪声方差

## 2.2 局部梯度场假设实体

定义第 $i$ 个局部实体状态：
\[
H_i^t=\big(\Omega_i,\phi_i,\mu_i^g,\Sigma_i^g,\rho_i,\mathbf c_i,\hat T_i,\Sigma_i^T,\mathcal E_i,\mathcal F_i,t_i\big) \tag{2}
\]

- $\Omega_i$：活跃体素域（多尺度哈希块）  
- $\phi_i:\Omega_i\to\mathbb R$：局部 SDF  
- $\mu_i^g(\mathbf v)\in\mathbb R^3$：体素梯度均值，近似 $\nabla\phi_i$  
- $\Sigma_i^g(\mathbf v)\in\mathbb R^{3\times 3}$：梯度协方差  
- $\rho_i(\mathbf v)\in\mathbb R_+$：证据置信度场  
- $\mathbf c_i(\mathbf v)=\nabla\rho_i(\mathbf v)$：证据梯度场  
- $(\hat T_i,\Sigma_i^T)$：实体位姿与协方差（SE(3) + 李代数协方差）  
- $\mathcal E_i$：证据样本集合（观测点/法向/权重/时间）  
- $\mathcal F_i$：前沿缓存（未确认边界证据）

这对应 CGPM 的 $H=(\mathcal M,\hat g,\Sigma_g,\mathcal E,\hat L,t_{last})$ 的 3D 梯度场迁移版 [C1][C3]。

## 2.3 场定义与关键约束

### 2.3.1 几何梯度场

\[
\mathbf g_i(\mathbf x)=\nabla \phi_i(\mathbf x),\quad
\hat{\mathbf n}_i(\mathbf x)=\frac{\mathbf g_i(\mathbf x)}{\|\mathbf g_i(\mathbf x)\|+\epsilon} \tag{3}
\]

### 2.3.2 证据置信度场（3D KDE）

\[
\rho_i(\mathbf x)=\sum_{e_m\in\mathcal E_i} w_m\,K_h(\|\mathbf x-\mathbf x_m\|),\quad
K_h(r)=\frac{1}{(2\pi h^2)^{3/2}}\exp\!\left(-\frac{r^2}{2h^2}\right) \tag{4}
\]

\[
\mathbf c_i(\mathbf x)=\nabla \rho_i(\mathbf x) \tag{5}
\]

式(4)(5)是 CGPM $\mu_E(s),\partial\mu_E/\partial s$ 的空间推广 [C2][C4]。

### 2.3.3 可积性与 Eikonal 约束

\[
\nabla\times \mathbf g_i \approx \mathbf 0,\qquad \|\nabla\phi_i\|\approx 1 \tag{6}
\]

用于保证梯度场可由一致标量场积分得到，避免局部扭曲。

## 2.4 关键更新方程

### 2.4.1 位姿预测

\[
\hat T_i^{t|t-1}=f(\hat T_i^{t-1},u_t,\Delta t),\quad
\Sigma_i^{T,t|t-1}=F_i\Sigma_i^{T,t-1}F_i^\top+Q_T \tag{7}
\]

借鉴 CGPM Predictor（STATIC/CV/CTRV）思想 [C5]。

### 2.4.2 场预测（warp + 衰减）

\[
\phi_i^{t|t-1}(\mathbf x)=\phi_i^{t-1}\!\left((\Delta T_i)^{-1}\mathbf x\right),\quad
\mu_i^{g,t|t-1}(\mathbf x)=R(\Delta T_i)\mu_i^{g,t-1}\!\left((\Delta T_i)^{-1}\mathbf x\right) \tag{8}
\]

\[
\rho_i^{t|t-1}(\mathbf x)=\lambda_\rho\,\rho_i^{t-1}\!\left((\Delta T_i)^{-1}\mathbf x\right),\quad
\Sigma_i^{g,t|t-1}=\Sigma_i^{g,t-1}+Q_g \tag{9}
\]

### 2.4.3 梯度场投影算子

\[
\Pi_i(\mathbf x_k)=\mathbf x_k-\frac{\phi_i^{t|t-1}(\mathbf x_k)}{\|\mu_i^{g,t|t-1}(\mathbf x_k)\|^2+\epsilon}\,\mu_i^{g,t|t-1}(\mathbf x_k) \tag{10}
\]

是 CGPM 投影算子 $\Pi_\gamma$ 的隐式场版本 [C6]。

### 2.4.4 关联残差与门限

\[
r_d=\phi_i^{t|t-1}(\mathbf x_k),\qquad
r_n=1-\mathbf n_k^\top\hat{\mathbf n}_i^{t|t-1}(\mathbf x_k) \tag{11}
\]

\[
\mathbf r_{ik}=[r_d,\;r_n]^\top,\quad
R_{ik}^{\text{eff}}=
\frac{R_0}{1+\alpha\|\mathbf c_i^{t|t-1}(\mathbf x_k)\|}
+\beta\,\mathrm{tr}\!\left(\Sigma_i^{g,t|t-1}(\mathbf x_k)\right)I_2 \tag{12}
\]

\[
d_{ik}^2=\mathbf r_{ik}^\top (R_{ik}^{\text{eff}})^{-1}\mathbf r_{ik},\quad
d_{ik}^2<\tau_i \Rightarrow \text{accept} \tag{13}
\]

式(12)是本框架关键创新：证据梯度越强，测量噪声越小；梯度不确定度越大，门限越保守。

### 2.4.5 证据累积

\[
w_k=\frac{\exp(-\frac12 d_{ik}^2)}{\sigma_{d,k}^2+\lambda_n\sigma_{n,k}^2},\quad
\mathcal E_i\leftarrow \mathcal E_i\cup\{(\mathbf x_k,\mathbf n_k,w_k,t)\} \tag{14}
\]

\[
\rho_i^{t}(\mathbf x)=\lambda_\rho \rho_i^{t|t-1}(\mathbf x)+\sum_{k\in\mathcal A_i^t}w_kK_h(\|\mathbf x-\mathbf x_k\|),\quad
\mathbf c_i^{t}=\nabla\rho_i^t \tag{15}
\]

### 2.4.6 梯度贝叶斯融合

对体素 $\mathbf v$ 的后验更新：
\[
K_{i,\mathbf v}=\Sigma_{i,\mathbf v}^{g,-}\left(\Sigma_{i,\mathbf v}^{g,-}+R_{i,\mathbf v}^{g,\text{eff}}\right)^{-1} \tag{16}
\]

\[
\mu_{i,\mathbf v}^{g,+}=\mu_{i,\mathbf v}^{g,-}+K_{i,\mathbf v}\left(\tilde{\mathbf g}_{\mathbf v}-\mu_{i,\mathbf v}^{g,-}\right),\quad
\Sigma_{i,\mathbf v}^{g,+}=(I-K_{i,\mathbf v})\Sigma_{i,\mathbf v}^{g,-} \tag{17}
\]

其中
\[
\tilde{\mathbf g}_{\mathbf v}=
\mathrm{normalize}\!\Big(
\eta_n\bar{\mathbf n}_{\mathbf v}
+\eta_r\bar{\mathbf g}_{\text{ray},\mathbf v}
+\eta_c\,\mathrm{normalize}(\mathbf c_{i,\mathbf v}^{t})
\Big),\quad \eta_n+\eta_r+\eta_c=1 \tag{18}
\]

式(16)(17)是 Bayesian 融合主干（对应评估中“概率融合”）[P5]。

### 2.4.7 SDF 重建（Poisson + Eikonal）

\[
\phi_i^t=
\arg\min_{\phi}
\Big[
\sum_{\mathbf v\in\Omega_i}\omega_{\mathbf v}\|\nabla\phi(\mathbf v)-\mu_{i,\mathbf v}^{g,+}\|_2^2
+\lambda_{\text{eik}}\sum_{\mathbf v}(\|\nabla\phi(\mathbf v)\|_2-1)^2
+\lambda_{\text{np}}\sum_{k\in\mathcal A_i^t}w_k\big(\phi(\mathbf x_k)-d_k^{\text{np}}\big)^2
\Big] \tag{19}
\]

该式融合了梯度域重建 [P4][P6] 与 non-projective 距离监督 [P1]。

### 2.4.8 位姿更新（IEKF）

\[
S_i=H_i\Sigma_i^{T,t|t-1}H_i^\top+R_i,\quad
K_i^T=\Sigma_i^{T,t|t-1}H_i^\top S_i^{-1} \tag{20}
\]

\[
\delta\xi_i=K_i^T \bar{\mathbf r}_i,\quad
\hat T_i^t=\hat T_i^{t|t-1}\exp(\delta\xi_i^\wedge),\quad
\Sigma_i^{T,t}=(I-K_i^T H_i)\Sigma_i^{T,t|t-1} \tag{21}
\]

其中 $\bar{\mathbf r}_i$ 是关联集合上的残差堆叠。

---

## 3. 状态演化流程设计

## 3.1 总流程（Predict → Associate → Update → Merge/Split）

### 算法 1：EGF-DHMap 主循环

```text
Input:
  Z_t: 当前帧观测
  H_{t-1}: 上一帧实体集合
  U_t: 运动输入（里程计/IMU）
Output:
  H_t: 更新后实体集合

1) Predict:
   for each entity H_i in H_{t-1}:
      pose predict via Eq.(7)
      field warp+decay via Eq.(8)(9)
      refresh frontier candidates from low-ρ / high-|∇ρ| zones

2) Associate:
   A_t, F_t = AssociateGradientObservations(Z_t, H_{t|t-1})  # 算法2

3) Update:
   for each entity H_i:
      pose IEKF update via Eq.(20)(21)
      evidence accumulation via Eq.(14)(15)
      voxel gradient Bayesian update via Eq.(16)(17)(18)
      local SDF reconstruction via Eq.(19)

4) Topology:
   H_t = TopologyManager(H_t, F_t)  # 算法3 (Merge/Split)

5) Lifecycle:
   prune stale entities; spawn new entities from persistent frontier evidence
```

## 3.2 预测阶段（Predict）

输入：$H_{t-1},U_t$；输出：$H_{t|t-1}$。  
关键步骤：

1. 位姿与协方差预测（式(7)），直接继承 CGPM 的运动模型体系 [C5]。  
2. 场状态随位姿做坐标变换（式(8)(9)）。  
3. 证据衰减（防止历史拖尾），并更新前沿候选：  
- 低覆盖：$\rho(\mathbf v)<\tau_\rho$  
- 前沿强度：$\|\nabla\rho(\mathbf v)\|>\tau_c$  
4. 若实体长期无证据支撑，进入“待删除队列”。

## 3.3 关联阶段（Associate）

输入：$Z_t,H_{t|t-1}$；输出：匹配集 $\mathcal A_t$ 与前沿缓存事件 $\mathcal F_t$。

### 算法 2：梯度场关联器

```text
for each observation z_k:
  1) coarse preselect top-K entities by distance to active voxel blocks
  2) for each candidate H_i:
       x*_ik = Π_i(x_k)                      # Eq.(10)
       compute residual r_ik=[r_d,r_n]       # Eq.(11)
       build R_eff with ∇ρ and Σ_g           # Eq.(12)
       compute Mahalanobis d^2_ik            # Eq.(13)
  3) choose best i* with minimal d^2 and pass gate
  4) if pass:
       add (k -> i*) to A_t
     else if near frontier and direction-consistent:
       push z_k into frontier cache F_t
```

### 3.3.1 梯度场投影算子

采用式(10)进行一次牛顿式投影；可做 2-3 次迭代提高稳定性。该设计与 CGPM “粗搜+refine 投影”在结构上对齐 [C6]。

### 3.3.2 残差定义

- 距离残差：$r_d$ 约束点到零水平集距离。  
- 法向残差：$r_n$ 约束观测法向与场梯度方向一致。  

对应“几何 + 方向”双约束，比仅用距离更抗误匹配。

### 3.3.3 门限设计（证据梯度调权）

核心是式(12)：  
- 若证据梯度强（边界清晰、观测充足），$R_{\text{eff}}$ 变小，接受更严格。  
- 若梯度后验不确定度高，$R_{\text{eff}}$ 变大，避免过拟合噪声。

这等价于把 CGPM 的马氏门限拓展为“证据-不确定度自适应门限” [C7]。

## 3.4 更新阶段（Update）

### 3.4.1 位姿更新

按式(20)(21)做 IEKF。沿用 CGPM 的“先估计位姿”思想 [C8]。

### 3.4.2 梯度场贝叶斯融合

对每个受观测影响的体素执行式(16)(17)，并用式(18)融合三类梯度线索：

- 法向观测主导（$\bar{\mathbf n}_{\mathbf v}$）  
- 射线/距离诱导梯度（$\bar{\mathbf g}_{\text{ray},\mathbf v}$）  
- 证据梯度引导（$\nabla\rho$）

### 3.4.3 SDF 重建与可积性修正

在局部 ROI 内最小化式(19)：  
- 第一项：拟合梯度后验  
- 第二项：Eikonal 正则  
- 第三项：non-projective 距离监督  

数值求解建议：多重网格预条件 + 共轭梯度，迭代上限固定保证实时性。

### 3.4.4 证据累积与权重更新

用式(14)(15)累计证据；对历史证据按时间衰减，保留“长期稳定结构”并抑制瞬时动态干扰。对应 CGPM 证据集持续积累思想 [C2][C7]。

## 3.5 拓扑管理（Merge/Split）

### 3.5.1 Merge 条件

对实体 $H_a,H_b$，定义：

\[
S_g=\exp\!\left(-\frac{1}{|\Gamma|}\sum_{\mathbf x\in\Gamma}
\frac{\|\hat{\mathbf n}_a(\mathbf x)-\hat{\mathbf n}_b(\mathbf x)\|^2}{\sigma_g^2}\right),\quad
S_\rho=\frac{\sum_{\mathbf v\in\Omega_a\cap\Omega_b}\min(\rho_a,\rho_b)}
{\sum_{\mathbf v\in\Omega_a\cup\Omega_b}\max(\rho_a,\rho_b)} \tag{22}
\]

\[
\chi_{\text{pose}}^2=\xi_{ab}^\top\Sigma_{ab}^{-1}\xi_{ab},\quad
\Sigma_{ab}=J_a\Sigma_a^TJ_a^\top+J_b\Sigma_b^TJ_b^\top \tag{23}
\]

Merge 判据：
\[
S_g>\tau_g,\;S_\rho>\tau_\rho,\;\chi_{\text{pose}}^2<\tau_{\text{pose}} \Rightarrow \text{Merge}(a,b) \tag{24}
\]

这正是对 CGPM README Merge 条件的梯度场化落地 [C1][C10]。

### 3.5.2 Split 条件

若同一实体内部出现以下任一现象则触发 Split 候选：

1. 残差分布双峰：$p(r)$ 通过 BIC 更偏向 2 高斯。  
2. 证据断裂带：存在连通切割使 $S_\rho<\tau_{\rho,\text{cut}}$。  
3. 梯度多模态：区域内法向主方向聚类数 $>1$ 且夹角 $>\tau_\theta$。  

### 算法 3：拓扑管理器

```text
Input: entity set H_t
1) build adjacency graph of entities by spatial overlap
2) for each adjacent pair (a,b):
     compute S_g, S_ρ, χ_pose^2 via Eq.(22)(23)
     if pass Eq.(24): merge and fuse fields/covariances/evidence
3) for each entity i:
     test split criteria (bimodal residual, evidence cut, gradient multimodality)
     if split:
        spectral partition active voxels into two sets
        spawn H_i1, H_i2 with re-initialized (T, Σ^T, μ_g, Σ_g, ρ)
4) apply hysteresis (N_confirm frames) to avoid oscillatory merge/split
```

---

## 4. 实现架构设计

## 4.1 数据结构

### 4.1.1 体素单元

```cpp
struct VoxelCell {
  float phi;              // SDF
  Vec3  g_mean;           // 梯度均值 μ_g
  Mat3  g_cov;            // 梯度协方差 Σ_g
  float rho;              // 证据置信度
  Vec3  c_rho;            // 证据梯度 ∇ρ
  float dyn_prob;         // 动态概率(可选)
  uint16_t age;
};
```

### 4.1.2 局部实体

```cpp
struct GradientEntity {
  EntityId id;
  SE3      T_hat;
  Mat6     Sigma_T;
  HashGridMultiScale<VoxelCell> blocks;
  EvidenceBuffer evidence;
  FrontierCache frontier;
  Timestamp last_update;
};
```

### 4.1.3 全局管理器

- `HypothesisPool`：实体生命周期管理（创建、激活、冻结、删除）  
- `TopologyGraph`：实体邻接与候选合并图

## 4.2 算法模块与 I/O

| 模块 | 输入 | 输出 | 核心算法 |
|---|---|---|---|
| Predictor | `H_{t-1}, U_t` | `H_{t|t-1}` | 位姿模型 + 场 warp/衰减（式(7)(8)(9)） |
| Associator | `Z_t, H_{t|t-1}` | `A_t, F_t` | 投影算子 + 马氏门限 + 前沿缓存（式(10)-(13)） |
| Updater | `A_t, F_t, H_{t|t-1}` | `H_t'` | IEKF + 证据累积 + 梯度后验 + SDF 重建（式(14)-(21)） |
| TopologyManager | `H_t'` | `H_t` | Merge/Split 判据与图优化（式(22)-(24)） |

## 4.3 工程优化策略

1. **多尺度哈希体素**：粗层负责关联/位姿，细层负责几何细节；借鉴 Gradient-SDF / MS²IS [P2][P3]。  
2. **局部 ROI 更新**：只更新被命中体素与邻域块，降低式(19)求解规模。  
3. **异步重建**：关联/滤波主线程 + Poisson/Eikonal 子线程。  
4. **鲁棒法向估计**：时空双边滤波 + Huber 代价，减轻梯度噪声放大。  
5. **动态抑制**：用高 $\mathrm{tr}(\Sigma_g)$ 与低 $\rho$ 标记可疑动态体素，参考 UN3 不确定度思想 [P1]。

---

## 5. 与 CGPM 的对比与创新分析

## 5.1 借鉴对照表（模块级）

| 设计模块 | CGPM 借鉴来源 | 迁移改造方式 | 结果 |
|---|---|---|---|
| 证据场 | 证据集/密度/密度导数 [C2][C4] | $\mu_E(s)\to \rho(\mathbf x), \partial\mu/\partial s\to \nabla\rho$ | 支持 3D 前沿感知与关联调权 |
| 假设实体 | Entity 结构 [C3] | 曲线实体改为局部梯度场实体 $H_i$ | 局部地图可独立生命周期管理 |
| Predict | Motion models [C5] | 位姿预测 + 场 warp/衰减 | 动态场景时序连续性 |
| Associate | 投影与马氏门限 [C6][C7] | 曲线投影改为 SDF 零水平集投影 + 法向残差 | 关联更几何一致、抗错配 |
| Update | EKF + 几何融合解耦 [C8][C9] | IEKF + 梯度贝叶斯融合 + Poisson/Eikonal | 稳定与精度兼顾 |
| 边界生长 | 缓存与边界扩展 [C2][C7][C9] | 前沿缓存驱动实体生长 | 未知区域扩展可控 |
| Merge | README 定义 [C1] | 梯度相似+证据重叠+位姿一致落地 | 补齐 C10 缺位 |

## 5.2 相比现有梯度场工作的增益

1. **对 UN3-Mapping [P1] 的增益**  
- UN3 强在隐式建图与不确定度；EGF-DHMap 增加了实体级生命周期和拓扑管理，适合多目标动态场景。

2. **对 Gradient-SDF [P2] 的增益**  
- Gradient-SDF 强在半隐式表达与优化；EGF-DHMap 增加证据梯度场 $\nabla\rho$，在“是否该更新”上更可解释。

3. **对 MS²IS [P3] 的增益**  
- MS²IS 强在多尺度效率；EGF-DHMap 在其上增加贝叶斯后验与 Merge/Split，强化长期一致性。

4. **对传统梯度融合 [P4][P5][P6] 的增益**  
- 传统工作多是离线融合 + 重建；EGF-DHMap 将其在线化并嵌入 SLAM 时序闭环。

---

## 6. 可行性与实现路线

## 6.1 理论可行性

1. 梯度域融合与 Poisson 重建有成熟基础 [P4][P6]。  
2. 随机/线性统计融合可支撑贝叶斯梯度更新 [P5]。  
3. 半隐式梯度场已在 3D 重建验证有效 [P2][P3]。  
4. 不确定度驱动动态抑制在神经隐式建图中已验证 [P1]。  
5. 概率梯度（log-prob gradient）解释可用于动态去噪 [P7]。

## 6.2 工程可行性

设每帧观测数 $N_o$，候选实体平均数 $K$，更新体素数 $N_v$，则：

- 关联复杂度：$O(N_oK)$（通过空间索引使 $K\ll |\mathcal H|$）  
- 梯度融合复杂度：$O(N_v)$  
- 局部重建复杂度：$O(I\cdot N_{\text{ROI}})$，$I$ 为固定迭代次数  
- 内存复杂度：$O(N_{\text{active voxels}})$（多尺度哈希）

在 ROI + 异步求解下可满足实时近似（10-20Hz，依赖硬件）。

## 6.3 分阶段实现路线

### Phase A：静态基线（4-6 周）

目标：
- 实现单实体 SDF+梯度体素表示  
- 完成 Predict/Associate/Update 主链路（不含拓扑）

验证：
- 重建误差（Chamfer, Normal Consistency）  
- 位姿误差（ATE/RPE）

### Phase B：双场耦合与动态抑制（4-6 周）

目标：
- 加入 $\rho,\nabla\rho$ 与式(12)自适应门限  
- 加入不确定度后验式(16)(17)

验证：
- 动态区域误关联率  
- 噪声场景稳定性（轨迹抖动、重建漂移）

### Phase C：拓扑管理（6-8 周）

目标：
- 实现 Merge/Split（算法3）  
- 生命周期管理与前沿生长闭环

验证：
- Merge Precision/Recall  
- Split Precision/Recall  
- 长时一致性（地图破碎率、ID 稳定性）

### Phase D：性能与工程化（持续）

目标：
- GPU 化、异步管线、参数自标定  
- 融合神经残差头（可选）

---

## 7. 风险分析与缓解策略

| 风险 | 表现 | 触发原因 | 缓解策略 |
|---|---|---|---|
| 梯度噪声放大 | 法向抖动、表面毛刺 | 观测法向不稳定 | 鲁棒法向估计 + Huber 权重 + 低证据区降权 |
| 双场耦合不稳定 | 过度平滑或误关联 | $\alpha,\beta,\eta_*$ 设置不当 | 分层调参（先关 $\eta_c$ 再逐步开启）+ 梯度裁剪 |
| 可积性破坏 | 局部拓扑撕裂 | 仅拟合梯度未约束旋度 | 强化 Eikonal/Poisson 项，周期性全局一致化 |
| Merge/Split 阈值敏感 | 实体振荡合并/分裂 | 边界场景多义性 | 双阈值滞回 + 连续 $N$ 帧确认 + 代价历史平滑 |
| 计算超预算 | 帧率下降 | 活跃体素暴增 | ROI 更新、块级 LOD、异步求解、预算驱动裁剪 |
| 动态物体拖尾 | 地图残影 | 旧证据衰减慢 | 时间衰减 $\lambda_\rho$ 自适应 + 动态概率剔除 |

---

## 8. 总结与展望

EGF-DHMap 的核心价值是：  
1. 把 CGPM 的成功工程范式（证据驱动、假设实体、贝叶斯时序）完整迁移到梯度场建图；  
2. 在梯度场侧加入“证据-不确定度联合融合 + 可积性约束重建 + 拓扑级 Merge/Split”三类关键创新；  
3. 面向动态场景提供可解释、可扩展、可落地的实现路径。

预期效果：
- 提升动态场景下的重建完整性和几何一致性；  
- 降低误关联和动态残影；  
- 在多实体长期运行中保持拓扑稳定。

未来方向：
1. 把 $\rho,\Sigma_g$ 接入主动感知策略（Next-Best-View）。  
2. 引入学习型先验预测 $\eta_*$ 与阈值，实现自适应参数控制。  
3. 扩展到多机器人共享实体图（跨平台 Merge/Split）。

---

## 参考文献与代码依据

### A. 梯度场与融合相关文献

[P1] Song, S., Zhao, J., Veas, E., et al. **UN3-Mapping: Uncertainty-Aware Neural Non-Projective Signed Distance Fields for 3D Mapping**. IEEE RA-L, 2025. DOI: https://doi.org/10.1109/LRA.2025.3588410  
[P2] Sommer, C., Sang, L., Schubert, D., Cremers, D. **Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction**. CVPR, 2022. DOI: https://doi.org/10.1109/CVPR52688.2022.00618  
[P3] Deng, Z., Wu, X., Wang, M. Y. **MS²IS: A Multi-Scale Semi-Implicit Surface Representation for 3D Reconstruction**. IEEE RA-L, 2025. DOI: https://doi.org/10.1109/LRA.2025.3597899  
[P4] Wen, J., Li, Y., Gong, H. **Remote Sensing Image Fusion on Gradient Field**. ICPR, 2006. DOI: https://doi.org/10.1109/ICPR.2006.995  
[P5] Sankaranarayanan, A. C., Chellappa, R. **Stochastic fusion of multi-view gradient fields**. ICIP, 2008. DOI: https://doi.org/10.1109/ICIP.2008.4712007  
[P6] Gu, B., Li, W., Wong, J., et al. **Gradient field multi-exposure images fusion for high dynamic range image visualization**. JVCIR, 2012. DOI: https://doi.org/10.1016/J.JVCIR.2012.02.009  
[P7] Hu, Q., Hu, W. **Dynamic Point Cloud Denoising via Gradient Fields**. TOMM, 2025. DOI: https://doi.org/10.1145/3721431  
[P8] Song, S., Zhao, J., Veas, E., et al. **N3-Mapping: Normal Guided Neural Non-Projective Signed Distance Fields for 3D Mapping**. arXiv, 2024. https://arxiv.org/abs/2401.03412  
[P9] Kalman, R. E. **A New Approach to Linear Filtering and Prediction Problems**. Journal of Basic Engineering, 1960. DOI: https://doi.org/10.1115/1.3662552

### B. CGPM 代码与文档证据（必须项）

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

