## 基于参数化流形与贝叶斯估计的动态局部感知系统形式化模型

原始参考链接：https://chatglm.cn/share/BFcqKu3x

### 1. 系统输入：射线测量空间

**定义 1.1（观测元 $z$）**：单个观测元 $z$ 定义为四元组：

$$
z = (\phi, r, \sigma^2, t)
$$

其中 $\phi \in [0, 2\pi)$ 为射线角度，$r \in \mathbb{R}^+$ 为测距值，$\sigma^2$ 为已知测距噪声方差，$t$ 为时间戳。

**定义 1.2（观测序列 $Z_{1:t}$）**：时刻 $t$ 的系统输入为历史观测集合：

$$
Z_{1:t} = \{ z_1, \dots, z_t \}
$$

### 2. 系统状态：假设实体集合

**定义 2.1（局部几何模型 $\mathcal{M}$）**：假设实体的几何形状由参数化流形表示。$\mathcal{M}$ 定义为一个三元组：

$$
\mathcal{M} = (P, \Phi, \theta)
$$

- $P = \{ \mathbf{p}_i \in \mathbb{R}^2 \}_{i=1}^N$：控制点集合。
- $\Phi$：基函数族，满足完备性条件——$\{\phi_i\}$ 在空间 $L^2[0, \hat{L}]$ 中线性无关，且能张成足够丰富的函数空间以逼近真实边界。
- $\theta$：形状参数向量。

由此生成的参数化曲线 $\gamma: [0, \hat{L}] \to \mathbb{R}^2$ 定义为：

$$
\gamma(s) = \sum_{i=1}^N \mathbf{p}_i \cdot \phi_i(s; \theta)
$$

**定义 2.2（证据集 $\mathcal{E}$）**：证据集是支持该假设的所有历史观测的带权结构化集合：

$$
\mathcal{E} = \{ e_k = (s_k, \mathbf{z}_k, w_k, t_k) \}
$$

- $s_k \in [0, \hat{L}]$：观测点在局部模型上的参数坐标（反投影结果），即 $s_k = \Pi_{\gamma}(\mathbf{z}_k)$，其中 $\Pi_{\gamma}$ 为定义于算子 3.2 中的投影算子。
- $\mathbf{z}_k \in \mathbb{R}^2$：该观测点的全局笛卡尔坐标 $\mathbf{z}_k = r_k \mathbf{u}(\phi_k)$。
- $w_k \in \mathbb{R}^+$：权重，通常与测距方差成反比 $w_k \propto 1/\sigma_k^2$。
- $t_k$：观测时间。

**定义 2.3（假设实体 $H$）**：一个假设实体 $H$ 是一个六元组：

$$
H = (\mathcal{M}, \hat{g}, \Sigma_g, \mathcal{E}, \hat{L}, t_{last})
$$

- $\hat{g} \in SE(2)$：从局部坐标系到全局坐标系的位姿估计。
- $\Sigma_g \in \mathbb{R}^{3 \times 3}$：位姿估计的协方差矩阵（定义在李代数 $\mathfrak{se}(2)$ 上）。
- $t_{last}$：最后一次更新时间。

**定义 2.4（端点集 $\mathcal{B}$）**：用于定义假设的开放边界。

$$
\mathcal{B} = \{ \hat{g} \cdot \gamma(0), \hat{g} \cdot \gamma(\hat{L}) \} \cup \{ \hat{g} \cdot \gamma(s) \mid s \in [0, \hat{L}], \tfrac{\partial \mu_E(s)}{\partial s} < \delta \}
$$

其中 $\mu_E(s)$ 是基于证据密度定义的函数（见定义 4.1）。低密度梯度点指示了开放边界。

### 3. 状态演化算子

系统状态转移由算子 $F$ 决定：

$$
\mathcal{H}(t) = F(\mathcal{H}(t-\Delta t), Z_t)
$$

**算子 3.1（预测 Predict）**：基于恒定速度模型（或其他运动模型 $u$）：

$$
\hat{g}(t) = \hat{g}(t-\Delta t) \cdot \exp(u \Delta t)
$$

协方差更新：

$$
\Sigma_g(t) = J \Sigma_g(t-\Delta t) J^T + Q
$$

其中 $J$ 为雅可比矩阵，$Q$ 为过程噪声协方差。

**算子 3.2（关联 Associate）**：

首先定义投影算子 $\Pi_{\gamma}: \mathbb{R}^2 \to [0, \hat{L}]$，对于全局点 $\mathbf{z}$：

$$
\Pi_{\gamma}(\mathbf{z}) = \underset{s \in [0, \hat{L}]}{\arg\min} \; \| \hat{g} \cdot \gamma(s) - \mathbf{z} \|^2
$$

若存在多个局部极小值（射线与模型相交于多点），选取使得切向夹角最小的解：

$$
s^* = \underset{s \in S_{min}}{\arg\min} \; \big| \angle(\mathbf{u}(\phi_{ray})) - \angle(\gamma'(s)) \big|
$$

其中 $S_{min}$ 是使距离达到全局最小值的参数集合。

基于此，定义观测模型 $h(H, \phi)$：计算射线 $R_\phi$ 与模型 $\gamma$ 的最近交点。设 $s^* = \Pi_{\gamma}(r \mathbf{u}(\phi))$ 为取得最小值的参数，预测观测值为 $\hat{z}_{pred} = \hat{g} \cdot \gamma(s^*)$。

对于新观测 $z_i$ 和假设 $H_j$，定义马氏距离：

$$
d_{ij}^2 = (\mathbf{z}_i - \hat{z}_{pred, j})^T \Sigma_{total}^{-1} (\mathbf{z}_i - \hat{z}_{pred, j})
$$

其中 $\Sigma_{total} = \Sigma_{obs} + J_{pose} \Sigma_g J_{pose}^T$。若 $d_{ij}^2 < \chi^2_{threshold}$，则建立关联。

**算子 3.3（更新 Update）**：采用解耦优化策略进行更新。

- 状态估计（EKF）：固定几何模型 $(P, \theta)$，仅优化位姿 $\hat{g}$ 和其协方差 $\Sigma_g$。
- 几何融合（形状拟合）：固定第一阶段更新后的位姿 $\hat{g}_{new}$，求解关于 $(P, \theta)$ 的带正则项加权最小二乘问题：

$$
\min_{P, \theta} \sum_{e_k \in \mathcal{E}} w_k \cdot \| \mathbf{z}_k - \hat{g}_{new} \cdot \gamma(s_k) \|^2 + \lambda \cdot \mathcal{R}(P)
$$

其中 $\lambda$ 为正则化系数，$\mathcal{R}(P)$ 为控制点的平滑正则项。若需更高精度，可采用坐标下降法在上述两阶段间进行迭代。

证据累积：$\mathcal{E} \leftarrow \mathcal{E} \cup \{ (s_{new}, \mathbf{z}_i, w_i, t) \}$。

**算子 3.4（拓扑合并 Merge $\Psi$）**

对于两个假设 $H_a, H_b$，若满足以下条件则执行合并 $\Psi(H_a, H_b) \to H_{merged}$：

1) 几何邻近性：存在 $b_a \in \mathcal{B}_a, b_b \in \mathcal{B}_b$，使得 $|b_a - b_b| < \epsilon_{dist}$。

2) 统计一致性检验：计算相对位姿 $\xi_{rel} = \log(\hat{g}_a^{-1} \hat{g}_b) \in \mathfrak{se}(2)$，设 $\xi_{rel}$ 的均值为 $\mathbf{0}$，协方差为 $\Sigma_{rel} = J_a \Sigma_{g_a} J_a^T + J_b \Sigma_{g_b} J_b^T$。若 $\xi_{rel}^T \Sigma_{rel}^{-1} \xi_{rel} < \chi^2_{merge}$，则判定为同一实体的不同部分。

合并执行：

- 参考系统一：选取 $H_a$ 为基准，计算变换 $T_{rel} = \hat{g}_a^{-1} \hat{g}_b$。
- 证据融合：

$$
\mathcal{E}_{merged} = \mathcal{E}_a \cup \{ (s, T_{rel}\mathbf{z}, w, t) \mid (s, \mathbf{z}, w, t) \in \mathcal{E}_b \}
$$

- 模型重构（自适应）：保持基函数族 $\Phi$ 的数学形式不变。根据融合后证据集 $\mathcal{E}_{merged}$ 在参数空间 $[0, \hat{L}_{new}]$ 中的分布密度，自适应调整控制点数量 $N_{new}$（例如 $N_{new} = \lceil \hat{L}_{new} / dL \rceil$），利用最小二乘法拟合新的 $P_{new}$ 和 $\theta_{new}$，生成 $\mathcal{M}_{merged}$。

### 4. 完备性与几何闭合

**定义 4.1（证据测度与覆盖）**：定义参数空间上的证据密度测度 $\mu_E: [0, \hat{L}] \to \mathbb{R}^+$：

$$
\mu_E(s) = \sum_{e_k \in \mathcal{E}} w_k \cdot K_h(s - s_k)
$$

其中 $K_h(\cdot)$ 为高斯核函数，其带宽 $h$ 根据测量噪声水平自适应确定：

$$
h = c \cdot \bar{\sigma}
$$

这里 $\bar{\sigma}$ 是当前证据集的平均测距标准差，$c$ 为经验常数（通常取 2-3）。这确保了核的尺度与不确定性相匹配。

**定义 4.2（几何闭合）**：假设实体 $H$ 达到几何闭合，当且仅当满足以下量化条件。其中参数 $s_{start}$ 与 $s_{end}$ 被严格限定为模型参数域的边界，即 $s_{start} = 0$ 且 $s_{end} = \hat{L}$。

- 高覆盖度：测度比 $\tfrac{m(I_{covered})}{\hat{L}} > 1 - \epsilon_{gap}$。
- 位置闭合：$\| \gamma(0) - \gamma(\hat{L}) \| < \epsilon_{pos}$。
- 切向闭合：$\| \gamma'(0) - \gamma'(\hat{L}) \| < \epsilon_{tan}$。
- 拓扑完整性（量化）：曲率积分满足：

$$
\left| \int_{0}^{\hat{L}} \kappa(s) ds - 2\pi \right| < \epsilon_{\kappa}
$$

### 5. 系统性质

**定义 5.1（系统一致性）**：若对于静态场景（输入 $Z_t$ 由固定生成过程产生），系统满足：

$$
\lim_{t \to \infty} \text{tr}(\Sigma_g(t)) = 0
$$

$$
\lim_{t \to \infty} \int_0^{\hat{L}(t)} \| \gamma(s; \theta_t) - \gamma_{true}(s) \|^2 ds = 0
$$

则称该系统是统计一致的。

**定义 5.2（可观测性）**：称实体 $H$ 是可观测的，如果存在输入序列 $Z_{1:t}$，使得从任意初始状态 $(\hat{g}_0, \Sigma_{g,0})$ 出发，估计误差 $(\hat{g}(t) - g_{true})$ 均方收敛至 0。
