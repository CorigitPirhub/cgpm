# DESIGN_A：Schrödinger–Agmon 对数振幅梯度场

## 1. 目标定位

方案 A 的目标是把“梯度场构建”转化成一个带势项、带各向异性张量的椭圆型场求解问题。其核心思想不是直接构造距离，而是先构造一个正场 `\psi_\varepsilon(x)`，再通过对数变换得到距离型函数：

\[
D_\varepsilon(x) = -\varepsilon \log \psi_\varepsilon(x),
\qquad
G_\varepsilon(x) = -\nabla D_\varepsilon(x).
\]

该思路与 `Log-GPIS-MOP` 的共同点是都采用“线性 PDE 解 + 对数变换”；不同点是这里对应的不是热方程，而是带势 Schrödinger 型方程，因此后续调用的不是 Varadhan 公式，而是 Agmon / WKB / 半经典衰减理论。

## 2. 数学原型

### 2.1 核心公式

在域 `\Omega \subset \mathbb{R}^d` 中，给定表面或目标集合 `\Gamma`，定义正场 `\psi_\varepsilon : \Omega \setminus \Gamma \to (0, +\infty)` 满足：

\[
-\varepsilon^2 \nabla \cdot (M(x) \nabla \psi_\varepsilon(x)) + V(x)\psi_\varepsilon(x) = 0,
\qquad x \in \Omega \setminus \Gamma,
\]

边界条件取：

\[
\psi_\varepsilon(x) = 1, \qquad x \in \Gamma.
\]

其中：

- `M(x) \succ 0` 是各向异性正定张量；
- `V(x) \ge 0` 是势函数；
- `\varepsilon > 0` 是尺度参数。

然后定义：

\[
D_\varepsilon(x) := -\varepsilon \log \psi_\varepsilon(x).
\]

### 2.2 严格等价推导

令

\[
\psi_\varepsilon(x)=\exp\left(-\frac{D_\varepsilon(x)}{\varepsilon}\right).
\]

则

\[
\nabla\psi_\varepsilon = -\frac{1}{\varepsilon}\exp\left(-\frac{D_\varepsilon}{\varepsilon}\right)\nabla D_\varepsilon.
\]

从而

\[
M\nabla\psi_\varepsilon = -\frac{1}{\varepsilon}\exp\left(-\frac{D_\varepsilon}{\varepsilon}\right)M\nabla D_\varepsilon.
\]

取散度得

\[
\nabla\cdot(M\nabla\psi_\varepsilon)
=
-\frac{1}{\varepsilon}\exp\left(-\frac{D_\varepsilon}{\varepsilon}\right)\nabla\cdot(M\nabla D_\varepsilon)
+
\frac{1}{\varepsilon^2}\exp\left(-\frac{D_\varepsilon}{\varepsilon}\right)
\langle \nabla D_\varepsilon, M\nabla D_\varepsilon\rangle.
\]

代回原方程：

\[
-\varepsilon^2\nabla\cdot(M\nabla\psi_\varepsilon)+V\psi_\varepsilon = 0,
\]

可得

\[
\varepsilon\exp\left(-\frac{D_\varepsilon}{\varepsilon}\right)\nabla\cdot(M\nabla D_\varepsilon)
-
\exp\left(-\frac{D_\varepsilon}{\varepsilon}\right)
\langle \nabla D_\varepsilon, M\nabla D_\varepsilon\rangle
+
V\exp\left(-\frac{D_\varepsilon}{\varepsilon}\right)=0.
\]

除以正因子 `\exp(-D_\varepsilon/\varepsilon)`，得到等价非线性 PDE：

\[
\boxed{
\langle \nabla D_\varepsilon, M\nabla D_\varepsilon\rangle
-
\varepsilon\nabla\cdot(M\nabla D_\varepsilon)
=
V(x).
}
\]

这是方案 A 最关键的一步：

> 以 `\psi_\varepsilon` 为未���量���线性椭圆方程，与以 `D_\varepsilon` 为未知量的粘性加权 Eikonal 方程严格等价。

### 2.3 极限行为与理论含义

当 `\varepsilon \to 0`，粘性项消失，得到：

\[
\boxed{
\langle \nabla D, M\nabla D\rangle = V(x).
}
\]

若定义 Riemann 型度量张量

\[
G(x) := V(x) M(x)^{-1},
\]

则对应的测地距离为

\[
 d_A(x,\Gamma) = \inf_{\gamma(0)\in\Gamma,\gamma(1)=x}
 \int_0^1 \sqrt{\dot\gamma(t)^\top G(\gamma(t))\dot\gamma(t)}\,dt.
\]

因此 `D` 是一个势函数控制的各向异性距离；`-\nabla D` 则是向 `\Gamma` 的最小衰减路径方向。

## 3. 物理等价与可套用定理

### 3.1 物理公式

该方程本质上是半经典 Schrödinger 问题中的基态/隧穿振幅衰减方程。物理上 `\psi_\varepsilon` 是“波函数振幅”，而 `D_\varepsilon` 是对数振幅的缩放。

### 3.2 物理定理

可调用的核心定理是 Agmon 衰减估计：

- 在 classical forbidden region 内，本征函数按 `\exp(-d_A/\varepsilon)` 衰减；
- `d_A` 是由势函数决定的 Agmon 距离。

因此，若我们把不确定区域、动态边界、几何不一致区域编码进 `V(x)`，则对数振幅就天然生成“远离高势垒”的梯度场。

## 4. 对映射梯度场的意义

### 4.1 如何映射到 `EGF_DHMap`

可令：

\[
V(x) = \lambda_1 r_{dyn}(x) + \lambda_2 r_{occ}(x) + \lambda_3 r_{vis}(x) + \lambda_4 r_{geo}(x),
\]

其中：

- `r_dyn`：动态风险；
- `r_occ`：占据冲突/残差；
- `r_vis`：可见性缺口；
- `r_geo`：法向/平面一致性代价。

再令 `M(x)` 把前后方向、切向/法向方向分别赋予不同导通性。

这样得到的梯度场会天然具有：

- 动态边界屏障感知；
- 对可见性缺口与几何一致性的各向异性偏好；
- 对“穿过高风险区域”施加指数惩罚。

### 4.2 预期效果

理论上应出现以下效应：

1. `Ghost` 抑制更强，因为高动态风险区被直接写入 `V(x)`；
2. `Acc` 更稳，因为 `M(x)` 可沿高置信平面切向扩散、在法向抑制漂移；
3. `Comp-R` 增益取决于 `V(x)` 是否过度保守，需实验平衡。

## 5. 实验计划

### 5.1 实现目标

在 `experiments/design_a/` 中落一个最小原型求解器，验证 `D_\varepsilon = -\varepsilon \log \psi_\varepsilon` 是否能从“风险场 + 表面边界”稳定恢复 barrier-aware distance。

### 5.2 代码骨架

- `experiments/design_a/prototype.py`
  - `log_amplitude_distance(psi, epsilon)`
- 后续建议新增：
  - `solver.py`：离散化 `-ε² div(M∇ψ)+Vψ=0`
  - `experiment.py`：生成 toy scene，输出距离场与梯度图

### 5.3 实验分三层

#### 层 1：解析/半解析 sanity check

构造 1D 势垒：

- `\Gamma = {0}`
- 分段常数 `V(x)`
- 常数 `M=1`

验证：

- `D_\varepsilon` 是否随 `V` 增大而加速增长；
- 小 `\varepsilon` 下是否逼近加权 Eikonal 解。

#### 层 2：2D 合成几何场景

构造：

- 一条目标边界 `\Gamma`
- 一块高风险动态区域
- 一块可见性不确定区域
- 一块高置信平面区域

实验观测：

- 梯度线是否绕开高 `V` 区；
- 修改 `M` 后，梯度是否沿切向传播更远；
- 与普通 Eikonal/TSDF 梯度对比。

#### 层 3：接入 `EGF_DHMap`

输入：

- 当前 `111/122/123/125` 的 `r_dyn / p_occ / visibility_deficit / plane confidence`

构造：

- `V(x)`：风险叠加势
- `M(x)`：平面/法向各向异性张量

输出：

- 一个新的 `design_a_gradient_score`
- 作为 candidate activation 的 prior

### 5.4 评价指标

1. 数学层：
- `|| \langle ∇D, M∇D\rangle - V ||`
- `|| -ε² div(M∇ψ)+Vψ ||`

2. 几何层：
- 梯度方向与目标法向夹角
- 穿越高势垒区域的比例

3. 系统层：
- Bonn `Acc / Comp-R / Ghost`
- 与 `125`、`130` 比较

### 5.5 成功标准

- 在 toy scene 上验证对数振幅距离与加权 Eikonal 一致；
- 在真实数据上，相比 `125` 至少实现以下之一：
  - `Ghost` 进一步下降且 `Comp-R` 不回退；
  - `Acc` 改善且 `Ghost` 不恶化。

## 6. 风险评估

- 若 `V(x)` 设计过强，会导致极端保守，`Comp-R` 明显下降；
- 若 `M(x)` 构造不稳定，可能带来数值各向异性震荡；
- 若边界条件选取不对，`ψ` 会退化为平凡场。

## 7. 当前建议

方案 A 最适合作为第一实现对象，因为：

- 数学等价最干净；
- 与现有 `122/123/125` 风险特征最容易对接；
- 可以先在离散图/网格上实现，再决定是否走 GP/kernel 化版本。
