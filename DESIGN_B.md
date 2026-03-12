# DESIGN_B：Fokker–Planck 准势梯度场

## 1. 目标定位

方案 B 试图把梯度场定义为 **非平衡随机系统的准势梯度**，而不是静态几何距离梯度。这里的核心对象不是波函数，也不是温度，而是稳态概率密度 `\rho_\varepsilon(x)`。

核心定义为：

\[
U_\varepsilon(x) := -\varepsilon \log \rho_\varepsilon(x),
\qquad
G_\varepsilon(x) := -\nabla U_\varepsilon(x).
\]

物理意义：

- `U_\varepsilon` 衡量“系统到达该点的罕见程度”；
- `-\nabla U_\varepsilon` 则指向“最可能的回流方向”或“最小作用量下降方向”。

## 2. 数学原型

### 2.1 核心公式

考虑稳态 Fokker–Planck 方程：

\[
-\nabla \cdot (b(x)\rho_\varepsilon(x)) + \varepsilon \Delta \rho_\varepsilon(x) = 0,
\qquad x\in\Omega.
\]

其中：

- `b(x)` 是漂移场；
- `\varepsilon` 是小噪声强度；
- `\rho_\varepsilon(x)` 是稳态概率密度。

然后定义：

\[
U_\varepsilon(x) = -\varepsilon \log \rho_\varepsilon(x).
\]

### 2.2 严格等价推导

令

\[
\rho_\varepsilon(x)=\exp\left(-\frac{U_\varepsilon(x)}{\varepsilon}\right).
\]

则有

\[
\nabla\rho_\varepsilon = -\frac{1}{\varepsilon}\rho_\varepsilon \nabla U_\varepsilon,
\]

\[
\Delta\rho_\varepsilon = \frac{1}{\varepsilon^2}\rho_\varepsilon\|\nabla U_\varepsilon\|^2 - \frac{1}{\varepsilon}\rho_\varepsilon\Delta U_\varepsilon.
\]

再计算漂移项：

\[
-\nabla\cdot(b\rho_\varepsilon)
= -\rho_\varepsilon \nabla\cdot b + \frac{1}{\varepsilon}\rho_\varepsilon b\cdot \nabla U_\varepsilon.
\]

代回 Fokker–Planck 方程：

\[
-\nabla\cdot(b\rho_\varepsilon)+\varepsilon\Delta\rho_\varepsilon=0,
\]

除以 `\rho_\varepsilon > 0`，可得：

\[
\boxed{
b\cdot\nabla U_\varepsilon + \|\nabla U_\varepsilon\|^2 - \varepsilon(\Delta U_\varepsilon + \nabla\cdot b)=0.
}
\]

这是稳态密度对数势与随机动力学之间的严格等价关系。

### 2.3 小噪声极限

当 `\varepsilon \to 0`，上式退化为：

\[
\boxed{
b(x)\cdot\nabla U + \|\nabla U\|^2 = 0.
}
\]

这就是 Freidlin–Wentzell 理论中的 quasipotential Hamilton–Jacobi 方程。

## 3. 物理等价与可套用定理

### 3.1 物理公式

该方程对应的是带小噪声漂移过程：

\[
dX_t = b(X_t)dt + \sqrt{2\varepsilon}\, dW_t.
\]

其稳态密度满足上述 Fokker–Planck 方程。

### 3.2 可调用定理

Freidlin–Wentzell 大偏差理论给出：

- 稳态密度满足大偏差原理；
- 其 rate function 是 quasipotential；
- 最小作用量路径由这个准势决定。

因此，`U = -\varepsilon \log \rho_\varepsilon` 不是普通势能，而是 **最小作用量势**。

## 4. 对映射梯度场的意义

### 4.1 与 `EGF_DHMap` 的对应

这里的关键不在距离，而在方向性。我们可以把漂移 `b(x)` 设计成：

\[
b(x) = -\alpha_1 n_{plane}(x) - \alpha_2 g_{vis}(x) - \alpha_3 g_{occ}(x) + \alpha_4 g_{dyn}(x),
\]

其中：

- `n_plane(x)`：高置信平面的几何法向或切向指引；
- `g_vis(x)`：visibility deficit 的修复方向；
- `g_occ(x)`：占据-熵机制给出的稳定激活方向；
- `g_dyn(x)`：动态风险的排斥漂移。

这样构造后：

- `U` 表示“从稳定吸引盆跨到该点需要多大作用量”；
- `-\nabla U` 就是“最可能回到稳定背景流形的方向”。

### 4.2 预期效果

方案 B 的最大优势是可表达：

- 前方观测 vs 后方缺口的非对称性；
- 动态尾迹与真实背景缺口的不可逆差异；
- 方向性先验。

预期在 `Ghost` 控制上比纯距离场更强。

## 5. 实验计划

### 5.1 第一阶段：1D/2D 随机动力学 sanity check

构造一个双吸引盆系统：

\[
b(x) = -\nabla W(x) + c(x),
\]

其中 `c(x)` 是一个非保守项。

验证：

- `-\varepsilon \log \rho_\varepsilon` 是否恢复出准势；
- 非保守漂移下，与普通势场法相比，是否能体现方向性差异。

### 5.2 第二阶段：几何场景原型

构造 2D 场景：

- 前方观测带；
- 后方可见缺口；
- 动态干扰带；
- 静态平面引导带。

定义：

- `b(x)` 把背景回归方向和动态排斥方向都编码进去；
- 数值求解稳态 `\rho_\varepsilon`；
- ��演 `U_\varepsilon` 与 `-\nabla U_\varepsilon`。

评价：

- 梯度是否天然绕开动态高风险区；
- 梯度是否朝向后方缺口而不是前方伪观测。

### 5.3 第三阶段：接入当前有效主线

输入使用当前主线已有量：

- `122/123` 的 `visibility_deficit_score`
- `116` 的 `p_occ / entropy / score`
- `125` 的 plane consistency / gap band 筛选

把这些量拼成漂移 `b(x)`，再求 `U`。

最终把：

\[
\text{activation score}(x) = \beta_1\,p_{occ}(x)e^{-H(x)} + \beta_2\,(-\nabla U(x))\cdot d_{gap}(x)
\]

用于 GT-free 激活。

### 5.4 评价指标

- 数学层：
  - Fokker–Planck 残差
  - `-\varepsilon \log \rho` 的稳定性
- 系统层：
  - Bonn `Acc`
  - Bonn `Comp-R`
  - Bonn `Ghost`
  - 与 `125/130` 对比

### 5.5 成功标准

- Toy scene 上能显示非对称 drift 对梯度方向的改变；
- 接入主线后，至少在 `Ghost` 不恶化的情况下提升 `Comp-R`。

## 6. 风险评估

- `b(x)` 过强时，系统会在局部吸引盆里卡死；
- 若 `\rho_\varepsilon` 动态范围太大，`-\varepsilon \log \rho_\varepsilon` 数值可能不稳；
- 需要稳态求解器，工程复杂度高于方案 A。

## 7. 当前建议

方案 B 更像是下一代非对称几何/动态风险场。若目标是解决“前后可见性与动态遮挡的方向性差异”，它比 A 更有研究冲击力，但工程代价也更高。
