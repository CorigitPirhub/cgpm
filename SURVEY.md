# SURVEY: 面向梯度场构建的“数学公式 ↔ 物理公式 ↔ 物理定理”三条新链路

日期：`2026-03-12`
项目：`EGF_DHMap`
阶段：`S2`
作者角色：核心算法研究员 / 代码考古专家

---

## 0. 任务目标与结论先行

本报告围绕如下目标展开：

- 不是做“物理类比”的灵感式设计；
- 而是要构造一条严格链路：
  1. 梯度场核心公式；
  2. 与某个物理方程的数学等价；
  3. 该物理方程已有的定理；
  4. 将该定理严格转译为梯度场构建中的可用结论。

本轮结论：

- 我调研并筛出了 **三个** 满足这一范式、且在已检索的机器人映射/梯度场文献中**未发现直接实现证据**的方案；
- 三个方案分别来自：
  1. **量子隧穿 / Schrödinger–Agmon 距离**；
  2. **小噪声随机动力学 / Fokker–Planck–Freidlin–Wentzell 准势**；
  3. **相场界面 / Allen–Cahn–Modica–Mortola 距离反演**；
- 三者都满足“核心公式 → 数学等价 → 定理套用”这一链条；
- 从严谨性与可实现性平衡看，**方案 A（Schrödinger–Agmon）**与**方案 B（Fokker–Planck 准势）**最强；
- 从与“显式界面/表面”直接关联的角度看，**方案 C（Allen–Cahn）**最贴近距离场与法向场的构建。

**重要说明**：

- “无人实现过”无法被逻辑上完全证明；本报告能做的是：在本轮检索覆盖的机器人映射/距离场/梯度场相关公开文献中，**未找到把这三条物理-数学链直接用于机器人梯度场构建的工作**。
- 因此下面的“创新性判断”是 **best-effort novelty screening**，不是绝对数学证明。

---

## 1. 参考基线：Log-GPIS-MOP 的真正方法论价值

用户给出的参考是正确的：`Log-GPIS-MOP` 的价值不在于“用了热度”本身，而在于它满足如下四段链：

1. 先构建一个可线性求解的场；
2. 该场对应某个物理 PDE；
3. 该 PDE 已有成熟定理；
4. 通过对数变换，把这个 PDE 的定理转译为距离场结论。

`Log-GPIS-MOP` 与 `Faithful Euclidean Distance Field from Log-GPIS` 明确指出：

- 他们对 GPIS 做对数变换，以恢复距离场；
- 推导中关键使用了 **Varadhan 公式**；
- 其要点是把非线性的 Eikonal/EDF 问题，转成了某个线性 PDE 的对数解读。

对应来源：

- `Log-GPIS-MOP`：Lan Wu et al., arXiv:2206.09506
- `Faithful Euclidean Distance Field from Log-GPIS`：Lan Wu et al., arXiv:2010.11487

其中 `Faithful Euclidean Distance Field from Log-GPIS` 的摘要明确写到：

- 通过对 GPIS 施加对数变换恢复 EDF；
- 使用 Varadhan 公式把非线性 Eikonal PDE 近似为线性 PDE 的对数形式；
- Matérn 家族成员可直接满足该线性 PDE。

这条方法论对我们最重要的启发是：

> 真正有研究价值的，不是“从某个物理量联想到梯度”，而是“找到一个可严格变换到物理 PDE 的核心公式，再借用物理定理”。

---

## 2. 检索与筛选标准

### 2.1 筛选标准

每个候选方案必须满足：

1. 存在一个明确的梯度场核心公式；
2. 这个核心公式能通过严格推导变形为某个物理方程；
3. 该物理方程有可直接调用的成熟定理；
4. 该定理可以转译为“距离、梯度、界面、最短路径、最小作用量、稳定方向”等与梯度场直接相关的结论。

### 2.2 明确排除

本轮明确不选：

- 热传导/Varadhan 直接同类；
- 电磁势/静电势/泊松场直观类比；
- 只有“物理像”但没有精确同构推导的方案；
- 机器人领域里已经大量出现的标准 Eikonal / Fast Marching / 电势场直接变体。

### 2.3 创新性检索（best effort）

本轮专门检索了如下组合词：

- `Agmon distance robotics mapping`
- `quasipotential robotics mapping`
- `Allen-Cahn signed distance mapping robotics`
- `Cole-Hopf robotics distance field`

在已检索结果中：

- 没有发现把 **Agmon 距离** 用作机器人梯度场构建核心的直接工作；
- 没有发现把 **Freidlin–Wentzell 准势** 用作机器人映射/距离场构建核心的直接工作；
- 没有发现把 **Allen–Cahn 相场反演为距离场** 用作机器人梯度场构建核心的直接工作。

因此这三条链在当前检索范围内具备较强创新性。

---

## 3. 方案 A：Schrödinger–Agmon 对数振幅梯度场

### 3.1 核心思想

我们定义一个正场 `ψ_ε(x) > 0`，它不是“温度”，而是 **量子隧穿振幅 / 基态振幅**。令其满足带势场的椭圆方程：

\[
-\varepsilon^2 \nabla\cdot(M(x)\nabla \psi_\varepsilon(x)) + V(x)\,\psi_\varepsilon(x) = 0,
\quad x\in\Omega\setminus \Gamma,
\]

并在观测边界/表面 `Γ` 上施加边界条件，例如 `ψ_ε|_Γ = 1`。

其中：

- `M(x) \succ 0`：位置相关的各向异性张量；
- `V(x) \ge 0`：势函数，可编码障碍、几何不确定性、局部观测惩罚；
- `ε > 0`：半经典尺度参数。

然后定义梯度场核心函数：

\[
D_\varepsilon(x) := -\varepsilon \log \psi_\varepsilon(x),
\qquad
G_\varepsilon(x) := -\nabla D_\varepsilon(x).
\]

### 3.2 与物理方程的严格等价推导

令

\[
\psi_\varepsilon(x) = e^{-D_\varepsilon(x)/\varepsilon}.
\]

则有

\[
\nabla \psi_\varepsilon = -\frac{1}{\varepsilon}e^{-D_\varepsilon/\varepsilon}\nabla D_\varepsilon.
\]

进一步

\[
\nabla\cdot(M\nabla\psi_\varepsilon)
= -\frac{1}{\varepsilon}e^{-D_\varepsilon/\varepsilon}\nabla\cdot(M\nabla D_\varepsilon)
+ \frac{1}{\varepsilon^2}e^{-D_\varepsilon/\varepsilon}
\langle \nabla D_\varepsilon, M\nabla D_\varepsilon\rangle.
\]

代回 Schrödinger 型方程：

\[
-\varepsilon^2\nabla\cdot(M\nabla\psi_\varepsilon)+V\psi_\varepsilon=0,
\]

消去公共因子 `e^{-D_ε/ε}`，得到 **完全等价** 的非线性 PDE：

\[
\boxed{
\langle \nabla D_\varepsilon, M\nabla D_\varepsilon\rangle
- \varepsilon\,\nabla\cdot(M\nabla D_\varepsilon)
= V(x).
}
\]

这一步是精确的，不是类比。

### 3.3 可套用的物理定理

对 Schrödinger 算子的经典结果是 **Agmon estimate**：

- 在经典禁阻区（`V > E`）中，本征函数/解会以指数方式衰减；
- 衰减速率由一个加权距离控制，这个加权距离就是 **Agmon 距离**。

参考来源：

- Stefan Steinerberger, *An Agmon estimate for Schrödinger operators on Graphs*, arXiv:2206.09521

该文摘要明确指出：

- Schrödinger 本征函数在 forbidden region 中指数衰减；
- 其大小被一个 weighted Agmon distance 控制。

### 3.4 从物理定理到梯度场结论

当 `ε \to 0` 时，上述精确 PDE 退化为：

\[
\boxed{
\langle \nabla D, M\nabla D\rangle = V(x).
}
\]

这就是一个 **加权 Eikonal / Hamilton–Jacobi** 方程。

相应的距离是：

\[
 d_A(x,\Gamma)
 = \inf_{\gamma(0)\in\Gamma,\,\gamma(1)=x}
 \int_0^1 \sqrt{\dot\gamma(t)^\top V(\gamma(t))M(\gamma(t))^{-1}\dot\gamma(t)}\,dt.
\]

因此，Agmon 定理给出的不是普通欧氏距离，而是：

> **带势垒与各向异性代价的“量子衰减距离”**。

于是梯度场可直接定义为：

\[
G(x) = -\nabla d_A(x,\Gamma),
\]

其意义是：

- 沿最容易穿过势垒/不确定区的方向下降；
- 当 `V` 编码几何惩罚或不确定性时，它天然给出 barrier-aware gradient field。

### 3.5 对机器人梯度场构建的创新性

与热度 → 对数 → EDF 不同，这里得到的是：

- **量子振幅 → 对数 → 加权禁阻距离**；
- 不是热扩散距离，而是 **隧穿衰减距离**；
- 定理支撑来自 Agmon，而不是 Varadhan。

### 3.6 优缺点

**优点**
- 数学链条最干净；
- 可自然支持各向异性；
- 与“障碍势阱/几何屏障/不确定性屏障”高度兼容。

**风险**
- 需要构造合理的 `V(x)`；
- 更像 weighted distance，而非严格欧氏距离。

---

## 4. 方案 B：Fokker–Planck 准势梯度场

### 4.1 核心思想

我们不再从“热度”出发，而从 **小噪声随机动力系统的稳态概率密度** 出发。

令 `ρ_ε(x) > 0` 满足稳态 Fokker–Planck 方程：

\[
-\nabla\cdot(b(x)\rho_\varepsilon(x)) + \varepsilon \Delta \rho_\varepsilon(x) = 0,
\quad x\in\Omega.
\]

其中：

- `b(x)` 是漂移场；
- `ε` 是小噪声强度；
- `ρ_ε(x)` 是系统平衡后在位置 `x` 的驻留密度。

定义核心公式：

\[
U_\varepsilon(x) := -\varepsilon \log \rho_\varepsilon(x),
\qquad
G_\varepsilon(x) := -\nabla U_\varepsilon(x).
\]

### 4.2 与物理方程的严格等价推导

令

\[
\rho_\varepsilon(x)=e^{-U_\varepsilon(x)/\varepsilon}.
\]

则

\[
\nabla\rho_\varepsilon = -\frac{1}{\varepsilon}\rho_\varepsilon\nabla U_\varepsilon,
\qquad
\Delta\rho_\varepsilon = \frac{1}{\varepsilon^2}\rho_\varepsilon\|\nabla U_\varepsilon\|^2 - \frac{1}{\varepsilon}\rho_\varepsilon\Delta U_\varepsilon.
\]

再计算

\[
-\nabla\cdot(b\rho_\varepsilon)
= -\rho_\varepsilon\nabla\cdot b + \frac{1}{\varepsilon}\rho_\varepsilon b\cdot \nabla U_\varepsilon.
\]

代入稳态 Fokker–Planck 方程：

\[
-\nabla\cdot(b\rho_\varepsilon)+\varepsilon\Delta\rho_\varepsilon=0,
\]

约去 `ρ_ε`，得到 **精确等价** 的非线性 PDE：

\[
\boxed{
b(x)\cdot\nabla U_\varepsilon + \|\nabla U_\varepsilon\|^2
- \varepsilon\big(\Delta U_\varepsilon + \nabla\cdot b\big) = 0.
}
\]

这一步也是严格推导得到的，不是启发。

### 4.3 可套用的物理定理

当 `ε \to 0` 时，上式退化为：

\[
\boxed{
b(x)\cdot\nabla U + \|\nabla U\|^2 = 0.
}
\]

这正是 Freidlin–Wentzell 小噪声理论中的 **quasipotential** 方程。

相关定理链是：

- 稳态测度满足大偏差原理；
- 其 rate function 由 quasipotential 给出；
- 最可能跃迁路径 / minimum action path 可由该准势及其梯度决定。

参考来源：

- Jonathan Farfan, *Static large deviations of boundary driven exclusion processes*, arXiv:0908.1798
- Nicholas Paskal, Maria Cameron, *An efficient jet marcher for computing the quasipotential for 2D SDEs*, arXiv:2109.03424
- Bo Lin, Qianxiao Li, Weiqing Ren, *A Data Driven Method for Computing Quasipotentials*, arXiv:2012.09111

其中：

- `0908.1798` 的摘要直接指出：稳态测度的大偏差 rate function 由 Freidlin–Wentzell quasipotential 给出；
- `2109.03424` 的摘要指出 quasipotential 及其梯度可高精度数值求解；
- `2012.09111` 指出 quasipotential 是非平衡系统中 energy function 的自然推广。

### 4.4 从物理定理到梯度场结论

于是：

- `U_ε = -ε log ρ_ε` 是“从稳态概率恢复出来的准能量地形”；
- `G = -∇U` 则是“沿最小作用量下降的方向场”。

如果把 `b(x)` 设计为：

- 由表面法向、遮挡方向、可见性不确定性、语义风险构成的漂移；

那么 `U` 就不再是欧氏距离，而是：

> **一个由随机动力学大偏差理论支持的“最小作用量距离场”**。

这条链的物理意义非常强：

- 热度型方法衡量“扩散到这里有多难”；
- 准势型方法衡量“系统从吸引盆跃迁到这里有多难”。

### 4.5 创新性

这条方案不是 heat kernel，也不是 electrostatic potential；
它来自 **非平衡随机动力学 / 罕见事件理论**。

### 4.6 优缺点

**优点**
- 能天然编码方向性与不可逆性；
- 特别适合“前方/后方/可见/被遮挡”这类非对称几何关系。

**风险**
- `b(x)` 的设计需要非常谨慎；
- 计算上比标准距离场更复杂。

---

## 5. 方案 C：Allen–Cahn 相场 Logit 距离场

### 5.1 核心思想

我们引入一个相场/序参量 `u_ε(x) \in (-1,1)`，令它满足稳态 Allen–Cahn 方程：

\[
\varepsilon \Delta u_\varepsilon - \frac{1}{\varepsilon}(u_\varepsilon^3-u_\varepsilon)=0.
\]

双稳势取为：

\[
W(u)=\frac{(1-u^2)^2}{4}.
\]

然后定义核心距离反演公式：

\[
D_\varepsilon(x) := \sqrt{2}\,\varepsilon\,\operatorname{artanh}(u_\varepsilon(x)),
\qquad
G_\varepsilon(x):=\nabla D_\varepsilon(x).
\]

这不是启发式拍脑袋，因为 Allen–Cahn 的一维异宿解恰好是 `tanh` 轮廓。

### 5.2 与物理方程的严格等价推导

设

\[
u = \tanh\left(\frac{D}{\sqrt{2}\varepsilon}\right).
\]

记

\[
zeta := \frac{D}{\sqrt{2}\varepsilon}, \qquad u = \tanh(\zeta).
\]

则

\[
\nabla u = (1-u^2)\nabla \zeta = \frac{1-u^2}{\sqrt{2}\varepsilon}\nabla D.
\]

进一步

\[
\Delta u = (1-u^2)\Delta \zeta - 2u(1-u^2)\|\nabla \zeta\|^2.
\]

代入 `\zeta = D/(\sqrt{2}\varepsilon)`，得

\[
\Delta u = \frac{1-u^2}{\sqrt{2}\varepsilon}\Delta D - \frac{u(1-u^2)}{\varepsilon^2}\|\nabla D\|^2.
\]

代回 Allen–Cahn 方程：

\[
\varepsilon\Delta u - \frac{1}{\varepsilon}(u^3-u)=0.
\]

又因为 `u^3-u = -u(1-u^2)`，于是得到：

\[
\frac{1-u^2}{\sqrt{2}}\Delta D - \frac{u(1-u^2)}{\varepsilon}\|\nabla D\|^2 + \frac{u(1-u^2)}{\varepsilon}=0.
\]

约去 `(1-u^2)`（在 `|u|<1` 区域非零），得 **精确等价** 形式：

\[
\boxed{
\frac{\varepsilon}{\sqrt{2}}\Delta D + u\,(1-\|\nabla D\|^2)=0,
\qquad
u = \tanh\!\left(\frac{D}{\sqrt{2}\varepsilon}\right).
}
\]

即

\[
\boxed{
\frac{\varepsilon}{\sqrt{2}}\Delta D
+ \tanh\!\left(\frac{D}{\sqrt{2}\varepsilon}\right)
(1-\|\nabla D\|^2)=0.
}
\]

这是核心公式与物理方程之间的严格数学等价。

### 5.3 可套用的物理/几何定理

Allen–Cahn 理论给出：

- 一维异宿轮廓 `H(r)=\tanh(r/\sqrt{2})`；
- 多维情况下，界面附近的解近似为 `H(dist(x,\Gamma)/(\sqrt{2}\varepsilon))`；
- 其能量 `Γ` 收敛到界面面积（Modica–Mortola / diffuse interface theory）。

参考来源：

- *Geometric Aspects of the Allen–Cahn Equation*（几何综述型主文献）
- `turn7view2` / `turn9view2` 中的材料明确给出：
  - 一维轮廓 `H(r)=tanh(r/\sqrt{2})`；
  - 界面解可写成 `H(dist(·,\Gamma)/\varepsilon)` 的形式；
  - 相场能量逼近界面几何量。

### 5.4 从物理定理到梯度场结论

如果界面附近满足

\[
u_\varepsilon(x) \approx \tanh\left(\frac{d(x,\Gamma)}{\sqrt{2}\varepsilon}\right),
\]

那么由反函数关系立刻得到

\[
D_\varepsilon(x) = \sqrt{2}\varepsilon\operatorname{artanh}(u_\varepsilon(x)) \approx d(x,\Gamma).
\]

也就是说：

> **相场的 logit 反演，本质上可恢复 signed distance。**

因此梯度场可以定义为

\[
G_\varepsilon(x)=\nabla D_\varepsilon(x),
\]

它在界面附近自然给出法向方向场。

### 5.5 创新性

这条链不是热扩散，不是电磁势，也不是传统 Eikonal 重初始化；
它来自 **相分离 / 界面热力学 / diffuse interface theory**。

### 5.6 优缺点

**优点**
- 与“表面/界面”天然耦合；
- 直接给出 signed-distance-like 结构；
- 对界面法向很自然。

**风险**
- 更适合界面附近的局部距离，而不是大尺度全局距离；
- 若远离界面，`u≈±1` 时数值反演需要稳定化处理。

---

## 6. 三个方案的比较

| 方案 | 核心变换 | 物理方程 | 套用定理 | 输出距离类型 | 优势 | 风险 |
|---|---|---|---|---|---|---|
| A | `D=-ε log ψ` | Schrödinger | Agmon / WKB | 势垒加权距离 | 数学最干净、可各向异性 | 需要设计势函数 `V` |
| B | `U=-ε log ρ` | 稳态 Fokker–Planck | Freidlin–Wentzell 准势 | 最小作用量距离 | 可表达方向性/非对称性 | 需要设计漂移 `b` |
| C | `D=√2 ε artanh(u)` | Allen–Cahn | Modica–Mortola / 几何 Allen–Cahn | signed-distance-like 界面距离 | 最贴近表面与法向 | 更局部，远场数值更难 |

---

## 7. 推荐优先级

### 第一优先：方案 A（Schrödinger–Agmon）

原因：

- “核心公式 ↔ 物理方程”同构最清晰；
- 物理定理最直接给出“对数衰减 ↔ 距离”；
- 比热传导更偏向“障碍势垒 / 几何屏障 / 不确定性屏障”。

### 第二优先：方案 B（Fokker–Planck 准势）

原因：

- 最适合表达我们项目里最难的非对称几何关系：
  - 可见 vs 被遮挡
  - 前侧 vs 后侧
  - 动态侵入 vs 静态稳定
- 从研究上也最有“机制创新”潜力。

### 第三优先：方案 C（Allen–Cahn）

原因：

- 它最像“界面层中的 signed distance 反演器”；
- 如果将来我们更强调“从稀疏表面样本恢复界面法向与局部距离”，它会很强；
- 但全局导航意义弱于 A/B。

---

## 8. 本轮最终建议

如果要继续推进成真正的新梯度场理论主线，我建议按下面顺序走：

1. 先做 **方案 A 的算子级原型**：
   - 定义 `L_Q = -ε² div(M∇·)+V`；
   - 验证 `D=-ε log ψ` 在简单场景下能否稳定恢复 barrier-aware distance；
2. 再做 **方案 B 的非对称场原型**：
   - 让 `b(x)` 直接编码“前后可见性/遮挡/时间稳定性”；
   - 看准势梯度是否能自然分离前侧噪声与后侧真实缺口；
3. 若需要贴近表面细化，再引入 **方案 C** 做局部 signed-distance refinement。

---

## 9. 参考文献（主来源）

### 机器人参考基线

1. Lan Wu, Ki Myung Brian Lee, Cedric Le Gentil, Teresa Vidal-Calleja. *Log-GPIS-MOP: A Unified Representation for Mapping, Odometry and Planning*. arXiv:2206.09506.  
2. Lan Wu, Ki Myung Brian Lee, Liyang Liu, Teresa Vidal-Calleja. *Faithful Euclidean Distance Field from Log-Gaussian Process Implicit Surfaces*. arXiv:2010.11487.

### 方案 A：Schrödinger / Agmon

3. Stefan Steinerberger. *An Agmon estimate for Schrödinger operators on Graphs*. arXiv:2206.09521.

### 方案 B：Fokker–Planck / 准势 / 大偏差

4. Jonathan Farfan. *Static large deviations of boundary driven exclusion processes*. arXiv:0908.1798.  
5. Nicholas Paskal, Maria Cameron. *An efficient jet marcher for computing the quasipotential for 2D SDEs*. arXiv:2109.03424.  
6. Bo Lin, Qianxiao Li, Weiqing Ren. *A Data Driven Method for Computing Quasipotentials*. arXiv:2012.09111.  
7. Nicola Gigli, Luca Tamanini, Dario Trevisan. *Viscosity solutions of Hamilton-Jacobi equation in RCD(K,∞) spaces and applications to large deviations*. arXiv:2203.11701.

### 方案 C：Allen–Cahn / 相场 / 界面几何

8. *Geometric Aspects of the Allen–Cahn Equation*（文中明确给出 `H(r)=tanh(r/√2)` 以及 `H(dist(·,Γ)/ε)` 型界面近似）。

---

## 10. 结语

本轮最重要的不是“找到了三个像物理的灵感”，而是确认了：

- 这种“核心公式 ↔ 物理方程 ↔ 物理定理 ↔ 梯度场构建”的研究范式，不只存在于热传导；
- 至少还存在 **量子衰减、非平衡准势、相场界面** 三条严谨可走的数学-物理通路；
- 其中方案 A/B 具备最强的研究突破潜力。

