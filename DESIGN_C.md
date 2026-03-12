# DESIGN_C：Allen–Cahn 相场 Logit 距离场

## 1. 目标定位

方案 C 从相场/界面动力学出发，不直接求距离，而是先求一个界面序参量 `u_\varepsilon(x)\in(-1,1)`，再通过反双曲正切把它反演成局部距离：

\[
D_\varepsilon(x)=\sqrt{2}\varepsilon\operatorname{artanh}(u_\varepsilon(x)).
\]

这条链的核心是：Allen–Cahn 界面解天然具有 `tanh` 轮廓，因此 logit 型反演不是启发，而是建立在已知异宿解之上的精确结构。

## 2. 数学原型

### 2.1 核心公式

考虑稳态 Allen–Cahn 方程：

\[
\varepsilon\Delta u_\varepsilon - \frac{1}{\varepsilon}(u_\varepsilon^3-u_\varepsilon)=0,
\qquad x\in\Omega.
\]

定义反演距离：

\[
D_\varepsilon(x)=\sqrt{2}\varepsilon\operatorname{artanh}(u_\varepsilon(x)),
\qquad
G_\varepsilon(x)=\nabla D_\varepsilon(x).
\]

### 2.2 严格等价推导

令

\[
u = \tanh\left(\frac{D}{\sqrt{2}\varepsilon}\right).
\]

记

\[
zeta = \frac{D}{\sqrt{2}\varepsilon}.
\]

则

\[
\nabla u = (1-u^2)\nabla\zeta = \frac{1-u^2}{\sqrt{2}\varepsilon}\nabla D.
\]

进一步

\[
\Delta u = (1-u^2)\Delta\zeta - 2u(1-u^2)\|\nabla\zeta\|^2.
\]

代入 `\zeta = D/(\sqrt{2}\varepsilon)` 得

\[
\Delta u = \frac{1-u^2}{\sqrt{2}\varepsilon}\Delta D - \frac{u(1-u^2)}{\varepsilon^2}\|\nabla D\|^2.
\]

再代回 Allen–Cahn：

\[
\varepsilon\Delta u - \frac{1}{\varepsilon}(u^3-u)=0.
\]

因为

\[
u^3-u = -u(1-u^2),
\]

于是得到

\[
\frac{1-u^2}{\sqrt{2}}\Delta D - \frac{u(1-u^2)}{\varepsilon}\|\nabla D\|^2 + \frac{u(1-u^2)}{\varepsilon}=0.
\]

约去 `1-u^2`，得到严格等价方程：

\[
\boxed{
\frac{\varepsilon}{\sqrt{2}}\Delta D + u(1-\|\nabla D\|^2)=0,
\qquad
u = \tanh\left(\frac{D}{\sqrt{2}\varepsilon}\right).
}
\]

因此，Allen–Cahn 界面场与一个带粘性项的距离方程严格对应。

### 2.3 界面极限

当 `\varepsilon` 小且点在界面附近时，有经典近似：

\[
u_\varepsilon(x) \approx \tanh\left(\frac{d(x,\Gamma)}{\sqrt{2}\varepsilon}\right).
\]

因此反演后：

\[
D_\varepsilon(x) \approx d(x,\Gamma).
\]

这说明 `D_\varepsilon` 是局部 signed-distance-like 场，`\nabla D_\varepsilon` 就是局部界面法向场。

## 3. 物理等价与可套用定理

### 3.1 物理公式

Allen–Cahn 方程来自相分离与界面热力学，其序参量 `u` 描述两相之间的过渡层。

### 3.2 可调用定理

可直接利用的理论包括：

- 一维异宿解 `H(r)=\tanh(r/\sqrt{2})`；
- 界面层解近似 `H(d(x,\Gamma)/(\sqrt{2}\varepsilon))`；
- Modica–Mortola `\Gamma` 收敛：相场能量收敛到界面面积。

因此，方案 C 的理论价值在于：

> 不是“相场像距离”，而是“相场在界面极限下可严格反演为局部距离”。

## 4. 对映射梯度场的意义

### 4.1 与当前任务的结合方式

该方案特别适合：

- 已经有较稳定表面支持，但需要恢复局部法向和局部缺口厚度；
- 想在 plane / surface 支撑附近生成局部 refined gradient。

可把当前 `125` 或未来候选表面视为界面 `\Gamma`，再以：

- `u>0`：背景相；
- `u<0`：空洞/未观测相；

构建一个相场反演距离。

### 4.2 预期效果

- 比纯点云后验收缩更有理论基础；
- 能输出局部 signed distance 与法向；
- 更适合做局部 geometry refinement，而不是全局 completeness recovery。

## 5. 实验计划

### 5.1 第一阶段：1D 界面反演

构造 1D 解：

\[
u(x)=\tanh\left(\frac{x-x_0}{\sqrt{2}\varepsilon}\right).
\]

验证：

\[
D(x)=\sqrt{2}\varepsilon\operatorname{artanh}(u(x)) = x-x_0.
\]

确认反演是精确的。

### 5.2 第二阶段：2D 界面层 toy scene

构造：

- 平面界面；
- 曲面界面；
- 带噪声界面。

验证：

- `D_\varepsilon` 是否在界面附近恢复 signed distance；
- `\nabla D_\varepsilon` 是否与真实法向一致。

### 5.3 第三阶段：接入现有主线

接入方式不应直接替代 `125`，而应作为后续 refinement 模块：

1. 先由 `125` 产生 GT-free 激活点；
2. 再在这些点与高置信平面之间拟合局部相场；
3. 用 `D_\varepsilon` 与 `\nabla D_\varepsilon` 修正点位与法向。

### 5.4 评价指标

- 局部法向一致性；
- 局部点面距离；
- 对 `Acc` 的影响；
- 是否破坏 `Ghost` 与 `Comp-R`。

### 5.5 成功标准

- 在 toy scene 上验证反演正确；
- 在真实数据上，作为局部 refinement 至少带来 `Acc` 改善而不恶化 `Ghost`。

## 6. 风险评估

- 该方案更偏局部界面层，难以单独承担全局完整性恢复；
- 若界面初值差，反演的 `D_\varepsilon` 会被初值误差放大；
- 更适合作为 refinement，而不是整个主线替换。

## 7. 当前建议

方案 C 最适合作为 A/B 之后的局部精修模块。若目标是构造全新主线，优先级低于 A/B；若目标是给现有 GT-free 主线补一个有理论支撑的局部法向/距离精修器，它很有价值。
