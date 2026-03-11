# S2 delayed-specific write target 定义页

版本：`2026-03-09`

## 1. 阶段目标
本阶段围绕 `delayed-branch write-time target synthesis` 做结构性增强：
- 不再把 delayed branch 视为 committed 的弱变体；
- 而是在写入期直接构造 delayed-specific surface target；
- 让 delayed branch 在 update 内部就携带不同于 committed/front 的几何假设。

## 2. 设计动机
当前现状表明：
- `no_synthesis / rear-buffer only` 能保 `Comp-R`，但 `Acc` 不足；
- `legacy wdsg/wdsgr/spg` 能明显压低 `Acc`，但容易结构性吃掉 `Comp-R`；
- 因此新的主问题不是“再加强 routing/export”，而是：
  - 如何在 write-time 就构造更可信、同时更保守的 delayed target；
  - 让 delayed target 既能矫正前景污染，又不把真实背景几何一起抹掉。

## 3. 统一形式
记当前观测点对某 voxel 的 legacy 三类 write-time target 为：
- `d_static_legacy`
- `d_transient_legacy`
- `d_rear_legacy`

本轮把 delayed-specific synthesis 扩展为四种模式：
- `legacy`
- `anchor`
- `counterfactual`
- `energy`

共同变量：
- `anchor_phi`：由 `phi_static / phi_geo / phi_rear / phi_bg / persistent_surface_readout` 聚合得到的 delayed anchor
- `front_conf`：由 `wod_front + wod_shell + q_dyn_obs` 聚合得到的前景污染强度
- `static_conf / rear_conf / dominance`：由 `PTDSF` 状态统计量给出的 delayed branch 静态可信度

## 4. 候选方法

### 4.1 Candidate-A: `anchor`
核心思想：
- 不直接把 legacy rear target 写入 delayed branch；
- 先用 persistent/static/geo/background 多源场读出一个 `anchor_phi`；
- 再让 `d_static / d_rear` 向这个 anchor 收敛，`d_transient` 则远离 anchor。

效果取向：
- 相对 `legacy` 更强调几何 anchor；
- 但仍保留原始 front/rear 分离结构。

### 4.2 Candidate-B: `counterfactual`
核心思想：
- 把 `d_transient - anchor_phi` 视作前景污染量；
- 估计“若移除前景污染，静态几何应落在何处”的 counterfactual target；
- 再把 `d_static` 向该 counterfactual target 收敛。

效果取向：
- 更激进地追求 `Acc`；
- 风险是 `Comp-R` 容易受损。

### 4.3 Candidate-C: `energy`
核心思想：
- 把 `legacy / rear / anchor / bg` 看作多个 delayed target 候选；
- 用 `static_conf / dominance / rear_conf / bg_conf / front_conf` 计算 soft energy；
- 通过 soft selection 得到一个 energy-calibrated delayed target。

效果取向：
- 更像“多候选 delayed field synthesis”；
- 追求更强的 field-level 创新性，但也最容易带来 completeness 崩塌。

## 5. 第二轮保守迭代
在第一轮三条候选之后，本轮又补了三条“保守 delayed synthesis”迭代：
- `anchor_noroute`
- `counterfactual_noroute`
- `anchor_lite_noroute`
- `anchor_ultralite_noroute`

这些迭代的核心不是改 export，而是：
- 保留 write-time synthesis 这条主命题；
- 但显式减小 `front/rear shift` 与 `front/rear mix`；
- 同时关闭 `route/spg`，验证“comp 崩塌究竟来自 synthesis 自身，还是来自 route/promote 联动”。

## 6. 当前结论
- 第一轮 `anchor / counterfactual / energy` 都证明了 write-time synthesis 对 `Acc` 有显著作用；
- 第二轮保守版本进一步证明：`Comp-R` 的主要问题来自 synthesis 振幅过大，而不只是 delayed branch 这一主线本身；
- 当前最值得保留的继续迭代配置是：`anchor_ultralite_noroute`。
