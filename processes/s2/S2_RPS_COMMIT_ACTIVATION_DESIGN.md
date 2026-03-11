# S2 RPS commit activation 设计

## 1. 目标
围绕 `25_rear_bg_coupled_formation`，恢复：
- `phi_rear_cand / rho_rear_cand`
- `rps_commit_score / rps_active`
- `phi_rear / rho_rear`

这条 committed rear-bank activation 链。

## 2. 理论依据
commit 评分应同时考虑：
- 证据强度：`rho_rear_cand` 与 `phi_rear_cand_w`
- 几何一致性：`phi_rear_cand` 与 `phi_geo / phi_static / phi_bg` 的一致性
- 背景支撑：`rho_bg / phi_bg_w`
- 前景惩罚：`wod_front / wod_shell / q_dyn_obs / assoc_risk`

只有当上述量形成“高背景支撑 + 足够几何一致 + 前景惩罚不高”的组合时，rear candidate 才应被提交为 committed rear bank。

## 3. 机制
- 新模块：`egf_dhmap3d/P10_method/rps_commit.py`
- 核心输出：
  - `rps_commit_score`
  - `rps_active`
- 提交条件：
  - `rps_commit_score >= rps_commit_threshold`
  - `rps_commit_age >= rps_commit_age_threshold`
  - `phi_rear_cand_w >= rps_commit_min_cand_w`
  - `rho_rear_cand >= rps_commit_min_cand_rho`
- 提交动作：
  - `phi_rear_cand -> phi_rear`
  - `rho_rear_cand -> rho_rear`

## 4. 受控计划
- 控制组：`25_rps_commit_control`
- 实验组：
  - `28_rps_commit_geom_bg_soft`
  - `29_rps_commit_geom_bg_mid`

协议固定：`frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
