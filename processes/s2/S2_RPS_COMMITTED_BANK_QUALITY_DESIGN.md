# S2 committed rear-bank quality enhancement design

日期：`2026-03-09`
协议：`frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
控制起点：`30_rps_commit_geom_bg_soft_bank`

## 1. 任务目标
本轮不再扩展 export-only 支线，只围绕 `30_rps_commit_geom_bg_soft_bank` 强化 committed rear bank 的两件事：
- 有效规模：让更多 rear candidate 以更高 transfer / rho 进入 committed rear bank；
- 几何质量：让 committed `phi_rear` 更稳定、更有 front-vs-rear separation。

目标仍是尝试推动 Bonn `ghost_reduction_vs_tsdf` 向 `S2` 门槛移动，同时不破坏 TUM `Comp-R`。

## 2. 当前瓶颈假设
上一轮 `30` 已经证明：
- `phi_rear_cand / rho_rear_cand -> phi_rear / rho_rear` 可以被打通；
- `rear_selected > 0` 已首次出现；
- 但 Bonn 主指标完全没有起量。

因此本轮假设是：
> 问题不再是“rear commit 链是否存在”，而是 committed rear bank 的 scale / geometry quality 还不够强，尚不足以改变 Bonn 的最终导出表面。

## 3. 方法设计
本轮新增一个受控开关：`rps_commit_quality_enable`。
开启后，仅在 `rps_commit.py` 的 commit 形成阶段做三类增强。

### 3.1 quality-aware commit scoring
在原有 `evidence / geometry / bg / static` 之外，额外引入：
- candidate mass (`weight_n`)；
- bank continuity (`bank_agree`)；
- quality score 的单独统计。

目的：避免 commit score 只看“有无证据”，而忽略 candidate 的实际体量与 rear bank 连续性。

### 3.2 committed target synthesis
commit 发生时，不再总是直接把 `phi_rear_cand` 原样写入 `phi_rear`；而是：
- 用 `phi_static / phi_geo / phi_bg / phi_rear` 与 candidate 做质量加权融合；
- 在保持 candidate 方向一致的前提下，施加一个最小 separation floor；
- 让 committed rear target 更稳定，同时尽量避免与 front readout 过度贴合。

### 3.3 quality-conditioned transfer / rho gain
commit 成功时，不再固定使用同一 transfer 强度；而是根据 quality score 调整：
- candidate -> committed rear 的 transfer 比例；
- `rho_rear` 的累计增益。

目的：把“高质量 rear candidate”更充分地写成 committed rear bank，而不是只做到激活但幅度过弱。

## 4. 诊断增强
本轮新增 `rps_commit_diag` 与补充 state 统计，统一写入 `summary.json`：
- `commit_ready`
- `commit_transfer_sum`
- `commit_w_sum`
- `commit_rho_sum`
- `commit_shift_sum`
- `rps_commit_score_ge_on`
- `rps_commit_age_ge_thr`

这些诊断用于判断：
- internal state 是否真的增强；
- 最终指标不变时，问题究竟卡在“没写进去”还是“写进去了但没导出来”。

## 5. 受控候选
- `30_rps_commit_geom_bg_soft_bank`：current-code control recheck
- `31_rps_commit_quality_bank_mid`：温和质量增强
- `32_rps_commit_quality_bank_geom`：更强几何融合
- `33_rps_commit_quality_bank_push`：更强 transfer / rho push

## 6. 判定原则
若候选满足以下三点，才可保留：
- Bonn `ghost_reduction_vs_tsdf` 出现净提升；
- TUM `Comp-R` 不发生结构性退化；
- 提升能够被 rear-bank 诊断量与最终表面选择共同支撑。

否则即使 internal state 变强，也必须判为 `abandon`。
