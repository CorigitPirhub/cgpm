# 主线 / 支线裁剪表

版本：`2026-03-08`
作用：给出当前项目的主线、保留支撑线、附录线与已退场支线的统一裁剪结果。

| 类别 | 分支/机制 | 当前状态 | 处理方式 | 原因 | 证据来源 |
|---|---|---|---|---|---|
| 主线 | `evidence-gradient` | 保留 | 主文 | 状态层解耦核心 | `TASK_LOCAL_TOPTIER.md` 3.1 |
| 主线 | `dual-state disentanglement` | 保留 | 主文 | 解释 `Acc/ghost` 耦合 | `TASK_LOCAL_TOPTIER.md` 3.1 |
| 主线 | `delayed branch` | 保留 | 主文 | conflict-isolated geometry path | `TASK_LOCAL_TOPTIER.md` 3.2 |
| 主线 | `write-time target synthesis` | 保留 | 主文主攻 | 最有希望成为核心新增贡献 | `TASK_LOCAL_TOPTIER.md` 3.3 |
| 支撑 | `precision profile` | 保留 | 主文/附录 | 量化 `Acc` 缺口 | `output/summary_tables/local_mapping_precision_profile.csv` |
| 支撑 | `canonical 5-seed` 主表 | 保留 | 主文 | 正式对外口径 | `README.md`, `summary_tables` |
| 诊断 | `quantile-calibrated support-gap` | 降级 | 附录/负结果链 | 机制有增益，指标过小 | `processes/p10/P10_METHOD_PROPOSALS.md` |
| 诊断 | `top-tail delayed-only` | 降级 | 附录/负结果链 | 打开 delayed-only，但收益不大 | `processes/p10/P10_METHOD_PROPOSALS.md` |
| 诊断 | `promotion hold/hysteresis` | 降级 | 附录/负结果链 | 机制正确，结果放大不足 | `processes/p10/P10_METHOD_PROPOSALS.md` |
| 诊断 | `residency-gated export` | 降级 | 附录/负结果链 | delayed tail 入 export，但不主导表面 | `processes/p10/P10_METHOD_PROPOSALS.md` |
| 诊断 | `local replacement / competition replacement` | 降级 | 附录/负结果链 | export trick 边际收益衰减 | `processes/p10/P10_METHOD_PROPOSALS.md` |
| 诊断 | `dedicated delayed banked readout` | 降级 | 附录/负结果链 | 更干净但不更强 | `processes/p10/P10_METHOD_PROPOSALS.md:2906` |
| 诊断 | `persistent delayed bank accumulation` | 降级 | 附录/负结果链 | 与上一轮几乎等价 | `processes/p10/P10_METHOD_PROPOSALS.md:2972` |
| 退场 | `OTV` | 退出 | 不进入主文 | 已证伪 / 收益不足 | `processes/governance/PROJECT_AUDIT_CHECKLIST.md` |
| 退场 | `CSR-XMap` | 退出 | 不进入主文 | 已证伪 / 收益不足 | `processes/governance/PROJECT_AUDIT_CHECKLIST.md` |
| 退场 | `XMem / BECM / RCCM` | 退出 | 不进入主文 | 已证伪 / 收益不足 | `processes/governance/PROJECT_AUDIT_CHECKLIST.md` |
| 退场 | `OBL-3D` | 退出 | 不进入主文 | 已证伪 / 收益不足 | `processes/governance/PROJECT_AUDIT_CHECKLIST.md` |
| 退场 | `CMCT` | 退出 | 不进入主文 | 已证伪 / 收益不足 | `processes/governance/PROJECT_AUDIT_CHECKLIST.md` |
| 退场 | `CGCC` | 退出 | 不进入主文 | 已证伪 / 收益不足 | `processes/governance/PROJECT_AUDIT_CHECKLIST.md` |
| 退场 | `PFVP / PFV-sharp / PFV-bank` | 退出 | 不进入主文 | 已证伪 / 收益不足 | `processes/governance/PROJECT_AUDIT_CHECKLIST.md` |

结论：
- 未来主文只允许围绕“状态层解耦 + delayed branch + write-time target synthesis”展开；
- 其余内容只能作为 supporting diagnostics chain 或 appendix。 
