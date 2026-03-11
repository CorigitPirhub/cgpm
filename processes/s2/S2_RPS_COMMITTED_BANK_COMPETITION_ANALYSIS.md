# S2 committed rear-bank competition analysis

日期：`2026-03-09`
协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_COMMITTED_BANK_COMPETITION_COMPARE.csv`
诊断页：`processes/s2/S2_RPS_COMMITTED_BANK_COMPETITION_DIAGNOSIS.md`

## 1. 结果概览
- `30_rps_commit_geom_bg_soft_bank`：Bonn `ghost_reduction_vs_tsdf = 13.25%`，`extract_rear_selected_sum = 7`
- `34_bonn_compete_bgscore`：Bonn `ghost_reduction_vs_tsdf = 13.25%`，`extract_rear_selected_sum = 7`
- `35_bonn_compete_softgap`：Bonn `ghost_reduction_vs_tsdf = 13.25%`，`extract_rear_selected_sum = 7`
- `36_bonn_compete_softgap_support`：Bonn `ghost_reduction_vs_tsdf = 13.25%`，`extract_rear_selected_sum = 7`

统一事实：
- 三个 competition 变体全部 strict zero-delta；
- TUM family mean 与 Bonn family mean 指标都没有任何可用变化；
- 因此 `34/35/36` 全部 `abandon`。

## 2. 哪个变体提升了 rear_selected
没有任何一个变体提升了 rear 胜出数：
- 控制组 `30`：`extract_rear_selected_sum = 7`
- `34`：`7`
- `35`：`7`
- `36`：`7`

因此本轮对 competition 的所有 Bonn-only 微调都没有产生正向效应。

## 3. 是否转化为 ghost 改善
没有。

Bonn `ghost_reduction_vs_tsdf` 在四个配置上完全相同：
- `13.25%`

因此：
- `rear_selected` 没有提升；
- `ghost_reduction_vs_tsdf` 也没有提升；
- 本轮不能声称 competition tuning 对动态抑制有任何正收益。

## 4. 本轮最重要的新结论
本轮最重要的价值，不是找到新配置，而是**排除了一个错误假设**：

> 当前 Bonn 的主瓶颈并不在最终 front-vs-rear competition boundary。

extract-only 诊断显示：
- 真正进入 competition 的 rear 候选数量很少；
- 但一旦进入，已经全部胜出；
- 所以当前 zero-delta 的原因不是 competition 太硬，而是 upstream admission 太弱。

这意味着当前主瓶颈应从：
- `competition boundary too hard`
修正为：
- `too few committed rear states survive into extract competition`

## 5. 是否达到 S2 门槛
仍然没有。

当前 family mean：
- TUM：`Acc = 0.9358 cm`，`Comp-R = 93.12%`
- Bonn：`Acc = 4.2389 cm`，`Comp-R = 70.87%`，`ghost_reduction_vs_tsdf = 13.25%`

对照门槛：
- TUM：`Acc <= 2.55 cm`，`Comp-R >= 98%`
- Bonn：`Acc <= 3.10 cm`，`Comp-R >= 98%`，`ghost_reduction_vs_tsdf >= 22%`

结论：
- 本轮仍未达到 `S2` 门槛；
- 绝对不能进入 `S3`。

## 6. 当前 best iterate 与后续判断
当前唯一继续配置仍然是：
- `30_rps_commit_geom_bg_soft_bank`

原因：
- `34/35/36` 没有带来任何指标或 competition 胜出数增益；
- 它们的价值仅在于证明：当前再做 competition-only 微调已经不是有效主线。

统一判断：
- `30_rps_commit_geom_bg_soft_bank`：`keep / iterate`
- `34_bonn_compete_bgscore`：`abandon`
- `35_bonn_compete_softgap`：`abandon`
- `36_bonn_compete_softgap_support`：`abandon`

## 7. 结论归因
本轮失败更应归因为：
- **方法方向判断错误**：把当前瓶颈误判成 competition boundary；
- 而不是简单的参数未收敛。

因此，若下一轮继续 `S2`，唯一合理的主线应是：
- 继续固定 `30_rps_commit_geom_bg_soft_bank`；
- 把主线重新上移到 extract competition 之前，追查为什么只有极少数 rear state 能够进入 extract competition；
- 不再继续扩展新的 competition-only 支线。
