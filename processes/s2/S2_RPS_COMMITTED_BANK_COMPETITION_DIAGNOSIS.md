# S2 Bonn committed rear-bank competition diagnosis

日期：`2026-03-09`
协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
控制组：`30_rps_commit_geom_bg_soft_bank`

## 1. 现象
本轮最初假设是：
> committed rear state 已经形成，但在 Bonn 上输在导出阶段的 front-vs-rear competition 边界。

为验证这一点，本轮新增了 extract-only 竞争诊断，把 `sync` 与 `extract` 的 persistent readout 分开统计。

## 2. 根因结论
结论：**当前主阻塞并不在 front-vs-rear competition 本身。**

更准确地说：
- 真正进入 extract 阶段 competition 的 Bonn rear 候选数量本来就极少；
- 而这些少量候选一旦进入 competition，已经全部胜出；
- 因此继续调 `margin / sep_gate / soft_gap` 不会带来增益；
- 当前更真实的瓶颈，是 **upstream 有效 rear candidate 进入 extract competition 的数量不足**，而不是最终 competition boundary 过硬。

## 3. 证据链
### 证据 A：extract-only 统计与旧 `rear_selected` 口径不同
控制组 `30` 在 Bonn all3 上：
- 旧口径 `rear_selected_sync_sum = 35`
- 新的 extract-only `extract_rear_selected_sum = 7`
- 新的 extract-only `extract_considered_sum = 7`

说明：
- 旧 `rear_selected` 混入了 sync/readout 期调用，不能直接当作“最终导出竞争胜出数”；
- 真正进入 extract competition 的 rear 候选，只有 `7` 个。

### 证据 B：进入 extract competition 的 rear 候选已经全部获胜
控制组 `30` 在 Bonn all3 三条序列上的 extract-only 统计：
- `rgbd_bonn_balloon2`：`extract_considered = 2`，`extract_rear_selected = 2`
- `rgbd_bonn_balloon`：`extract_considered = 1`，`extract_rear_selected = 1`
- `rgbd_bonn_crowd2`：`extract_considered = 4`，`extract_rear_selected = 4`

合计：
- `extract_considered_sum = 7`
- `extract_rear_selected_sum = 7`

因此：
> 当前不是“rear 候选进入 competition 之后输给了 front”，而是“绝大多数 rear 状态根本没有走到 extract competition 这一步”。

### 证据 C：competition failure diagnostics 基本全零
控制组 `30` 的三条 Bonn 序列上：
- `extract_fail_sep = 0`
- `extract_fail_margin = 0`
- `extract_fail_min = 0`
- `extract_fail_soft_support = 0`
- `extract_close_gap = 0`

这直接否定了“margin 太硬 / separation 太硬 / near-tie 太多”的原始假设。

### 证据 D：rear 分数本身并不低
控制组 `30` 的三条 Bonn 序列上：
- `front_score_mean ≈ 0.142 - 0.201`
- `rear_score_mean ≈ 0.455 - 0.463`
- `rear_gap_mean ≈ 0.262 - 0.313`
- `rear_sep_mean = 1.0`

即：
- 只要 rear 候选真的进入 extract competition，rear 分数是显著高于 front 的；
- 因此 competition score 设计不是当前最窄瓶颈。

## 4. 对本轮三个变体的解释
`34/35/36` 全部 zero-delta，不是因为参数没有生效，而是因为：
- competition 参数已正确透传并写入 summary；
- 但 Bonn extract-only competition 本就已经是“7/7 全胜”；
- 所以继续软化 competition，不会创造新的可赢样本。

## 5. 当前统一结论
本轮应统一采用如下说法：

> 本轮关于“Bonn 输在 front-vs-rear competition 边界”的工作假设，已被 extract-only 诊断证伪。\
> 当前 Bonn 的真正瓶颈，不是 rear candidate 在最终 competition 中打不过 front，\
> 而是只有极少数 committed rear state 能够真正走到 extract competition。\
> 因此，下一轮若继续 `S2`，主线必须重新上移到 extract competition 之前的 state admission / candidate admission / pre-extract filtering，而不能继续在 competition 参数上打转。
