# S2 Bonn local clipping / calibration refinement 分析

日期：`2026-03-09`

## 1. 本轮目标
本轮按用户要求优先完成：
1. `current-code re-baseline`
2. `canonical refresh`
3. `drift source` 定位
4. 在此基础上再判断是否继续沿 `16` 做 Bonn-only 微调

## 2. 关键修正：协议漂移已定位
本轮首先确认，上一轮“current-code recheck”之所以与 historical `14` 严重不一致，第一原因不是方法本身，而是**协议漏项**：
- historical `14` 输出中：`reference_points = 3000`
- 若 `frames=5`，这意味着真实评测口径不是上一轮误用的 `max_points_per_frame=1500`
- 结合当前评测实现，`reference_points=3000` 对应的实际协议是：
  - `frames=5`
  - `stride=3`
  - `seed=7`
  - `max_points_per_frame=600`

因此，本轮将 `max_points_per_frame=600` 固定为当前 `S2 dev quick` 的 explicit canonical 协议，并已把相关 runner 默认值改为 `600`。

## 3. current-code canonical 重建结果
表：`processes/s2/S2_BONN_LOCALCLIP_REFINEMENT_COMPARE.md`

`14/15/16` 在 current-code canonical 协议下结果如下：
- `14_bonn_localclip_drive_recheck`
  - `TUM Acc = 0.9355 cm`
  - `TUM Comp-R = 68.53%`
  - `Bonn Acc = 2.8864 cm`
  - `Bonn Comp-R = 83.57%`
  - `Bonn ghost_reduction_vs_tsdf = -8.00%`
- `15_bonn_localclip_band_relax`
  - 与 `14` 基本完全相同，`abandon`
- `16_bonn_localclip_pfv_rearexpand`
  - 与 `14` 基本完全相同，`abandon`

## 4. 漂移来源定位
### 4.1 第一层：协议漂移
已确认 historical `S2` 文档缺失了关键协议项：
- `max_points_per_frame=600`

这会直接导致：
- `reference_points` 从 historical `3000` 变成误跑时的 `7500`
- 从而使上一轮 recheck 口径不一致

### 4.2 第二层：实现漂移
在修正协议后，residual drift 仍然存在，而且已经可以明确定位到**实现链条而非评测口径**：
- historical `14`：
  - `TUM Acc = 2.6008 cm`, `Comp-R = 99.13%`
  - `Bonn Acc = 3.7550 cm`, `Comp-R = 83.87%`
  - `dynamic_score_mean(TUM/Bonn) > 0`
- current-code `14@600`：
  - `TUM Acc = 0.9355 cm`, `Comp-R = 68.53%`
  - `Bonn Acc = 2.8864 cm`, `Comp-R = 83.57%`
  - `dynamic_score_mean(TUM/Bonn) = 0`

更强的定位证据：
- 本轮额外重跑了 `05_anchor_noroute_recheck600`：
  - `TUM = 0.9292 / 68.50%`
  - `Bonn = 2.8865 / 83.60%`
- 与 current-code `14@600` 的差值只有：
  - `TUM Acc +0.0063 cm`
  - `TUM Comp-R +0.0333%`
  - `Bonn Acc -0.0001 cm`
  - `Bonn Comp-R -0.0333%`

结论：
- **current-code 的 `14/15/16` 已经退化为 `05_anchor_noroute` 等价行为**；
- 也就是说，historical `05 -> 14` 的那段收益链条在当前实现里已经失活；
- 该失活不是 Bonn-only calibration 的问题，而是更上游的 delayed synthesis / dynamic cue / state update 链条问题。

## 5. 漂移模块定位
结合当前代码与工作树 diff，本轮将 residual drift 定位到以下模块：
- `egf_dhmap3d/P10_method/ptdsf.py`
- `egf_dhmap3d/modules/updater.py`
- `egf_dhmap3d/modules/pipeline.py`

当前最直接的症状是：
- `dynamic_score_mean` 在 current-code `14` 中掉到 `0.0`；
- `14/15/16` 对 Bonn-only 微调完全不敏感；
- 说明依赖 `q_dyn_obs / state stats / delayed target` 的后续细化链条当前基本没有被真正激活。

## 6. 对 `16` 方向的判断
结论：**本轮不应继续沿 `16` 做 Bonn-only 微调。**

理由：
- 在 explicit canonical 协议下，`16` 与 `14` 完全无有效差异；
- 当前主矛盾已不再是“Bonn-side clipping 调得不够细”，而是“让 `14` 区别于 `05` 的实现链条已经失活”；
- 在该问题解决前，继续做 `16` 方向只会得到 zero-delta 实验。

## 7. 当前阶段判断
- 已完成 `current-code re-baseline + canonical refresh`
- 已定位协议漂移与实现漂移
- 已完成是否继续 `16` 的决策：`否`
- `S2` 仍然**不能通过，也绝对不能进入 `S3`**

## 8. 下一步建议
下一步唯一合理主线：
1. 保持 `S2 dev quick canonical = frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
2. 以 `05_anchor_noroute_recheck600` 与 `14_bonn_localclip_drive_recheck` 的等价性为靶点
3. 专门恢复 `historical 05 -> 14` 那段实现收益链条
4. 只有在 `14` 再次显著区别于 `05` 后，才重新讨论 `16` 这类 Bonn-only 微调
