# S2 pairing evidence analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_PAIRING_EVIDENCE_COMPARE.csv`

## 1. 前后表面对配是否保护了 TB
- 控制组 `80`: TB=`4`, Ghost=`19`, Noise=`96`
- 最佳候选 `91_occlusion_depth_hypothesis_tb_protection`: TB=`4`, Ghost=`19`, Noise=`96`
- `91_occlusion_depth_hypothesis_tb_protection` 受保护点=`115`，实际吸附点=`8`；若 TB 与控制组持平，说明保护逻辑至少阻止了再次过清洗，但没有新增 TB 证据。

## 2. 证据积累是否避免了过度清洗
- 控制组 Comp-R / Acc: `70.20% / 4.317 cm`
- 候选 Comp-R / Acc: `70.20% / 4.317 cm`
- TB-Noise 相关系数：`0.991`
- 解释：当前为正相关，说明本轮仍主要是削减 Noise，而不是把 Noise 转成 TB。

## 3. 结论
- 若 `TB` 没有超过控制组，说明当前保守平面吸附仍未真正实现 `Noise -> TB` 转化。
- 若 `Comp-R` 没有继续下跌，则说明“有证据保留”至少避免了上一轮 `TB=0` 的过度清洗灾难。
- 本轮相关性为正，说明证据配对仍停留在“保守保留/轻微清洗”，没有形成明确的 `Noise -> TB` 转化链路。

## 4. 阶段判断
- 若未达到 `TB > 6`、`ghost_reduction_vs_tsdf >= 22%`、`Ghost <= 25` 与 `Comp-R >= 70%`，则 `S2` 仍未通过，绝对不能进入 `S3`。
