# S2 ray consistency analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_RAY_CONSISTENCY_COMPARE.csv`

## 1. 新特征是否摆脱饱和
- 控制组 `72`: penetration=`0.099`, observation=`0.183`, static_coherence=`0.823`
- `best TB` `80_ray_penetration_consistency`: penetration=`0.338`, observation=`0.168`, static_coherence=`0.882`
- 控制组 TB vs Noise: penetration=`0.156` vs `0.105`, observation=`0.246` vs `0.183`, coherence=`0.750` vs `0.801`
- 厚度/空洞跨度代理 `penetration_free_span_mean` 也已摆脱饱和：控制组 `72 = 0.112 m`，`80 = 0.414 m`，`82 = 0.383 m`。
- 新特征已经不再是全零/全饱和，但分离力并不稳定：`observation_support` 仍能拉开 TB/Noise，而 `penetration_score` 在 `80/82` 上没有把 TB 排到 Noise 之前。

## 2. Noise 与 Ghost 是否被削减
- 控制组 `72`: rear=`234`, TB=`4`, Ghost=`27`, Noise=`203` (`noise_ratio=0.868`), `ghost_reduction_vs_tsdf=22.64%`
- `80_ray_penetration_consistency` 把 `Ghost` 压到最低：rear=`119`, TB=`4`, Ghost=`15`, Noise=`100` (`noise_ratio=0.840`), `ghost_reduction_vs_tsdf=22.66%`
- `81_unobserved_space_veto` 没有提升 TB：rear=`231`, TB=`4`, Ghost=`25`, Noise=`202` (`noise_ratio=0.874`), `ghost_reduction_vs_tsdf=22.18%`
- `82_static_neighborhood_coherence` 在 `ghost_reduction_vs_tsdf` 上最高：TB=`4`, Ghost=`18`, Noise=`93` (`noise_ratio=0.809`), `ghost_reduction_vs_tsdf=22.70%`
- 结论：`80/82` 已经把 `hole_or_noise_sum` 压到 `180` 以下，并把 `Ghost` 保持在 `25` 以下，但 `TB` 仍停在 `4`。

## 3. 诊断结论
- 若 `penetration_score` 只能帮助压低 `Ghost/Noise`，却不能把 `TB` 从 `4` 推高，说明当前候选生成本身仍缺少 true-background 命中密度，筛选只能做有限净化。
- 若 `observation_support` 成功拉开 TB 与 Noise，但 ghost 仍居高，说明 synthetic-only 噪声被清掉了，剩余误差主要来自已观测动态残留。
- 若 `static_coherence` 对 TB 与 Noise 可分，但 `noise_ratio` 仍高，说明当前孤立噪声并不完全缺邻域结构，后续还需要更强的 front/back 几何冲突特征。

## 4. 阶段判断
- 若未达到 `TB >= 6`、`noise_ratio < 0.75`、`Ghost <= 25` 与 `ghost_reduction_vs_tsdf >= 22%`，则 `S2` 仍未通过，绝对不能进入 `S3`。
