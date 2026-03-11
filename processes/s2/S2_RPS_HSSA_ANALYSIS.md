# S2 HSSA analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_HSSA_COMPARE.csv`

## 1. 支持度特征是否区分了 TB 与 Noise
- 控制组 `80`：`support_score(TB/Noise) = 0.210 / 0.146`，`history_reactivate(TB/Noise) = 0.230 / 0.208`。
- 最佳候选 `93_spatial_neighborhood_density_clustering`：`support_score(TB/Noise) = 0.241 / 0.146`，差值=`0.095`。

## 2. HSSA 是否打破了 TB=4 的停滞
- 控制组 `80`：TB=`4`, Ghost=`19`, Noise=`96`。
- 最佳候选 `93_spatial_neighborhood_density_clustering`：TB=`13`, Ghost=`19`, Noise=`96`, 新增支持点=`9`。

## 3. 相关性是否解耦
- 控制组相关性：`0.991`。
- 候选相关性：`0.991`。
- 若相关性仍接近 `1.0`，说明支持度聚合只在单个序列形成局部增益，尚未在 family 层打破 `TB-Noise` 耦合。

## 4. 阶段判断
- 若未同时满足 `TB > 6`、`tb_noise_correlation < 0.9`、`support_score(TB) > support_score(Noise)`、`ghost_reduction_vs_tsdf >= 22%`，则 `S2` 仍未通过，绝对不能进入 `S3`。
