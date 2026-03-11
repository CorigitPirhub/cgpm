# S2 balloon cluster analysis

日期：`2026-03-10`
协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`
对比表：`processes/s2/S2_RPS_BALLOON_CLUSTER_COMPARE.csv`

## 1. 簇级一致性是否区分了真假 TB
- 控制组 `93`：`fit(TB/Noise) = 0.0023 / 0.0074`，`geo(TB/Noise) = 0.1782 / 0.0543`。
- 最佳候选 `96_support_cluster_model_fitting`：`fit(TB/Noise) = 0.0023 / 0.0074`，`geo(TB/Noise) = 0.1782 / 0.0543`。
- 当前分离主因来自 `cluster_fitting_error`：TB 簇的平面拟合误差显著低于 Noise 簇；`geodesic_smoothness` 单独使用时并不构成有效门槛，这也是 `95` 退化为零保留的原因。

## 2. 相关性是否打破死锁
- 控制组 `93`：TB=`13`, Noise=`96`, `tb_noise_correlation=0.991`。
- 最佳候选 `96_support_cluster_model_fitting`：TB=`12`, Noise=`23`, `tb_noise_correlation=-0.756`。

## 3. 结论
- 若 `tb_noise_correlation < 0.9`，说明簇级验证至少打破了 `crowd2` 单序列过拟合对 family 统计的绑架。
- 若 `TB` 仍保持 `>= 8` 且 `ghost_reduction_vs_tsdf >= 22%`，说明这条支链已从“局部加点”推进到“全局一致性过滤”。
- 即便本轮局部过线，也不代表 `S2` 整体通过；`Acc/Comp-R` 全局门槛仍未满足时，绝对不能进入 `S3`。
