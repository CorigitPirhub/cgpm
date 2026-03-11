# S2 唯一继续配置冻结页

版本：`2026-03-10`

## 1. historical archive
historical `S2` 中，唯一继续保留的 branch label 是：
- `14_bonn_localclip_drive`

它在 historical 文档中的冻结理由是：
- 相对 `10_anchor_ultralite_bonn_relaxed`：
  - TUM `Acc: 2.724 -> 2.601 cm`
  - TUM `Comp-R: 98.9% -> 99.13%`
  - Bonn `ghost_reduction_vs_tsdf: 32.09% -> 35.69%`
  - Bonn `Acc: 3.759 -> 3.755 cm`

## 2. current-code canonical lock
经本轮 `re-baseline` 确认，current-code `S2 dev quick` canonical 必须显式写为：
- `frames=5`
- `stride=3`
- `seed=7`
- `max_points_per_frame=600`

在该 canonical 下，当前角色应区分为：
- `14_bonn_localclip_drive_recheck`：`current-code control label`
- `25_rear_bg_coupled_formation`：rear/bg formation 恢复基线
- `30_rps_commit_geom_bg_soft_bank`：当前唯一新的 `iterate` 候选
- `31/32/33_rps_commit_quality_bank_*`：current-code committed rear-bank quality enhancement negative chain
- `34/35/36_bonn_compete_*`：current-code competition-only negative chain

## 3. current-code 数值
`14_bonn_localclip_drive_recheck`
- `TUM Acc = 0.9355 cm`
- `TUM Comp-R = 68.53%`
- `Bonn Acc = 2.8864 cm`
- `Bonn Comp-R = 83.57%`
- `Bonn ghost_reduction_vs_tsdf = -8.00%`

`25_rear_bg_coupled_formation`
- `TUM Acc = 0.9400 cm`
- `TUM Comp-R = 91.90%`
- `Bonn Acc = 2.8867 cm`
- `Bonn Comp-R = 83.57%`
- `Bonn ghost_reduction_vs_tsdf = -7.56%`

`30_rps_commit_geom_bg_soft_bank`
- `TUM Acc = 0.9413 cm`
- `TUM Comp-R = 91.93%`
- `Bonn Acc = 2.8868 cm`
- `Bonn Comp-R = 83.57%`
- `Bonn ghost_reduction_vs_tsdf = -7.56%`

## 4. 本轮冻结判断
- `15_bonn_localclip_band_relax`：`abandon`
- `16_bonn_localclip_pfv_rearexpand`：`abandon`
- `17-24`：全部 `abandon`
- `25_rear_bg_coupled_formation`：`superseded-iterate-source`
- `30_rps_commit_geom_bg_soft_bank`：`iterate`
- `31_rps_commit_quality_bank_mid`：`abandon`
- `32_rps_commit_quality_bank_geom`：`abandon`
- `33_rps_commit_quality_bank_push`：`abandon`
- `34_bonn_compete_bgscore`：`abandon`
- `35_bonn_compete_softgap`：`abandon`
- `36_bonn_compete_softgap_support`：`abandon`
- `37_bonn_admission_gate_relax`：`abandon`
- `38_bonn_state_protect`：`iterate`
- `39_bonn_admission_gate_plus_protect`：`abandon`
- `40_bonn_geometry_aligned_admission`：`abandon`
- `41_bonn_geometry_occlusion_admission`：`abandon`
- `42_bonn_geometry_density_gate`：`abandon`
- `43_history_guided_background_location`：`abandon`
- `44_history_plus_ghost_suppress`：`abandon`
- `45_visual_evidence_anchor_strict`：`abandon`
- `46_history_background_only_admission`：`abandon`
- `47_history_visible_obstructed_manifold`：`abandon`
- `48_stable_background_memory_state`：`abandon`
- `49_relaxed_manifold_guided_generation`：`abandon`
- `50_dense_background_propagation`：`abandon`
- `51_geometry_guided_manifold_completion`：`abandon`
- `52_dual_scale_manifold_fusion`：`abandon`
- `53_surface_adjacent_propagation`：`abandon`
- `54_normal_guided_manifold_extension`：`abandon`
- `55_surface_constrained_ray_projection`：`abandon`
- `56_temporal_occlusion_tunneling`：`abandon`
- `57_historical_surface_rear_projection`：`abandon`
- `58_ghost_aware_surface_inpainting`：`abandon`
- `59_relaxed_occlusion_tunneling`：`abandon`
- `60_cone_based_rear_projection`：`abandon`
- `61_hybrid_confidence_gating`：`abandon`
- `62_dense_patch_projection`：`abandon`
- `63_multi_hypothesis_depth_sampling`：`abandon`
- `64_patch_depth_hybrid_generation`：`abandon`
- `65_ghost_risk_prediction_filter`：`abandon`
- `66_geometry_constrained_admission`：`abandon`
- `67_topk_selective_generation`：`abandon`
- `68_rear_front_score_competition`：`abandon`
- `69_depth_gap_validation`：`abandon`
- `70_fused_discriminator_topk`：`abandon`
- `71_occlusion_order_consistency`：`abandon`
- `72_local_geometric_conflict_resolution`：`abandon`
- `73_front_residual_aware_suppression`：`abandon`
- `74_static_history_weight_boosting`：`abandon`
- `75_dynamic_shell_masking`：`abandon`
- `76_surface_persistent_anchoring`：`abandon`
- `77_hybrid_boost_conflict`：`abandon`
- `78_conservative_anchoring`：`abandon`
- `79_feature_weighted_topk`：`abandon`
- `80_ray_penetration_consistency`：`abandon`
- `81_unobserved_space_veto`：`abandon`
- `82_static_neighborhood_coherence`：`abandon`
- `83_minimum_thickness_topology_filter`：`abandon`
- `84_front_back_normal_consistency`：`abandon`
- `85_occlusion_ray_convergence_constraint`：`abandon`
- `86_rear_plane_clustering_snapping`：`abandon`
- `87_front_mask_guided_back_projection`：`abandon`
- `88_occlusion_depth_hypothesis_validation`：`abandon`
- `89_front_back_surface_pairing_guard`：`abandon`
- `90_background_plane_evidence_accumulation`：`abandon`
- `91_occlusion_depth_hypothesis_tb_protection`：`abandon`
- `92_multi_view_ray_support_aggregation`：`abandon`
- `93_spatial_neighborhood_density_clustering`：`abandon`
- `94_historical_tsdf_consistency_reactivation`：`abandon`
- `95_geodesic_balloon_consistency`：`abandon`
- `96_support_cluster_model_fitting`：`positive-subchain / not-promoted`
- `97_global_map_anchoring`：`positive-subchain / not-promoted`
- `98_geodesic_support_diffusion`：`positive-subchain / not-promoted`
- `99_manhattan_plane_completion`：`positive-subchain / not-promoted`
- `100_cluster_view_inpainting`：`positive-subchain / not-promoted`
- `101_manhattan_plane_projection_hard_snapping`：`abandon`
- `102_scale_drift_correction`：`abandon`
- `103_local_cluster_refinement`：`abandon`
- `104_depth_bias_minus1cm`：`upstream-signal / not-promoted`
- `105_depth_bias_plus1cm`：`upstream-signal / not-promoted`
- `108_geometry_chain_coupled_direct`：`positive-subchain / not-promoted`
- `109_geometry_chain_coupled_projected`：`positive-subchain / not-promoted`
- `110_geometry_chain_coupled_conservative`：`abandon`
- `111_native_geometry_chain_direct`：`native-mainline-validated / not-promoted`
- `112_native_geometry_chain_projected`：`native-mainline-validated / not-promoted`
- 因此当前下一步**不应继续沿 `16` 做 Bonn-only 微调**，也不应继续堆 purely internal committed-bank strengthening、competition-only tuning、宽松 space-redirect、过硬的 strict HVO gate，或当前这版稀疏/无差别 dense-manifold state；当前唯一合理继续方向仍是围绕 `38` 的 rear-state geometry quality / spatial placement 主线

## 5. 漂移结论
本轮已定位两层 drift：
1. **协议漂移**：historical 文档缺失 `max_points_per_frame=600`
2. **实现漂移**：修正协议后，current-code `14` 仍退化为与 `05_anchor_noroute` 等价的行为族

## 6. 阶段状态
- `S2` 仍然**没有通过**
- 当前更准确的状态是：`re-baselined / drift-localized / not-pass`
- **绝对禁止进入 `S3`**

## 本轮 downstream chain 攻击结果
- `17_banked_export_compete`：`abandon`
- `18_banked_dual_compete_export`：`abandon`
- `19_banked_nonpersistent_export`：`abandon`
- `20_soft_rear_bank_export`：`abandon`
- `21_soft_rear_bank_dual_compete`：`abandon`
- `22_soft_rear_bank_nonpersistent`：`abandon`

补充判断：
- 当前 downstream `sync/export` 侧的 family 已被验证为统一 zero-delta；
- 随后的 competition-only family（`34/35/36`）也已被证伪；
- 因此 current-code control label 仍只能是 `14_bonn_localclip_drive_recheck`；
- 下一步若继续 `S2`，主线必须固定在 `38_bonn_state_protect` 对应的 admission-aware rear-state persistence / pre-extract admission，而不是再回到 export-only family 或更早的无关支线。


## 当前唯一继续配置（更新于 2026-03-10）
- `38_bonn_state_protect`

补充说明：
- `80_ray_penetration_consistency` 仅作为 `80`-based rear post-filter / topology / pairing 子链的控制基数组，不构成新的全局唯一继续配置；
- `86-112` 已进一步证明：即使切回 `80` 作为支链工作基数，也尚未产出新的可接受全局 `iterate` 候选。

更新理由：
- `30_rps_commit_geom_bg_soft_bank` 仍是 admission 主线的起点配置；
- 本轮 control 诊断进一步证明：Bonn 的主流失点在 extract admission 的 `score_gate`，而不是最终 competition；
- `38_bonn_state_protect` 在不引入明显额外结构性退化的前提下，把 Bonn `extract_rear_selected_sum` 从 `7` 提升到 `108`，并把 Bonn family-mean `ghost_reduction_vs_tsdf` 从 `13.25%` 提升到 `15.47%`；
- `39` 虽然把 admission 数量推得更高，但最终 ghost 表现反而弱于 `38`，说明更强的 admission 并不等于更好的几何质量；
- 本轮 `40/41/42` 又进一步证明：即使增加几何一致性、遮挡历史和轻量密度门控，rear points 仍主要落在 `hole/noise` 区域，尚未形成足够多的 true-background coverage；
- 随后的 `43/44/45` space-redirect 复核则进一步证明：即使提高历史背景引导和视觉锚定，true-background rear points 虽有小幅增加，但 ghost 区误入同步明显膨胀，因此最终指标仍不如 `38`；
- 本轮 `46/47` 的 strict history-visible + currently-obstructed 约束又进一步证明：若直接把历史可见性/遮挡约束做成硬门槛，rear points 会被整体清零，虽然 `Comp-R` 出现极小数值波动，但动态抑制明显回退到 `13.29%`，因此这条硬约束也不能成立；
- 本轮 `48/49` 的稳定背景流形状态则进一步证明：显式背景记忆能够把 `ghost` 从 `10` 压到 `3-4`、把 `hole/noise` 从 `97` 压到 `25-34`，但同时把 rear 总量压缩到 `29-39`，且 `true_background` 仍停在 `1`，因此当前这版 manifold state 依然没有学成可用的背景流形；
- 本轮 `50/51/52` 的 dense manifold 传播又进一步证明：局部传播可以恢复 rear 数量、减少部分 hole/noise，但并没有把支持有效重定向到 true-background 表面，最佳候选也只能把 `true_background` 提到 `2`；
- 本轮 `53/54/55` 的 surface-constrained 传播则进一步证明：沿表面约束传播能把 `true_background` 从 `1` 提到 `3`，但仍远未达到目标，且 ghost 仍保持在 `12-14`，因此表面约束本身还不足以完成有效重定向；
- 本轮 `56/57/58` 的 occlusion-aware bridging 又进一步证明：桥接机制能把 `ghost` 控制回 `9`，并把 `true_background` 稍微提升到 `2-3`，但同时 rear 总量显著下降到 `57-58`，因此当前桥接覆盖面仍不足以替代 `38`；
- 本轮 `59/60/61` 的 high-coverage bridge 恢复又进一步证明：即使放宽桥接条件、加入锥形投影或混合筛选，rear 总量也只能恢复到 `65` 左右，`true_background` 仍然只有 `2`；
- 本轮 `62/63/64` 的 multi-candidate generation 则进一步证明：即使直接在 bridge 目标上做 dense patch / depth hypothesis / hybrid 扩张，rear 总量可以恢复到 `250-442`，`true_background` 也能提高到 `3-8`，并把 Bonn `ghost_reduction_vs_tsdf` 最多推到 `16.48%`，但 `ghost` 会同步膨胀到 `23-60`、`Comp-R` 仍停在 `70.80-70.82%`，因此“以量换面”的方向依然没有学会真正的 true-background targeting；
- 本轮 `65/66/67` 的 ghost-capped selectivity 则进一步证明：当前在线 selectivity 信号还不足以把 `64` 恢复出的覆盖量重新压回 true-background manifold；`65` 的风险过滤没有切掉任何 rear points，`66` 的 geometry-constrained admission 反而把 rear 放大到 `555` 并把 `ghost` 推到 `84`，`67` 的 Top-K 虽然把 Bonn `ghost_reduction_vs_tsdf` 推到 `22.16%`，但仍只做到 `rear = 231` / `true_background = 3` / `ghost = 38`，因此目前还不能把它接受为可继续配置；
- 本轮 `68/69/70` 的 discriminative fusion 则进一步证明：`rear-front score competition` 确实能继续把 Bonn `ghost_reduction_vs_tsdf` 推高到 `23.52%`，并把 `ghost` 从 `41` 压到 `32`，但 `true_background` 同时跌到 `2`；而 `69/70` 的 depth-gap / fused top-k 在当前在线特征下直接退化成 `rear = 0` 的零点解，因此当前 `rear_score/front_score/rear_gap` 这组判别特征还不足以形成可接受的在线分类器；
- 本轮 `71/72/73` 的 semantic-aware occlusion/conflict 方向则进一步证明：高阶时空特征确实开始提供增益；其中 `72_local_geometric_conflict_resolution` 在保持 Bonn `ghost_reduction_vs_tsdf = 27.27%` 的同时，把 `ghost` 压到 `24`、把 `true_background` 从 `2` 提到 `4`，成为当前这条 semantic branch 的最佳工作基数，但仍远未达到 `Ghost <= 15` 与 `TB >= 8`，说明当前在线特征仍不足以完成最终分布收敛；
- 本轮 `74/75/76` 的 static-persistence anchoring 则进一步证明：直接把历史静态持久性拉高并不能自动恢复真实背景表面；`74` 只能把 `true_background` 提到 `4` 且丢掉 `22%` 门槛，`75/76` 虽把 ghost 压到 `28-29`，但 `Comp-R` 同时跌到 `68.8%` 左右，因此当前静态锚定仍没有学会把 `hole/noise` 转化为有效背景覆盖；
- 本轮 `77/78/79` 的 hybrid optimization 则进一步证明：静态锚定与局部冲突的线性组合可以小幅改善局部分布，但仍没有找到可继续配置；其中 `77_hybrid_boost_conflict` 把 `true_background` 从 `3` 提到 `4` 并把 `ghost` 从 `30` 降到 `29`，但 `ghost_reduction_vs_tsdf` 从 `22.10%` 掉到 `21.83%`；`79_feature_weighted_topk` 守住了 `ghost_reduction_vs_tsdf = 22.24%`，但 `true_background` 仍只有 `3`，未优于 `72`；`78_conservative_anchoring` 则同时丢掉动态抑制和 ghost 控制；
- 同时，本轮已经修复静态锚定特征的统计链路：`history_anchor_mean` 在 Bonn `72/77/78/79` 上恢复为 `0.246 / 0.247 / 0.246 / 0.250`，`surface_anchor_mean` 为 `1.000`，因此后续不应再把“统计全零”当作方法失败原因；当前失败原因已经明确回到“特征虽生效，但判别力不足”；
- 本轮 `80/81/82` 的 ray-consistency 方向则进一步证明：`penetration_score` 与 `observation_support` 已经不再全零，并对 TB / Noise 产生了有限分离；例如控制组 `72` 在 Bonn 上已有 `penetration(TB/Noise) = 0.156 / 0.105`、`observation(TB/Noise) = 0.246 / 0.183`。但这种分离还不足以转化为真正的地图分布突破：`80_ray_penetration_consistency` 把 `ghost` 压到 `15` 且 `ghost_reduction_vs_tsdf = 22.66%`，`82_static_neighborhood_coherence` 把 `ghost` 压到 `18` 且 `ghost_reduction_vs_tsdf = 22.70%`，但两者 `true_background` 都仍只有 `4`，`noise_ratio` 仍停在 `0.84 / 0.81`；`81_unobserved_space_veto` 也只能把 `ghost` 压到 `25`，没有恢复 TB。说明当前问题已经从“特征缺失”转成“方向性拓扑约束仍不足”，现有 cell-local 物理特征还不能把大块 hole/noise 转化为 true-background coverage；
- 本轮 `83/84/85` 的 topology-constraint 方向则进一步证明：结构级约束比此前的 cell-local 标量特征更擅长清理 `Noise/Ghost`。`83_minimum_thickness_topology_filter` 把 `Noise` 压到 `106`、`Ghost` 压到 `16`，`85_occlusion_ray_convergence_constraint` 把 `Noise` 压到 `99`、`Ghost` 压到 `22`，并且二者都维持 `ghost_reduction_vs_tsdf > 22%`；同时 `85` 的 kept/dropped 拓扑统计已经形成明显分离：`thickness = 0.383 / 0.055 m`、`normal = 0.940 / 0.694`、`convergence = 0.729 / 0.424`。但 `TB` 仍稳定停在 `4`，说明当前拓扑约束只能净化已有候选，尚不能把被遮挡的真实背景重新定位出来；
- 本轮 `86/87/88` 的 plane-attribution 方向则进一步证明：激进的平面吸附/归属在当前 `80` 基数组上会直接走向过清洗灾难；三组变体都把 `TB` 从 `3-4` 清到 `0`，同时把 Bonn `Comp-R` 压到 `67-69%`，说明“先假定是噪声、再做平面拟合”会误杀仅存的 true background；
- 本轮 `89/90/91` 的 pairing-evidence 方向则进一步证明：保守的前后表面对配可以避免再次出现 `TB=0`，但仍没有形成 `Noise -> TB` 转化。`89_front_back_surface_pairing_guard` 在保护 `115` 个点、吸附 `96` 个点后仍只做到 `TB=4 / Ghost=20 / Noise=95 / ghost_reduction_vs_tsdf=22.61%`；`90_background_plane_evidence_accumulation` 把 `Noise` 进一步降到 `91`、`Comp-R` 稍回升到 `70.24%`，但 `ghost_reduction_vs_tsdf` 掉到 `21.58%`；`91_occlusion_depth_hypothesis_tb_protection` 只吸附 `8` 个点，却守住了 `TB=4 / Ghost=19 / Noise=96 / ghost_reduction_vs_tsdf=23.52%`，说明当前 evidence-based pairing 只能“避免过清洗”，还不能“恢复 TB”；
- 本轮 `92/93/94` 的 HSSA 方向则进一步证明：支持聚合已经开始具备“正向造面”能力。`92_multi_view_ray_support_aggregation` 在不牺牲 Ghost/Comp-R 的前提下把 `TB` 从 `4` 提到 `5`；`93_spatial_neighborhood_density_clustering` 通过 flat support cluster patch 把 `TB` 直接提到 `13`，并守住 `ghost_reduction_vs_tsdf = 23.59% / Ghost = 19`；`94_historical_tsdf_consistency_reactivation` 也把 `TB` 提到 `8`，并守住 `ghost_reduction_vs_tsdf = 22.45% / Ghost = 19`。这说明 `80`-based 支链已经不再只是“防止 TB 归零”，而是能够在局部序列上恢复隐藏背景表面；
- 但 `92/93/94` 的共同失败点也同样明确：`tb_noise_correlation` 仍停在 `0.991` 左右，说明所有新增支持几乎都集中在单个序列，尚未在 family 级别打破 `TB-Noise` 正相关耦合；因此它们仍不能升级为新的全局 `iterate` 配置；
- 本轮 `95/96/97` 的 balloon-cluster 方向则进一步证明：cluster-level validation 已经足以真正打破相关性死锁。`95_geodesic_balloon_consistency` 因 retain 门槛过严而退化为与 `93` 等价；但 `96_support_cluster_model_fitting` 通过 plane-band 过滤把 `TB/Noise/Ghost` 压到 `12/23/6`，并把 `tb_noise_correlation` 拉到 `-0.756`；`97_global_map_anchoring` 进一步把 `TB/Noise/Ghost` 调整到 `10/22/5`，并把 `tb_noise_correlation` 拉到 `-0.786`、`ghost_reduction_vs_tsdf` 推到 `34.06%`。这说明“簇级一致性验证”已经是正解，family 级解耦问题已被解决；
- 但 `96/97` 也清楚暴露了新的瓶颈：为了打破相关性，它们把 Bonn `Comp-R` 一并压到了 `68.36% / 68.90%`。因此它们只能记为 `positive-subchain / not-promoted`，而不能提升为新的全局 continue config；
- 本轮 `98/99/100` 的 deep-explore 方向则进一步证明：`purified cluster completeness recovery` 已经成立。`98_geodesic_support_diffusion` 把 Bonn `Comp-R` 拉回 `70.08%`，并保持 `tb_noise_correlation = -0.327 / ghost_reduction_vs_tsdf = 28.00% / TB = 40`；`99_manhattan_plane_completion` 也把 `Comp-R` 拉回 `70.08%`，并保持 `tb_noise_correlation = -0.225 / ghost_reduction_vs_tsdf = 28.28% / TB = 39`；`100_cluster_view_inpainting` 同样守住 `Comp-R = 70.08%` 与负相关，但 `ghost` 回升更明显。因此“purified cluster -> structured completion”已成为当前最强的局部正链；
- 但 `98/99/100` 仍只能记为 `positive-subchain / not-promoted`：虽然它们已经满足本轮 local target，却依然远未触及 `S2` 全局硬门槛（尤其是 `Acc` 与 `98% Comp-R`），也还没有完成对 `RB-Core` 的完整净正闭环；
- 本轮 `101/102/103` 的 Acc-tightening 方向则进一步证明：当前 `Acc` 缺口不在“点云厚度噪声”这一层。`101_manhattan_plane_projection_hard_snapping` 虽把 `mean_distance_to_plane` 从 `0.2208` 压到 `0.2143`，但 Bonn `Acc` 仍是 `4.347 cm`；`102_scale_drift_correction` 则把 `mean_distance_to_plane_after` 放大到 `0.5763`，并把 Bonn `Acc` 直接恶化到 `5.269 cm`，说明没有可用的 geometry-only scale drift 证据；`103_local_cluster_refinement` 只带来 `0.0002` 量级的平面厚度变化，Bonn `Acc` 仍是 `4.346 cm`。因此这条后验几何紧化链已经被证伪；
- 本轮 `104/105` 的 upstream-geometry 方向则进一步证明：真实上游 depth bias 确实存在可测响应。`104_depth_bias_minus1cm` 把 Bonn `Acc` 从参考 `99` 的 `4.346 cm` 拉到 `4.238 cm`，并把同口径平面距离从 `0.0980` 压到 `0.0746`；`105_depth_bias_plus1cm` 也让 Bonn `Acc` 落在 `4.238 cm`，但平面距离回到 `0.0983`。这说明系统确实对上游 depth bias 敏感，但目前可执行的 raw upstream 前驱并不能保住 downstream rear branch，因此它们只能记为 `upstream-signal / not-promoted`；
- 本轮 `108/109/110` 的 geometry-chain coupling 方向则进一步证明：结构断裂本身已经被修复。`108_geometry_chain_coupled_direct` 直接把 `104` corrected geometry 喂给 `99` 的 rear donor/completion，就恢复了 `TB=39` 并把 Bonn `Acc` 压到 `4.233 cm`；`109_geometry_chain_coupled_projected` 进一步把 donor rear points 投到 corrected plane frame，得到同轮最优 Bonn `Acc = 4.233 cm` 与 `Comp-R = 70.84%`；`110_geometry_chain_coupled_conservative` 则说明只保留 purified cluster 会把 `TB` 压回 `10`。因此当前问题已不再是“能否修复 104→99 断裂”，而是“如何把这条 one-way repaired chain 内化回真实可执行主线”；
- 本轮 `111/112` 的 Native Mainline Integration 则进一步证明：原生可执行主线已经完成重置。`111_native_geometry_chain_direct` 与 `108` 的 Bonn `Acc` 偏差仅 `7.7e-14 cm`；`112_native_geometry_chain_projected` 与 `109` 的 Bonn `Acc` 偏差仅 `0.00079 cm`。这说明耦合逻辑已经成功内化进标准管道，而临时 runner 仅保留为诊断脚本，不再是必需执行路径；
- 因此当前主瓶颈已进一步收敛为：`80`-based 结构约束、证据保护、HSSA、balloon validation、purified completion 与 native integration 已经能做到 “TB 恢复 + 相关性解耦 + Comp-R >= 70 + Ghost 保持”，当前唯一剩余缺口是上游几何形成畸变导致的 `Acc` 硬缺口，以及 support-cluster completion 的可触达范围不足；
- 因此当前最值得继续的唯一 `iterate` 候选仍然是 `38_bonn_state_protect`。
