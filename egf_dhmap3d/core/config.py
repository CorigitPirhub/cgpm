from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    fx: float = 517.3
    fy: float = 516.5
    cx: float = 318.6
    cy: float = 255.3
    depth_scale: float = 5000.0
    depth_min: float = 0.2
    depth_max: float = 4.5


@dataclass
class Map3DConfig:
    voxel_size: float = 0.05
    truncation: float = 0.15
    init_grad_cov: float = 0.40
    min_cov: float = 1e-4
    max_cov: float = 5.0
    rho_decay: float = 0.992
    phi_w_decay: float = 0.992
    # Geometry-channel confidence decay (kept milder than dynamic channel by default).
    phi_geo_w_decay: float = 0.996
    cov_inflation: float = 1.004


@dataclass
class Assoc3DConfig:
    gate_threshold: float = 14.0
    relax_gate_scale: float = 3.0
    sigma_d0: float = 0.04
    sigma_n0: float = 0.18
    alpha_evidence: float = 1.0
    beta_uncert: float = 0.06
    huber_delta_n: float = 0.20
    strict_surface_weight: float = 0.8
    search_radius_cells: int = 2
    frontier_relax_boost: float = 1.5
    frontier_activate_thresh: float = 1.2
    use_normal_residual: bool = True
    use_evidence_in_noise: bool = True
    seed_fallback_enable: bool = True
    seed_fallback_low_support_scale: float = 0.7
    seed_fallback_frontier_scale: float = 0.7
    # Heteroscedastic noise model (incidence/depth/normal residual aware).
    hetero_enable: bool = False
    hetero_inc_ref_cos: float = 0.65
    hetero_depth_ref_m: float = 2.5
    hetero_normal_ref: float = 0.20
    hetero_k_inc: float = 0.45
    hetero_k_depth: float = 0.12
    hetero_k_normal: float = 0.55
    hetero_good_cos: float = 0.90
    hetero_good_bonus: float = 0.20
    hetero_sigma_d_min_scale: float = 0.75
    hetero_sigma_d_max_scale: float = 1.75
    hetero_sigma_n_min_scale: float = 0.70
    hetero_sigma_n_max_scale: float = 2.20
    # Pre-association contradiction gating:
    # move dynamic contradiction handling upstream to association stage.
    contra_gate_enable: bool = True
    contra_stmem_weight: float = 0.65
    contra_visibility_weight: float = 0.20
    contra_residual_weight: float = 0.15
    contra_free_ratio_ref: float = 1.0
    contra_rho_ref: float = 1.6
    contra_static_guard: float = 0.70
    contra_rho_guard: float = 0.55
    contra_d2_boost_max: float = 2.2


@dataclass
class Update3DConfig:
    rho_sigma: float = 0.08
    eta_normal: float = 0.75
    eta_evidence: float = 0.25
    poisson_lr: float = 0.08
    poisson_iters: int = 1
    eikonal_lambda: float = 0.02
    dyn_score_alpha: float = 0.10
    dyn_d2_ref: float = 8.0
    dyn_forget_gain: float = 0.12
    forget_mode: str = "local"  # local | global | off
    frontier_boost: float = 0.45
    frontier_decay: float = 0.92
    surf_band_ratio: float = 0.45
    evidence_decay: float = 0.985
    rho_osc_ref: float = 0.8
    dscore_ema: float = 0.12
    residual_score_weight: float = 0.25
    # Structural decoupling dynamic state:
    # independent dynamic probability used by suppression operator.
    dyn_state_alpha: float = 0.16
    dyn_state_conflict_weight: float = 0.45
    dyn_state_visibility_weight: float = 0.25
    dyn_state_residual_weight: float = 0.20
    dyn_state_osc_weight: float = 0.10
    # SSE-EM: latent static/dynamic responsibility update (module-1).
    sse_em_enable: bool = False
    sse_em_prior_temp: float = 0.9
    sse_em_assoc_weight: float = 0.30
    sse_em_residual_weight: float = 0.30
    sse_em_free_weight: float = 0.20
    sse_em_rho_weight: float = 0.10
    sse_em_visibility_weight: float = 0.10
    sse_em_mstep_alpha: float = 0.20
    sse_em_static_floor: float = 0.05
    sse_em_dynamic_ceil: float = 0.95
    # Dynamic-state SDF channel (phi_dyn) integration:
    # explicit transient geometry state to decouple suppression from geometry debias.
    dyn_channel_enable: bool = True
    # PT-DSF: persistent-transient dual surface field formalization.
    ptdsf_enable: bool = False
    ptdsf_rho_alpha: float = 0.18
    ptdsf_static_blend: float = 0.55
    ptdsf_commit_age_ref: float = 3.0
    ptdsf_commit_bonus: float = 0.08
    ptdsf_rollback_bonus: float = 0.06
    # WOD: write-time occlusion decomposition.
    wod_enable: bool = False
    wod_alpha: float = 0.18
    wod_front_margin_vox: float = 0.35
    wod_rear_margin_vox: float = 0.35
    wod_shell_margin_vox: float = 1.10
    wod_history_mix: float = 0.55
    wod_rear_consistency_ref: float = 0.04
    wod_front_transient_boost: float = 0.35
    wod_rear_static_boost: float = 0.55
    wod_shell_weight: float = 0.35
    wod_geo_front_suppress: float = 0.65
    wod_geo_rear_boost: float = 0.25
    wod_dyn_front_boost: float = 0.45
    wod_dyn_shell_boost: float = 0.20
    wod_shell_free_gain: float = 0.55
    # RPS: rear-persistent surface buffer.
    rps_enable: bool = False
    rps_rho_alpha: float = 0.18
    rps_consistency_ref: float = 0.03
    rps_front_suppress: float = 0.60
    rps_geo_mix: float = 0.65
    rps_rear_boost: float = 0.85
    rps_read_weight_gain: float = 0.70
    rps_rho_ref: float = 1.0
    rps_hard_commit_enable: bool = False
    # Surface-bank readout: treat committed rear geometry as a discrete bank
    # and only export it when it wins an explicit front-vs-rear competition.
    rps_surface_bank_enable: bool = False
    rps_bank_margin: float = 0.08
    rps_bank_separation_ref: float = 0.04
    rps_bank_rear_min_score: float = 0.52
    # WDSG: Write-time Dual Surface Generation.
    # Instead of routing the same signed distance into all channels, synthesize
    # front-transient and rear-static hypotheses directly at update time.
    wdsg_enable: bool = False
    wdsg_front_shift_vox: float = 0.90
    wdsg_rear_shift_vox: float = 1.10
    wdsg_shell_shift_vox: float = 0.40
    wdsg_max_shift_vox: float = 1.80
    wdsg_front_mix_gain: float = 0.95
    wdsg_rear_mix_gain: float = 1.00
    wdsg_proj_floor: float = 0.35
    # WDSG-R: separation-aware routing. Use dual-surface separation to reroute
    # front-dominant observations away from static/geo and into transient/dyn.
    wdsg_route_enable: bool = False
    wdsg_route_static_suppress: float = 0.85
    wdsg_route_geo_suppress: float = 0.75
    wdsg_route_transient_boost: float = 0.75
    wdsg_route_dyn_boost: float = 0.65
    wdsg_route_rear_recover: float = 0.18
    # SPG: Static Promotion Gate. Candidate geometry remains permissive, but only
    # time-consistent static geometry is promoted into the exported front bank.
    spg_enable: bool = False
    spg_route_relax: float = 0.45
    spg_score_alpha: float = 0.18
    spg_commit_on: float = 0.62
    spg_commit_off: float = 0.40
    spg_commit_age_ref: float = 1.5
    spg_commit_blend: float = 0.60
    spg_read_weight_gain: float = 0.85
    spg_candidate_mix: float = 0.22
    spg_bank_decay: float = 0.96
    # OTV: Observation-Time Transient Veto. Detect a separated front transient
    # supported by an already stable rear surface, and reserve it into a
    # short-lived transient bank before it contaminates persistent geometry.
    otv_enable: bool = False
    otv_sep_ref_vox: float = 0.90
    otv_score_alpha: float = 0.18
    otv_support_ref: float = 0.26
    otv_commit_on: float = 0.58
    otv_commit_off: float = 0.38
    otv_age_ref: float = 1.0
    otv_static_veto: float = 0.92
    otv_geo_veto: float = 0.95
    otv_transient_boost: float = 0.85
    otv_dyn_boost: float = 0.75
    otv_decay: float = 0.96
    rps_score_alpha: float = 0.18
    rps_commit_on: float = 0.62
    rps_commit_off: float = 0.40
    rps_commit_age_ref: float = 2.0
    rps_candidate_gate_min: float = 0.16
    rps_commit_blend: float = 0.78
    rps_candidate_decay: float = 0.92
    rps_active_decay: float = 0.97
    dyn_channel_min_weight_ratio: float = 0.08
    dyn_channel_obs_weight: float = 0.50
    dyn_channel_residual_weight: float = 0.30
    dyn_channel_risk_weight: float = 0.20
    dyn_channel_static_suppress: float = 0.55
    dyn_channel_div_weight: float = 0.20
    dyn_channel_div_ref: float = 0.04
    # Explicit dynamic latent state (z_dyn):
    # a dedicated temporal state for suppression routing, decoupled from
    # geometry debias and geometry-channel integration.
    zdyn_enable: bool = False
    zdyn_alpha_up: float = 0.26
    zdyn_alpha_down: float = 0.10
    zdyn_decay: float = 0.985
    zdyn_conflict_weight: float = 0.40
    zdyn_visibility_weight: float = 0.25
    zdyn_residual_weight: float = 0.20
    zdyn_osc_weight: float = 0.10
    zdyn_free_ratio_weight: float = 0.05
    zdyn_free_ratio_ref: float = 1.0
    # Short-term contradiction memory (suppression-only channel).
    stmem_enable: bool = True
    stmem_alpha: float = 0.22
    stmem_decay: float = 0.94
    stmem_conflict_weight: float = 0.42
    stmem_visibility_weight: float = 0.26
    stmem_clear_weight: float = 0.18
    stmem_residual_weight: float = 0.14
    stmem_free_ratio_ref: float = 1.0
    stmem_rho_ref: float = 1.5
    integration_radius_scale: float = 1.0
    integration_min_radius_vox: float = 1.2
    # Apply decay every N frames with mathematically equivalent compounded factors.
    decay_interval_frames: int = 1
    # Optional free-space clearing along sensor rays.
    raycast_clear_gain: float = 0.0
    raycast_step_scale: float = 1.0
    raycast_end_margin: float = 0.12
    raycast_max_rays: int = 600
    raycast_rho_max: float = 5.0
    raycast_phiw_max: float = 40.0
    raycast_dyn_boost: float = 0.25
    # LZCD: local zero-crossing debias (online local phi bias correction).
    lzcd_enable: bool = False
    lzcd_interval: int = 2
    lzcd_radius_cells: int = 1
    lzcd_min_neighbors: int = 6
    lzcd_min_phi_w: float = 0.40
    lzcd_min_rho: float = 0.05
    lzcd_max_dscore: float = 0.85
    lzcd_neighbor_phi_gate: float = 0.25
    lzcd_normal_cos_min: float = 0.45
    lzcd_bias_alpha: float = 0.18
    lzcd_correction_gain: float = 0.35
    lzcd_max_bias: float = 0.06
    lzcd_max_step: float = 0.02
    # Robust trimming ratio for local zero-crossing debias (A-stage: Acc de-bias).
    lzcd_trim_quantile: float = 0.75
    # Prefer geometry-only SDF channel in debias estimation when available.
    lzcd_use_geo_channel: bool = True
    # Geometry-only debias convergence solver (decoupled from dynamic suppressor).
    lzcd_solver_iters: int = 3
    lzcd_solver_lambda_smooth: float = 0.35
    lzcd_solver_step: float = 0.85
    lzcd_solver_tol: float = 5e-4
    # Residual anchor from association signed residual (source voxel).
    lzcd_residual_anchor_weight: float = 0.25
    lzcd_residual_alpha: float = 0.12
    lzcd_residual_hit_ref: float = 10.0
    lzcd_residual_max_abs: float = 0.10
    # LZCD affine debias:
    # fit local phi as phi ~= a * <n, delta_x> + b and use intercept b as
    # robust local zero-crossing estimate.
    lzcd_affine_enable: bool = True
    lzcd_affine_mix: float = 0.45
    lzcd_affine_slope_min: float = 0.65
    lzcd_affine_slope_max: float = 1.35
    lzcd_affine_min_samples: int = 8
    # Runtime cap for LZCD candidate voxels per frame (0 means no cap).
    lzcd_max_candidates: int = 6000
    # ZCBF: block-level zero-crossing bias field on persistent geometry.
    zcbf_enable: bool = False
    zcbf_block_size_cells: int = 6
    zcbf_min_rho: float = 0.25
    zcbf_min_phi_w: float = 0.6
    zcbf_max_dscore: float = 0.55
    zcbf_alpha: float = 0.18
    zcbf_trim_quantile: float = 0.70
    zcbf_apply_gain: float = 0.30
    zcbf_max_bias: float = 0.04
    zcbf_static_rho_ref: float = 1.0
    # LBR-3D: local bias regression for geometry-only debias (module-2).
    lbr_enable: bool = False
    lbr_alpha: float = 0.14
    lbr_max_bias: float = 0.05
    lbr_depth_ref: float = 2.5
    lbr_inc_weight: float = 0.45
    lbr_depth_weight: float = 0.35
    lbr_res_weight: float = 0.20
    lbr_apply_gain: float = 0.35
    # STCG: spatio-temporal contradiction accumulation.
    stcg_enable: bool = False
    stcg_alpha: float = 0.12
    stcg_conflict_weight: float = 0.60
    stcg_residual_weight: float = 0.25
    stcg_osc_weight: float = 0.15
    stcg_free_ratio_ref: float = 0.9
    # Contradiction shell accumulation:
    # add free-space contradiction evidence in a wider shell around uncertain
    # observations without changing phi integration support.
    stcg_shell_enable: bool = False
    stcg_shell_min_radius_vox: float = 2.0
    stcg_shell_max_radius_m: float = 0.12
    stcg_shell_residual_gain: float = 0.9
    stcg_shell_free_weight: float = 0.45
    stcg_shell_clear_boost: float = 0.10
    # Hysteresis thresholds for contradiction gate stabilization.
    stcg_on_thresh: float = 0.58
    stcg_off_thresh: float = 0.42
    # VCR: visibility-contradiction decomposition gate (module-3).
    vcr_enable: bool = False
    vcr_alpha: float = 0.16
    vcr_free_weight: float = 0.35
    vcr_occ_weight: float = 0.35
    vcr_res_weight: float = 0.20
    vcr_vis_weight: float = 0.10
    vcr_on_thresh: float = 0.55
    vcr_off_thresh: float = 0.40
    # Dual-state fusion:
    # static channel stores long-term stable geometry;
    # transient channel absorbs dynamic/inconsistent observations.
    dual_state_enable: bool = False
    dual_state_assoc_weight: float = 0.45
    dual_state_free_weight: float = 0.25
    dual_state_residual_weight: float = 0.15
    dual_state_osc_weight: float = 0.10
    dual_state_pose_weight: float = 0.05
    dual_state_bias: float = 0.45
    dual_state_temp: float = 0.25
    dual_pose_var_ref: float = 0.05
    dual_state_static_ema: float = 0.12
    dual_state_min_static_ratio: float = 0.06
    dual_state_commit_thresh: float = 0.70
    dual_state_rollback_thresh: float = 0.32
    dual_state_commit_gain: float = 0.25
    dual_state_rollback_gain: float = 0.10
    dual_state_static_protect_rho: float = 0.90
    dual_state_static_protect_ratio: float = 1.6
    dual_state_static_decay_mult: float = 1.0
    dual_state_transient_decay_mult: float = 2.2
    # RBI: local re-integration buffer (module-4).
    rbi_enable: bool = False
    rbi_decay: float = 0.92
    rbi_commit_static_p: float = 0.72
    rbi_dyn_gate: float = 0.35
    rbi_min_weight: float = 1.5
    rbi_recover_gain: float = 0.30
    rbi_max_step: float = 0.02
    enable_evidence: bool = True
    enable_gradient_fusion: bool = True


@dataclass
class Predict3DConfig:
    process_noise_trans: float = 0.01
    process_noise_rot: float = 0.01
    slam_anchor_with_first_gt: bool = True
    # Fairness default: SLAM front-end should not consume GT motion increments.
    slam_use_gt_delta_odom: bool = False
    icp_voxel_size: float = 0.04
    icp_max_corr: float = 0.12
    icp_max_iters: int = 30
    icp_min_fitness: float = 0.10
    icp_max_rmse: float = 0.10
    icp_max_trans_step: float = 0.35
    icp_max_rot_deg_step: float = 30.0


@dataclass
class Surface3DConfig:
    phi_thresh: float = 0.04
    rho_thresh: float = 0.20
    min_weight: float = 1.5
    max_d_score: float = 1.0
    max_age_frames: int = 1_000_000_000
    max_free_ratio: float = 1e9
    prune_free_min: float = 1e9
    prune_residual_min: float = 1e9
    max_clear_hits: float = 1e9
    poisson_depth: int = 8
    # Surface extraction debiasing: project voxel centers to local zero-level set
    # along fused gradient direction: x_s = c - phi * n.
    use_zero_crossing: bool = True
    zero_crossing_max_offset: float = 0.06
    zero_crossing_phi_gate: float = 0.05
    # Use geometry-only SDF channel for surface extraction (dual-channel decoupling).
    use_phi_geo_channel: bool = False
    # Local surface-consistency gate (mainly for static denoising).
    consistency_enable: bool = False
    consistency_radius: int = 1
    consistency_min_neighbors: int = 4
    consistency_normal_cos: float = 0.55
    consistency_phi_diff: float = 0.04
    # SNEF-3D local: dynamic confidence quantile clipping in local blocks.
    snef_local_enable: bool = False
    snef_block_size_cells: int = 8
    snef_dscore_quantile: float = 0.80
    snef_dscore_margin: float = 0.05
    snef_free_ratio_quantile: float = 0.85
    snef_free_ratio_margin: float = 0.10
    snef_abs_phi_quantile: float = 1.00
    snef_abs_phi_margin: float = 0.00
    snef_min_keep_per_block: int = 16
    snef_min_keep_ratio_per_block: float = 0.0
    snef_min_candidates_per_block: int = 10
    snef_anchor_rho_quantile: float = 0.90
    snef_anchor_dscore_quantile: float = 0.25
    snef_anchor_min_per_block: int = 2
    # Two-stage extraction:
    # Stage-A relaxes geometric gate to protect completeness;
    # Stage-B only denoises locally in dynamic-confidence blocks.
    two_stage_enable: bool = False
    two_stage_geom_margin: float = 0.02
    two_stage_dynamic_dscore_quantile: float = 0.70
    two_stage_dynamic_free_quantile: float = 0.70
    # Rho-aware two-stage dynamic gating:
    # only treat a local block as dynamic when dynamic cues are high
    # and (optionally) confidence rho is low.
    two_stage_dynamic_rho_quantile: float = 0.40
    two_stage_dynamic_rho_margin: float = 0.0
    two_stage_dynamic_require_low_rho: bool = True
    # Adaptive extraction (useful for highly dynamic Bonn scenes):
    # tighten unstable cells while preserving high-confidence static support.
    adaptive_enable: bool = False
    adaptive_rho_ref: float = 2.0
    adaptive_phi_min_scale: float = 0.55
    adaptive_phi_max_scale: float = 1.15
    adaptive_min_weight_gain: float = 0.8
    adaptive_free_ratio_gain: float = 0.5
    # LZCD surface-time debias application.
    lzcd_apply_in_extraction: bool = False
    lzcd_bias_scale: float = 1.0
    # PT-DSF extraction: export persistent geometry as first-class surface.
    ptdsf_persistent_only_enable: bool = False
    ptdsf_persistent_min_rho: float = 0.15
    ptdsf_static_rho_weight: float = 0.35
    # ZCBF extraction-time bias application.
    zcbf_apply_in_extraction: bool = False
    zcbf_bias_scale: float = 1.0
    # STCG soft gating in extraction.
    stcg_enable: bool = False
    # DCCM extraction soft gate: only affects dynamic suppressor, not geometry candidate formation.
    dccm_enable: bool = False
    dccm_commit_weight: float = 0.30
    dccm_static_guard: float = 0.65
    dccm_drop_gain: float = 0.22
    stcg_min_score: float = 0.35
    stcg_rho_ref: float = 1.8
    stcg_free_shrink: float = 0.45
    stcg_phi_shrink: float = 0.25
    stcg_dscore_shrink: float = 0.30
    stcg_weight_gain: float = 0.50
    stcg_static_protect: float = 0.70
    # Prefer static dual-state channel during extraction when available.
    use_dual_static_channel: bool = False
    dual_p_static_min: float = 0.0
    # Structural decoupling:
    # geometry extraction from debiased geometry channel + independent dynamic suppressor.
    structural_decouple_enable: bool = True
    decouple_min_geo_weight_ratio: float = 0.35
    decouple_dyn_drop_thresh: float = 0.78
    decouple_dyn_rho_guard: float = 1.2
    decouple_dyn_free_ratio_thresh: float = 1.10
    # Cross-channel contradiction gate:
    # when geometry channel disagrees with legacy/static channel, increase dynamic suppression.
    decouple_channel_div_enable: bool = False
    decouple_channel_div_thresh: float = 0.04
    decouple_channel_div_weight: float = 0.35
    decouple_channel_div_static_guard: float = 0.70
    # Decoupled short-memory gate (only affects suppression gate).
    decouple_stmem_enable: bool = True
    decouple_stmem_weight: float = 0.55
    decouple_stmem_static_guard: float = 0.55
    decouple_stmem_rho_guard: float = 0.45
    decouple_stmem_free_shrink: float = 0.12
    # Explicit dual-layer extraction:
    # stage-1 extracts geometry candidates (Acc-oriented) from geometry/static channels;
    # stage-2 applies dynamic suppressor mask (Ghost-oriented) from dynamic states only.
    dual_layer_extract_enable: bool = False
    dual_layer_geo_min_weight_ratio: float = 0.30
    dual_layer_dyn_use_zdyn: bool = True
    dual_layer_dyn_prob_weight: float = 0.38
    dual_layer_dyn_stmem_weight: float = 0.22
    dual_layer_dyn_contra_weight: float = 0.20
    dual_layer_dyn_transient_weight: float = 0.20
    dual_layer_dyn_phi_div_weight: float = 0.16
    dual_layer_dyn_phi_ratio_weight: float = 0.10
    dual_layer_dyn_phi_div_ref: float = 0.04
    dual_layer_dyn_use_phi_dyn: bool = True
    dual_layer_compete_enable: bool = False
    dual_layer_compete_margin: float = 0.08
    dual_layer_compete_geo_weight: float = 0.62
    dual_layer_compete_dyn_mix_weight: float = 0.55
    dual_layer_compete_dyn_conf_weight: float = 0.25
    dual_layer_dyn_drop_thresh: float = 0.72
    dual_layer_dyn_free_ratio_min: float = 0.90
    dual_layer_static_anchor_rho: float = 0.90
    dual_layer_static_anchor_p: float = 0.70
    dual_layer_static_anchor_ratio: float = 1.70
    # OMHS: occlusion-aware multi-hypothesis surface readout.
    omhs_enable: bool = False
    # EBCut: local energy-based extraction (module-5).
    ebcut_enable: bool = False
    ebcut_energy_thresh: float = 0.58
    ebcut_w_phi: float = 0.30
    ebcut_w_dyn: float = 0.35
    ebcut_w_free: float = 0.20
    ebcut_w_conf: float = 0.15
    ebcut_w_smooth: float = 0.10
    ebcut_smooth_radius: int = 1
    # MOPC: online multi-objective local controller (module-6).
    mopc_enable: bool = False
    mopc_step: float = 0.02
    mopc_dyn_target: float = 0.20
    mopc_rej_target: float = 0.08
    mopc_drop_min: float = 0.60
    mopc_drop_max: float = 0.90
    mopc_maxd_min: float = 0.70
    mopc_maxd_max: float = 1.00


@dataclass
class EGF3DConfig:
    camera: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    map3d: Map3DConfig = field(default_factory=Map3DConfig)
    assoc: Assoc3DConfig = field(default_factory=Assoc3DConfig)
    update: Update3DConfig = field(default_factory=Update3DConfig)
    predict: Predict3DConfig = field(default_factory=Predict3DConfig)
    surface: Surface3DConfig = field(default_factory=Surface3DConfig)
