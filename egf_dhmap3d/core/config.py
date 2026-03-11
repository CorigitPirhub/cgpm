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
    # PFAG: PFV-guided association gating. Disabled by default after
    # focused probes showed that hard pre-association rejection hurts late-frame
    # dynamic suppression despite improving geometry.
    pfv_gate_enable: bool = False
    pfv_seed_block_on: float = 0.42
    pfv_gate_on: float = 0.30
    pfv_gate_off: float = 0.18
    pfv_d2_boost_max: float = 2.8
    pfv_static_guard: float = 0.84
    pfv_bg_rho_guard: float = 0.82
    pfv_view_align_weight: float = 0.25


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
    rps_bank_sep_gate: float = 0.22
    rps_bank_bg_support_gain: float = 0.0
    rps_bank_front_dyn_penalty_gain: float = 0.0
    rps_bank_rear_score_bias: float = 0.0
    rps_bank_soft_competition_enable: bool = False
    rps_bank_soft_competition_gap: float = 0.0
    rps_bank_soft_sep_relax: float = 0.0
    rps_bank_soft_rear_min_relax: float = 0.0
    rps_bank_soft_support_min: float = 0.45
    rps_bank_soft_local_support_gain: float = 1.0
    rps_soft_bank_export_enable: bool = False
    rps_soft_bank_min_score: float = 0.18
    rps_soft_bank_gain: float = 0.65
    rps_soft_bank_commit_relax: float = 0.70
    rps_candidate_rescue_enable: bool = False
    rps_candidate_support_gain: float = 0.28
    rps_candidate_bg_gain: float = 0.22
    rps_candidate_rho_gain: float = 0.18
    rps_candidate_front_relax: float = 0.20
    rps_commit_activation_enable: bool = False
    rps_commit_threshold: float = 0.62
    rps_commit_release: float = 0.40
    rps_commit_age_threshold: float = 2.0
    rps_commit_rho_ref: float = 0.08
    rps_commit_weight_ref: float = 0.80
    rps_commit_min_cand_rho: float = 0.02
    rps_commit_min_cand_w: float = 0.08
    rps_commit_evidence_weight: float = 0.34
    rps_commit_geometry_weight: float = 0.28
    rps_commit_bg_weight: float = 0.20
    rps_commit_static_weight: float = 0.18
    rps_commit_front_penalty: float = 0.22
    rps_admission_support_enable: bool = False
    rps_admission_support_on: float = 0.42
    rps_admission_support_gain: float = 0.55
    rps_admission_score_relax: float = 0.10
    rps_admission_active_floor: float = 0.32
    rps_admission_rho_ref: float = 0.08
    rps_admission_weight_ref: float = 0.35
    rps_admission_geometry_enable: bool = False
    rps_admission_geometry_weight: float = 0.25
    rps_admission_geometry_floor: float = 0.40
    rps_admission_occlusion_enable: bool = False
    rps_admission_occlusion_weight: float = 0.12
    rps_space_redirect_history_enable: bool = False
    rps_space_redirect_history_weight: float = 0.32
    rps_space_redirect_history_bg_weight: float = 0.60
    rps_space_redirect_history_static_weight: float = 0.40
    rps_space_redirect_history_floor: float = 0.30
    rps_space_redirect_ghost_suppress_enable: bool = False
    rps_space_redirect_ghost_suppress_weight: float = 0.22
    rps_space_redirect_visual_anchor_enable: bool = False
    rps_space_redirect_visual_anchor_weight: float = 0.16
    rps_space_redirect_visual_anchor_min: float = 0.36
    rps_history_obstructed_gate_enable: bool = False
    rps_history_visible_min: float = 0.45
    rps_obstruction_min: float = 0.28
    rps_non_hole_min: float = 0.30
    rps_history_manifold_enable: bool = False
    rps_history_manifold_visible_min: float = 0.45
    rps_history_manifold_obstruction_min: float = 0.28
    rps_history_manifold_bg_weight: float = 0.50
    rps_history_manifold_geo_weight: float = 0.30
    rps_history_manifold_static_weight: float = 0.20
    rps_history_manifold_blend: float = 0.75
    rps_history_manifold_max_offset: float = 0.04
    rps_bg_manifold_state_enable: bool = False
    rps_bg_manifold_alpha_up: float = 0.08
    rps_bg_manifold_alpha_down: float = 0.02
    rps_bg_manifold_rho_alpha: float = 0.10
    rps_bg_manifold_weight_gain: float = 0.55
    rps_bg_manifold_rho_ref: float = 0.08
    rps_bg_manifold_weight_ref: float = 0.35
    rps_bg_manifold_history_weight: float = 0.30
    rps_bg_manifold_obstruction_weight: float = 0.20
    rps_bg_manifold_visible_lo: float = 0.25
    rps_bg_manifold_visible_hi: float = 0.50
    rps_bg_manifold_decay: float = 0.992
    rps_bg_manifold_mem_decay: float = 0.996
    rps_bg_dense_state_enable: bool = False
    rps_bg_dense_neighbor_radius: int = 1
    rps_bg_dense_neighbor_weight: float = 0.55
    rps_bg_dense_geometry_weight: float = 0.30
    rps_bg_dense_max_weight: float = 1.0
    rps_bg_dense_support_floor: float = 0.18
    rps_bg_dense_decay: float = 0.996
    rps_bg_surface_constrained_enable: bool = False
    rps_bg_surface_min_conf: float = 0.12
    rps_bg_surface_agree_weight: float = 0.40
    rps_bg_surface_tangent_enable: bool = False
    rps_bg_surface_tangent_weight: float = 0.65
    rps_bg_surface_tangent_floor: float = 0.15
    rps_bg_bridge_enable: bool = False
    rps_bg_bridge_min_visible: float = 0.35
    rps_bg_bridge_min_obstruction: float = 0.30
    rps_bg_bridge_min_step: int = 1
    rps_bg_bridge_max_step: int = 3
    rps_bg_bridge_gain: float = 0.65
    rps_bg_bridge_phi_blend: float = 0.85
    rps_bg_bridge_target_dyn_max: float = 0.35
    rps_bg_bridge_target_surface_max: float = 0.35
    rps_bg_bridge_ghost_suppress_enable: bool = False
    rps_bg_bridge_ghost_suppress_weight: float = 0.22
    rps_bg_bridge_relaxed_dyn_max: float = 0.45
    rps_bg_bridge_keep_multi_hits: bool = False
    rps_bg_bridge_max_hits_per_source: int = 3
    rps_bg_bridge_cone_enable: bool = False
    rps_bg_bridge_cone_radius_cells: int = 1
    rps_bg_bridge_cone_gain_scale: float = 0.65
    rps_bg_bridge_patch_radius_cells: int = 0
    rps_bg_bridge_patch_gain_scale: float = 0.55
    rps_bg_bridge_depth_hypothesis_count: int = 0
    rps_bg_bridge_depth_step_scale: float = 0.50
    rps_bg_bridge_rear_synth_enable: bool = False
    rps_bg_bridge_rear_support_gain: float = 0.28
    rps_bg_bridge_rear_rho_gain: float = 0.10
    rps_bg_bridge_rear_phi_blend: float = 0.80
    rps_bg_bridge_rear_score_floor: float = 0.22
    rps_bg_bridge_rear_active_floor: float = 0.52
    rps_bg_bridge_rear_age_floor: float = 1.0
    rps_rear_hybrid_filter_enable: bool = False
    rps_rear_hybrid_bridge_support_min: float = 0.20
    rps_rear_hybrid_dyn_max: float = 0.22
    rps_rear_hybrid_manifold_min: float = 0.25
    rps_rear_density_gate_enable: bool = False
    rps_rear_density_radius_cells: int = 1
    rps_rear_density_min_neighbors: int = 2
    rps_rear_density_support_min: float = 0.45
    rps_rear_selectivity_enable: bool = False
    rps_rear_selectivity_support_weight: float = 0.18
    rps_rear_selectivity_history_weight: float = 0.24
    rps_rear_selectivity_static_weight: float = 0.16
    rps_rear_selectivity_geom_weight: float = 0.22
    rps_rear_selectivity_bridge_weight: float = 0.10
    rps_rear_selectivity_density_weight: float = 0.10
    rps_rear_selectivity_rear_score_weight: float = 0.28
    rps_rear_selectivity_front_score_weight: float = 0.28
    rps_rear_selectivity_competition_weight: float = 0.34
    rps_rear_selectivity_competition_alpha: float = 0.80
    rps_rear_selectivity_gap_weight: float = 0.18
    rps_rear_selectivity_sep_weight: float = 0.08
    rps_rear_selectivity_dyn_weight: float = 0.22
    rps_rear_selectivity_ghost_weight: float = 0.18
    rps_rear_selectivity_front_weight: float = 0.16
    rps_rear_selectivity_geom_risk_weight: float = 0.22
    rps_rear_selectivity_history_risk_weight: float = 0.16
    rps_rear_selectivity_density_risk_weight: float = 0.10
    rps_rear_selectivity_bridge_relief_weight: float = 0.10
    rps_rear_selectivity_static_relief_weight: float = 0.08
    rps_rear_selectivity_gap_risk_weight: float = 0.18
    rps_rear_selectivity_score_min: float = 0.46
    rps_rear_selectivity_risk_max: float = 0.45
    rps_rear_selectivity_geom_floor: float = 0.48
    rps_rear_selectivity_history_floor: float = 0.36
    rps_rear_selectivity_bridge_floor: float = 0.12
    rps_rear_selectivity_competition_floor: float = -0.02
    rps_rear_selectivity_front_score_max: float = 0.92
    rps_rear_selectivity_gap_min: float = 0.018
    rps_rear_selectivity_gap_max: float = 0.090
    rps_rear_selectivity_gap_valid_min: float = 0.28
    rps_rear_selectivity_occlusion_order_weight: float = 0.0
    rps_rear_selectivity_occlusion_order_floor: float = 0.0
    rps_rear_selectivity_occlusion_order_risk_weight: float = 0.0
    rps_rear_selectivity_local_conflict_weight: float = 0.0
    rps_rear_selectivity_local_conflict_max: float = 1.5
    rps_rear_selectivity_front_residual_weight: float = 0.0
    rps_rear_selectivity_front_residual_max: float = 1.5
    rps_rear_selectivity_occluder_protect_weight: float = 0.0
    rps_rear_selectivity_occluder_protect_floor: float = 0.0
    rps_rear_selectivity_occluder_relief_weight: float = 0.0
    rps_rear_selectivity_dynamic_trail_weight: float = 0.0
    rps_rear_selectivity_dynamic_trail_max: float = 1.5
    rps_rear_selectivity_dynamic_trail_relief_weight: float = 0.0
    rps_rear_selectivity_history_anchor_weight: float = 0.0
    rps_rear_selectivity_history_anchor_floor: float = 0.0
    rps_rear_selectivity_history_anchor_relief_weight: float = 0.0
    rps_rear_selectivity_surface_anchor_weight: float = 0.0
    rps_rear_selectivity_surface_anchor_floor: float = 0.0
    rps_rear_selectivity_surface_anchor_risk_weight: float = 0.0
    rps_rear_selectivity_surface_distance_ref: float = 0.05
    rps_rear_selectivity_dynamic_shell_weight: float = 0.0
    rps_rear_selectivity_dynamic_shell_max: float = 1.5
    rps_rear_selectivity_dynamic_shell_gap_ref: float = 0.05
    rps_rear_selectivity_conflict_radius_cells: int = 1
    rps_rear_selectivity_conflict_front_score_min: float = 0.20
    rps_rear_selectivity_conflict_static_score_min: float = 0.35
    rps_rear_selectivity_conflict_dist_scale: float = 1.2
    rps_rear_selectivity_conflict_gap_ref: float = 0.06
    rps_rear_selectivity_conflict_ref: float = 1.8
    rps_rear_selectivity_trail_radius_cells: int = 1
    rps_rear_selectivity_trail_ref: float = 2.0
    rps_rear_selectivity_density_radius_cells: int = 1
    rps_rear_selectivity_density_ref: int = 8
    rps_rear_selectivity_topk: int = 0
    rps_rear_selectivity_rank_risk_weight: float = 0.55
    rps_rear_selectivity_penetration_weight: float = 0.0
    rps_rear_selectivity_penetration_floor: float = 0.0
    rps_rear_selectivity_penetration_risk_weight: float = 0.0
    rps_rear_selectivity_penetration_free_ref: float = 0.05
    rps_rear_selectivity_penetration_max_steps: int = 10
    rps_rear_selectivity_observation_weight: float = 0.0
    rps_rear_selectivity_observation_floor: float = 0.0
    rps_rear_selectivity_observation_risk_weight: float = 0.0
    rps_rear_selectivity_observation_count_ref: float = 6.0
    rps_rear_selectivity_observation_min_count: float = 0.0
    rps_rear_selectivity_unobserved_veto_enable: bool = False
    rps_rear_selectivity_static_coherence_weight: float = 0.0
    rps_rear_selectivity_static_coherence_floor: float = 0.0
    rps_rear_selectivity_static_coherence_relief_weight: float = 0.0
    rps_rear_selectivity_static_coherence_radius_cells: int = 1
    rps_rear_selectivity_static_coherence_ref: float = 0.35
    rps_rear_selectivity_static_neighbor_min_weight: float = 0.20
    rps_rear_selectivity_static_neighbor_dyn_max: float = 0.35
    rps_rear_selectivity_thickness_weight: float = 0.0
    rps_rear_selectivity_thickness_floor: float = 0.0
    rps_rear_selectivity_thickness_risk_weight: float = 0.0
    rps_rear_selectivity_thickness_ref: float = 0.08
    rps_rear_selectivity_normal_consistency_weight: float = 0.0
    rps_rear_selectivity_normal_consistency_floor: float = 0.0
    rps_rear_selectivity_normal_consistency_relief_weight: float = 0.0
    rps_rear_selectivity_normal_consistency_radius_cells: int = 1
    rps_rear_selectivity_normal_consistency_dyn_max: float = 0.35
    rps_rear_selectivity_ray_convergence_weight: float = 0.0
    rps_rear_selectivity_ray_convergence_floor: float = 0.0
    rps_rear_selectivity_ray_convergence_relief_weight: float = 0.0
    rps_rear_selectivity_ray_convergence_radius_cells: int = 1
    rps_rear_selectivity_ray_convergence_gap_ref: float = 0.06
    rps_rear_selectivity_ray_convergence_normal_cos: float = 0.75
    rps_rear_selectivity_ray_convergence_thickness_ref: float = 0.08
    rps_rear_selectivity_ray_convergence_ref: float = 2.0
    rps_rear_state_protect_enable: bool = False
    rps_rear_state_decay_relax: float = 0.45
    rps_rear_state_active_floor: float = 0.28
    rps_commit_quality_enable: bool = False
    rps_commit_quality_transfer_gain: float = 0.18
    rps_commit_quality_rho_gain: float = 0.55
    rps_commit_quality_geom_blend: float = 0.35
    rps_commit_quality_sep_scale: float = 0.65
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
    wdsg_synth_mode: str = "legacy"  # legacy | anchor | counterfactual | energy
    wdsg_synth_anchor_gain: float = 0.55
    wdsg_synth_geo_gain: float = 0.35
    wdsg_synth_bg_gain: float = 0.20
    wdsg_synth_counterfactual_gain: float = 0.45
    wdsg_synth_front_repel_gain: float = 0.35
    wdsg_synth_energy_temp: float = 0.18
    wdsg_synth_clip_vox: float = 2.40
    wdsg_conservative_enable: bool = False
    wdsg_conservative_ref_vox: float = 0.60
    wdsg_conservative_min_clip_scale: float = 0.20
    wdsg_conservative_static_gain: float = 0.45
    wdsg_conservative_rear_gain: float = 0.25
    wdsg_conservative_geo_gain: float = 0.20
    wdsg_conservative_front_penalty: float = 0.35
    wdsg_local_clip_enable: bool = False
    wdsg_local_clip_min_scale: float = 0.70
    wdsg_local_clip_max_scale: float = 1.18
    wdsg_local_clip_risk_gain: float = 0.52
    wdsg_local_clip_expand_gain: float = 0.22
    wdsg_local_clip_front_gate: float = 0.48
    wdsg_local_clip_support_gate: float = 0.52
    wdsg_local_clip_ambiguity_gate: float = 0.12
    wdsg_local_clip_pfv_gain: float = 0.20
    wdsg_local_clip_bg_gain: float = 0.18
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
    # XMem: Exclusion Memory. Unlike OTV, it keeps a reversible dynamic-front
    # exclusion state using both front occupancy support and later free-space
    # clearing evidence, so suppression is decided at state-time rather than
    # only at export-time.
    xmem_enable: bool = False
    xmem_sep_ref_vox: float = 0.90
    xmem_occ_alpha: float = 0.18
    xmem_free_alpha: float = 0.14
    xmem_score_alpha: float = 0.20
    xmem_support_ref: float = 0.24
    xmem_commit_on: float = 0.60
    xmem_commit_off: float = 0.42
    xmem_age_ref: float = 1.0
    xmem_static_guard: float = 0.78
    xmem_free_gain: float = 0.85
    xmem_static_veto: float = 0.96
    xmem_geo_veto: float = 0.92
    xmem_transient_boost: float = 0.90
    xmem_dyn_boost: float = 0.82
    xmem_decay: float = 0.97
    xmem_clear_alpha: float = 0.18
    xmem_clear_on: float = 0.42
    xmem_clear_off: float = 0.28
    xmem_clear_static_release: float = 0.76
    xmem_clear_weight_decay: float = 0.92
    xmem_raycast_gain: float = 0.24
    xmem_raycast_gate: float = 0.22
    xmem_raycast_static_decay: float = 0.32
    # OBL-3D: write a dedicated background layer under front/rear separation.
    obl_enable: bool = False
    obl_sep_ref_vox: float = 0.90
    obl_rho_alpha: float = 0.20
    obl_score_alpha: float = 0.18
    obl_rear_gain: float = 0.92
    obl_static_gain: float = 0.38
    obl_commit_on: float = 0.58
    obl_commit_off: float = 0.40
    obl_age_ref: float = 1.0
    obl_static_veto: float = 0.78
    obl_geo_veto: float = 0.62
    obl_extract_gain: float = 1.10
    obl_dyn_static_guard: float = 0.60
    # DMBG-3D: true dual-map background/foreground split.
    dual_map_enable: bool = False
    dual_map_bg_front_veto: float = 0.90
    dual_map_bg_rear_gain: float = 1.00
    dual_map_bg_static_floor: float = 0.08
    dual_map_fg_front_boost: float = 1.10
    dual_map_fg_static_leak: float = 0.04
    dual_map_fg_dynamic_score_bias: float = 0.20
    # PFVP: PFV-guided proposal routing. Observations are first associated, then
    # routed to background / foreground / dual write based on persistent free-space
    # prior and current foreground history instead of being dropped upfront.
    pfvp_enable: bool = False
    pfvp_margin: float = 0.08
    pfvp_fg_on: float = 0.38
    pfvp_bg_on: float = 0.28
    pfvp_bg_keep_floor: float = 0.08
    pfvp_pfv_weight: float = 0.55
    pfvp_fg_hist_weight: float = 0.25
    pfvp_assoc_weight: float = 0.20
    pfvp_static_weight: float = 0.55
    pfvp_bg_rho_weight: float = 0.25
    pfvp_bg_obl_weight: float = 0.20
    # CMCT: Cross-Map Contradiction Transfer. Stable foreground contradiction
    # is projected back to the background map as explicit negative evidence.
    cmct_enable: bool = False
    cmct_alpha: float = 0.18
    cmct_commit_on: float = 0.46
    cmct_commit_off: float = 0.30
    cmct_bg_decay: float = 0.68
    cmct_geo_decay: float = 0.52
    cmct_rho_decay: float = 0.40
    cmct_static_guard: float = 0.78
    cmct_bg_rho_protect: float = 0.85
    cmct_radius_cells: int = 1
    # CGCC: Cross-Map Geometric Carving Corridor. Expand a stable foreground
    # surface into a short ray-aligned corridor and carve weak background support
    # in that corridor. This moves contradiction transfer from voxel-local to
    # geometric/line-of-sight domain.
    cgcc_enable: bool = False
    cgcc_conf_on: float = 0.42
    cgcc_conf_off: float = 0.28
    cgcc_front_margin_vox: float = 0.35
    cgcc_rear_margin_vox: float = 1.40
    cgcc_step_scale: float = 0.75
    cgcc_lateral_radius_cells: int = 1
    cgcc_bg_decay: float = 0.72
    cgcc_geo_decay: float = 0.58
    cgcc_rho_decay: float = 0.46
    cgcc_bg_layer_decay: float = 0.22
    cgcc_static_guard: float = 0.84
    cgcc_fg_weight_floor: float = 0.18
    # PFV: Persistent Free-space Volume. Stable background hits accumulate a
    # persistent free-space bank along the line of sight; background surfaces
    # are only exported when they are not contradicted by the accumulated free volume.
    pfv_enable: bool = False
    pfv_alpha: float = 0.18
    pfv_commit_on: float = 0.44
    pfv_commit_off: float = 0.28
    pfv_step_scale: float = 0.75
    pfv_end_margin: float = 0.18
    pfv_bg_support_ref: float = 0.38
    pfv_fg_guard: float = 0.35
    pfv_static_guard: float = 0.82
    pfv_extract_thresh: float = 0.36
    pfv_bg_decay: float = 0.55
    pfv_geo_decay: float = 0.42
    pfv_rho_decay: float = 0.28
    pfv_long_alpha: float = 0.10
    pfv_long_on: float = 0.34
    pfv_release_rho: float = 0.92
    pfv_depth_gain: float = 0.30
    pfv_cluster_radius: int = 1
    pfv_cluster_weight: float = 0.35
    pfv_cluster_thresh: float = 0.42
    pfv_bg_decay_long: float = 0.68
    pfv_geo_decay_long: float = 0.52
    # Banked PFV: decompose persistent free-space into near / mid / far banks.
    # This lets export distinguish a truly cleared foreground corridor from generic
    # weak free-space support and is more structural than sharpening one scalar.
    pfv_bank_alpha: float = 0.16
    pfv_bank_near_split: float = 0.35
    pfv_bank_far_split: float = 0.72
    pfv_bank_weight_near: float = 0.95
    pfv_bank_weight_mid: float = 0.80
    pfv_bank_weight_far: float = 0.55
    pfv_bank_cluster_weight: float = 0.45
    pfv_bank_extract_thresh: float = 0.48
    # PFV exclusivity: separate export-only cleared-corridor state.
    pfv_exclusive_enable: bool = False
    pfv_exclusive_alpha: float = 0.18
    pfv_exclusive_on: float = 0.32
    pfv_exclusive_off: float = 0.18
    pfv_exclusive_fg_weight: float = 0.65
    pfv_exclusive_long_weight: float = 0.75
    pfv_exclusive_static_guard: float = 0.72
    pfv_exclusive_extract_thresh: float = 0.30
    pfv_exclusive_anchor_guard: float = 0.92
    pfv_exclusive_rho_guard: float = 1.20
    pfv_exclusive_free_ratio_min: float = 0.45
    # PFV-conditioned background commit delay: suppress background-map writes
    # when persistent free-space repeatedly covers a voxel without stable
    # background support. This acts at update-time instead of export-time.
    pfv_commit_delay_enable: bool = False
    pfv_commit_delay_on: float = 0.34
    pfv_commit_delay_static_weight: float = 0.85
    pfv_commit_delay_bg_weight: float = 0.95
    pfv_commit_delay_geo_weight: float = 0.70
    pfv_commit_delay_rho_weight: float = 0.55
    pfv_commit_delay_support_guard: float = 0.78
    pfv_commit_delay_rho_guard: float = 0.95
    pfv_commit_delay_free_ratio_ref: float = 0.65
    pfv_commit_delay_min_scale: float = 0.02
    # Delayed background candidate state: when PFV strongly contradicts the
    # current background write, route it to a separate candidate buffer instead
    # of directly committing into `phi_bg / phi_static`.
    pfv_bg_candidate_enable: bool = False
    pfv_bg_candidate_on: float = 0.30
    joint_bg_state_enable: bool = False
    joint_bg_state_on: float = 0.20
    joint_bg_state_gain: float = 0.55
    joint_bg_state_rho_gain: float = 0.20
    joint_bg_state_front_penalty: float = 0.22
    pfv_bg_candidate_off: float = 0.18
    pfv_bg_candidate_alpha: float = 0.18
    pfv_bg_candidate_gain: float = 1.00
    pfv_bg_candidate_leak: float = 0.04
    pfv_bg_candidate_promote_on: float = 0.72
    pfv_bg_candidate_promote_rho: float = 0.85
    pfv_bg_candidate_promote_blend: float = 0.45
    pfv_bg_candidate_decay: float = 0.94
    # Tri-map background architecture: split the old background role into
    # committed / delayed / foreground maps and let PFV decide write target.
    tri_map_enable: bool = False
    tri_map_delay_on: float = 0.32
    tri_map_delay_margin: float = 0.04
    tri_map_pfv_weight: float = 0.45
    tri_map_fg_weight: float = 0.35
    tri_map_assoc_weight: float = 0.20
    tri_map_bg_support_weight: float = 0.55
    tri_map_bg_rho_weight: float = 0.45
    tri_map_promote_on: float = 0.72
    tri_map_promote_fg_guard: float = 0.28
    tri_map_promote_blend: float = 0.40
    # Conflict-band routing: keep delayed routing in a middle band instead of
    # either sending too much (v2) or almost nothing (criterion redesign).
    tri_map_conflict_strong_margin: float = 0.08
    tri_map_conflict_soft_margin: float = -0.02
    tri_map_support_max_strong: float = 0.72
    tri_map_support_max_soft: float = 0.90
    # Front-occupancy anchored tri-map routing: route to delayed map only when
    # a persistent foreground surface is established and background support is weak.
    tri_map_front_occ_on: float = 0.58
    tri_map_front_occ_soft_on: float = 0.34
    tri_map_front_occ_weight: float = 0.60
    tri_map_front_occ_occ_weight: float = 0.20
    tri_map_front_occ_pfv_weight: float = 0.20
    tri_map_bg_support_guard_strong: float = 0.65
    tri_map_bg_support_guard_soft: float = 0.82
    tri_map_bg_rho_guard_strong: float = 0.70
    tri_map_bg_rho_guard_soft: float = 0.90
    # Hybrid conflict-score routing.
    tri_map_hybrid_pfv_weight: float = 0.40
    tri_map_hybrid_front_weight: float = 0.40
    tri_map_hybrid_assoc_weight: float = 0.10
    tri_map_hybrid_bg_deficit_weight: float = 0.10
    tri_map_hybrid_strong_on: float = 0.14
    tri_map_hybrid_soft_on: float = 0.08
    tri_map_hybrid_front_floor: float = 0.05
    tri_map_hybrid_pfv_floor: float = 0.05

    # Support-gap calibrated tri-map routing: route delayed writes using
    # front-vs-background support gap, then use PFV/assoc/background deficit
    # only as small calibration terms instead of a primary linear score.
    tri_map_support_gap_bg_support_weight: float = 0.60
    tri_map_support_gap_bg_rho_weight: float = 0.40
    tri_map_support_gap_pfv_gain: float = 0.16
    tri_map_support_gap_assoc_gain: float = 0.06
    tri_map_support_gap_bg_deficit_gain: float = 0.08
    tri_map_support_gap_front_bonus_gain: float = 0.06
    tri_map_support_gap_strong_on: float = 0.11
    tri_map_support_gap_soft_on: float = 0.04
    tri_map_support_gap_front_floor: float = 0.06
    tri_map_support_gap_front_soft_floor: float = 0.03
    tri_map_support_gap_pfv_floor: float = 0.04
    tri_map_support_gap_assoc_floor: float = 0.08
    # Zero-centered normalized support-gap routing.
    tri_map_support_gap_center_bg_ratio: float = 0.38
    tri_map_support_gap_norm_floor: float = 0.28
    tri_map_support_gap_zero_bias: float = 0.02
    tri_map_support_gap_centered_floor_strong: float = 0.00
    tri_map_support_gap_centered_floor_soft: float = -0.03

    # Quantile-calibrated support-gap routing.
    tri_map_support_gap_quantile_strong_q: float = 0.82
    tri_map_support_gap_quantile_soft_q: float = 0.985
    tri_map_support_gap_quantile_floor_strong: float = 0.028
    tri_map_support_gap_quantile_floor_soft: float = 0.02
    tri_map_support_gap_quantile_sep_margin: float = 0.008
    tri_map_support_gap_strong_budget_ratio: float = 0.25
    tri_map_support_gap_soft_budget_ratio: float = 0.015
    tri_map_support_gap_strong_min_budget: int = 1
    tri_map_support_gap_soft_min_budget: int = 8
    tri_map_support_gap_strong_max_budget: int = 6
    tri_map_support_gap_soft_max_budget: int = 96
    # Escalation-aware delayed residency hold / promotion hysteresis.
    tri_map_escalation_hold_frames: float = 3.0
    tri_map_escalation_hold_max_frames: float = 6.0
    tri_map_escalation_hysteresis_gain: float = 1.0
    tri_map_escalation_hysteresis_decay: float = 0.85
    tri_map_escalation_promote_bonus: float = 0.12
    tri_map_escalation_blend_suppress: float = 0.35
    # Residency-gated delayed export participation.
    tri_map_residency_export_enable: bool = True
    tri_map_residency_export_hold_min: float = 0.5
    tri_map_residency_export_hysteresis_min: float = 0.05
    tri_map_residency_export_support_on: float = 0.45
    tri_map_residency_export_fg_guard: float = 0.65
    tri_map_residency_export_commit_max: float = 0.92
    tri_map_residency_export_route_score_min: float = 0.02
    tri_map_residency_export_radius_cells: int = 1
    # Export-time local replacement around delayed tail.
    tri_map_residency_replace_enable: bool = True
    tri_map_residency_replace_radius_cells: int = 2
    tri_map_residency_replace_distance_vox: float = 2.2
    tri_map_residency_replace_commit_max: float = 1.0
    tri_map_residency_replace_support_margin: float = -1.0
    tri_map_residency_replace_max_per_delayed: int = 2
    # Competition-scored local replacement.
    tri_map_residency_compete_route_ref: float = 0.08
    tri_map_residency_compete_normal_cos_min: float = 0.25
    tri_map_residency_compete_margin: float = 0.03
    tri_map_residency_compete_delayed_support_weight: float = 0.55
    tri_map_residency_compete_route_weight: float = 0.25
    tri_map_residency_compete_residency_weight: float = 0.10
    tri_map_residency_compete_normal_weight: float = 0.15
    tri_map_residency_compete_commit_support_weight: float = 0.60
    tri_map_residency_compete_distance_weight: float = 0.20
    # Delayed-branch geometry refinement before export competition.
    tri_map_delayed_refine_enable: bool = True
    tri_map_delayed_refine_radius_cells: int = 1
    tri_map_delayed_refine_blend: float = 0.65
    tri_map_delayed_refine_normal_blend: float = 0.75
    tri_map_delayed_refine_max_offset_vox: float = 1.25
    tri_map_delayed_refine_support_weight: float = 0.50
    tri_map_delayed_refine_route_weight: float = 0.25
    tri_map_delayed_refine_residency_weight: float = 0.25
    tri_map_delayed_refine_phi_static_weight: float = 0.45
    tri_map_delayed_refine_phi_bg_weight: float = 0.35
    tri_map_delayed_refine_phi_geo_weight: float = 0.20
    tri_map_delayed_refine_neighbor_min: int = 2
    # Delayed-branch dedicated surface readout / banked field refinement.
    tri_map_delayed_bank_enable: bool = True
    tri_map_delayed_bank_radius_cells: int = 1
    tri_map_delayed_bank_support_on: float = 0.40
    tri_map_delayed_bank_route_on: float = 0.01
    tri_map_delayed_bank_residency_on: float = 0.10
    tri_map_delayed_bank_phi_thresh_scale: float = 1.15
    tri_map_delayed_bank_min_weight_scale: float = 0.60
    tri_map_delayed_bank_phi_static_weight: float = 0.50
    tri_map_delayed_bank_phi_bg_weight: float = 0.35
    tri_map_delayed_bank_phi_geo_weight: float = 0.15
    tri_map_delayed_bank_support_weight: float = 0.45
    tri_map_delayed_bank_route_weight: float = 0.25
    tri_map_delayed_bank_residency_weight: float = 0.30
    tri_map_delayed_bank_normal_blend: float = 0.80
    tri_map_delayed_bank_max_offset_vox: float = 1.50
    tri_map_delayed_bank_neighbor_min: int = 2
    # Persistent delayed surface bank accumulation.
    tri_map_delay_bank_accum_enable: bool = True
    tri_map_delay_bank_conf_on: float = 0.35
    tri_map_delay_bank_conf_alpha: float = 0.18
    tri_map_delay_bank_decay: float = 0.96
    tri_map_delay_bank_rho_alpha: float = 0.12
    tri_map_delay_bank_phi_static_weight: float = 0.50
    tri_map_delay_bank_phi_bg_weight: float = 0.35
    tri_map_delay_bank_phi_geo_weight: float = 0.15
    tri_map_delay_bank_min_weight: float = 0.20
    tri_map_delay_bank_gain: float = 1.35
    tri_map_delay_bank_rear_weight: float = 0.65
    tri_map_delay_bank_bg_weight: float = 0.25
    tri_map_delay_bank_static_weight: float = 0.10
    tri_map_delay_bank_normal_history_weight: float = 0.55
    tri_map_delay_bank_normal_obs_weight: float = 0.45
    tri_map_delay_bank_route_ref: float = 0.08
    # Promotion-aware rescue: lower promotion threshold and increase blend
    # when committed map has a local hole but delayed map has stable support.
    tri_map_promotion_rescue_enable: bool = False
    tri_map_promotion_hole_weight: float = 0.35
    tri_map_promotion_min_on: float = 0.55
    # Hole-only rescue at export: delayed map only fills committed-map holes.
    tri_map_hole_rescue_enable: bool = False
    tri_map_hole_support_on: float = 0.72
    tri_map_hole_commit_max: float = 0.25
    tri_map_hole_fg_guard: float = 0.28
    tri_map_hole_radius_cells: int = 1
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
    depth_bias_offset_m: float = 0.0
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
    point_bias_along_normal_m: float = 0.0
    geometry_chain_coupling_enable: bool = False
    geometry_chain_coupling_mode: str = "direct"
    geometry_chain_coupling_donor_root: str = ""
    geometry_chain_coupling_project_dist: float = 0.05
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
    # CSR-XMap: Counterfactual Static Readout + explicit dynamic exclusion map.
    # CSR builds a static-only readout from persistent channels; XMap builds a
    # front transient surface from dynamic channels and uses geometric
    # separation, rather than scalar gating, to exclude dynamic fronts.
    csr_enable: bool = False
    csr_min_score: float = 0.38
    csr_geo_blend: float = 0.18
    csr_geo_agree_min: float = 0.70
    xmap_enable: bool = False
    xmap_dyn_min_score: float = 0.52
    xmap_static_min_score: float = 0.42
    xmap_sep_ref_vox: float = 0.90
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
