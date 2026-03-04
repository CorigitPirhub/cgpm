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
    integration_radius_scale: float = 1.0
    # Optional free-space clearing along sensor rays.
    raycast_clear_gain: float = 0.0
    raycast_step_scale: float = 1.0
    raycast_end_margin: float = 0.12
    raycast_max_rays: int = 600
    raycast_rho_max: float = 5.0
    raycast_phiw_max: float = 40.0
    raycast_dyn_boost: float = 0.25
    enable_evidence: bool = True
    enable_gradient_fusion: bool = True


@dataclass
class Predict3DConfig:
    process_noise_trans: float = 0.01
    process_noise_rot: float = 0.01
    slam_anchor_with_first_gt: bool = True
    slam_use_gt_delta_odom: bool = True
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


@dataclass
class EGF3DConfig:
    camera: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    map3d: Map3DConfig = field(default_factory=Map3DConfig)
    assoc: Assoc3DConfig = field(default_factory=Assoc3DConfig)
    update: Update3DConfig = field(default_factory=Update3DConfig)
    predict: Predict3DConfig = field(default_factory=Predict3DConfig)
    surface: Surface3DConfig = field(default_factory=Surface3DConfig)
