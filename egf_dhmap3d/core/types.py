from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VoxelCell3D:
    phi: float = 0.0
    phi_w: float = 0.0
    # Dual-state SDF channels.
    phi_static: float = 0.0
    phi_static_w: float = 0.0
    phi_transient: float = 0.0
    phi_transient_w: float = 0.0
    p_static: float = 0.5
    # Geometry-only SDF channel (decoupled from dynamic suppression channel).
    phi_geo: float = 0.0
    phi_geo_w: float = 0.0
    # Dynamic-only SDF channel (stores transient/dynamic geometry evidence).
    phi_dyn: float = 0.0
    phi_dyn_w: float = 0.0
    # PT-DSF: evidence split for persistent/transient surface states.
    rho_static: float = 0.0
    rho_transient: float = 0.0
    ptdsf_commit_age: float = 0.0
    ptdsf_rollback_age: float = 0.0
    rho: float = 0.0
    g_mean: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    g_cov: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    c_rho: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    # Local dynamic memory for selective forgetting.
    d_score: float = 0.0
    # Structural decoupling: dynamic suppression state (independent from geometry debias).
    dyn_prob: float = 0.0
    # Explicit dynamic latent state used by dual-layer extraction routing.
    z_dyn: float = 0.0
    # SSE-EM latent responsibilities (static/dynamic).
    resp_static: float = 0.5
    resp_dynamic: float = 0.5
    # Short-term contradiction memory for dynamic suppression only.
    st_mem: float = 0.0
    # LBR-3D local debias estimate (geometry channel only).
    lbr_bias: float = 0.0
    lbr_hits: float = 0.0
    # VCR: visibility-contradiction decomposition.
    vcr_free: float = 0.0
    vcr_surface: float = 0.0
    vcr_occ: float = 0.0
    vcr_score: float = 0.0
    # RBI: lightweight local re-integration buffer state.
    rbi_sum_phi: float = 0.0
    rbi_sum_w: float = 0.0
    rbi_dyn_ema: float = 0.0
    surf_evidence: float = 0.0
    free_evidence: float = 0.0
    residual_evidence: float = 0.0
    # Local zero-crossing debias estimate (EMA of signed phi bias).
    phi_bias: float = 0.0
    # Geometry-channel local debias estimate (EMA of signed phi_geo bias).
    phi_geo_bias: float = 0.0
    # ZCBF: block-level persistent geometry debias field.
    zcbf_bias: float = 0.0
    zcbf_bias_conf: float = 0.0
    # Geometry residual anchor (association signed-distance residual history).
    geo_res_ema: float = 0.0
    geo_res_hits: float = 0.0
    # Spatio-temporal contradiction score (free/surface inconsistency).
    stcg_score: float = 0.0
    # Hysteresis gate state for contradiction activation.
    stcg_active: float = 0.0
    # DCCM: delayed-commit contradiction memory.
    dccm_free: float = 0.0
    dccm_surface: float = 0.0
    dccm_rear: float = 0.0
    dccm_age: float = 0.0
    dccm_commit: float = 0.0
    # OMHS: occlusion-aware multi-hypothesis surface state.
    omhs_front_conf: float = 0.0
    omhs_rear_conf: float = 0.0
    omhs_gap: float = 0.0
    omhs_active: float = 0.0
    # WOD: write-time occlusion decomposition state.
    wod_front_conf: float = 0.0
    wod_rear_conf: float = 0.0
    wod_shell_conf: float = 0.0
    # RPS: rear-persistent surface bank written at update time.
    phi_rear: float = 0.0
    phi_rear_w: float = 0.0
    rho_rear: float = 0.0
    # Hard rear-commit candidate state.
    phi_rear_cand: float = 0.0
    phi_rear_cand_w: float = 0.0
    rho_rear_cand: float = 0.0
    rps_commit_score: float = 0.0
    rps_commit_age: float = 0.0
    rps_active: float = 0.0
    # SPG: static promotion gate bank for export-side persistent front geometry.
    phi_spg: float = 0.0
    phi_spg_w: float = 0.0
    rho_spg: float = 0.0
    spg_score: float = 0.0
    spg_age: float = 0.0
    spg_active: float = 0.0
    # OTV: observation-time transient veto bank. It stores front-layer transient
    # geometry that is explicitly prevented from entering persistent geometry.
    phi_otv: float = 0.0
    phi_otv_w: float = 0.0
    rho_otv: float = 0.0
    otv_score: float = 0.0
    otv_age: float = 0.0
    otv_active: float = 0.0
    # XMem: write-time exclusion memory. It stores dynamic-front occupancy
    # evidence and later free-space clearing evidence as a reversible state.
    xmem_occ: float = 0.0
    xmem_free: float = 0.0
    xmem_score: float = 0.0
    xmem_age: float = 0.0
    xmem_active: float = 0.0
    xmem_clear: float = 0.0
    xmem_clear_age: float = 0.0
    xmem_clear_active: float = 0.0
    # OBL-3D: independent occlusion-buffered background layer.
    phi_bg: float = 0.0
    phi_bg_w: float = 0.0
    rho_bg: float = 0.0
    # Delayed background candidate state.
    phi_bg_cand: float = 0.0
    phi_bg_cand_w: float = 0.0
    rho_bg_cand: float = 0.0
    phi_bg_memory: float = 0.0
    phi_bg_memory_w: float = 0.0
    rho_bg_stable: float = 0.0
    bg_visible_mem: float = 0.0
    bg_obstruction_mem: float = 0.0
    bg_dense_support: float = 0.0
    bg_dense_phi: float = 0.0
    bg_dense_w: float = 0.0
    bg_cand_score: float = 0.0
    bg_cand_age: float = 0.0
    bg_cand_active: float = 0.0
    # Tri-map escalation-aware delayed residency state.
    trimap_hold_frames: float = 0.0
    trimap_hysteresis: float = 0.0
    trimap_route_score: float = 0.0
    trimap_escalated: float = 0.0
    # Persistent delayed surface bank.
    phi_delayed_bank: float = 0.0
    phi_delayed_bank_w: float = 0.0
    rho_delayed_bank: float = 0.0
    g_delayed_bank: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    g_delayed_bank_w: float = 0.0
    delayed_bank_conf: float = 0.0
    delayed_bank_age: float = 0.0
    delayed_bank_active: float = 0.0
    obl_score: float = 0.0
    obl_age: float = 0.0
    obl_active: float = 0.0
    # CMCT: foreground-to-background contradiction transfer state.
    cmct_score: float = 0.0
    cmct_age: float = 0.0
    cmct_active: float = 0.0
    # CGCC: cross-map geometric carving corridor state.
    cgcc_score: float = 0.0
    cgcc_age: float = 0.0
    cgcc_active: float = 0.0
    # PFV: persistent free-space volume state.
    pfv_score: float = 0.0
    pfv_age: float = 0.0
    pfv_active: float = 0.0
    pfv_long: float = 0.0
    pfv_long_age: float = 0.0
    pfv_near: float = 0.0
    pfv_mid: float = 0.0
    pfv_far: float = 0.0
    pfv_near_age: float = 0.0
    pfv_mid_age: float = 0.0
    pfv_far_age: float = 0.0
    # PFV exclusivity: export-oriented persistent cleared-corridor state.
    pfv_exclusive: float = 0.0
    pfv_exclusive_age: float = 0.0
    pfv_exclusive_active: float = 0.0
    rho_prev: float = 0.0
    rho_osc: float = 0.0
    free_hit_ema: float = 0.0
    occ_hit_ema: float = 0.0
    visibility_contradiction: float = 0.0
    clear_hits: float = 0.0
    # Frontier activation score (cold-start growth support).
    frontier_score: float = 0.0
    # Number of direct measurement integrations that touched this voxel.
    observation_count: float = 0.0
    # Last frame index when the voxel was updated by a measurement.
    last_seen: int = 0


@dataclass
class PoseSE3State:
    t_wc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    r_wc: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    cov: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=float) * 0.01)

    def as_matrix(self) -> np.ndarray:
        out = np.eye(4, dtype=float)
        out[:3, :3] = self.r_wc
        out[:3, 3] = self.t_wc
        return out

    def set_matrix(self, t_wc: np.ndarray) -> None:
        self.r_wc = np.asarray(t_wc[:3, :3], dtype=float)
        self.t_wc = np.asarray(t_wc[:3, 3], dtype=float)
