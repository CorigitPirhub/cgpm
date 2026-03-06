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
    rho_prev: float = 0.0
    rho_osc: float = 0.0
    free_hit_ema: float = 0.0
    occ_hit_ema: float = 0.0
    visibility_contradiction: float = 0.0
    clear_hits: float = 0.0
    # Frontier activation score (cold-start growth support).
    frontier_score: float = 0.0
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
