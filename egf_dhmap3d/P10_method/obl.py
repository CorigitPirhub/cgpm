from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `obl`."""

def update_obl_state(
    self,
    voxel_map: VoxelHashMap3D,
    cell: VoxelCell3D,
    *,
    w_obs: float,
    d_static_obs: float,
    d_rear_obs: float,
    wod_front: float,
    wod_rear: float,
    wod_shell: float,
    q_dyn_obs: float,
    assoc_risk: float,
) -> Tuple[float, float, float]:
    if not bool(getattr(self.cfg.update, 'obl_enable', False)):
        return 0.0, 1.0, 1.0
    if w_obs <= 1e-12:
        return 0.0, 1.0, 1.0

    sep_ref = float(max(0.25, getattr(self.cfg.update, 'obl_sep_ref_vox', 0.90)))
    sep_n = float(
        np.clip(
            abs(float(d_rear_obs) - float(d_static_obs)) / max(1e-6, sep_ref * voxel_map.voxel_size),
            0.0,
            1.5,
        )
    )
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    p_static = float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0))
    rho_s = float(max(0.0, getattr(cell, 'rho_static', 0.0)))
    rho_t = float(max(0.0, getattr(cell, 'rho_transient', 0.0)))
    ptdsf_dom = float(np.clip(rho_s / max(1e-6, rho_s + rho_t), 0.0, 1.0)) if (rho_s + rho_t) > 1e-8 else p_static
    rear_rho = float(np.clip(getattr(cell, 'rho_rear', 0.0) / rho_ref, 0.0, 1.0))
    spg_score = float(np.clip(getattr(cell, 'spg_score', 0.0), 0.0, 1.0))
    spg_active = float(np.clip(getattr(cell, 'spg_active', 0.0), 0.0, 1.0))
    surf = float(max(1e-6, getattr(cell, 'surf_evidence', 0.0)))
    free = float(max(0.0, getattr(cell, 'free_evidence', 0.0)))
    static_occ = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
    xmem_n = float(np.clip(max(getattr(cell, 'xmem_active', 0.0), getattr(cell, 'xmem_score', 0.0)), 0.0, 1.0))
    dyn_front = float(
        np.clip(
            0.30 * wod_front
            + 0.18 * wod_shell
            + 0.18 * float(np.clip(q_dyn_obs, 0.0, 1.0))
            + 0.12 * float(np.clip(getattr(cell, 'dyn_prob', 0.0), 0.0, 1.0))
            + 0.10 * float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0))
            + 0.08 * xmem_n
            + 0.04 * float(np.clip(assoc_risk, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    rear_support = float(
        np.clip(
            0.24 * wod_rear
            + 0.16 * rear_rho
            + 0.14 * spg_active
            + 0.12 * spg_score
            + 0.12 * p_static
            + 0.12 * ptdsf_dom
            + 0.10 * static_occ
            + 0.10 * float(np.clip(getattr(cell, 'obl_active', 0.0), 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    stable_support = float(np.clip(max(static_occ, p_static, ptdsf_dom, rear_rho, spg_active), 0.0, 1.0))
    rear_obs = float(np.clip(rear_support * (0.30 + 0.70 * min(1.0, sep_n)), 0.0, 1.0))
    static_obs = float(np.clip(stable_support * (0.20 + 0.80 * max(float(np.clip(getattr(cell, 'obl_active', 0.0), 0.0, 1.0)), 0.45)) * (1.0 - 0.55 * dyn_front), 0.0, 1.0))
    score_raw = float(np.clip(max(rear_obs, static_obs) - float(getattr(self.cfg.update, 'obl_dyn_static_guard', 0.60)) * dyn_front + 0.08 * stable_support, -1.0, 1.0))
    score_obs = float(self._sigmoid(4.8 * (score_raw - 0.20)))
    a_score = float(np.clip(getattr(self.cfg.update, 'obl_score_alpha', 0.18), 0.01, 0.8))
    cell.obl_score = float(np.clip((1.0 - a_score) * float(getattr(cell, 'obl_score', 0.0)) + a_score * score_obs, 0.0, 1.0))

    active_obs = bool(
        score_obs >= max(0.50, float(getattr(self.cfg.update, 'obl_commit_off', 0.40)))
        and max(rear_obs, static_obs) >= 0.22
    )
    if active_obs:
        cell.obl_age = float(min(20.0, float(getattr(cell, 'obl_age', 0.0)) + 1.0))
    else:
        cell.obl_age = float(max(0.0, 0.96 * float(getattr(cell, 'obl_age', 0.0))))

    on = float(np.clip(getattr(self.cfg.update, 'obl_commit_on', 0.58), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'obl_commit_off', 0.40), 0.0, 1.0))
    age_ref = float(max(1.0, getattr(self.cfg.update, 'obl_age_ref', 1.0)))
    active_prev = float(np.clip(getattr(cell, 'obl_active', 0.0), 0.0, 1.0))
    active = bool((cell.obl_score >= on and cell.obl_age >= age_ref) or (active_prev >= 0.5 and cell.obl_score >= off))
    cell.obl_active = 1.0 if active else float(0.96 * active_prev)

    q_route = float(np.clip(max(cell.obl_score, cell.obl_active, rear_obs), 0.0, 1.0))
    if q_route <= 1e-6:
        return 0.0, 1.0, 1.0

    rear_mode = bool(rear_obs >= static_obs and sep_n >= 0.25)
    d_bg_obs = float(d_rear_obs if rear_mode else d_static_obs)
    if rear_mode:
        gain = float(max(0.0, getattr(self.cfg.update, 'obl_rear_gain', 0.92)))
        support = float(rear_obs)
    else:
        gain = float(max(0.0, getattr(self.cfg.update, 'obl_static_gain', 0.38)))
        support = float(static_obs)
    w_bg = float(w_obs * q_route * gain * (0.25 + 0.75 * support))
    w_bg_new = float(float(getattr(cell, 'phi_bg_w', 0.0)) + w_bg)
    if w_bg_new > 1e-12:
        cell.phi_bg = float((float(getattr(cell, 'phi_bg_w', 0.0)) * float(getattr(cell, 'phi_bg', 0.0)) + w_bg * d_bg_obs) / w_bg_new)
        cell.phi_bg_w = float(min(5000.0, w_bg_new))
        rho_alpha = float(np.clip(getattr(self.cfg.update, 'obl_rho_alpha', 0.20), 0.01, 1.0))
        cell.rho_bg = float(float(getattr(cell, 'rho_bg', 0.0)) + rho_alpha * w_bg)

    strength = float(q_route * (0.35 + 0.65 * dyn_front) * (0.25 + 0.75 * min(1.0, sep_n)))
    static_keep = float(np.clip(1.0 - float(getattr(self.cfg.update, 'obl_static_veto', 0.78)) * strength, 0.08, 1.0))
    geo_keep = float(np.clip(1.0 - float(getattr(self.cfg.update, 'obl_geo_veto', 0.62)) * strength, 0.10, 1.0))
    return q_route, static_keep, geo_keep

def obl_conf(self, cell: VoxelCell3D) -> float:
    score = float(np.clip(getattr(cell, 'obl_score', 0.0), 0.0, 1.0))
    active = float(np.clip(getattr(cell, 'obl_active', 0.0), 0.0, 1.0))
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    rho_n = float(np.clip(getattr(cell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.5))
    w = float(max(0.0, getattr(cell, 'phi_bg_w', 0.0)))
    w_n = float(np.clip(w / max(1e-6, w + 1.5), 0.0, 1.0))
    return float(np.clip(max(score, active, 0.55 * rho_n) * (0.25 + 0.75 * w_n), 0.0, 1.0))

