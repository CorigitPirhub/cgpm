from __future__ import annotations

from typing import Tuple

import numpy as np


def bg_candidate_route_score(
    self,
    voxel_map,
    cell,
    *,
    static_ratio: float,
    rear_bg: float,
    front_dyn: float,
    q_dyn_obs: float,
    assoc_risk: float,
) -> float:
    if not bool(getattr(self.cfg.update, 'pfv_bg_candidate_enable', False)):
        return 0.0
    pfv_conf = float(voxel_map._pfv_conf(cell)) if hasattr(voxel_map, '_pfv_conf') else 0.0
    pfv_excl = float(voxel_map._pfv_exclusive_conf(cell)) if hasattr(voxel_map, '_pfv_exclusive_conf') else 0.0
    pfv_n = float(max(pfv_conf, pfv_excl))
    if pfv_n <= 1e-6:
        return 0.0
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    static_support = float(np.clip(max(
        float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0)),
        float(np.clip(static_ratio, 0.0, 1.0)),
        float(np.clip(getattr(cell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
        float(np.clip(getattr(cell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
        float(rear_bg),
    ), 0.0, 1.0))
    score = float(np.clip(
        pfv_n
        * (0.55 + 0.30 * front_dyn + 0.10 * float(np.clip(q_dyn_obs, 0.0, 1.0)) + 0.05 * float(np.clip(assoc_risk, 0.0, 1.0)))
        * max(0.0, 1.0 - 0.85 * static_support),
        0.0,
        1.0,
    ))
    return score


def update_bg_candidate_state(
    self,
    voxel_map,
    cell,
    *,
    d_bg_obs: float,
    w_obs: float,
    route_score: float,
) -> Tuple[float, float]:
    if not bool(getattr(self.cfg.update, 'pfv_bg_candidate_enable', False)):
        return 1.0, 0.0
    on = float(np.clip(getattr(self.cfg.update, 'pfv_bg_candidate_on', 0.30), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'pfv_bg_candidate_off', 0.18), 0.0, 1.0))
    alpha = float(np.clip(getattr(self.cfg.update, 'pfv_bg_candidate_alpha', 0.18), 0.01, 0.8))
    gain = float(max(0.0, getattr(self.cfg.update, 'pfv_bg_candidate_gain', 1.00)))
    leak = float(np.clip(getattr(self.cfg.update, 'pfv_bg_candidate_leak', 0.04), 0.0, 1.0))

    prev_active = float(np.clip(getattr(cell, 'bg_cand_active', 0.0), 0.0, 1.0))
    cell.bg_cand_score = float(np.clip((1.0 - alpha) * float(getattr(cell, 'bg_cand_score', 0.0)) + alpha * route_score, 0.0, 1.0))
    if route_score >= off:
        cell.bg_cand_age = float(min(20.0, float(getattr(cell, 'bg_cand_age', 0.0)) + 1.0))
    else:
        cell.bg_cand_age = float(max(0.0, 0.96 * float(getattr(cell, 'bg_cand_age', 0.0))))
    cand_eff = float(np.clip(cell.bg_cand_score * (0.55 + 0.45 * min(1.0, float(getattr(cell, 'bg_cand_age', 0.0)) / 3.0)), 0.0, 1.0))
    active = bool((cand_eff >= on and cell.bg_cand_age >= 1.0) or (prev_active >= 0.5 and cand_eff >= off))
    cell.bg_cand_active = 1.0 if active else float(0.96 * prev_active)
    cand_n = float(max(cand_eff, cell.bg_cand_active))

    if cand_n > 1e-6:
        w_cand = float(max(0.0, gain * cand_n * w_obs))
        w_new = float(float(getattr(cell, 'phi_bg_cand_w', 0.0)) + w_cand)
        if w_new > 1e-12:
            cell.phi_bg_cand = float((float(getattr(cell, 'phi_bg_cand_w', 0.0)) * float(getattr(cell, 'phi_bg_cand', 0.0)) + w_cand * d_bg_obs) / w_new)
            cell.phi_bg_cand_w = float(min(5000.0, w_new))
            cell.rho_bg_cand = float(float(getattr(cell, 'rho_bg_cand', 0.0)) + w_cand)
        return leak, cand_n
    return 1.0, 0.0


def maybe_promote_bg_candidate(
    self,
    voxel_map,
    cell,
    *,
    static_ratio: float,
    rear_bg: float,
) -> float:
    if not bool(getattr(self.cfg.update, 'pfv_bg_candidate_enable', False)):
        return 0.0
    w_cand = float(max(0.0, getattr(cell, 'phi_bg_cand_w', 0.0)))
    rho_cand = float(max(0.0, getattr(cell, 'rho_bg_cand', 0.0)))
    if w_cand <= 1e-8 or rho_cand <= 1e-8:
        return 0.0
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    support = float(np.clip(max(
        float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0)),
        float(np.clip(static_ratio, 0.0, 1.0)),
        float(np.clip(rear_bg, 0.0, 1.0)),
        float(np.clip(getattr(cell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
        float(np.clip(getattr(cell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
    ), 0.0, 1.0))
    promote_on = float(np.clip(getattr(self.cfg.update, 'pfv_bg_candidate_promote_on', 0.72), 0.0, 1.0))
    promote_rho = float(np.clip(getattr(self.cfg.update, 'pfv_bg_candidate_promote_rho', 0.85), 0.0, 2.0))
    if support < promote_on or (rho_cand / rho_ref) < promote_rho:
        return 0.0
    blend = float(np.clip(getattr(self.cfg.update, 'pfv_bg_candidate_promote_blend', 0.45), 0.0, 1.0))
    w_commit = float(w_cand * blend)
    if w_commit <= 1e-12:
        return 0.0
    w_bg_old = float(max(0.0, getattr(cell, 'phi_bg_w', 0.0)))
    w_bg_new = float(w_bg_old + w_commit)
    if w_bg_new > 1e-12:
        cell.phi_bg = float((w_bg_old * float(getattr(cell, 'phi_bg', 0.0)) + w_commit * float(getattr(cell, 'phi_bg_cand', 0.0))) / w_bg_new)
        cell.phi_bg_w = float(min(5000.0, w_bg_new))
        cell.rho_bg = float(float(getattr(cell, 'rho_bg', 0.0)) + blend * rho_cand)
    cell.phi_bg_cand_w = float((1.0 - blend) * w_cand)
    cell.rho_bg_cand = float((1.0 - blend) * rho_cand)
    cell.bg_cand_score = float((1.0 - blend) * float(getattr(cell, 'bg_cand_score', 0.0)))
    cell.bg_cand_active = float((1.0 - blend) * float(getattr(cell, 'bg_cand_active', 0.0)))
    return blend
