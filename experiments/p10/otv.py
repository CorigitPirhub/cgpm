from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `otv`."""

def update_otv_state(
    self,
    voxel_map: VoxelHashMap3D,
    cell: VoxelCell3D,
    *,
    w_obs: float,
    d_transient_obs: float,
    d_rear_obs: float,
    wod_front: float,
    wod_rear: float,
    wod_shell: float,
    q_dyn_obs: float,
    assoc_risk: float,
) -> Tuple[float, float, float, float]:
    if not bool(self.cfg.update.otv_enable):
        return 0.0, 1.0, 1.0, 0.0
    if w_obs <= 1e-12:
        return 0.0, 1.0, 1.0, 0.0

    sep_ref = float(max(0.25, getattr(self.cfg.update, 'otv_sep_ref_vox', 0.90)))
    sep_n = float(
        np.clip(
            abs(float(d_rear_obs) - float(d_transient_obs))
            / max(1e-6, sep_ref * voxel_map.voxel_size),
            0.0,
            1.5,
        )
    )
    front_dom = float(np.clip(wod_front + 0.55 * wod_shell + 0.20 * q_dyn_obs, 0.0, 1.0))
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    spg_score = float(np.clip(getattr(cell, 'spg_score', 0.0), 0.0, 1.0))
    spg_active = float(np.clip(getattr(cell, 'spg_active', 0.0), 0.0, 1.0))
    spg_rho = float(np.clip(getattr(cell, 'rho_spg', 0.0) / rho_ref, 0.0, 1.0))
    rear_rho = float(np.clip(getattr(cell, 'rho_rear', 0.0) / rho_ref, 0.0, 1.0))
    p_static = float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0))
    rho_s = float(max(0.0, getattr(cell, 'rho_static', 0.0)))
    rho_t = float(max(0.0, getattr(cell, 'rho_transient', 0.0)))
    ptdsf_dom = float(np.clip(rho_s / max(1e-6, rho_s + rho_t), 0.0, 1.0)) if (rho_s + rho_t) > 1e-8 else p_static
    surf = float(max(1e-6, getattr(cell, 'surf_evidence', 0.0)))
    free = float(max(0.0, getattr(cell, 'free_evidence', 0.0)))
    static_occ = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
    rear_support = float(
        np.clip(
            0.22 * spg_active
            + 0.20 * spg_score
            + 0.16 * spg_rho
            + 0.14 * rear_rho
            + 0.14 * p_static
            + 0.08 * ptdsf_dom
            + 0.06 * wod_rear
            + 0.10 * static_occ,
            0.0,
            1.0,
        )
    )
    score_raw = float(
        np.clip(
            front_dom * (0.55 + 0.45 * min(1.0, sep_n)) * rear_support
            + 0.08 * assoc_risk,
            0.0,
            1.0,
        )
    )
    score_ref = float(np.clip(getattr(self.cfg.update, 'otv_support_ref', 0.26), 0.05, 0.85))
    score_obs = float(self._sigmoid(5.2 * (score_raw - score_ref)))
    alpha = float(np.clip(getattr(self.cfg.update, 'otv_score_alpha', 0.18), 0.01, 0.8))
    cell.otv_score = float(np.clip((1.0 - alpha) * float(getattr(cell, 'otv_score', 0.0)) + alpha * score_obs, 0.0, 1.0))

    active_obs = bool(score_obs >= max(0.50, float(getattr(self.cfg.update, 'otv_commit_off', 0.38))) and rear_support >= 0.40 and front_dom >= 0.35 and sep_n >= 0.35)
    if active_obs:
        cell.otv_age = float(min(20.0, float(getattr(cell, 'otv_age', 0.0)) + 1.0))
    else:
        decay = float(np.clip(getattr(self.cfg.update, 'otv_decay', 0.96), 0.80, 1.0))
        cell.otv_age = float(max(0.0, decay * float(getattr(cell, 'otv_age', 0.0))))

    on = float(np.clip(getattr(self.cfg.update, 'otv_commit_on', 0.58), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'otv_commit_off', 0.38), 0.0, 1.0))
    age_ref = float(max(1.0, getattr(self.cfg.update, 'otv_age_ref', 1.0)))
    active_prev = float(np.clip(getattr(cell, 'otv_active', 0.0), 0.0, 1.0))
    active = bool((score_obs >= on and cell.otv_age >= age_ref) or (active_prev >= 0.5 and cell.otv_score >= off))
    if active:
        cell.otv_active = 1.0
    else:
        decay = float(np.clip(getattr(self.cfg.update, 'otv_decay', 0.96), 0.80, 1.0))
        cell.otv_active = float(decay * active_prev)

    q_route = float(np.clip(max(score_obs, cell.otv_score, cell.otv_active), 0.0, 1.0))
    if q_route <= 1e-6:
        return 0.0, 1.0, 1.0, 0.0

    w_otv = float(w_obs * q_route * np.clip(0.35 + 0.65 * front_dom, 0.20, 1.0))
    w_new = float(float(getattr(cell, 'phi_otv_w', 0.0)) + w_otv)
    if w_new > 1e-12:
        cell.phi_otv = float((float(getattr(cell, 'phi_otv_w', 0.0)) * float(getattr(cell, 'phi_otv', 0.0)) + w_otv * float(d_transient_obs)) / w_new)
        cell.phi_otv_w = float(min(5000.0, w_new))
        cell.rho_otv = float(float(getattr(cell, 'rho_otv', 0.0)) + 0.18 * w_otv)

    strength = float(q_route * (0.55 + 0.45 * min(1.0, sep_n)))
    static_keep = float(np.clip(1.0 - getattr(self.cfg.update, 'otv_static_veto', 0.92) * strength, 0.02, 1.0))
    transient_boost = float(np.clip(1.0 + getattr(self.cfg.update, 'otv_transient_boost', 0.85) * strength, 1.0, 2.0))
    dyn_boost = float(np.clip(getattr(self.cfg.update, 'otv_dyn_boost', 0.75) * strength, 0.0, 1.0))
    return q_route, static_keep, transient_boost, dyn_boost

def otv_conf(self, cell: VoxelCell3D) -> float:
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    return float(
        np.clip(
            max(
                float(np.clip(getattr(cell, 'otv_score', 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, 'otv_active', 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, 'rho_otv', 0.0) / rho_ref, 0.0, 1.0)),
            ),
            0.0,
            1.0,
        )
    )

def otv_surface_conf(self, cell: VoxelCell3D) -> float:
    base = self._otv_conf(cell)
    w = float(max(0.0, getattr(cell, 'phi_otv_w', 0.0)))
    if base <= 1e-6 or w <= 1e-8:
        return 0.0
    phi_ref = float(max(1e-6, 2.5 * self.voxel_size))
    near = float(np.exp(-0.5 * (float(getattr(cell, 'phi_otv', 0.0)) / phi_ref) ** 2))
    w_n = float(np.clip(w / max(1e-6, w + 1.5), 0.0, 1.0))
    return float(np.clip(base * (0.25 + 0.75 * near) * (0.35 + 0.65 * w_n), 0.0, 1.0))

