from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `spg`."""

def update_spg_state(
    self,
    voxel_map: VoxelHashMap3D,
    cell,
    w_static: float,
    w_geo: float,
    static_mass: float,
    wod_front: float,
    wod_rear: float,
    wod_shell: float,
    q_dyn_obs: float,
    assoc_risk: float,
) -> None:
    if not (bool(self.cfg.update.spg_enable) and bool(self.cfg.update.dual_state_enable) and bool(self.cfg.update.ptdsf_enable)):
        return
    ws = float(max(0.0, getattr(cell, 'phi_static_w', 0.0)))
    wg = float(max(0.0, getattr(cell, 'phi_geo_w', 0.0)))
    if ws <= 1e-12 and wg <= 1e-12:
        return
    cons_ref = float(max(voxel_map.voxel_size, getattr(self.cfg.update, 'rps_consistency_ref', 0.03)))
    if ws > 1e-12 and wg > 1e-12:
        geo_agree = float(np.exp(-0.5 * ((float(cell.phi_geo) - float(cell.phi_static)) / cons_ref) ** 2))
        w_s = float(ws * (1.0 + 0.20 * static_mass + 0.10 * wod_rear))
        w_g = float(wg * (0.22 + 0.78 * geo_agree))
        phi_cand = float((w_s * float(cell.phi_static) + w_g * float(cell.phi_geo)) / max(1e-9, w_s + w_g))
        w_cand = float(w_s + w_g)
    elif ws > 1e-12:
        geo_agree = 0.82
        phi_cand = float(cell.phi_static)
        w_cand = float(ws)
    else:
        geo_agree = 0.82
        phi_cand = float(cell.phi_geo)
        w_cand = float(wg)
    age_ref = float(max(1.0, getattr(self.cfg.update, 'ptdsf_commit_age_ref', 3.0)))
    commit_n = float(np.clip(getattr(cell, 'ptdsf_commit_age', 0.0) / age_ref, 0.0, 1.0))
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    rho_n = float(np.clip(float(getattr(cell, 'rho', 0.0)) / rho_ref, 0.0, 1.5))
    p_static = float(np.clip(getattr(cell, 'p_static', 0.5), 0.0, 1.0))
    front_dyn = float(
        np.clip(
            max(
                wod_front + 0.40 * wod_shell,
                float(np.clip(q_dyn_obs, 0.0, 1.0)),
                float(np.clip(assoc_risk, 0.0, 1.0)),
                float(np.clip(getattr(cell, 'dyn_prob', 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, 'st_mem', 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, 'visibility_contradiction', 0.0), 0.0, 1.0)),
            ),
            0.0,
            1.0,
        )
    )
    support_obs = float(
        np.clip(
            0.28 * static_mass
            + 0.22 * p_static
            + 0.16 * commit_n
            + 0.14 * min(1.0, rho_n)
            + 0.10 * geo_agree
            + 0.10 * wod_rear
            + 0.08 * (1.0 - front_dyn),
            0.0,
            1.0,
        )
    )
    penalty_obs = float(
        np.clip(
            0.38 * wod_front
            + 0.14 * wod_shell
            + 0.14 * float(np.clip(q_dyn_obs, 0.0, 1.0))
            + 0.10 * float(np.clip(assoc_risk, 0.0, 1.0))
            + 0.12 * float(np.clip(getattr(cell, 'visibility_contradiction', 0.0), 0.0, 1.0))
            + 0.06 * float(np.clip(getattr(cell, 'st_mem', 0.0), 0.0, 1.0))
            + 0.06 * float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    score_obs = float(self._sigmoid(3.4 * (support_obs - penalty_obs - 0.04)))
    alpha = float(np.clip(getattr(self.cfg.update, 'spg_score_alpha', 0.18), 0.01, 0.8))
    cell.spg_score = float(np.clip((1.0 - alpha) * float(getattr(cell, 'spg_score', 0.0)) + alpha * score_obs, 0.0, 1.0))
    if score_obs >= 0.54 and support_obs >= (penalty_obs + 0.04):
        cell.spg_age = float(min(20.0, float(getattr(cell, 'spg_age', 0.0)) + 1.0))
    else:
        cell.spg_age = float(max(0.0, 0.82 * float(getattr(cell, 'spg_age', 0.0))))
    on = float(np.clip(getattr(self.cfg.update, 'spg_commit_on', 0.62), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'spg_commit_off', 0.40), 0.0, 1.0))
    age_gate = float(max(1.0, getattr(self.cfg.update, 'spg_commit_age_ref', 1.5)))
    active_prev = float(np.clip(getattr(cell, 'spg_active', 0.0), 0.0, 1.0))
    commit_ready = bool(
        ((cell.spg_score >= on and cell.spg_age >= age_gate) or (active_prev >= 0.5 and cell.spg_score >= off))
        and w_cand > 1e-6
    )
    if commit_ready:
        blend = float(np.clip(getattr(self.cfg.update, 'spg_commit_blend', 0.60), 0.0, 1.0))
        w_commit = float(blend * max(0.35 * w_static, w_geo, 0.30 * w_cand))
        w_new = float(float(getattr(cell, 'phi_spg_w', 0.0)) + w_commit)
        if w_new > 1e-12:
            cell.phi_spg = float((float(getattr(cell, 'phi_spg_w', 0.0)) * float(getattr(cell, 'phi_spg', 0.0)) + w_commit * phi_cand) / w_new)
            cell.phi_spg_w = float(min(5000.0, w_new))
            cell.rho_spg = float(float(getattr(cell, 'rho_spg', 0.0)) + 0.18 * w_commit)
        cell.spg_active = 1.0
    else:
        if cell.spg_score < off or penalty_obs > (support_obs + 0.10):
            decay = float(np.clip(getattr(self.cfg.update, 'spg_bank_decay', 0.96), 0.80, 1.0))
            cell.phi_spg_w = float(decay * float(getattr(cell, 'phi_spg_w', 0.0)))
            cell.rho_spg = float(decay * float(getattr(cell, 'rho_spg', 0.0)))
            cell.spg_active = float(0.70 * active_prev)
        else:
            cell.spg_active = active_prev

