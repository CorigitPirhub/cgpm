from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `rps`."""

def update_rps_state(
    self,
    voxel_map: VoxelHashMap3D,
    cell,
    d_geo: float,
    w_obs: float,
    w_static: float,
    w_geo: float,
    static_mass: float,
    wod_front: float,
    wod_rear: float,
    wod_shell: float,
    q_dyn_obs: float,
    assoc_risk: float,
) -> None:
    if not (bool(self.cfg.update.rps_enable) and bool(self.cfg.update.dual_state_enable) and bool(self.cfg.update.wod_enable)):
        return
    cons_ref = float(max(voxel_map.voxel_size, getattr(self.cfg.update, 'rps_consistency_ref', 0.03)))
    static_cons = 0.5
    if float(getattr(cell, 'phi_static_w', 0.0)) > 1e-8:
        static_cons = float(np.exp(-0.5 * ((d_geo - float(cell.phi_static)) / cons_ref) ** 2))
    geo_cons = 0.5
    if float(getattr(cell, 'phi_geo_w', 0.0)) > 1e-8:
        geo_cons = float(np.exp(-0.5 * ((d_geo - float(cell.phi_geo)) / cons_ref) ** 2))
    bank_cons = 0.5
    if float(getattr(cell, 'phi_rear_w', 0.0)) > 1e-8:
        bank_cons = float(np.exp(-0.5 * ((d_geo - float(getattr(cell, 'phi_rear', 0.0))) / cons_ref) ** 2))
    cand_cons = 0.5
    if float(getattr(cell, 'phi_rear_cand_w', 0.0)) > 1e-8:
        cand_cons = float(np.exp(-0.5 * ((d_geo - float(getattr(cell, 'phi_rear_cand', 0.0))) / cons_ref) ** 2))
    rear_obs = float(
        np.clip(
            0.34 * wod_rear
            + 0.20 * static_mass
            + 0.16 * max(static_cons, geo_cons)
            + 0.10 * bank_cons
            + 0.10 * cand_cons
            + 0.10 * float(np.clip(getattr(cell, 'dccm_rear', 0.0), 0.0, 1.0))
            + 0.10 * (1.0 - float(np.clip(q_dyn_obs, 0.0, 1.0))),
            0.0,
            1.0,
        )
    )
    front_pen = float(
        np.clip(
            wod_front
            + 0.60 * wod_shell
            + 0.35 * float(np.clip(q_dyn_obs, 0.0, 1.0))
            + 0.15 * float(np.clip(assoc_risk, 0.0, 1.0)),
            0.0,
            1.5,
        )
    )
    rear_gate = float(np.clip(rear_obs * (1.0 - float(getattr(self.cfg.update, 'rps_front_suppress', 0.60)) * front_pen), 0.0, 1.0))
    w_base = float(max(w_geo, 0.35 * w_static))
    if not bool(getattr(self.cfg.update, 'rps_hard_commit_enable', False)):
        if rear_gate > 1e-6 and w_base > 1e-12:
            w_rps = float(
                w_base
                * (float(getattr(self.cfg.update, 'rps_geo_mix', 0.65)) + (1.0 - float(getattr(self.cfg.update, 'rps_geo_mix', 0.65))) * static_mass)
                * (0.12 + float(getattr(self.cfg.update, 'rps_rear_boost', 0.85)) * rear_gate)
            )
            w_r_new = float(cell.phi_rear_w + w_rps)
            if w_r_new > 1e-12:
                cell.phi_rear = float((cell.phi_rear_w * float(getattr(cell, 'phi_rear', 0.0)) + w_rps * d_geo) / w_r_new)
                cell.phi_rear_w = float(min(5000.0, w_r_new))
                cell.rho_rear = float(cell.rho_rear + float(getattr(self.cfg.update, 'rps_rho_alpha', 0.18)) * w_obs * rear_gate)
        return

    cand_min = float(np.clip(getattr(self.cfg.update, 'rps_candidate_gate_min', 0.16), 0.0, 0.8))
    cand_gate = float(np.clip((rear_gate - cand_min) / max(1e-6, 1.0 - cand_min), 0.0, 1.0))
    support_obs = float(
        np.clip(
            0.34 * rear_gate
            + 0.18 * float(np.clip(getattr(cell, 'visibility_contradiction', 0.0), 0.0, 1.0))
            + 0.14 * float(np.clip(getattr(cell, 'dccm_rear', 0.0), 0.0, 1.0))
            + 0.10 * wod_shell
            + 0.10 * cand_cons
            + 0.08 * bank_cons
            + 0.06 * (1.0 - float(np.clip(q_dyn_obs, 0.0, 1.0))),
            0.0,
            1.0,
        )
    )
    penalty = float(
        np.clip(
            0.58 * wod_front
            + 0.18 * float(np.clip(q_dyn_obs, 0.0, 1.0))
            + 0.14 * float(np.clip(assoc_risk, 0.0, 1.0))
            + 0.10 * max(0.0, 1.0 - max(static_cons, geo_cons)),
            0.0,
            1.0,
        )
    )
    score_obs = float(self._sigmoid(3.2 * (support_obs - penalty - 0.10)))
    alpha = float(np.clip(getattr(self.cfg.update, 'rps_score_alpha', 0.18), 0.01, 0.8))
    cell.rps_commit_score = float(np.clip((1.0 - alpha) * float(getattr(cell, 'rps_commit_score', 0.0)) + alpha * score_obs, 0.0, 1.0))
    if cand_gate > 1e-6 and w_base > 1e-12:
        w_cand = float(
            w_base
            * (float(getattr(self.cfg.update, 'rps_geo_mix', 0.65)) + (1.0 - float(getattr(self.cfg.update, 'rps_geo_mix', 0.65))) * static_mass)
            * (0.10 + float(getattr(self.cfg.update, 'rps_rear_boost', 0.85)) * cand_gate)
        )
        w_c_new = float(float(getattr(cell, 'phi_rear_cand_w', 0.0)) + w_cand)
        if w_c_new > 1e-12:
            cell.phi_rear_cand = float((float(getattr(cell, 'phi_rear_cand_w', 0.0)) * float(getattr(cell, 'phi_rear_cand', 0.0)) + w_cand * d_geo) / w_c_new)
            cell.phi_rear_cand_w = float(min(5000.0, w_c_new))
            cell.rho_rear_cand = float(float(getattr(cell, 'rho_rear_cand', 0.0)) + float(getattr(self.cfg.update, 'rps_rho_alpha', 0.18)) * w_obs * cand_gate)
    age_ref = float(max(1.0, getattr(self.cfg.update, 'rps_commit_age_ref', 2.0)))
    if cand_gate >= 0.22 and support_obs >= (penalty + 0.06):
        cell.rps_commit_age = float(min(20.0, float(getattr(cell, 'rps_commit_age', 0.0)) + 1.0))
    else:
        cell.rps_commit_age = float(max(0.0, 0.82 * float(getattr(cell, 'rps_commit_age', 0.0))))
    on = float(np.clip(getattr(self.cfg.update, 'rps_commit_on', 0.62), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'rps_commit_off', 0.40), 0.0, 1.0))
    active_prev = float(np.clip(getattr(cell, 'rps_active', 0.0), 0.0, 1.0))
    commit_ready = bool(
        cell.rps_commit_score >= on
        and cell.rps_commit_age >= age_ref
        and float(getattr(cell, 'phi_rear_cand_w', 0.0)) > 1e-6
        and float(getattr(cell, 'rho_rear_cand', 0.0)) >= 0.10
    )
    if commit_ready:
        blend = float(np.clip(getattr(self.cfg.update, 'rps_commit_blend', 0.78), 0.0, 1.0))
        w_commit = float(float(getattr(cell, 'phi_rear_cand_w', 0.0)) * blend)
        w_bank_new = float(float(getattr(cell, 'phi_rear_w', 0.0)) + w_commit)
        if w_bank_new > 1e-12:
            cell.phi_rear = float((float(getattr(cell, 'phi_rear_w', 0.0)) * float(getattr(cell, 'phi_rear', 0.0)) + w_commit * float(getattr(cell, 'phi_rear_cand', 0.0))) / w_bank_new)
            cell.phi_rear_w = float(min(5000.0, w_bank_new))
            cell.rho_rear = float(float(getattr(cell, 'rho_rear', 0.0)) + blend * float(getattr(cell, 'rho_rear_cand', 0.0)))
        cell.phi_rear_cand_w = float((1.0 - blend) * float(getattr(cell, 'phi_rear_cand_w', 0.0)))
        cell.rho_rear_cand = float((1.0 - blend) * float(getattr(cell, 'rho_rear_cand', 0.0)))
        cell.rps_active = 1.0
    elif cell.rps_commit_score < off:
        cell.rps_active = float(0.5 * active_prev)
    else:
        cell.rps_active = active_prev

