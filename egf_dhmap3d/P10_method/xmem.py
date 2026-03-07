from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `xmem`."""

def update_xmem_state(
    self,
    voxel_map: VoxelHashMap3D,
    cell: VoxelCell3D,
    *,
    d_signed: float,
    surf_band: float,
    trunc_eff: float,
    w_obs: float,
    d_transient_obs: float,
    d_rear_obs: float,
    wod_front: float,
    wod_rear: float,
    wod_shell: float,
    q_dyn_obs: float,
    assoc_risk: float,
) -> Tuple[float, float, float, float, float]:
    if not bool(getattr(self.cfg.update, 'xmem_enable', False)):
        return 0.0, 1.0, 1.0, 1.0, 0.0
    if w_obs <= 1e-12:
        return 0.0, 1.0, 1.0, 1.0, 0.0

    sep_ref = float(max(0.25, getattr(self.cfg.update, 'xmem_sep_ref_vox', 0.90)))
    sep_n = float(
        np.clip(
            abs(float(d_rear_obs) - float(d_transient_obs)) / max(1e-6, sep_ref * voxel_map.voxel_size),
            0.0,
            1.5,
        )
    )
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    p_static = float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0))
    rho_s = float(max(0.0, getattr(cell, 'rho_static', 0.0)))
    rho_t = float(max(0.0, getattr(cell, 'rho_transient', 0.0)))
    ptdsf_dom = float(np.clip(rho_s / max(1e-6, rho_s + rho_t), 0.0, 1.0)) if (rho_s + rho_t) > 1e-8 else p_static
    static_occ = float(np.clip(float(getattr(cell, 'surf_evidence', 0.0)) / max(1e-6, float(getattr(cell, 'surf_evidence', 0.0)) + float(getattr(cell, 'free_evidence', 0.0))), 0.0, 1.0))
    spg_score = float(np.clip(getattr(cell, 'spg_score', 0.0), 0.0, 1.0))
    spg_active = float(np.clip(getattr(cell, 'spg_active', 0.0), 0.0, 1.0))
    spg_rho = float(np.clip(getattr(cell, 'rho_spg', 0.0) / rho_ref, 0.0, 1.0))
    rear_rho = float(np.clip(getattr(cell, 'rho_rear', 0.0) / rho_ref, 0.0, 1.0))
    static_support = float(
        np.clip(
            0.22 * spg_active
            + 0.18 * spg_score
            + 0.16 * rear_rho
            + 0.14 * p_static
            + 0.12 * ptdsf_dom
            + 0.10 * spg_rho
            + 0.08 * static_occ
            + 0.08 * wod_rear,
            0.0,
            1.0,
        )
    )
    surf_like = bool(abs(float(d_signed)) <= float(surf_band))
    free_like = bool(abs(float(d_signed)) > float(surf_band))
    front_signal = float(
        np.clip(
            0.34 * wod_front
            + 0.18 * wod_shell
            + 0.16 * float(np.clip(q_dyn_obs, 0.0, 1.0))
            + 0.12 * float(np.clip(assoc_risk, 0.0, 1.0))
            + 0.10 * float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0))
            + 0.10 * min(1.0, sep_n),
            0.0,
            1.0,
        )
    )
    free_ratio_obs = float(np.clip((abs(float(d_signed)) - float(surf_band)) / max(1e-6, float(trunc_eff) - float(surf_band)), 0.0, 1.0)) if free_like else 0.0
    free_signal = float(
        np.clip(
            0.40 * float(np.clip(getattr(cell, 'visibility_contradiction', 0.0), 0.0, 1.0))
            + 0.28 * float(np.clip(getattr(cell, 'free_hit_ema', 0.0), 0.0, 1.0))
            + 0.16 * float(np.clip(getattr(cell, 'clear_hits', 0.0) / 3.0, 0.0, 1.0))
            + 0.16 * free_ratio_obs,
            0.0,
            1.0,
        )
    )
    occ_obs = float(np.clip(front_signal * (0.30 + 0.70 * min(1.0, sep_n)) * (1.0 - float(getattr(self.cfg.update, 'xmem_static_guard', 0.78)) * static_support), 0.0, 1.0))
    if not surf_like:
        occ_obs *= 0.25
    free_obs = float(np.clip(free_signal * (0.40 + 0.60 * max(float(np.clip(getattr(cell, 'xmem_active', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'xmem_occ', 0.0), 0.0, 1.0)))), 0.0, 1.0))
    if surf_like:
        free_obs *= 0.35

    a_occ = float(np.clip(getattr(self.cfg.update, 'xmem_occ_alpha', 0.18), 0.01, 0.8))
    a_free = float(np.clip(getattr(self.cfg.update, 'xmem_free_alpha', 0.14), 0.01, 0.8))
    cell.xmem_occ = float(np.clip((1.0 - a_occ) * float(getattr(cell, 'xmem_occ', 0.0)) + a_occ * occ_obs, 0.0, 1.0))
    cell.xmem_free = float(np.clip((1.0 - a_free) * float(getattr(cell, 'xmem_free', 0.0)) + a_free * free_obs, 0.0, 1.0))

    support_ref = float(np.clip(getattr(self.cfg.update, 'xmem_support_ref', 0.24), 0.05, 0.9))
    free_gain = float(max(0.0, getattr(self.cfg.update, 'xmem_free_gain', 0.85)))
    score_raw = float(np.clip(float(getattr(cell, 'xmem_occ', 0.0)) - free_gain * float(getattr(cell, 'xmem_free', 0.0)) - 0.45 * static_support + 0.08 * min(1.0, sep_n), -1.0, 1.0))
    score_obs = float(self._sigmoid(5.2 * (score_raw - support_ref)))
    a_score = float(np.clip(getattr(self.cfg.update, 'xmem_score_alpha', 0.20), 0.01, 0.8))
    cell.xmem_score = float(np.clip((1.0 - a_score) * float(getattr(cell, 'xmem_score', 0.0)) + a_score * score_obs, 0.0, 1.0))

    active_obs = bool(
        score_obs >= max(0.50, float(getattr(self.cfg.update, 'xmem_commit_off', 0.42)))
        and float(getattr(cell, 'xmem_occ', 0.0)) >= float(getattr(cell, 'xmem_free', 0.0)) + 0.05
        and front_signal >= 0.25
        and min(1.0, sep_n) >= 0.25
    )
    if active_obs:
        cell.xmem_age = float(min(20.0, float(getattr(cell, 'xmem_age', 0.0)) + 1.0))
    else:
        decay = float(np.clip(getattr(self.cfg.update, 'xmem_decay', 0.97), 0.80, 1.0))
        cell.xmem_age = float(max(0.0, decay * float(getattr(cell, 'xmem_age', 0.0))))

    on = float(np.clip(getattr(self.cfg.update, 'xmem_commit_on', 0.60), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'xmem_commit_off', 0.42), 0.0, 1.0))
    age_ref = float(max(1.0, getattr(self.cfg.update, 'xmem_age_ref', 1.0)))
    active_prev = float(np.clip(getattr(cell, 'xmem_active', 0.0), 0.0, 1.0))
    active = bool((cell.xmem_score >= on and cell.xmem_age >= age_ref) or (active_prev >= 0.5 and cell.xmem_score >= off))
    if active:
        cell.xmem_active = 1.0
    else:
        decay = float(np.clip(getattr(self.cfg.update, 'xmem_decay', 0.97), 0.80, 1.0))
        cell.xmem_active = float(decay * active_prev)

    # BECM: split the old reversible score into two independent memories.
    # `xmem_active` models front-layer exclusion, while `xmem_clear_active`
    # models the later free-space clearing lock. Free evidence should not
    # cancel exclusion; it should transition the voxel into a background
    # recovery phase that keeps geometry writes/readout suppressed until
    # static support is rebuilt.
    clear_obs = float(
        np.clip(
            float(getattr(cell, 'xmem_occ', 0.0))
            * float(getattr(cell, 'xmem_free', 0.0))
            * (0.45 + 0.55 * free_signal)
            * (0.35 + 0.65 * min(1.0, sep_n)),
            0.0,
            1.0,
        )
    )
    a_clear = float(np.clip(getattr(self.cfg.update, 'xmem_clear_alpha', 0.18), 0.01, 0.8))
    cell.xmem_clear = float(
        np.clip(
            (1.0 - a_clear) * float(getattr(cell, 'xmem_clear', 0.0)) + a_clear * clear_obs,
            0.0,
            1.0,
        )
    )
    clear_obs_active = bool(
        clear_obs >= max(0.30, float(getattr(self.cfg.update, 'xmem_clear_off', 0.28)))
        and free_signal >= 0.18
        and float(getattr(cell, 'xmem_occ', 0.0)) >= 0.20
    )
    if clear_obs_active:
        cell.xmem_clear_age = float(min(20.0, float(getattr(cell, 'xmem_clear_age', 0.0)) + 1.0))
    else:
        decay = float(np.clip(getattr(self.cfg.update, 'xmem_decay', 0.97), 0.80, 1.0))
        cell.xmem_clear_age = float(max(0.0, decay * float(getattr(cell, 'xmem_clear_age', 0.0))))
    clear_on = float(np.clip(getattr(self.cfg.update, 'xmem_clear_on', 0.42), 0.0, 1.0))
    clear_off = float(np.clip(getattr(self.cfg.update, 'xmem_clear_off', 0.28), 0.0, 1.0))
    clear_release = bool(
        static_support >= float(np.clip(getattr(self.cfg.update, 'xmem_clear_static_release', 0.76), 0.0, 1.0))
        and ptdsf_dom >= 0.60
        and rear_rho >= 0.22
    )
    clear_prev = float(np.clip(getattr(cell, 'xmem_clear_active', 0.0), 0.0, 1.0))
    clear_active = bool(
        (cell.xmem_clear >= clear_on and cell.xmem_clear_age >= 1.0)
        or (clear_prev >= 0.5 and cell.xmem_clear >= clear_off)
    ) and (not clear_release)
    if clear_active:
        cell.xmem_clear_active = 1.0
        if free_signal >= 0.18:
            decay_w = float(np.clip(getattr(self.cfg.update, 'xmem_clear_weight_decay', 0.92), 0.50, 1.0))
            cell.phi_w = float(max(0.0, decay_w * cell.phi_w))
            cell.phi_static_w = float(max(0.0, decay_w * cell.phi_static_w))
            cell.phi_geo_w = float(max(0.0, decay_w * cell.phi_geo_w))
            cell.rho = float(max(0.0, decay_w * cell.rho))
    else:
        decay = float(np.clip(getattr(self.cfg.update, 'xmem_decay', 0.97), 0.80, 1.0))
        cell.xmem_clear_active = float(decay * clear_prev)

    clear_conf = float(np.clip(max(float(getattr(cell, 'xmem_clear', 0.0)), float(getattr(cell, 'xmem_clear_active', 0.0))), 0.0, 1.0))
    q_route = float(
        np.clip(
            max(
                float(getattr(cell, 'xmem_score', 0.0)),
                float(getattr(cell, 'xmem_active', 0.0)),
                clear_conf,
                float(np.clip(float(getattr(cell, 'xmem_occ', 0.0)) - 0.5 * float(getattr(cell, 'xmem_free', 0.0)), 0.0, 1.0)),
            ),
            0.0,
            1.0,
        )
    )
    if q_route <= 1e-6:
        return 0.0, 1.0, 1.0, 1.0, 0.0

    strength = float(q_route * (0.55 + 0.45 * min(1.0, sep_n)) * (1.0 - 0.25 * static_support))
    if clear_conf > 1e-6:
        strength = max(strength, float(clear_conf * (0.70 + 0.30 * free_signal) * (1.0 - 0.15 * static_support)))
    static_keep = float(np.clip(1.0 - float(getattr(self.cfg.update, 'xmem_static_veto', 0.96)) * strength, 0.02, 1.0))
    geo_keep = float(np.clip(1.0 - float(getattr(self.cfg.update, 'xmem_geo_veto', 0.92)) * strength, 0.02, 1.0))
    transient_boost = float(np.clip(1.0 + float(getattr(self.cfg.update, 'xmem_transient_boost', 0.90)) * strength, 1.0, 2.0))
    dyn_boost = float(np.clip(float(getattr(self.cfg.update, 'xmem_dyn_boost', 0.82)) * strength, 0.0, 1.0))
    if clear_conf > 1e-6:
        static_keep = float(min(static_keep, np.clip(1.0 - 1.10 * clear_conf, 0.02, 1.0)))
        geo_keep = float(min(geo_keep, np.clip(1.0 - 1.05 * clear_conf, 0.02, 1.0)))
        transient_boost = float(max(transient_boost, np.clip(1.0 + 0.65 * clear_conf, 1.0, 2.0)))
        dyn_boost = float(max(dyn_boost, np.clip(0.35 + 0.55 * clear_conf + 0.10 * free_signal, 0.0, 1.0)))
    return q_route, static_keep, geo_keep, transient_boost, dyn_boost

def xmem_clear_conf(self, cell: VoxelCell3D) -> float:
    clear = float(np.clip(getattr(cell, 'xmem_clear', 0.0), 0.0, 1.0))
    clear_active = float(np.clip(getattr(cell, 'xmem_clear_active', 0.0), 0.0, 1.0))
    return float(np.clip(max(clear, clear_active), 0.0, 1.0))

def xmem_conf(self, cell: VoxelCell3D) -> float:
    occ = float(np.clip(getattr(cell, 'xmem_occ', 0.0), 0.0, 1.0))
    free = float(np.clip(getattr(cell, 'xmem_free', 0.0), 0.0, 1.0))
    score = float(np.clip(getattr(cell, 'xmem_score', 0.0), 0.0, 1.0))
    active = float(np.clip(getattr(cell, 'xmem_active', 0.0), 0.0, 1.0))
    clear = float(self._xmem_clear_conf(cell))
    return float(np.clip(max(score, active, clear, np.clip(occ - 0.5 * free, 0.0, 1.0)), 0.0, 1.0))

