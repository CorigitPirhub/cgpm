from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `pfv`."""

def update_persistent_free_space_volume(
    self,
    background_map: VoxelHashMap3D,
    foreground_map: VoxelHashMap3D,
    accepted: List[AssocMeasurement3D],
) -> dict:
    if not bool(getattr(self.cfg.update, 'pfv_enable', False)):
        return {
            "pfv_applied": 0.0,
            "pfv_cells": 0.0,
            "pfv_mean_score": 0.0,
        }
    if not accepted:
        return {
            "pfv_applied": 0.0,
            "pfv_cells": 0.0,
            "pfv_mean_score": 0.0,
        }

    alpha = float(np.clip(getattr(self.cfg.update, 'pfv_alpha', 0.18), 0.01, 0.8))
    alpha_long = float(np.clip(getattr(self.cfg.update, 'pfv_long_alpha', 0.10), 0.01, 0.8))
    on = float(np.clip(getattr(self.cfg.update, 'pfv_commit_on', 0.44), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'pfv_commit_off', 0.28), 0.0, 1.0))
    long_on = float(np.clip(getattr(self.cfg.update, 'pfv_long_on', 0.34), 0.0, 1.0))
    step_scale = float(max(0.1, getattr(self.cfg.update, 'pfv_step_scale', 0.75)))
    end_margin = float(max(0.5 * background_map.voxel_size, getattr(self.cfg.update, 'pfv_end_margin', 0.18)))
    bg_support_ref = float(np.clip(getattr(self.cfg.update, 'pfv_bg_support_ref', 0.38), 0.05, 1.0))
    fg_guard = float(np.clip(getattr(self.cfg.update, 'pfv_fg_guard', 0.35), 0.0, 1.0))
    static_guard = float(np.clip(getattr(self.cfg.update, 'pfv_static_guard', 0.82), 0.0, 1.0))
    bg_decay = float(np.clip(getattr(self.cfg.update, 'pfv_bg_decay', 0.55), 0.0, 1.0))
    geo_decay = float(np.clip(getattr(self.cfg.update, 'pfv_geo_decay', 0.42), 0.0, 1.0))
    rho_decay = float(np.clip(getattr(self.cfg.update, 'pfv_rho_decay', 0.28), 0.0, 1.0))
    bg_decay_long = float(np.clip(getattr(self.cfg.update, 'pfv_bg_decay_long', 0.68), 0.0, 1.0))
    geo_decay_long = float(np.clip(getattr(self.cfg.update, 'pfv_geo_decay_long', 0.52), 0.0, 1.0))
    release_rho = float(np.clip(getattr(self.cfg.update, 'pfv_release_rho', 0.92), 0.0, 2.0))
    depth_gain = float(np.clip(getattr(self.cfg.update, 'pfv_depth_gain', 0.30), 0.0, 1.0))
    excl_enable = bool(getattr(self.cfg.update, 'pfv_exclusive_enable', False))
    excl_alpha = float(np.clip(getattr(self.cfg.update, 'pfv_exclusive_alpha', 0.18), 0.01, 0.8))
    excl_on = float(np.clip(getattr(self.cfg.update, 'pfv_exclusive_on', 0.46), 0.0, 1.0))
    excl_off = float(np.clip(getattr(self.cfg.update, 'pfv_exclusive_off', 0.32), 0.0, 1.0))
    excl_fg_weight = float(np.clip(getattr(self.cfg.update, 'pfv_exclusive_fg_weight', 0.45), 0.0, 1.0))
    excl_long_weight = float(np.clip(getattr(self.cfg.update, 'pfv_exclusive_long_weight', 0.55), 0.0, 1.0))
    excl_static_guard = float(np.clip(getattr(self.cfg.update, 'pfv_exclusive_static_guard', 0.78), 0.0, 1.0))
    rho_ref = float(max(1e-6, self.cfg.update.dual_state_static_protect_rho))
    step = float(max(0.5 * background_map.voxel_size, step_scale * background_map.voxel_size))

    touched: Set[VoxelIndex] = set()
    scored: List[float] = []
    excl_scored: List[float] = []
    for m in accepted:
        if m.sensor_origin is None:
            continue
        p = np.asarray(m.point_world, dtype=float).reshape(3)
        origin = np.asarray(m.sensor_origin, dtype=float).reshape(3)
        v = p - origin
        dist = float(np.linalg.norm(v))
        if dist <= end_margin + step:
            continue
        endpoint_idx = background_map.world_to_index(p)
        cb_end = background_map.get_cell(endpoint_idx)
        if cb_end is None:
            continue
        bg_support = float(np.clip(max(
            float(np.clip(getattr(cb_end, 'p_static', 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cb_end, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
            float(np.clip(getattr(cb_end, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
            float(background_map._obl_conf(cb_end)) if hasattr(background_map, '_obl_conf') else 0.0,
        ), 0.0, 1.0))
        if bg_support < bg_support_ref:
            continue
        d = v / dist
        s = step
        while s < dist - end_margin:
            x = origin + s * d
            bidx = background_map.world_to_index(x)
            cb = background_map.get_cell(bidx)
            if cb is None:
                s += step
                continue
            cf = foreground_map.get_cell(foreground_map.world_to_index(x))
            fg_conf = 0.0
            if cf is not None:
                rho_s = float(max(0.0, getattr(cf, 'rho_static', 0.0)))
                rho_t = float(max(0.0, getattr(cf, 'rho_transient', 0.0)))
                split = float(np.clip(rho_t / max(1e-6, rho_s + rho_t), 0.0, 1.0)) if (rho_s + rho_t) > 1e-8 else float(np.clip(getattr(cf, 'dyn_prob', 0.0), 0.0, 1.0))
                fg_conf = float(np.clip(max(split, getattr(cf, 'dyn_prob', 0.0), getattr(cf, 'z_dyn', 0.0), getattr(cf, 'visibility_contradiction', 0.0), foreground_map._xmem_conf(cf) if hasattr(foreground_map, '_xmem_conf') else 0.0), 0.0, 1.0))
            local_bg = float(np.clip(max(
                float(np.clip(getattr(cb, 'p_static', 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cb, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
                float(np.clip(getattr(cb, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
                float(background_map._obl_conf(cb)) if hasattr(background_map, '_obl_conf') else 0.0,
            ), 0.0, 1.0))
            axial_phase = float(np.clip(s / max(1e-6, dist - end_margin), 0.0, 1.0))
            depth_term = float(np.clip(0.45 + depth_gain * axial_phase, 0.0, 1.5))
            obs = float(np.clip(bg_support * depth_term * (1.0 - fg_guard * fg_conf) * (1.0 - static_guard * local_bg), 0.0, 1.0))
            if obs <= 1e-6:
                s += step
                continue
            cb.pfv_score = float(np.clip((1.0 - alpha) * float(getattr(cb, 'pfv_score', 0.0)) + alpha * obs, 0.0, 1.0))
            if obs >= off:
                cb.pfv_age = float(min(20.0, float(getattr(cb, 'pfv_age', 0.0)) + 1.0))
            else:
                cb.pfv_age = float(max(0.0, 0.96 * float(getattr(cb, 'pfv_age', 0.0))))
            bg_rho_n = float(np.clip(max(getattr(cb, 'rho_bg', 0.0), getattr(cb, 'rho_static', 0.0)) / rho_ref, 0.0, 1.5))
            long_obs = float(np.clip(obs * (0.35 + 0.65 * bg_support) * (1.0 - 0.55 * fg_conf), 0.0, 1.0))
            bank_alpha = float(np.clip(getattr(self.cfg.update, 'pfv_bank_alpha', 0.16), 0.01, 0.8))
            near_split = float(np.clip(getattr(self.cfg.update, 'pfv_bank_near_split', 0.35), 0.05, 0.9))
            far_split = float(np.clip(getattr(self.cfg.update, 'pfv_bank_far_split', 0.72), near_split + 0.05, 0.98))
            near_obs = 0.0
            mid_obs = 0.0
            far_obs = 0.0
            if axial_phase <= near_split:
                near_obs = obs
            elif axial_phase <= far_split:
                mid_obs = obs
            else:
                far_obs = obs
            cb.pfv_near = float(np.clip((1.0 - bank_alpha) * float(getattr(cb, 'pfv_near', 0.0)) + bank_alpha * near_obs, 0.0, 1.0))
            cb.pfv_mid = float(np.clip((1.0 - bank_alpha) * float(getattr(cb, 'pfv_mid', 0.0)) + bank_alpha * mid_obs, 0.0, 1.0))
            cb.pfv_far = float(np.clip((1.0 - bank_alpha) * float(getattr(cb, 'pfv_far', 0.0)) + bank_alpha * far_obs, 0.0, 1.0))
            if near_obs >= long_on:
                cb.pfv_near_age = float(min(20.0, float(getattr(cb, 'pfv_near_age', 0.0)) + 1.0))
            else:
                cb.pfv_near_age = float(max(0.0, 0.97 * float(getattr(cb, 'pfv_near_age', 0.0))))
            if mid_obs >= long_on:
                cb.pfv_mid_age = float(min(20.0, float(getattr(cb, 'pfv_mid_age', 0.0)) + 1.0))
            else:
                cb.pfv_mid_age = float(max(0.0, 0.97 * float(getattr(cb, 'pfv_mid_age', 0.0))))
            if far_obs >= long_on:
                cb.pfv_far_age = float(min(20.0, float(getattr(cb, 'pfv_far_age', 0.0)) + 1.0))
            else:
                cb.pfv_far_age = float(max(0.0, 0.97 * float(getattr(cb, 'pfv_far_age', 0.0))))
            if bg_rho_n >= release_rho:
                cb.pfv_long = float(max(0.0, 0.92 * float(getattr(cb, 'pfv_long', 0.0))))
                cb.pfv_long_age = float(max(0.0, 0.96 * float(getattr(cb, 'pfv_long_age', 0.0))))
            else:
                cb.pfv_long = float(np.clip((1.0 - alpha_long) * float(getattr(cb, 'pfv_long', 0.0)) + alpha_long * long_obs, 0.0, 1.0))
                if long_obs >= long_on:
                    cb.pfv_long_age = float(min(20.0, float(getattr(cb, 'pfv_long_age', 0.0)) + 1.0))
                else:
                    cb.pfv_long_age = float(max(0.0, 0.97 * float(getattr(cb, 'pfv_long_age', 0.0))))
            active_prev = float(np.clip(getattr(cb, 'pfv_active', 0.0), 0.0, 1.0))
            near_w = float(getattr(self.cfg.update, 'pfv_bank_weight_near', 0.95))
            mid_w = float(getattr(self.cfg.update, 'pfv_bank_weight_mid', 0.80))
            far_w = float(getattr(self.cfg.update, 'pfv_bank_weight_far', 0.55))
            bank_eff = float(np.clip(max(
                near_w * float(getattr(cb, 'pfv_near', 0.0)) * (0.55 + 0.45 * min(1.0, float(getattr(cb, 'pfv_near_age', 0.0)) / 3.0)),
                mid_w * float(getattr(cb, 'pfv_mid', 0.0)) * (0.55 + 0.45 * min(1.0, float(getattr(cb, 'pfv_mid_age', 0.0)) / 3.0)),
                far_w * float(getattr(cb, 'pfv_far', 0.0)) * (0.55 + 0.45 * min(1.0, float(getattr(cb, 'pfv_far_age', 0.0)) / 3.0)),
            ), 0.0, 1.0))
            pfv_eff = float(max(cb.pfv_score, min(1.0, cb.pfv_long * (0.55 + 0.45 * min(1.0, float(getattr(cb, 'pfv_long_age', 0.0)) / 4.0))), bank_eff))
            active = bool((pfv_eff >= on and cb.pfv_age >= 1.0) or (active_prev >= 0.5 and pfv_eff >= off))
            cb.pfv_active = 1.0 if active else float(0.96 * active_prev)
            pfv_n = float(max(pfv_eff, cb.pfv_active))
            excl_n = 0.0
            if excl_enable:
                excl_prev = float(np.clip(getattr(cb, 'pfv_exclusive_active', 0.0), 0.0, 1.0))
                excl_obs = float(
                    np.clip(
                        max(obs, 0.82 * bank_eff, 0.70 * long_obs)
                        * (0.55 + excl_fg_weight * max(fg_conf, 1.0 - local_bg))
                        * (0.45 + excl_long_weight * max(long_obs, bank_eff))
                        * (1.0 - excl_static_guard * local_bg),
                        0.0,
                        1.0,
                    )
                )
                if bg_rho_n >= release_rho or local_bg >= 0.72:
                    cb.pfv_exclusive = float(max(0.0, 0.92 * float(getattr(cb, 'pfv_exclusive', 0.0))))
                    cb.pfv_exclusive_age = float(max(0.0, 0.96 * float(getattr(cb, 'pfv_exclusive_age', 0.0))))
                else:
                    cb.pfv_exclusive = float(np.clip((1.0 - excl_alpha) * float(getattr(cb, 'pfv_exclusive', 0.0)) + excl_alpha * excl_obs, 0.0, 1.0))
                    if excl_obs >= excl_on:
                        cb.pfv_exclusive_age = float(min(20.0, float(getattr(cb, 'pfv_exclusive_age', 0.0)) + 1.0))
                    else:
                        cb.pfv_exclusive_age = float(max(0.0, 0.97 * float(getattr(cb, 'pfv_exclusive_age', 0.0))))
                excl_eff = float(np.clip(float(getattr(cb, 'pfv_exclusive', 0.0)) * (0.55 + 0.45 * min(1.0, float(getattr(cb, 'pfv_exclusive_age', 0.0)) / 3.0)), 0.0, 1.0))
                excl_active = bool((excl_eff >= excl_on and float(getattr(cb, 'pfv_exclusive_age', 0.0)) >= 1.0) or (excl_prev >= 0.5 and excl_eff >= excl_off))
                cb.pfv_exclusive_active = 1.0 if excl_active else float(0.96 * excl_prev)
                excl_n = float(max(excl_eff, float(getattr(cb, 'pfv_exclusive_active', 0.0))))
            if pfv_n > 1e-6 or excl_n > 1e-6:
                short_strength = float(np.clip(pfv_n * (1.0 - 0.70 * local_bg), 0.0, 1.0))
                long_strength = float(np.clip(float(getattr(cb, 'pfv_long', 0.0)) * (1.0 - 0.55 * local_bg), 0.0, 1.0))
                excl_strength = float(np.clip(excl_n * (1.0 - 0.45 * local_bg), 0.0, 1.0))
                cb.phi_bg_w = float(max(0.0, getattr(cb, 'phi_bg_w', 0.0) * (1.0 - bg_decay * short_strength) * (1.0 - bg_decay_long * long_strength) * (1.0 - 0.35 * bg_decay * excl_strength)))
                cb.phi_static_w = float(max(0.0, cb.phi_static_w * (1.0 - 0.50 * bg_decay * short_strength) * (1.0 - 0.50 * bg_decay_long * long_strength) * (1.0 - 0.22 * bg_decay * excl_strength)))
                cb.phi_geo_w = float(max(0.0, cb.phi_geo_w * (1.0 - geo_decay * short_strength) * (1.0 - geo_decay_long * long_strength) * (1.0 - 0.28 * geo_decay * excl_strength)))
                cb.rho_bg = float(max(0.0, getattr(cb, 'rho_bg', 0.0) * (1.0 - rho_decay * short_strength) * (1.0 - 0.85 * rho_decay * long_strength) * (1.0 - 0.42 * rho_decay * excl_strength)))
                cb.rho_static = float(max(0.0, getattr(cb, 'rho_static', 0.0) * (1.0 - 0.75 * rho_decay * short_strength) * (1.0 - 0.60 * rho_decay * long_strength) * (1.0 - 0.24 * rho_decay * excl_strength)))
                cb.rho = float(max(0.0, cb.rho * (1.0 - 0.5 * rho_decay * short_strength) * (1.0 - 0.40 * rho_decay * long_strength) * (1.0 - 0.18 * rho_decay * excl_strength)))
                background_map._sync_legacy_channels(cb)
            touched.add(bidx)
            scored.append(float(obs))
            if excl_enable:
                excl_scored.append(float(excl_n))
            s += step
    return {
        "pfv_applied": 1.0,
        "pfv_cells": float(len(touched)),
        "pfv_mean_score": float(np.mean(scored)) if scored else 0.0,
        "pfv_exclusive_mean": float(np.mean(excl_scored)) if excl_scored else 0.0,
    }

def pfv_conf(self, cell: VoxelCell3D) -> float:
    score = float(np.clip(getattr(cell, 'pfv_score', 0.0), 0.0, 1.0))
    active = float(np.clip(getattr(cell, 'pfv_active', 0.0), 0.0, 1.0))
    long_term = float(np.clip(getattr(cell, 'pfv_long', 0.0), 0.0, 1.0))
    long_age = float(np.clip(getattr(cell, 'pfv_long_age', 0.0) / 4.0, 0.0, 1.0))
    long_eff = float(np.clip(long_term * (0.55 + 0.45 * long_age), 0.0, 1.0))
    return float(np.clip(max(score, active, long_eff), 0.0, 1.0))

def pfv_commit_delay_scales(
    self,
    voxel_map: VoxelHashMap3D,
    cell: VoxelCell3D,
    *,
    static_ratio: float,
    q_dyn_obs: float,
    assoc_risk: float,
) -> Tuple[float, float, float, float, float]:
    if not bool(getattr(self.cfg.update, 'pfv_commit_delay_enable', False)):
        return 1.0, 1.0, 1.0, 1.0, 0.0
    pfv_score = float(np.clip(voxel_map._pfv_conf(cell), 0.0, 1.0)) if hasattr(voxel_map, '_pfv_conf') else 0.0
    pfv_excl = float(np.clip(voxel_map._pfv_exclusive_conf(cell), 0.0, 1.0)) if hasattr(voxel_map, '_pfv_exclusive_conf') else 0.0
    pfv_n = float(max(pfv_score, pfv_excl))
    if pfv_n <= 1e-6:
        return 1.0, 1.0, 1.0, 1.0, 0.0

    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    surf = float(max(1e-6, getattr(cell, 'surf_evidence', 0.0)))
    free = float(max(0.0, getattr(cell, 'free_evidence', 0.0)))
    free_ratio = float(free / surf)
    free_ref = float(max(1e-6, getattr(self.cfg.update, 'pfv_commit_delay_free_ratio_ref', 0.65)))
    free_n = float(np.clip(free_ratio / free_ref, 0.0, 1.5))
    static_conf = float(np.clip(max(
        float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0)),
        float(np.clip(getattr(cell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
        float(np.clip(getattr(cell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
        float(voxel_map._obl_conf(cell)) if hasattr(voxel_map, '_obl_conf') else 0.0,
        float(np.clip(static_ratio, 0.0, 1.0)),
    ), 0.0, 1.0))
    rho_n = float(np.clip(max(getattr(cell, 'rho_bg', 0.0), getattr(cell, 'rho_static', 0.0), getattr(cell, 'rho', 0.0)) / rho_ref, 0.0, 1.5))
    support_guard = float(np.clip(getattr(self.cfg.update, 'pfv_commit_delay_support_guard', 0.78), 0.0, 1.0))
    rho_guard = float(np.clip(getattr(self.cfg.update, 'pfv_commit_delay_rho_guard', 0.95), 0.0, 2.0))
    score = float(np.clip(
        pfv_n
        * (0.60 + 0.22 * float(np.clip(assoc_risk, 0.0, 1.0)) + 0.18 * float(np.clip(q_dyn_obs, 0.0, 1.0)) + 0.20 * min(1.0, free_n))
        * max(0.0, 1.0 - support_guard * static_conf)
        * max(0.0, 1.0 - min(1.0, rho_n / max(1e-6, rho_guard))),
        0.0,
        1.0,
    ))
    if score < float(np.clip(getattr(self.cfg.update, 'pfv_commit_delay_on', 0.34), 0.0, 1.0)):
        return 1.0, 1.0, 1.0, 1.0, score

    min_scale = float(np.clip(getattr(self.cfg.update, 'pfv_commit_delay_min_scale', 0.02), 0.0, 1.0))
    static_keep = float(np.clip(1.0 - getattr(self.cfg.update, 'pfv_commit_delay_static_weight', 0.85) * score, min_scale, 1.0))
    bg_keep = float(np.clip(1.0 - getattr(self.cfg.update, 'pfv_commit_delay_bg_weight', 0.95) * score, min_scale, 1.0))
    geo_keep = float(np.clip(1.0 - getattr(self.cfg.update, 'pfv_commit_delay_geo_weight', 0.70) * score, min_scale, 1.0))
    rho_keep = float(np.clip(1.0 - getattr(self.cfg.update, 'pfv_commit_delay_rho_weight', 0.55) * score, min_scale, 1.0))
    return static_keep, bg_keep, geo_keep, rho_keep, score


def pfv_exclusive_conf(self, cell: VoxelCell3D) -> float:
    score = float(np.clip(getattr(cell, 'pfv_exclusive', 0.0), 0.0, 1.0))
    age = float(np.clip(getattr(cell, 'pfv_exclusive_age', 0.0) / 3.0, 0.0, 1.0))
    active = float(np.clip(getattr(cell, 'pfv_exclusive_active', 0.0), 0.0, 1.0))
    return float(np.clip(max(score * (0.55 + 0.45 * age), active), 0.0, 1.0))


def pfv_cluster_conf(self, idx: VoxelIndex) -> float:
    radius = int(max(0, getattr(self.cfg.update, 'pfv_cluster_radius', 1)))
    if radius <= 0:
        cell = self.get_cell(idx)
        return 0.0 if cell is None else self._pfv_conf(cell)
    values = []
    bank_values = []
    near_w = float(getattr(self.cfg.update, 'pfv_bank_weight_near', 0.95))
    mid_w = float(getattr(self.cfg.update, 'pfv_bank_weight_mid', 0.80))
    far_w = float(getattr(self.cfg.update, 'pfv_bank_weight_far', 0.55))
    for nidx in self.neighbor_indices(idx, radius):
        cell = self.get_cell(nidx)
        if cell is None:
            continue
        values.append(self._pfv_conf(cell))
        bank_values.append(max(
            near_w * float(np.clip(getattr(cell, 'pfv_near', 0.0), 0.0, 1.0)),
            mid_w * float(np.clip(getattr(cell, 'pfv_mid', 0.0), 0.0, 1.0)),
            far_w * float(np.clip(getattr(cell, 'pfv_far', 0.0), 0.0, 1.0)),
        ))
    if not values:
        return 0.0
    values_arr = np.asarray(values, dtype=float)
    banks_arr = np.asarray(bank_values, dtype=float)
    return float(np.clip(max(np.quantile(values_arr, 0.75), np.quantile(banks_arr, 0.75)), 0.0, 1.0))

def cross_map_fg_conf(self, foreground_map: 'VoxelHashMap3D', idx: VoxelIndex) -> float:
    best = 0.0
    for nidx in foreground_map.neighbor_indices(idx, 1):
        cf = foreground_map.get_cell(nidx)
        if cf is None:
            continue
        rho_s = float(max(0.0, getattr(cf, 'rho_static', 0.0)))
        rho_t = float(max(0.0, getattr(cf, 'rho_transient', 0.0)))
        split = float(np.clip(rho_t / max(1e-6, rho_s + rho_t), 0.0, 1.0)) if (rho_s + rho_t) > 1e-8 else float(np.clip(getattr(cf, 'dyn_prob', 0.0), 0.0, 1.0))
        dyn = float(
            np.clip(
                max(
                    float(np.clip(getattr(cf, 'dyn_prob', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(cf, 'z_dyn', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(cf, 'visibility_contradiction', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(cf, 'st_mem', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(cf, 'wod_front_conf', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(cf, 'wod_shell_conf', 0.0), 0.0, 1.0)),
                    float(self._xmem_conf(cf)),
                    float(self._otv_conf(cf)),
                    float(self._otv_surface_conf(cf)),
                ),
                0.0,
                1.0,
            )
        )
        w = float(max(getattr(cf, 'phi_transient_w', 0.0), getattr(cf, 'phi_dyn_w', 0.0), getattr(cf, 'phi_otv_w', 0.0)))
        w_n = float(np.clip(w / max(1e-6, w + 1.5), 0.0, 1.0))
        conf = float(np.clip((0.72 * dyn + 0.28 * split) * (0.25 + 0.75 * w_n), 0.0, 1.0))
        best = max(best, conf)
    return float(np.clip(best, 0.0, 1.0))

def map_static_support(self, voxel_map: VoxelHashMap3D, cell) -> float:
    if cell is None:
        return 0.0
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    surf = float(max(1e-6, getattr(cell, 'surf_evidence', 0.0)))
    free = float(max(0.0, getattr(cell, 'free_evidence', 0.0)))
    occ = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
    p_static = float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0))
    rho_static = float(np.clip(getattr(cell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0))
    rho_bg = float(np.clip(getattr(cell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0))
    obl_conf = float(voxel_map._obl_conf(cell)) if hasattr(voxel_map, '_obl_conf') else 0.0
    return float(np.clip(max(0.34 * occ + 0.26 * p_static + 0.20 * rho_static + 0.20 * rho_bg + 0.16 * obl_conf, max(occ, p_static, rho_static, rho_bg, obl_conf)), 0.0, 1.0))

def route_measurements_with_pfv(
    self,
    accepted: List,
) -> Tuple[List, List, Dict[str, float]]:
    if self.foreground_map is None or not bool(getattr(self.cfg.update, 'pfvp_enable', False)):
        return accepted, accepted if self.foreground_map is not None else accepted, {
            'pfvp_bg_only': 0.0,
            'pfvp_fg_only': 0.0,
            'pfvp_dual': float(len(accepted)) if self.foreground_map is not None else 0.0,
        }

    bg_only: List = []
    fg_only: List = []
    dual: List = []
    bg_map = self.background_map
    fg_map = self.foreground_map
    margin = float(max(0.0, getattr(self.cfg.update, 'pfvp_margin', 0.08)))
    fg_on = float(np.clip(getattr(self.cfg.update, 'pfvp_fg_on', 0.38), 0.0, 1.0))
    bg_on = float(np.clip(getattr(self.cfg.update, 'pfvp_bg_on', 0.28), 0.0, 1.0))
    bg_floor = float(np.clip(getattr(self.cfg.update, 'pfvp_bg_keep_floor', 0.08), 0.0, 1.0))
    w_pfv = float(max(0.0, getattr(self.cfg.update, 'pfvp_pfv_weight', 0.55)))
    w_hist = float(max(0.0, getattr(self.cfg.update, 'pfvp_fg_hist_weight', 0.25)))
    w_assoc = float(max(0.0, getattr(self.cfg.update, 'pfvp_assoc_weight', 0.20)))
    w_static = float(max(0.0, getattr(self.cfg.update, 'pfvp_static_weight', 0.55)))
    w_rho = float(max(0.0, getattr(self.cfg.update, 'pfvp_bg_rho_weight', 0.25)))
    w_obl = float(max(0.0, getattr(self.cfg.update, 'pfvp_bg_obl_weight', 0.20)))
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))

    counts = {'pfvp_bg_only': 0.0, 'pfvp_fg_only': 0.0, 'pfvp_dual': 0.0}
    for measurement in accepted:
        idx = bg_map.world_to_index(np.asarray(measurement.point_world, dtype=float).reshape(3))
        pfv_conf = 0.0
        bg_support = 0.0
        bg_rho = 0.0
        bg_obl = 0.0
        for nidx in bg_map.neighbor_indices(idx, 1):
            cell = bg_map.get_cell(nidx)
            if cell is None:
                continue
            bg_support = max(bg_support, self._map_static_support(bg_map, cell))
            if hasattr(bg_map, '_pfv_conf'):
                pfv_conf = max(pfv_conf, float(bg_map._pfv_conf(cell)))
            bg_rho = max(bg_rho, float(np.clip(max(getattr(cell, 'rho_bg', 0.0), getattr(cell, 'rho_static', 0.0)) / rho_ref, 0.0, 1.0)))
            if hasattr(bg_map, '_obl_conf'):
                bg_obl = max(bg_obl, float(bg_map._obl_conf(cell)))
        fg_hist = float(bg_map._cross_map_fg_conf(fg_map, idx)) if hasattr(bg_map, '_cross_map_fg_conf') else 0.0
        assoc_term = float(np.clip(getattr(measurement, 'assoc_risk', 0.0), 0.0, 1.0))
        fg_score = float(np.clip(w_pfv * pfv_conf + w_hist * fg_hist + w_assoc * assoc_term, 0.0, 1.0))
        bg_score = float(np.clip(w_static * bg_support + w_rho * bg_rho + w_obl * bg_obl + bg_floor * (1.0 - min(1.0, fg_score)), 0.0, 1.0))

        route_fg = bool(fg_score >= fg_on and fg_score > bg_score + margin)
        route_bg = bool(bg_score >= bg_on)
        if route_fg and not route_bg:
            fg_only.append(measurement)
            counts['pfvp_fg_only'] += 1.0
        elif route_bg and not route_fg:
            bg_only.append(measurement)
            counts['pfvp_bg_only'] += 1.0
        else:
            dual.append(measurement)
            counts['pfvp_dual'] += 1.0

    bg_accepted = bg_only + dual
    fg_accepted = fg_only + dual
    return bg_accepted, fg_accepted, counts

