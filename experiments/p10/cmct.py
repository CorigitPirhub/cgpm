from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `cmct`."""

def transfer_cross_map_contradiction(
    self,
    background_map: VoxelHashMap3D,
    foreground_map: VoxelHashMap3D,
    accepted: List[AssocMeasurement3D],
) -> dict:
    if not bool(getattr(self.cfg.update, 'cmct_enable', False)):
        return {
            "cmct_applied": 0.0,
            "cmct_cells": 0.0,
            "cmct_mean_score": 0.0,
        }
    if not accepted:
        return {
            "cmct_applied": 0.0,
            "cmct_cells": 0.0,
            "cmct_mean_score": 0.0,
        }

    radius = int(max(0, getattr(self.cfg.update, 'cmct_radius_cells', 1)))
    alpha = float(np.clip(getattr(self.cfg.update, 'cmct_alpha', 0.18), 0.01, 0.8))
    on = float(np.clip(getattr(self.cfg.update, 'cmct_commit_on', 0.46), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'cmct_commit_off', 0.30), 0.0, 1.0))
    bg_decay = float(np.clip(getattr(self.cfg.update, 'cmct_bg_decay', 0.68), 0.0, 1.0))
    geo_decay = float(np.clip(getattr(self.cfg.update, 'cmct_geo_decay', 0.52), 0.0, 1.0))
    rho_decay = float(np.clip(getattr(self.cfg.update, 'cmct_rho_decay', 0.40), 0.0, 1.0))
    static_guard = float(np.clip(getattr(self.cfg.update, 'cmct_static_guard', 0.78), 0.0, 1.0))
    bg_rho_protect = float(np.clip(getattr(self.cfg.update, 'cmct_bg_rho_protect', 0.85), 0.0, 1.0))
    rho_ref = float(max(1e-6, self.cfg.update.dual_state_static_protect_rho))

    scored: List[float] = []
    touched: Set[VoxelIndex] = set()
    seen_fg: Set[VoxelIndex] = set()
    for m in accepted:
        seed_idx = foreground_map.world_to_index(np.asarray(m.point_world, dtype=float).reshape(3))
        for fidx in foreground_map.neighbor_indices(seed_idx, radius):
            if fidx in seen_fg:
                continue
            seen_fg.add(fidx)
            cf = foreground_map.get_cell(fidx)
            if cf is None:
                continue
            rho_s = float(max(0.0, getattr(cf, 'rho_static', 0.0)))
            rho_t = float(max(0.0, getattr(cf, 'rho_transient', 0.0)))
            split = float(np.clip(rho_t / max(1e-6, rho_s + rho_t), 0.0, 1.0)) if (rho_s + rho_t) > 1e-8 else float(np.clip(getattr(cf, 'dyn_prob', 0.0), 0.0, 1.0))
            xmem_n = float(foreground_map._xmem_conf(cf)) if hasattr(foreground_map, '_xmem_conf') else 0.0
            fg_dyn = float(
                np.clip(
                    0.22 * float(np.clip(getattr(cf, 'dyn_prob', 0.0), 0.0, 1.0))
                    + 0.18 * float(np.clip(getattr(cf, 'z_dyn', 0.0), 0.0, 1.0))
                    + 0.16 * float(np.clip(getattr(cf, 'visibility_contradiction', 0.0), 0.0, 1.0))
                    + 0.12 * float(np.clip(getattr(cf, 'st_mem', 0.0), 0.0, 1.0))
                    + 0.10 * float(np.clip(getattr(cf, 'wod_front_conf', 0.0), 0.0, 1.0))
                    + 0.08 * float(np.clip(getattr(cf, 'wod_shell_conf', 0.0), 0.0, 1.0))
                    + 0.08 * xmem_n
                    + 0.06 * float(np.clip(getattr(cf, 'dccm_commit', 0.0), 0.0, 1.0)),
                    0.0,
                    1.0,
                )
            )
            fg_support = float(
                np.clip(
                    0.28 * split
                    + 0.20 * float(np.clip(getattr(cf, 'phi_transient_w', 0.0) / (getattr(cf, 'phi_transient_w', 0.0) + 1.5), 0.0, 1.0))
                    + 0.16 * float(np.clip(getattr(cf, 'phi_dyn_w', 0.0) / (getattr(cf, 'phi_dyn_w', 0.0) + 1.5), 0.0, 1.0))
                    + 0.14 * float(np.clip(getattr(cf, 'visibility_contradiction', 0.0), 0.0, 1.0))
                    + 0.12 * float(np.clip(getattr(cf, 'wod_front_conf', 0.0), 0.0, 1.0))
                    + 0.10 * float(np.clip(getattr(cf, 'clear_hits', 0.0) / 3.0, 0.0, 1.0)),
                    0.0,
                    1.0,
                )
            )
            transfer = float(np.clip(fg_dyn * fg_support, 0.0, 1.0))
            if transfer < 0.16:
                continue

            for bidx in background_map.neighbor_indices(fidx, radius):
                cb = background_map.get_cell(bidx)
                if cb is None:
                    continue
                bg_p = float(np.clip(getattr(cb, 'p_static', 0.0), 0.0, 1.0))
                bg_rho = float(np.clip(max(getattr(cb, 'rho_bg', 0.0), getattr(cb, 'rho_static', 0.0), getattr(cb, 'rho_rear', 0.0)) / rho_ref, 0.0, 1.5))
                surf = float(max(1e-6, getattr(cb, 'surf_evidence', 0.0)))
                free = float(max(0.0, getattr(cb, 'free_evidence', 0.0)))
                bg_occ = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
                obl_n = float(background_map._obl_conf(cb)) if hasattr(background_map, '_obl_conf') else 0.0
                bg_support = float(np.clip(max(bg_p, bg_occ, 0.70 * bg_rho, obl_n), 0.0, 1.0))
                obs = float(np.clip(transfer * (1.0 - static_guard * bg_support), 0.0, 1.0))
                if obs <= 1e-6:
                    continue
                cb.cmct_score = float(np.clip((1.0 - alpha) * float(getattr(cb, 'cmct_score', 0.0)) + alpha * obs, 0.0, 1.0))
                if obs >= off:
                    cb.cmct_age = float(min(20.0, float(getattr(cb, 'cmct_age', 0.0)) + 1.0))
                else:
                    cb.cmct_age = float(max(0.0, 0.96 * float(getattr(cb, 'cmct_age', 0.0))))
                active_prev = float(np.clip(getattr(cb, 'cmct_active', 0.0), 0.0, 1.0))
                active = bool((cb.cmct_score >= on and cb.cmct_age >= 1.0) or (active_prev >= 0.5 and cb.cmct_score >= off))
                cb.cmct_active = 1.0 if active else float(0.96 * active_prev)
                cmct_n = float(max(cb.cmct_score, cb.cmct_active))
                if cmct_n <= 1e-6:
                    continue
                protect = float(np.clip(bg_rho_protect * bg_support, 0.0, 0.95))
                strength = float(np.clip(cmct_n * (1.0 - protect), 0.0, 1.0))
                if strength <= 1e-6:
                    continue
                cb.phi_static_w = float(max(0.0, cb.phi_static_w * (1.0 - bg_decay * strength)))
                cb.phi_geo_w = float(max(0.0, cb.phi_geo_w * (1.0 - geo_decay * strength)))
                cb.phi_w = float(max(0.0, cb.phi_w * (1.0 - 0.5 * geo_decay * strength)))
                cb.rho_static = float(max(0.0, getattr(cb, 'rho_static', 0.0) * (1.0 - rho_decay * strength)))
                cb.rho = float(max(0.0, cb.rho * (1.0 - 0.5 * rho_decay * strength)))
                bg_layer_decay = float(np.clip(0.40 * strength * max(0.0, 1.0 - bg_support), 0.0, 0.60))
                if bg_layer_decay > 1e-6:
                    cb.phi_bg_w = float(max(0.0, getattr(cb, 'phi_bg_w', 0.0) * (1.0 - bg_layer_decay)))
                    cb.rho_bg = float(max(0.0, getattr(cb, 'rho_bg', 0.0) * (1.0 - 0.75 * bg_layer_decay)))
                background_map._sync_legacy_channels(cb)
                touched.add(bidx)
                scored.append(float(obs))
    return {
        "cmct_applied": 1.0,
        "cmct_cells": float(len(touched)),
        "cmct_mean_score": float(np.mean(scored)) if scored else 0.0,
    }

def cmct_conf(self, cell: VoxelCell3D) -> float:
    score = float(np.clip(getattr(cell, 'cmct_score', 0.0), 0.0, 1.0))
    active = float(np.clip(getattr(cell, 'cmct_active', 0.0), 0.0, 1.0))
    return float(np.clip(max(score, active), 0.0, 1.0))

