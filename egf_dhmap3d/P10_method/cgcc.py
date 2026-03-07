from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `cgcc`."""

def transfer_cross_map_geometric_corridor(
    self,
    background_map: VoxelHashMap3D,
    foreground_map: VoxelHashMap3D,
    accepted: List[AssocMeasurement3D],
) -> dict:
    if not bool(getattr(self.cfg.update, 'cgcc_enable', False)):
        return {
            "cgcc_applied": 0.0,
            "cgcc_cells": 0.0,
            "cgcc_mean_score": 0.0,
        }
    if not accepted:
        return {
            "cgcc_applied": 0.0,
            "cgcc_cells": 0.0,
            "cgcc_mean_score": 0.0,
        }

    on = float(np.clip(getattr(self.cfg.update, 'cgcc_conf_on', 0.42), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'cgcc_conf_off', 0.28), 0.0, 1.0))
    front_margin = float(max(0.0, getattr(self.cfg.update, 'cgcc_front_margin_vox', 0.35))) * background_map.voxel_size
    rear_margin = float(max(0.0, getattr(self.cfg.update, 'cgcc_rear_margin_vox', 1.40))) * background_map.voxel_size
    step = float(max(0.5 * background_map.voxel_size, getattr(self.cfg.update, 'cgcc_step_scale', 0.75) * background_map.voxel_size))
    lateral_radius = int(max(0, getattr(self.cfg.update, 'cgcc_lateral_radius_cells', 1)))
    bg_decay = float(np.clip(getattr(self.cfg.update, 'cgcc_bg_decay', 0.72), 0.0, 1.0))
    geo_decay = float(np.clip(getattr(self.cfg.update, 'cgcc_geo_decay', 0.58), 0.0, 1.0))
    rho_decay = float(np.clip(getattr(self.cfg.update, 'cgcc_rho_decay', 0.46), 0.0, 1.0))
    bg_layer_decay = float(np.clip(getattr(self.cfg.update, 'cgcc_bg_layer_decay', 0.22), 0.0, 1.0))
    static_guard = float(np.clip(getattr(self.cfg.update, 'cgcc_static_guard', 0.84), 0.0, 1.0))
    fg_weight_floor = float(np.clip(getattr(self.cfg.update, 'cgcc_fg_weight_floor', 0.18), 0.0, 1.0))
    rho_ref = float(max(1e-6, self.cfg.update.dual_state_static_protect_rho))

    scored: List[float] = []
    touched: Set[VoxelIndex] = set()
    for m in accepted:
        if m.sensor_origin is None:
            continue
        p = np.asarray(m.point_world, dtype=float).reshape(3)
        origin = np.asarray(m.sensor_origin, dtype=float).reshape(3)
        v = p - origin
        dist = float(np.linalg.norm(v))
        if dist <= 1e-8:
            continue
        d = v / dist
        fidx = foreground_map.world_to_index(p)
        cf = foreground_map.get_cell(fidx)
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
                    float(foreground_map._xmem_conf(cf)) if hasattr(foreground_map, '_xmem_conf') else 0.0,
                    float(foreground_map._otv_conf(cf)) if hasattr(foreground_map, '_otv_conf') else 0.0,
                    float(foreground_map._otv_surface_conf(cf)) if hasattr(foreground_map, '_otv_surface_conf') else 0.0,
                ),
                0.0,
                1.0,
            )
        )
        fg_w = float(max(getattr(cf, 'phi_transient_w', 0.0), getattr(cf, 'phi_dyn_w', 0.0), getattr(cf, 'phi_otv_w', 0.0)))
        fg_w_n = float(np.clip(fg_w / max(1e-6, fg_w + 1.5), 0.0, 1.0))
        fg_conf = float(np.clip((0.68 * dyn + 0.32 * split) * max(fg_weight_floor, 0.25 + 0.75 * fg_w_n), 0.0, 1.0))
        if fg_conf < off:
            continue

        s = -front_margin
        while s <= rear_margin + 1e-9:
            x = p + s * d
            bidx0 = background_map.world_to_index(x)
            for bidx in background_map.neighbor_indices(bidx0, lateral_radius):
                cb = background_map.get_cell(bidx)
                if cb is None:
                    continue
                center = background_map.index_to_center(bidx)
                rel = center - p
                axial = float(np.dot(rel, d))
                if axial < -front_margin - 1e-9 or axial > rear_margin + 1e-9:
                    continue
                lateral = float(np.linalg.norm(rel - axial * d))
                lateral_ref = float(max(1e-6, (lateral_radius + 0.5) * background_map.voxel_size))
                lateral_n = float(np.clip(1.0 - lateral / lateral_ref, 0.0, 1.0))
                if lateral_n <= 1e-6:
                    continue
                bg_support = float(np.clip(max(
                    float(np.clip(getattr(cb, 'p_static', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(cb, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
                    float(np.clip(getattr(cb, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
                    float(background_map._obl_conf(cb)) if hasattr(background_map, '_obl_conf') else 0.0,
                ), 0.0, 1.0))
                axial_phase = float(np.clip((axial + front_margin) / max(1e-6, front_margin + rear_margin), 0.0, 1.0))
                obs = float(np.clip(fg_conf * (0.55 + 0.45 * lateral_n) * (0.35 + 0.65 * axial_phase) * (1.0 - static_guard * bg_support), 0.0, 1.0))
                if obs <= 1e-6:
                    continue
                alpha_c = 0.18
                cb.cgcc_score = float(np.clip((1.0 - alpha_c) * float(getattr(cb, 'cgcc_score', 0.0)) + alpha_c * obs, 0.0, 1.0))
                if obs >= off:
                    cb.cgcc_age = float(min(20.0, float(getattr(cb, 'cgcc_age', 0.0)) + 1.0))
                else:
                    cb.cgcc_age = float(max(0.0, 0.96 * float(getattr(cb, 'cgcc_age', 0.0))))
                active_prev = float(np.clip(getattr(cb, 'cgcc_active', 0.0), 0.0, 1.0))
                active = bool((cb.cgcc_score >= on and cb.cgcc_age >= 1.0) or (active_prev >= 0.5 and cb.cgcc_score >= off))
                cb.cgcc_active = 1.0 if active else float(0.96 * active_prev)
                cgcc_n = float(max(cb.cgcc_score, cb.cgcc_active))
                if cgcc_n <= 1e-6:
                    continue
                strength = float(np.clip(cgcc_n * (1.0 - 0.65 * bg_support), 0.0, 1.0))
                if strength <= 1e-6:
                    continue
                cb.phi_static_w = float(max(0.0, cb.phi_static_w * (1.0 - bg_decay * strength)))
                cb.phi_geo_w = float(max(0.0, cb.phi_geo_w * (1.0 - geo_decay * strength)))
                cb.phi_w = float(max(0.0, cb.phi_w * (1.0 - 0.5 * geo_decay * strength)))
                cb.rho_static = float(max(0.0, getattr(cb, 'rho_static', 0.0) * (1.0 - rho_decay * strength)))
                cb.rho = float(max(0.0, cb.rho * (1.0 - 0.5 * rho_decay * strength)))
                cb.phi_bg_w = float(max(0.0, getattr(cb, 'phi_bg_w', 0.0) * (1.0 - bg_layer_decay * strength)))
                cb.rho_bg = float(max(0.0, getattr(cb, 'rho_bg', 0.0) * (1.0 - 0.75 * bg_layer_decay * strength)))
                background_map._sync_legacy_channels(cb)
                touched.add(bidx)
                scored.append(float(obs))
            s += step
    return {
        "cgcc_applied": 1.0,
        "cgcc_cells": float(len(touched)),
        "cgcc_mean_score": float(np.mean(scored)) if scored else 0.0,
    }

def cgcc_conf(self, cell: VoxelCell3D) -> float:
    score = float(np.clip(getattr(cell, 'cgcc_score', 0.0), 0.0, 1.0))
    active = float(np.clip(getattr(cell, 'cgcc_active', 0.0), 0.0, 1.0))
    return float(np.clip(max(score, active), 0.0, 1.0))

