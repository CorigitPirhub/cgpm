from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `omhs`."""

def update_omhs_state(self, voxel_map: VoxelHashMap3D, cell) -> None:
    if not bool(self.cfg.surface.omhs_enable):
        cell.omhs_front_conf *= 0.96
        cell.omhs_rear_conf *= 0.96
        cell.omhs_gap *= 0.96
        cell.omhs_active *= 0.94
        return
    if not (bool(self.cfg.update.dual_state_enable) and bool(self.cfg.update.ptdsf_enable)):
        cell.omhs_front_conf *= 0.96
        cell.omhs_rear_conf *= 0.96
        cell.omhs_gap *= 0.96
        cell.omhs_active *= 0.94
        return

    ws = float(max(0.0, cell.phi_static_w))
    wt = float(max(0.0, cell.phi_transient_w))
    if ws <= 1e-8 or wt <= 1e-8:
        cell.omhs_front_conf *= 0.95
        cell.omhs_rear_conf *= 0.96
        cell.omhs_gap *= 0.95
        cell.omhs_active *= 0.92
        return

    stats = voxel_map._ptdsf_state_stats(cell) if hasattr(voxel_map, '_ptdsf_state_stats') else {
        'static_conf': 0.0,
        'transient_conf': 0.0,
        'dominance': 0.0,
        'rho_static': 0.0,
        'occ_frac': 0.0,
    }
    surf = float(max(1e-6, cell.surf_evidence))
    free = float(max(0.0, cell.free_evidence))
    occ_frac = float(np.clip(stats.get('occ_frac', surf / max(1e-6, surf + free)), 0.0, 1.0))
    dyn_n = float(np.clip(max(cell.dyn_prob, getattr(cell, 'z_dyn', 0.0)), 0.0, 1.0))
    trans_ratio = float(np.clip(wt / max(1e-6, ws + wt), 0.0, 1.0))
    rho_rear = float(np.clip(float(stats.get('rho_rear', 0.0)) / max(1e-6, self.cfg.update.dual_state_static_protect_rho), 0.0, 1.5))
    wg = float(max(0.0, cell.phi_geo_w))
    if wg > 1e-8:
        div_ref = float(max(1e-6, 2.0 * voxel_map.voxel_size))
        div = float(abs(float(cell.phi_geo) - float(cell.phi_static)))
        geo_agree = float(np.exp(-0.5 * (div / div_ref) * (div / div_ref)))
    else:
        geo_agree = 0.85

    gap = float(abs(float(cell.phi_static) - float(cell.phi_transient)))
    gap_ref = float(max(1.25 * voxel_map.voxel_size, 1e-3))
    gap_n = float(np.clip(gap / gap_ref, 0.0, 1.5))
    wod_front = float(np.clip(getattr(cell, 'wod_front_conf', 0.0), 0.0, 1.0))
    wod_rear = float(np.clip(getattr(cell, 'wod_rear_conf', 0.0), 0.0, 1.0))
    wod_shell = float(np.clip(getattr(cell, 'wod_shell_conf', 0.0), 0.0, 1.0))
    front_obs = float(
        np.clip(
            0.24 * float(stats.get('transient_conf', 0.0))
            + 0.18 * dyn_n
            + 0.18 * float(np.clip(getattr(cell, 'dccm_commit', 0.0), 0.0, 1.0))
            + 0.12 * float(np.clip(getattr(cell, 'dccm_rear', 0.0), 0.0, 1.0))
            + 0.10 * trans_ratio
            + 0.12 * wod_front
            + 0.06 * wod_shell,
            0.0,
            1.0,
        )
    )
    rear_obs = float(
        np.clip(
            0.24 * float(stats.get('static_conf', 0.0))
            + 0.22 * float(stats.get('dominance', 0.0))
            + 0.14 * occ_frac
            + 0.14 * min(1.0, rho_rear)
            + 0.12 * geo_agree
            + 0.14 * wod_rear,
            0.0,
            1.0,
        )
    )
    alpha = 0.18
    cell.omhs_front_conf = float(np.clip((1.0 - alpha) * cell.omhs_front_conf + alpha * front_obs, 0.0, 1.0))
    cell.omhs_rear_conf = float(np.clip((1.0 - alpha) * cell.omhs_rear_conf + alpha * rear_obs, 0.0, 1.0))
    cell.omhs_gap = float((1.0 - alpha) * cell.omhs_gap + alpha * gap)
    active_obs = 1.0 if (gap_n >= 0.75 and front_obs >= 0.30 and rear_obs >= 0.40) or (wod_front >= 0.32 and wod_rear >= 0.32) else 0.0
    cell.omhs_active = float(np.clip(0.82 * cell.omhs_active + 0.18 * active_obs, 0.0, 1.0))
    if rear_obs > 0.72 and front_obs < 0.18:
        cell.omhs_active *= 0.90

