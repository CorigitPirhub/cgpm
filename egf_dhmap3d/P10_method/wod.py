from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `wod`."""

def write_time_occlusion_split(
    self,
    voxel_map: VoxelHashMap3D,
    cell,
    measurement: AssocMeasurement3D,
    rel: np.ndarray,
    d_signed: float,
    q_dyn_obs: float,
    assoc_risk: float,
) -> tuple[float, float, float]:
    if not (bool(self.cfg.update.wod_enable) and bool(self.cfg.update.dual_state_enable)):
        return 0.0, 0.0, 0.0

    n = self._normalize(np.asarray(measurement.normal_world, dtype=float))
    view_axis, view_signed, _proj = self._measurement_view_components(measurement, rel, n)
    voxel = float(max(1e-6, voxel_map.voxel_size))
    front_margin = float(max(0.1 * voxel, self.cfg.update.wod_front_margin_vox * voxel))
    rear_margin = float(max(0.1 * voxel, self.cfg.update.wod_rear_margin_vox * voxel))
    shell_margin = float(max(0.6 * voxel, self.cfg.update.wod_shell_margin_vox * voxel))
    front_pos = self._sigmoid((-view_signed - front_margin) / max(1e-6, shell_margin))
    rear_pos = self._sigmoid((view_signed - rear_margin) / max(1e-6, shell_margin))
    shell_pos = float(np.exp(-0.5 * (view_signed / max(1e-6, shell_margin)) ** 2))

    rho_s_prev = float(max(0.0, getattr(cell, 'rho_static', 0.0)))
    rho_t_prev = float(max(0.0, getattr(cell, 'rho_transient', 0.0)))
    rho_r_prev = float(max(0.0, getattr(cell, 'rho_rear', 0.0)))
    split_ratio_prev = float(np.clip(rho_s_prev / max(1e-6, rho_s_prev + rho_t_prev), 0.0, 1.0)) if (rho_s_prev + rho_t_prev) > 1e-8 else float(np.clip(cell.p_static, 0.0, 1.0))
    rho_ref = float(max(1e-6, self.cfg.update.dual_state_static_protect_rho))
    rho_stat_n = float(np.clip(rho_s_prev / rho_ref, 0.0, 1.5))
    rho_rear_n = float(np.clip(rho_r_prev / max(1e-6, getattr(self.cfg.update, 'rps_rho_ref', rho_ref)), 0.0, 1.5))
    hist_mix = float(np.clip(self.cfg.update.wod_history_mix, 0.0, 1.0))
    rear_hist = float(
        np.clip(
            hist_mix * max(float(np.clip(cell.p_static, 0.0, 1.0)), split_ratio_prev)
            + (1.0 - hist_mix) * min(1.0, rho_stat_n)
            + 0.18 * min(1.0, rho_rear_n),
            0.0,
            1.0,
        )
    )
    front_hist = float(
        np.clip(
            0.45 * max(float(np.clip(getattr(cell, 'omhs_front_conf', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'wod_front_conf', 0.0), 0.0, 1.0)))
            + 0.35 * float(np.clip(rho_t_prev / max(1e-6, rho_s_prev + rho_t_prev), 0.0, 1.0))
            + 0.20 * float(np.clip(getattr(cell, 'dccm_commit', 0.0), 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    cons_ref = float(max(voxel, self.cfg.update.wod_rear_consistency_ref))
    rear_cons = 0.5
    if float(getattr(cell, 'phi_static_w', 0.0)) > 1e-8:
        rear_cons = float(np.exp(-0.5 * ((d_signed - float(cell.phi_static)) / cons_ref) ** 2))
    if float(getattr(cell, 'phi_geo_w', 0.0)) > 1e-8:
        geo_cons = float(np.exp(-0.5 * ((d_signed - float(cell.phi_geo)) / cons_ref) ** 2))
        rear_cons = float(0.55 * rear_cons + 0.45 * geo_cons)
    if float(getattr(cell, 'phi_rear_w', 0.0)) > 1e-8:
        rear_buf_cons = float(np.exp(-0.5 * ((d_signed - float(getattr(cell, 'phi_rear', 0.0))) / cons_ref) ** 2))
        rear_cons = float(0.60 * rear_cons + 0.40 * rear_buf_cons)
    trans_cons = 0.5
    if float(getattr(cell, 'phi_transient_w', 0.0)) > 1e-8:
        trans_cons = float(np.exp(-0.5 * ((d_signed - float(cell.phi_transient)) / cons_ref) ** 2))

    front_raw = float(np.clip(0.38 * front_pos + 0.24 * q_dyn_obs + 0.14 * assoc_risk + 0.14 * front_hist + 0.10 * trans_cons, 0.0, 1.5))
    front_raw *= float(np.clip(1.0 - 0.45 * rear_hist * rear_cons, 0.15, 1.0))
    rear_raw = float(
        np.clip(
            0.34 * rear_pos
            + 0.30 * rear_hist
            + 0.20 * rear_cons
            + 0.10 * max(float(np.clip(getattr(cell, 'omhs_rear_conf', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'wod_rear_conf', 0.0), 0.0, 1.0))),
            0.0,
            1.5,
        )
    )
    rear_raw *= float(np.clip(1.0 - 0.30 * q_dyn_obs, 0.20, 1.0))
    overlap = float(np.sqrt(max(0.0, front_raw) * max(0.0, rear_raw)))
    shell_raw = float(np.clip(shell_pos * (0.70 * overlap + 0.30 * assoc_risk), 0.0, 1.5))
    if rear_hist > 0.65 and rear_cons > 0.70:
        shell_raw = float(max(shell_raw, 0.35 * front_raw * shell_pos))

    raw_sum = float(front_raw + rear_raw + shell_raw)
    if raw_sum <= 1e-9:
        return 0.0, 0.0, 0.0
    return (float(front_raw / raw_sum), float(rear_raw / raw_sum), float(shell_raw / raw_sum))

