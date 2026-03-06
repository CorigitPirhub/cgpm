from __future__ import annotations

from typing import Iterable, List, Set

import numpy as np

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.core.voxel_hash import VoxelHashMap3D, VoxelIndex
from egf_dhmap3d.modules.associator import AssocMeasurement3D


class Updater3D:
    def __init__(self, cfg: EGF3DConfig):
        self.cfg = cfg
        # Geometry-only global debias state (independent from dynamic suppression state).
        self.geo_bias_global = 0.0

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.zeros_like(v)
        return v / n

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        if weights.size != values.size:
            return float(np.median(values))
        order = np.argsort(values)
        v = values[order]
        w = np.clip(weights[order], 0.0, None)
        sw = float(np.sum(w))
        if sw <= 1e-12:
            return float(np.median(values))
        cdf = np.cumsum(w) / sw
        i = int(np.searchsorted(cdf, 0.5, side="left"))
        i = int(np.clip(i, 0, v.size - 1))
        return float(v[i])

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -20.0, 20.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def _update_omhs_state(self, voxel_map: VoxelHashMap3D, cell) -> None:
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

    def _sse_em_responsibility(
        self,
        cell,
        res_obs: float,
        assoc_risk: float,
    ) -> float:
        if not bool(self.cfg.update.sse_em_enable):
            return float(np.clip(cell.dyn_prob, 0.0, 1.0))
        sf = float(np.clip(self.cfg.update.sse_em_static_floor, 0.0, 0.49))
        dc = float(np.clip(self.cfg.update.sse_em_dynamic_ceil, 0.51, 1.0))
        if dc <= sf:
            sf, dc = 0.05, 0.95
        prior_dyn = float(np.clip(cell.resp_dynamic, sf, dc))
        temp = float(max(1e-3, self.cfg.update.sse_em_prior_temp))
        prior_logit = float(np.log(prior_dyn / max(1e-6, 1.0 - prior_dyn)) / temp)
        surf = float(max(1e-6, cell.surf_evidence))
        free = float(max(0.0, cell.free_evidence))
        free_ratio = float(free / surf)
        free_n = float(np.clip((free_ratio - 0.30) / 0.90, 0.0, 1.0))
        rho_ref = float(max(1e-6, self.cfg.update.dual_state_static_protect_rho))
        rho_n = float(np.clip(cell.rho / rho_ref, 0.0, 1.5))
        vis_n = float(np.clip(cell.visibility_contradiction, 0.0, 1.0))
        w_assoc = float(max(0.0, self.cfg.update.sse_em_assoc_weight))
        w_res = float(max(0.0, self.cfg.update.sse_em_residual_weight))
        w_free = float(max(0.0, self.cfg.update.sse_em_free_weight))
        w_rho = float(max(0.0, self.cfg.update.sse_em_rho_weight))
        w_vis = float(max(0.0, self.cfg.update.sse_em_visibility_weight))
        ws = max(1e-6, w_assoc + w_res + w_free + w_rho + w_vis)
        obs_dyn = float(
            np.clip(
                (w_assoc * assoc_risk + w_res * res_obs + w_free * free_n + w_rho * (1.0 - min(1.0, rho_n)) + w_vis * vis_n)
                / ws,
                0.0,
                1.0,
            )
        )
        logit = float(prior_logit + 2.4 * (obs_dyn - 0.5))
        r_dyn = float(np.clip(self._sigmoid(logit), sf, dc))
        alpha = float(np.clip(self.cfg.update.sse_em_mstep_alpha, 0.01, 0.8))
        cell.resp_dynamic = float(np.clip((1.0 - alpha) * cell.resp_dynamic + alpha * r_dyn, sf, dc))
        cell.resp_static = float(np.clip(1.0 - cell.resp_dynamic, 1.0 - dc, 1.0 - sf))
        return float(cell.resp_dynamic)

    def _lbr_observed_bias(self, measurement: AssocMeasurement3D, d_signed: float, res_obs: float) -> float:
        if not bool(self.cfg.update.lbr_enable):
            return 0.0
        if measurement.sensor_origin is None:
            return 0.0
        origin = np.asarray(measurement.sensor_origin, dtype=float).reshape(3)
        p = np.asarray(measurement.point_world, dtype=float).reshape(3)
        v = p - origin
        depth = float(np.linalg.norm(v))
        if depth < 1e-6:
            return 0.0
        view = v / depth
        n = self._normalize(np.asarray(measurement.normal_world, dtype=float))
        if float(np.linalg.norm(n)) < 1e-8:
            return 0.0
        cos_inc = float(np.clip(abs(np.dot(n, -view)), 0.0, 1.0))
        inc_pen = float(1.0 - cos_inc)
        depth_ref = float(max(1e-6, self.cfg.update.lbr_depth_ref))
        depth_n = float(np.clip(depth / depth_ref, 0.0, 2.0))
        wi = float(max(0.0, self.cfg.update.lbr_inc_weight))
        wd = float(max(0.0, self.cfg.update.lbr_depth_weight))
        wr = float(max(0.0, self.cfg.update.lbr_res_weight))
        ws = max(1e-6, wi + wd + wr)
        score = float(np.clip((wi * inc_pen + wd * depth_n + wr * res_obs) / ws, 0.0, 1.2))
        bmax = float(max(1e-6, self.cfg.update.lbr_max_bias))
        return float(np.clip(np.sign(d_signed) * min(abs(d_signed), bmax) * score, -bmax, bmax))

    def _estimate_dynamic_prob(
        self,
        voxel_map: VoxelHashMap3D,
        measurement: AssocMeasurement3D,
        pose_cov: np.ndarray | None,
    ) -> float:
        if not bool(self.cfg.update.dual_state_enable):
            return 0.0
        gate = float(max(1.0, self.cfg.assoc.gate_threshold))
        assoc = float(np.clip(measurement.d2 / gate, 0.0, 1.0))
        if measurement.seed:
            assoc = max(assoc, 0.65)
        ref = voxel_map.get_cell(measurement.source_index) or voxel_map.get_cell(measurement.voxel_index)
        if ref is None:
            free_n = 0.5
            res_n = 0.5 * assoc
            osc_n = 0.0
            contradiction_n = 0.0
            static_anchor = 0.0
        else:
            surf = float(max(1e-6, ref.surf_evidence))
            free = float(max(0.0, ref.free_evidence))
            free_ratio = float(free / surf)
            free_n = float(np.clip((free_ratio - 0.35) / 0.95, 0.0, 1.0))
            res_n = float(np.clip(ref.residual_evidence, 0.0, 1.0))
            osc_n = float(np.clip(ref.rho_osc / max(1e-6, self.cfg.update.rho_osc_ref), 0.0, 1.0))
            contradiction_n = float(
                np.clip(
                    max(
                        float(ref.stcg_score),
                        float(getattr(ref, "vcr_score", 0.0)),
                    ),
                    0.0,
                    1.0,
                )
            )
            static_anchor = 1.0 if (ref.rho >= self.cfg.update.dual_state_static_protect_rho and surf > self.cfg.update.dual_state_static_protect_ratio * free) else 0.0
        pose_n = 0.0
        if pose_cov is not None:
            try:
                cov = np.asarray(pose_cov, dtype=float)
                if cov.shape == (6, 6):
                    trans_var = float(np.trace(cov[:3, :3]))
                    pose_std = float(np.sqrt(max(0.0, trans_var)))
                    pose_n = float(np.clip(pose_std / max(1e-6, self.cfg.update.dual_pose_var_ref), 0.0, 1.0))
            except Exception:
                pose_n = 0.0
        w_assoc = float(max(0.0, self.cfg.update.dual_state_assoc_weight))
        w_free = float(max(0.0, self.cfg.update.dual_state_free_weight))
        w_res = float(max(0.0, self.cfg.update.dual_state_residual_weight))
        w_osc = float(max(0.0, self.cfg.update.dual_state_osc_weight))
        w_pose = float(max(0.0, self.cfg.update.dual_state_pose_weight))
        ws = max(1e-6, w_assoc + w_free + w_res + w_osc + w_pose)
        score = float(
            (w_assoc * assoc + w_free * free_n + w_res * max(res_n, contradiction_n) + w_osc * osc_n + w_pose * pose_n) / ws
        )
        if measurement.frontier:
            score = float(min(1.0, score + 0.08))
        if static_anchor > 0.5:
            score = float(max(0.0, score - 0.22))
        bias = float(np.clip(self.cfg.update.dual_state_bias, 0.0, 1.0))
        temp = float(max(1e-3, self.cfg.update.dual_state_temp))
        q = self._sigmoid((score - bias) / temp)
        if bool(self.cfg.update.sse_em_enable) and ref is not None:
            q = float(np.clip(0.60 * q + 0.40 * np.clip(ref.resp_dynamic, 0.0, 1.0), 0.0, 1.0))
        return float(np.clip(q, 0.0, 1.0))

    def _measurement_view_components(
        self,
        measurement: AssocMeasurement3D,
        rel: np.ndarray,
        n: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        if measurement.sensor_origin is not None:
            v = np.asarray(measurement.point_world, dtype=float).reshape(3) - np.asarray(measurement.sensor_origin, dtype=float).reshape(3)
            vn = float(np.linalg.norm(v))
            view_axis = v / vn if vn > 1e-8 else -n
        else:
            view_axis = -n
        if float(np.linalg.norm(view_axis)) < 1e-8:
            view_axis = -n
        if float(np.linalg.norm(view_axis)) < 1e-8:
            view_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        view_signed = float(np.dot(rel, view_axis))
        proj = float(np.clip(np.dot(view_axis, n), -1.0, 1.0))
        return view_axis, view_signed, proj

    def _write_time_dual_surface_targets(
        self,
        voxel_map: VoxelHashMap3D,
        measurement: AssocMeasurement3D,
        rel: np.ndarray,
        d_signed: float,
        wod_front: float,
        wod_rear: float,
        wod_shell: float,
        q_dyn_obs: float,
    ) -> tuple[float, float, float]:
        if not (bool(self.cfg.update.wdsg_enable) and bool(self.cfg.update.wod_enable) and bool(self.cfg.update.dual_state_enable)):
            return d_signed, d_signed, d_signed
        n = self._normalize(np.asarray(measurement.normal_world, dtype=float))
        if float(np.linalg.norm(n)) < 1e-8:
            return d_signed, d_signed, d_signed
        _view_axis, _view_signed, proj = self._measurement_view_components(measurement, rel, n)
        proj_floor = float(max(0.05, getattr(self.cfg.update, 'wdsg_proj_floor', 0.35)))
        if abs(proj) < proj_floor:
            proj_eff = (-proj_floor if proj <= 0.0 else proj_floor)
        else:
            proj_eff = proj
        voxel = float(max(1e-6, voxel_map.voxel_size))
        max_shift_vox = float(max(0.2, getattr(self.cfg.update, 'wdsg_max_shift_vox', 1.8)))
        front_delta_vox = float(np.clip(
            getattr(self.cfg.update, 'wdsg_front_shift_vox', 0.9) * wod_front
            + 0.5 * getattr(self.cfg.update, 'wdsg_shell_shift_vox', 0.4) * wod_shell
            + 0.15 * q_dyn_obs,
            0.0,
            max_shift_vox,
        ))
        rear_delta_vox = float(np.clip(
            getattr(self.cfg.update, 'wdsg_rear_shift_vox', 1.1) * wod_rear
            + 0.75 * getattr(self.cfg.update, 'wdsg_shell_shift_vox', 0.4) * wod_shell
            + 0.12 * (1.0 - float(np.clip(q_dyn_obs, 0.0, 1.0))),
            0.0,
            max_shift_vox,
        ))
        front_delta = float(voxel * front_delta_vox)
        rear_delta = float(voxel * rear_delta_vox)
        d_front = float(d_signed + front_delta * proj_eff)
        d_rear = float(d_signed - rear_delta * proj_eff)
        front_mix = float(np.clip(getattr(self.cfg.update, 'wdsg_front_mix_gain', 0.95) * (wod_front + 0.5 * wod_shell), 0.0, 0.95))
        rear_mix = float(np.clip(getattr(self.cfg.update, 'wdsg_rear_mix_gain', 1.0) * (wod_rear + 0.5 * wod_shell), 0.0, 0.95))
        d_static = float((1.0 - rear_mix) * d_signed + rear_mix * d_rear)
        d_transient = float((1.0 - front_mix) * d_signed + front_mix * d_front)
        return d_static, d_transient, d_rear

    def _write_time_occlusion_split(
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


    def _update_rps_state(
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


    def _update_spg_state(
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

    def _update_otv_state(
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

    def _integrate_measurement(
        self,
        voxel_map: VoxelHashMap3D,
        measurement: AssocMeasurement3D,
        touched: Set[VoxelIndex],
        frame_id: int,
    ) -> None:
        p = measurement.point_world
        n = self._normalize(measurement.normal_world)
        if float(np.linalg.norm(n)) < 1e-8:
            return

        trunc = float(self.cfg.map3d.truncation)
        trunc_scale = float(np.clip(self.cfg.update.integration_radius_scale, 0.20, 1.0))
        min_radius_vox = float(max(0.6, self.cfg.update.integration_min_radius_vox))
        min_trunc = float(min_radius_vox * voxel_map.voxel_size)
        rho_sigma = max(1e-4, float(self.cfg.update.rho_sigma))
        gate = float(self.cfg.assoc.gate_threshold)
        weight_assoc = np.exp(-0.5 * min(measurement.d2, gate))
        # Upstream contradiction-aware association risk attenuation.
        assoc_risk = float(np.clip(getattr(measurement, "assoc_risk", 0.0), 0.0, 1.0))
        if assoc_risk > 1e-8:
            weight_assoc *= float(np.clip(1.0 - 0.65 * assoc_risk, 0.12, 1.0))
        if measurement.seed:
            weight_assoc *= 0.55
            trunc_eff = max(min_trunc, 0.06)
            neighbor_iter = [measurement.voxel_index]
        else:
            trunc_eff = max(min_trunc, trunc * trunc_scale)
            neighbor_iter = voxel_map.neighbor_indices_for_point(p, radius_m=trunc_eff)
            # Geometry-only residual anchor:
            # persist signed association residual on source surface voxels.
            if bool(self.cfg.update.lzcd_enable):
                src = voxel_map.get_cell(measurement.source_index)
                if src is not None:
                    res_cap = float(max(1e-4, self.cfg.update.lzcd_residual_max_abs))
                    alpha_res = float(np.clip(self.cfg.update.lzcd_residual_alpha, 0.01, 0.8))
                    res = float(np.clip(measurement.phi, -res_cap, res_cap))
                    src.geo_res_ema = float((1.0 - alpha_res) * src.geo_res_ema + alpha_res * res)
                    src.geo_res_hits = float(min(200.0, src.geo_res_hits + 1.0))
        surf_band = float(np.clip(self.cfg.update.surf_band_ratio, 0.1, 0.9)) * trunc_eff
        if measurement.sensor_origin is not None:
            _view_vec = np.asarray(p, dtype=float).reshape(3) - np.asarray(measurement.sensor_origin, dtype=float).reshape(3)
            _view_norm = float(np.linalg.norm(_view_vec))
            view_axis = _view_vec / _view_norm if _view_norm > 1e-8 else -n
        else:
            view_axis = -n

        for nidx in neighbor_iter:
            center = voxel_map.index_to_center(nidx)
            rel = center - p
            d_signed = 0.0 if measurement.seed else float(np.dot(rel, n))
            if abs(d_signed) > trunc_eff:
                continue

            cell = voxel_map.get_or_create(nidx)
            dist2 = float(np.dot(rel, rel))
            rho_w = 0.0
            if self.cfg.update.enable_evidence:
                rho_w = float(np.exp(-0.5 * dist2 / (rho_sigma * rho_sigma)))
                cell.rho += float(rho_w)
            else:
                cell.rho = 1.0

            w_obs = float(np.exp(-0.5 * (d_signed / max(1e-6, trunc_eff)) ** 2) * weight_assoc)
            if w_obs <= 1e-12:
                continue
            # Residual-driven dynamic cue (shared by state updates and phi_dyn channel).
            d2_ref = float(max(1e-6, self.cfg.update.dyn_d2_ref))
            res_obs = float(np.clip(measurement.d2 / d2_ref, 0.0, 1.0))
            if measurement.seed:
                res_obs *= 0.5
            q_dyn_obs = float(np.clip(measurement.dynamic_prob, 0.0, 1.0))
            if bool(self.cfg.update.sse_em_enable):
                q_dyn_obs = self._sse_em_responsibility(cell, res_obs=res_obs, assoc_risk=assoc_risk)

            wod_front = 0.0
            wod_rear = 0.0
            wod_shell = 0.0
            w_shell = 0.0
            q_otv_route = 0.0
            otv_dyn_boost = 0.0
            if bool(self.cfg.update.dual_state_enable):
                q_dyn = float(np.clip(q_dyn_obs, 0.0, 1.0))
                min_static = float(np.clip(self.cfg.update.dual_state_min_static_ratio, 0.0, 0.5))
                static_prior = float(np.clip(1.0 - q_dyn, min_static, 1.0))
                if bool(self.cfg.update.ptdsf_enable):
                    age_ref = float(max(1.0, self.cfg.update.ptdsf_commit_age_ref))
                    commit_n = float(np.clip(cell.ptdsf_commit_age / age_ref, 0.0, 1.0))
                    rollback_n = float(np.clip(cell.ptdsf_rollback_age / age_ref, 0.0, 1.0))
                    rho_s_prev = float(max(0.0, getattr(cell, 'rho_static', 0.0)))
                    rho_t_prev = float(max(0.0, getattr(cell, 'rho_transient', 0.0)))
                    split_ratio_prev = float(np.clip(rho_s_prev / max(1e-6, rho_s_prev + rho_t_prev), 0.0, 1.0)) if (rho_s_prev + rho_t_prev) > 1e-8 else float(np.clip(cell.p_static, 0.0, 1.0))
                    hist_static = float(
                        np.clip(
                            0.42 * float(np.clip(cell.p_static, 0.0, 1.0))
                            + 0.28 * split_ratio_prev
                            + 0.18 * commit_n
                            + 0.12 * (1.0 - rollback_n),
                            0.0,
                            1.0,
                        )
                    )
                    static_ratio = float(np.clip(max(min_static, 0.60 * static_prior + 0.40 * hist_static), min_static, 1.0))
                else:
                    static_ratio = float(np.clip(max(min_static, static_prior), min_static, 1.0))
                transient_ratio = float(np.clip(1.0 - static_ratio, 0.0, 1.0))

                if bool(self.cfg.update.wod_enable):
                    wod_front, wod_rear, wod_shell = self._write_time_occlusion_split(
                        voxel_map,
                        cell,
                        measurement,
                        rel,
                        d_signed,
                        q_dyn_obs=q_dyn_obs,
                        assoc_risk=assoc_risk,
                    )
                    alpha_wod = float(np.clip(self.cfg.update.wod_alpha, 0.01, 0.8))
                    cell.wod_front_conf = float(np.clip((1.0 - alpha_wod) * cell.wod_front_conf + alpha_wod * wod_front, 0.0, 1.0))
                    cell.wod_rear_conf = float(np.clip((1.0 - alpha_wod) * cell.wod_rear_conf + alpha_wod * wod_rear, 0.0, 1.0))
                    cell.wod_shell_conf = float(np.clip((1.0 - alpha_wod) * cell.wod_shell_conf + alpha_wod * wod_shell, 0.0, 1.0))
                    static_ratio = float(np.clip(static_ratio * (1.0 - 0.55 * wod_front) + self.cfg.update.wod_rear_static_boost * wod_rear, min_static, 1.0))
                    transient_ratio = float(np.clip(transient_ratio * (1.0 - 0.35 * wod_rear) + self.cfg.update.wod_front_transient_boost * wod_front, 0.0, 1.0))
                    shell_ratio = float(np.clip(self.cfg.update.wod_shell_weight * wod_shell, 0.0, 0.75))
                    ratio_sum = float(max(1.0, static_ratio + transient_ratio + shell_ratio))
                    static_ratio /= ratio_sum
                    transient_ratio /= ratio_sum
                    shell_ratio /= ratio_sum
                    if self.cfg.update.enable_evidence and rho_w > 1e-12:
                        rho_mix = float(np.clip(1.0 - 0.28 * wod_shell - 0.10 * wod_front + 0.08 * wod_rear, 0.55, 1.10))
                        cell.rho = float(max(0.0, cell.rho + rho_w * (rho_mix - 1.0)))
                else:
                    shell_ratio = 0.0

                w_static = float(w_obs * static_ratio)
                w_transient = float(w_obs * transient_ratio)
                w_shell = float(w_obs * shell_ratio)
                static_mass = float(np.clip(w_static / max(1e-9, w_obs), 0.0, 1.0))
                transient_mass = float(np.clip(w_transient / max(1e-9, w_obs), 0.0, 1.0))
                shell_mass = float(np.clip(w_shell / max(1e-9, w_obs), 0.0, 1.0))
                d_static_obs = float(d_signed)
                d_transient_obs = float(d_signed)
                d_rear_obs = float(d_signed)
                if bool(self.cfg.update.wdsg_enable) and bool(self.cfg.update.wod_enable):
                    d_static_obs, d_transient_obs, d_rear_obs = self._write_time_dual_surface_targets(
                        voxel_map,
                        measurement,
                        rel,
                        d_signed,
                        wod_front=wod_front,
                        wod_rear=wod_rear,
                        wod_shell=wod_shell,
                        q_dyn_obs=q_dyn_obs,
                    )
                if bool(self.cfg.update.wdsg_route_enable) and bool(self.cfg.update.wdsg_enable) and bool(self.cfg.update.wod_enable):
                    sep_n = float(np.clip(abs(d_rear_obs - d_transient_obs) / max(1e-6, voxel_map.voxel_size), 0.0, 1.5))
                    front_dom = float(np.clip(wod_front + 0.5 * wod_shell + 0.25 * q_dyn_obs, 0.0, 1.0))
                    rear_dom = float(np.clip(wod_rear + 0.25 * (1.0 - float(np.clip(q_dyn_obs, 0.0, 1.0))), 0.0, 1.0))
                    static_keep = float(np.clip(1.0 - self.cfg.update.wdsg_route_static_suppress * front_dom * sep_n + self.cfg.update.wdsg_route_rear_recover * rear_dom * sep_n, 0.02, 1.10))
                    if bool(self.cfg.update.spg_enable):
                        static_floor = float(np.clip(self.cfg.update.spg_route_relax * (0.55 + 0.45 * rear_dom), 0.18, 0.75))
                        static_keep = max(static_keep, static_floor)
                    transient_boost = float(np.clip(1.0 + self.cfg.update.wdsg_route_transient_boost * front_dom * sep_n, 1.0, 1.80))
                    w_static *= static_keep
                    w_transient *= transient_boost
                    w_sum_route = float(w_static + w_transient + w_shell)
                    if w_sum_route > max(1e-9, w_obs):
                        route_scale = float(w_obs / w_sum_route)
                        w_static *= route_scale
                        w_transient *= route_scale
                        w_shell *= route_scale
                    static_mass = float(np.clip(w_static / max(1e-9, w_obs), 0.0, 1.0))
                    transient_mass = float(np.clip(w_transient / max(1e-9, w_obs), 0.0, 1.0))
                    shell_mass = float(np.clip(w_shell / max(1e-9, w_obs), 0.0, 1.0))

                if bool(self.cfg.update.otv_enable) and bool(self.cfg.update.wod_enable):
                    q_otv_route, otv_static_keep, otv_transient_boost, otv_dyn_boost = self._update_otv_state(
                        voxel_map,
                        cell,
                        w_obs=w_obs,
                        d_transient_obs=d_transient_obs,
                        d_rear_obs=d_rear_obs,
                        wod_front=wod_front,
                        wod_rear=wod_rear,
                        wod_shell=wod_shell,
                        q_dyn_obs=q_dyn_obs,
                        assoc_risk=assoc_risk,
                    )
                    if q_otv_route > 1e-6:
                        w_static *= otv_static_keep
                        w_transient *= otv_transient_boost
                        w_sum_otv = float(w_static + w_transient + w_shell)
                        if w_sum_otv > max(1e-9, w_obs):
                            otv_scale = float(w_obs / w_sum_otv)
                            w_static *= otv_scale
                            w_transient *= otv_scale
                            w_shell *= otv_scale
                        static_mass = float(np.clip(w_static / max(1e-9, w_obs), 0.0, 1.0))
                        transient_mass = float(np.clip(w_transient / max(1e-9, w_obs), 0.0, 1.0))
                        shell_mass = float(np.clip(w_shell / max(1e-9, w_obs), 0.0, 1.0))

                if bool(self.cfg.update.ptdsf_enable):
                    rho_alpha = float(np.clip(self.cfg.update.ptdsf_rho_alpha, 0.01, 1.0))
                    cell.rho_static = float(cell.rho_static + rho_alpha * w_static)
                    cell.rho_transient = float(cell.rho_transient + rho_alpha * (w_transient + w_shell))
                    if static_mass >= 0.65 and shell_mass <= 0.20:
                        cell.ptdsf_commit_age = float(min(20.0, cell.ptdsf_commit_age + 1.0))
                        cell.ptdsf_rollback_age = float(max(0.0, 0.75 * cell.ptdsf_rollback_age))
                    elif (transient_mass + 0.75 * shell_mass) >= 0.55 and wod_front >= wod_rear:
                        cell.ptdsf_rollback_age = float(min(20.0, cell.ptdsf_rollback_age + 1.0))
                        cell.ptdsf_commit_age = float(max(0.0, 0.75 * cell.ptdsf_commit_age))

                if w_static > 1e-12:
                    w_s_new = float(cell.phi_static_w + w_static)
                    if w_s_new > 1e-12:
                        cell.phi_static = float((cell.phi_static_w * cell.phi_static + w_static * d_static_obs) / w_s_new)
                        cell.phi_static_w = float(min(5000.0, w_s_new))
                if w_transient > 1e-12:
                    w_t_new = float(cell.phi_transient_w + w_transient)
                    if w_t_new > 1e-12:
                        cell.phi_transient = float((cell.phi_transient_w * cell.phi_transient + w_transient * d_transient_obs) / w_t_new)
                        cell.phi_transient_w = float(min(5000.0, w_t_new))

                voxel_map._sync_legacy_channels(cell)
            else:
                w_new = cell.phi_w + w_obs
                if w_new <= 1e-9:
                    continue
                cell.phi = float((cell.phi_w * cell.phi + w_obs * d_signed) / w_new)
                cell.phi_w = float(min(5000.0, w_new))
            # Geometry-only channel: integrate near-surface observations only,
            # decoupled from dynamic suppression / free-space contradiction updates.
            geo_band = float(max(0.6 * voxel_map.voxel_size, 0.35 * trunc_eff))
            if abs(d_signed) <= geo_band:
                d_geo = float(d_signed)
                if bool(self.cfg.update.lbr_enable):
                    b_obs = self._lbr_observed_bias(measurement, d_signed=d_signed, res_obs=res_obs)
                    alpha_b = float(np.clip(self.cfg.update.lbr_alpha, 0.01, 0.8))
                    cell.lbr_bias = float(np.clip((1.0 - alpha_b) * cell.lbr_bias + alpha_b * b_obs, -self.cfg.update.lbr_max_bias, self.cfg.update.lbr_max_bias))
                    cell.lbr_hits = float(min(200.0, cell.lbr_hits + 1.0))
                    d_geo = float(d_geo - cell.lbr_bias)
                d_geo_write = float(d_geo)
                d_rear_write = float(d_geo)
                geo_route_front = 0.0
                if bool(self.cfg.update.wdsg_enable) and bool(self.cfg.update.wod_enable) and bool(self.cfg.update.dual_state_enable):
                    d_geo_write, _d_trans_geo, d_rear_write = self._write_time_dual_surface_targets(
                        voxel_map,
                        measurement,
                        rel,
                        d_geo,
                        wod_front=wod_front,
                        wod_rear=wod_rear,
                        wod_shell=wod_shell,
                        q_dyn_obs=q_dyn_obs,
                    )
                    d_geo = float(d_geo_write)
                    geo_route_front = float(np.clip(wod_front + 0.5 * wod_shell + 0.25 * q_dyn_obs, 0.0, 1.0))
                w_geo = float(w_obs * np.exp(-0.5 * (d_signed / max(1e-6, geo_band)) ** 2))
                if measurement.seed:
                    w_geo *= 0.55
                if bool(self.cfg.update.dual_state_enable):
                    # Geometry channel is static-oriented under dual-state fusion.
                    q_geo_dyn = float(np.clip(q_dyn if 'q_dyn' in locals() else q_dyn_obs, 0.0, 1.0))
                    w_geo *= float(np.clip(1.0 - q_geo_dyn, 0.10, 1.0))
                    if bool(self.cfg.update.wod_enable):
                        geo_scale = float(
                            np.clip(
                                1.0
                                - self.cfg.update.wod_geo_front_suppress * (wod_front + 0.5 * wod_shell)
                                + self.cfg.update.wod_geo_rear_boost * wod_rear,
                                0.05,
                                1.25,
                            )
                        )
                        w_geo *= geo_scale
                    if bool(self.cfg.update.wdsg_route_enable) and bool(self.cfg.update.wdsg_enable) and bool(self.cfg.update.wod_enable):
                        sep_geo = float(np.clip(abs(d_rear_write - d_geo) / max(1e-6, voxel_map.voxel_size), 0.0, 1.5))
                        geo_keep = float(np.clip(1.0 - self.cfg.update.wdsg_route_geo_suppress * geo_route_front * sep_geo + self.cfg.update.wdsg_route_rear_recover * wod_rear * sep_geo, 0.05, 1.10))
                        if bool(self.cfg.update.spg_enable):
                            geo_floor = float(np.clip(self.cfg.update.spg_route_relax * (0.65 + 0.35 * wod_rear), 0.20, 0.80))
                            geo_keep = max(geo_keep, geo_floor)
                        w_geo *= geo_keep
                # ST-Mem upstream decoupling:
                # high short-term contradiction suppresses geometry-channel updates
                # without touching dynamic suppression states.
                st_mem_n = float(np.clip(cell.st_mem, 0.0, 1.0))
                if st_mem_n > 1e-6:
                    w_geo *= float(np.clip(1.0 - 0.75 * st_mem_n, 0.05, 1.0))
                if assoc_risk > 1e-8:
                    w_geo *= float(np.clip(1.0 - 0.80 * assoc_risk, 0.05, 1.0))
                if bool(self.cfg.update.otv_enable) and q_otv_route > 1e-6:
                    w_geo *= float(np.clip(1.0 - self.cfg.update.otv_geo_veto * q_otv_route, 0.02, 1.0))
                w_geo_new = cell.phi_geo_w + w_geo
                if w_geo_new > 1e-9:
                    cell.phi_geo = float((cell.phi_geo_w * cell.phi_geo + w_geo * d_geo) / w_geo_new)
                    cell.phi_geo_w = float(min(5000.0, w_geo_new))
                    if bool(self.cfg.update.lbr_enable):
                        gain = float(np.clip(self.cfg.update.lbr_apply_gain, 0.0, 1.0))
                        corr = float(np.clip(cell.lbr_bias, -self.cfg.update.lbr_max_bias, self.cfg.update.lbr_max_bias))
                        cell.phi_geo = float(np.clip(cell.phi_geo - gain * corr, -0.8, 0.8))
                self._update_rps_state(
                    voxel_map,
                    cell,
                    d_geo=d_rear_write,
                    w_obs=w_obs,
                    w_static=w_static,
                    w_geo=w_geo,
                    static_mass=static_mass,
                    wod_front=wod_front,
                    wod_rear=wod_rear,
                    wod_shell=wod_shell,
                    q_dyn_obs=q_dyn_obs,
                    assoc_risk=assoc_risk,
                )
                self._update_spg_state(
                    voxel_map,
                    cell,
                    w_static=w_static,
                    w_geo=w_geo,
                    static_mass=static_mass,
                    wod_front=wod_front,
                    wod_rear=wod_rear,
                    wod_shell=wod_shell,
                    q_dyn_obs=q_dyn_obs,
                    assoc_risk=assoc_risk,
                )
            # Dynamic-only channel:
            # accumulate transient/dynamic geometry independently from geometry channel.
            if bool(self.cfg.update.dyn_channel_enable):
                wq = float(np.clip(self.cfg.update.dyn_channel_obs_weight, 0.0, 1.0))
                wr = float(np.clip(self.cfg.update.dyn_channel_residual_weight, 0.0, 1.0))
                wk = float(np.clip(self.cfg.update.dyn_channel_risk_weight, 0.0, 1.0))
                ws_dyn = max(1e-6, wq + wr + wk)
                dyn_obs = float((wq * q_dyn_obs + wr * res_obs + wk * assoc_risk) / ws_dyn)
                if bool(self.cfg.update.dual_state_enable):
                    dyn_obs *= float(np.clip(1.0 - self.cfg.update.dyn_channel_static_suppress * np.clip(cell.p_static, 0.0, 1.0), 0.05, 1.0))
                    if bool(self.cfg.update.wod_enable):
                        dyn_obs = float(np.clip(max(dyn_obs, self.cfg.update.wod_dyn_front_boost * wod_front + self.cfg.update.wod_dyn_shell_boost * wod_shell), 0.0, 1.0))
                if bool(self.cfg.update.dccm_enable):
                    dyn_obs = float(np.clip(max(dyn_obs, 0.45 * float(np.clip(getattr(cell, 'dccm_commit', 0.0), 0.0, 1.0))), 0.0, 1.0))
                if bool(self.cfg.update.otv_enable) and q_otv_route > 1e-6:
                    dyn_obs = float(np.clip(max(dyn_obs, otv_dyn_boost), 0.0, 1.0))
                if bool(self.cfg.update.wdsg_route_enable) and bool(self.cfg.update.wdsg_enable) and bool(self.cfg.update.wod_enable):
                    sep_dyn = float(np.clip(abs(d_rear_obs - d_transient_obs) / max(1e-6, voxel_map.voxel_size), 0.0, 1.5))
                    dyn_obs = float(np.clip(dyn_obs + self.cfg.update.wdsg_route_dyn_boost * sep_dyn * float(np.clip(wod_front + 0.5 * wod_shell, 0.0, 1.0)), 0.0, 1.0))
                min_ratio = float(np.clip(self.cfg.update.dyn_channel_min_weight_ratio, 0.0, 0.6))
                w_dyn = float(w_obs * np.clip(min_ratio + (1.0 - min_ratio) * dyn_obs, min_ratio, 1.0))
                if bool(self.cfg.update.wod_enable):
                    w_dyn *= float(np.clip(0.30 + 0.70 * (wod_front + 0.50 * wod_shell), 0.10, 1.0))
                d_dyn_obs = float(d_signed)
                if bool(self.cfg.update.wdsg_enable) and bool(self.cfg.update.wod_enable) and bool(self.cfg.update.dual_state_enable):
                    _d_stat_dyn, d_dyn_obs, _d_rear_dyn = self._write_time_dual_surface_targets(
                        voxel_map,
                        measurement,
                        rel,
                        d_signed,
                        wod_front=wod_front,
                        wod_rear=wod_rear,
                        wod_shell=wod_shell,
                        q_dyn_obs=q_dyn_obs,
                    )
                if abs(d_signed) <= trunc_eff and w_dyn > 1e-12:
                    w_d_new = float(cell.phi_dyn_w + w_dyn)
                    if w_d_new > 1e-12:
                        cell.phi_dyn = float((cell.phi_dyn_w * cell.phi_dyn + w_dyn * d_dyn_obs) / w_d_new)
                        cell.phi_dyn_w = float(min(5000.0, w_d_new))
            # Update local dynamic evidence (surface vs free-space observation consistency).
            if self.cfg.update.enable_evidence and abs(d_signed) <= surf_band:
                shell_free = float(self.cfg.update.wod_shell_free_gain * w_shell) if bool(self.cfg.update.wod_enable) else 0.0
                surf_w = float(max(0.0, w_obs - shell_free))
                cell.surf_evidence += surf_w
                if shell_free > 1e-12:
                    cell.free_evidence += shell_free
                obs_sum = float(max(1e-9, surf_w + shell_free))
                occ_obs = float(np.clip(surf_w / obs_sum, 0.0, 1.0))
                free_obs = float(np.clip(shell_free / obs_sum, 0.0, 1.0))
                cell.occ_hit_ema = float(np.clip(0.92 * cell.occ_hit_ema + 0.08 * occ_obs, 0.0, 1.0))
                cell.free_hit_ema = float(np.clip(0.92 * cell.free_hit_ema + 0.08 * free_obs, 0.0, 1.0))
            else:
                if self.cfg.update.enable_evidence:
                    free_w = w_obs * float(np.clip(abs(d_signed) / max(1e-6, trunc_eff), 0.0, 1.0))
                    if bool(self.cfg.update.wod_enable) and w_shell > 1e-12:
                        free_w += float(self.cfg.update.wod_shell_free_gain * w_shell)
                    cell.free_evidence += free_w
                    cell.free_hit_ema = float(np.clip(0.92 * cell.free_hit_ema + 0.08 * 1.0, 0.0, 1.0))
                    cell.occ_hit_ema = float(np.clip(0.96 * cell.occ_hit_ema, 0.0, 1.0))
            cell.visibility_contradiction = float(np.clip(cell.free_hit_ema - cell.occ_hit_ema, 0.0, 1.0))
            # Residual-driven dynamic cue: high association inconsistency raises local dynamicity.
            res_alpha = float(np.clip(self.cfg.update.dyn_score_alpha, 0.01, 0.8))
            cell.residual_evidence = float((1.0 - res_alpha) * cell.residual_evidence + res_alpha * res_obs)
            if bool(self.cfg.update.rbi_enable):
                dec = float(np.clip(self.cfg.update.rbi_decay, 0.60, 0.999))
                cell.rbi_sum_phi = float(dec * cell.rbi_sum_phi + w_obs * d_signed)
                cell.rbi_sum_w = float(dec * cell.rbi_sum_w + w_obs)
                cell.rbi_dyn_ema = float(np.clip(dec * cell.rbi_dyn_ema + (1.0 - dec) * q_dyn_obs, 0.0, 1.0))

            if self.cfg.update.enable_gradient_fusion:
                c_dir = self._normalize(cell.c_rho)
                g_obs = self.cfg.update.eta_normal * n + self.cfg.update.eta_evidence * c_dir
                g_obs = self._normalize(g_obs)
                if float(np.linalg.norm(g_obs)) < 1e-8:
                    g_obs = n

                sigma = cell.g_cov
                mu = cell.g_mean
                r_eff = np.eye(3, dtype=float) * max(self.cfg.map3d.min_cov, measurement.sigma_n * measurement.sigma_n + 0.01)
                try:
                    k = sigma @ np.linalg.inv(sigma + r_eff)
                except np.linalg.LinAlgError:
                    k = np.zeros((3, 3), dtype=float)
                mu_new = mu + k @ (g_obs - mu)
                sigma_new = (np.eye(3, dtype=float) - k) @ sigma
                cell.g_mean = mu_new
                cell.g_cov = np.clip(sigma_new, self.cfg.map3d.min_cov, self.cfg.map3d.max_cov)
            else:
                # Keep a simple observed normal for output normals, without Bayesian gradient fusion.
                cell.g_mean = n
                cell.g_cov = np.eye(3, dtype=float) * self.cfg.map3d.max_cov
            cell.frontier_score *= 0.75
            cell.last_seen = int(frame_id)
            touched.add(nidx)

        # STCG contradiction shell:
        # For uncertain measurements, accumulate free-space contradiction
        # in a wider local shell (existing voxels only) to improve dynamic
        # discrimination without widening phi integration support.
        self._accumulate_contradiction_shell(
            voxel_map=voxel_map,
            measurement=measurement,
            touched=touched,
            trunc_eff=trunc_eff,
            surf_band=surf_band,
        )

    def _accumulate_contradiction_shell(
        self,
        voxel_map: VoxelHashMap3D,
        measurement: AssocMeasurement3D,
        touched: Set[VoxelIndex],
        trunc_eff: float,
        surf_band: float,
    ) -> None:
        if not bool(self.cfg.update.stcg_shell_enable):
            return
        if not bool(self.cfg.update.enable_evidence):
            return
        n = self._normalize(measurement.normal_world)
        if float(np.linalg.norm(n)) < 1e-8:
            return

        d2_ref = float(max(1e-6, self.cfg.update.dyn_d2_ref))
        res_n = float(np.clip(measurement.d2 / d2_ref, 0.0, 1.0))
        if res_n <= 0.12:
            return

        min_r = float(max(0.8 * voxel_map.voxel_size, self.cfg.update.stcg_shell_min_radius_vox * voxel_map.voxel_size))
        max_r = float(max(min_r, self.cfg.update.stcg_shell_max_radius_m))
        shell_r = float(
            np.clip(
                trunc_eff * (1.0 + self.cfg.update.stcg_shell_residual_gain * res_n),
                min_r,
                max_r,
            )
        )
        if shell_r <= surf_band * 1.02:
            return

        p = measurement.point_world
        band = float(max(1e-6, shell_r - surf_band))
        w_base = float(max(0.0, self.cfg.update.stcg_shell_free_weight))
        clear_boost = float(max(0.0, self.cfg.update.stcg_shell_clear_boost))
        for nidx in voxel_map.neighbor_indices_for_point(p, radius_m=shell_r):
            cell = voxel_map.get_cell(nidx)
            if cell is None:
                continue
            center = voxel_map.index_to_center(nidx)
            ad = abs(float(np.dot(center - p, n)))
            if ad <= surf_band * 1.02 or ad > shell_r:
                continue
            t = float(np.clip((ad - surf_band) / band, 0.0, 1.5))
            w_shell = float(w_base * res_n * np.exp(-1.8 * t * t))
            if w_shell <= 1e-8:
                continue
            cell.free_evidence += w_shell
            cell.clear_hits += clear_boost * w_shell
            cell.residual_evidence = float(np.clip(0.9 * cell.residual_evidence + 0.1 * res_n, 0.0, 1.0))
            # STCG-v2: shell contradiction also updates visibility history so
            # contradiction can rise before hard dynamic pruning is needed.
            cell.free_hit_ema = float(np.clip(0.90 * cell.free_hit_ema + 0.10 * 1.0, 0.0, 1.0))
            cell.occ_hit_ema = float(np.clip(0.97 * cell.occ_hit_ema, 0.0, 1.0))
            cell.visibility_contradiction = float(np.clip(cell.free_hit_ema - cell.occ_hit_ema, 0.0, 1.0))
            touched.add(nidx)

    def _refresh_evidence_gradient(self, voxel_map: VoxelHashMap3D, touched: Iterable[VoxelIndex]) -> None:
        vs = voxel_map.voxel_size
        touched_ext: Set[VoxelIndex] = set()
        for idx in touched:
            touched_ext.add(idx)
            touched_ext.update(voxel_map.neighbor_indices(idx, 1))

        for ix, iy, iz in touched_ext:
            c = voxel_map.get_cell((ix, iy, iz))
            if c is None:
                continue
            if self.cfg.update.enable_evidence:
                drdx = (voxel_map.get_rho((ix + 1, iy, iz)) - voxel_map.get_rho((ix - 1, iy, iz))) / (2.0 * vs)
                drdy = (voxel_map.get_rho((ix, iy + 1, iz)) - voxel_map.get_rho((ix, iy - 1, iz))) / (2.0 * vs)
                drdz = (voxel_map.get_rho((ix, iy, iz + 1)) - voxel_map.get_rho((ix, iy, iz - 1))) / (2.0 * vs)
                c.c_rho = np.array([drdx, drdy, drdz], dtype=float)
            else:
                c.c_rho = np.zeros(3, dtype=float)
                c.rho = 1.0
            # Per-voxel dynamic score from occupancy inconsistency and rho oscillation.
            rho_now = float(c.rho)
            rho_delta = abs(rho_now - float(c.rho_prev))
            c.rho_prev = rho_now
            c.rho_osc = 0.9 * c.rho_osc + 0.1 * rho_delta
            if bool(self.cfg.update.dccm_enable):
                alpha_dccm = float(np.clip(self.cfg.update.dccm_alpha, 0.01, 0.8))
                surf_hint = float(np.clip(c.surf_evidence / max(1e-6, c.surf_evidence + c.free_evidence), 0.0, 1.0))
                c.dccm_surface = float(np.clip((1.0 - alpha_dccm) * c.dccm_surface + alpha_dccm * surf_hint, 0.0, 1.0))
                c.dccm_age *= float(np.clip(self.cfg.update.dccm_age_decay, 0.70, 1.0))
                rho_n_dccm = float(np.clip(rho_now / max(1e-6, self.cfg.update.dual_state_static_protect_rho), 0.0, 1.0))
                score = (
                    float(self.cfg.update.dccm_free_weight) * float(np.clip(c.dccm_free, 0.0, 1.0))
                    + float(self.cfg.update.dccm_rear_weight) * float(np.clip(c.dccm_rear, 0.0, 1.0))
                    + float(self.cfg.update.dccm_age_weight) * float(np.clip(c.dccm_age, 0.0, self.cfg.update.ptdsf_commit_age_ref) / max(1e-6, self.cfg.update.ptdsf_commit_age_ref))
                    - float(self.cfg.update.dccm_surface_weight) * float(np.clip(c.dccm_surface, 0.0, 1.0))
                    - float(self.cfg.update.dccm_rho_weight) * rho_n_dccm
                )
                c.dccm_commit = float(np.clip((1.0 - alpha_dccm) * c.dccm_commit + alpha_dccm * self._sigmoid(2.6 * (score - 0.15)), 0.0, 1.0))
                if c.surf_evidence > 1.8 * c.free_evidence and rho_now > 0.9:
                    c.dccm_commit *= 0.88
            if self.cfg.update.enable_evidence:
                # Dynamic occupancy cue:
                # combine bounded occupancy fraction with an amplified free/surface ratio
                # so dynamic voxels can rise to high d_score bands.
                surf = float(max(1e-6, c.surf_evidence))
                free = float(max(0.0, c.free_evidence))
                free_ratio = free / surf
                dyn_occ_soft = free / max(1e-6, free + surf)
                dyn_occ_ratio = float(np.clip((free_ratio - 0.35) / 0.95, 0.0, 1.0))
                dyn_occ = max(dyn_occ_soft, dyn_occ_ratio)
                osc = float(np.clip(c.rho_osc / max(1e-6, self.cfg.update.rho_osc_ref), 0.0, 1.0))
            else:
                dyn_occ = 0.0
                osc = 0.0
            residual = float(np.clip(c.residual_evidence, 0.0, 1.0))
            frontier = float(np.clip(c.frontier_score / 3.0, 0.0, 1.0))
            clear = float(np.clip(c.clear_hits / 2.0, 0.0, 1.0))
            w_res = float(np.clip(self.cfg.update.residual_score_weight, 0.0, 0.5))
            w_occ = max(0.35, 0.65 - w_res)
            w_osc = max(0.10, 0.22 - 0.3 * w_res)
            w_clear = 0.05
            w_front = max(0.0, 1.0 - (w_occ + w_osc + w_res + w_clear))
            target = w_occ * dyn_occ + w_osc * osc + w_res * residual + w_clear * clear + w_front * frontier
            ema = float(np.clip(self.cfg.update.dscore_ema, 0.01, 0.5))
            c.d_score = float(np.clip((1.0 - ema) * c.d_score + ema * target, 0.0, 1.0))
            # Static-anchor suppression: persistent static support quickly damps dynamic score.
            if self.cfg.update.enable_evidence and c.surf_evidence > 1.8 * c.free_evidence and rho_now > 0.8:
                c.d_score *= 0.90
            elif c.surf_evidence > 1.3 * c.free_evidence:
                c.d_score *= 0.95

            # STCG: spatio-temporal contradiction accumulation.
            if bool(self.cfg.update.stcg_enable) and self.cfg.update.enable_evidence:
                surf = float(max(1e-6, c.surf_evidence))
                free = float(max(0.0, c.free_evidence))
                # Bounded conflict: high only when free and surface evidence co-exist.
                conflict = float(4.0 * free * surf / max(1e-6, (free + surf) * (free + surf)))
                free_ratio = float(free / surf)
                fr_ref = float(max(1e-6, self.cfg.update.stcg_free_ratio_ref))
                free_boost = float(np.clip(free_ratio / fr_ref, 0.0, 2.0))
                conflict = float(np.clip(conflict * (0.55 + 0.45 * free_boost), 0.0, 1.0))
                residual_n = float(np.clip(c.residual_evidence, 0.0, 1.0))
                osc_n = float(np.clip(c.rho_osc / max(1e-6, self.cfg.update.rho_osc_ref), 0.0, 1.0))
                vis_n = float(np.clip(c.visibility_contradiction, 0.0, 1.0))
                clear_n = float(np.clip(c.clear_hits / 2.0, 0.0, 1.0))
                wc = float(np.clip(self.cfg.update.stcg_conflict_weight, 0.0, 1.0))
                wr = float(np.clip(self.cfg.update.stcg_residual_weight, 0.0, 1.0))
                wo = float(np.clip(self.cfg.update.stcg_osc_weight, 0.0, 1.0))
                # STCG-v2: contradiction includes visibility and clear-history.
                wv = 0.18
                wcl = 0.12
                ws = max(1e-6, wc + wr + wo + wv + wcl)
                stcg_obs = float((wc * conflict + wr * residual_n + wo * osc_n + wv * vis_n + wcl * clear_n) / ws)
                if free_ratio >= 1.2 and vis_n >= 0.5:
                    stcg_obs = float(np.clip(stcg_obs + 0.08 * min(1.0, free_ratio / 2.0), 0.0, 1.0))
                stcg_alpha = float(np.clip(self.cfg.update.stcg_alpha, 0.01, 0.6))
                c.stcg_score = float(np.clip((1.0 - stcg_alpha) * c.stcg_score + stcg_alpha * stcg_obs, 0.0, 1.0))
                # Hysteresis gate to avoid rapid toggling in borderline regions.
                on_th = float(np.clip(self.cfg.update.stcg_on_thresh, 0.05, 0.95))
                off_th = float(np.clip(self.cfg.update.stcg_off_thresh, 0.01, 0.90))
                if off_th > on_th:
                    off_th = max(0.01, 0.85 * on_th)
                if c.stcg_active >= 0.5:
                    if c.stcg_score <= off_th:
                        c.stcg_active = 0.0
                    else:
                        c.stcg_active = 1.0
                else:
                    if c.stcg_score >= on_th:
                        c.stcg_active = 1.0
                # Static anchor suppression: stable high-rho support suppresses contradiction.
                static_anchor = bool(c.surf_evidence > 1.8 * c.free_evidence and rho_now > 0.9)
                if bool(self.cfg.update.dual_state_enable):
                    if c.p_static >= 0.72 and rho_now >= max(0.8, 0.9 * float(self.cfg.update.dual_state_static_protect_rho)):
                        static_anchor = True
                if static_anchor:
                    c.stcg_score *= 0.84
                    c.stcg_active *= 0.92
            else:
                c.stcg_score *= 0.98
                c.stcg_active *= 0.95

            # VCR: visibility-contradiction decomposition.
            if bool(self.cfg.update.vcr_enable) and self.cfg.update.enable_evidence:
                surf = float(max(1e-6, c.surf_evidence))
                free = float(max(0.0, c.free_evidence))
                free_ratio = float(free / surf)
                free_n = float(np.clip(free_ratio / max(1e-6, self.cfg.update.stcg_free_ratio_ref), 0.0, 1.5))
                surf_n = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
                occ_n = float(np.clip(c.occ_hit_ema, 0.0, 1.0))
                vis_n = float(np.clip(c.visibility_contradiction, 0.0, 1.0))
                res_n = float(np.clip(c.residual_evidence, 0.0, 1.0))
                alpha_v = float(np.clip(self.cfg.update.vcr_alpha, 0.01, 0.8))
                c.vcr_free = float(np.clip((1.0 - alpha_v) * c.vcr_free + alpha_v * free_n, 0.0, 1.5))
                c.vcr_surface = float(np.clip((1.0 - alpha_v) * c.vcr_surface + alpha_v * surf_n, 0.0, 1.0))
                c.vcr_occ = float(np.clip((1.0 - alpha_v) * c.vcr_occ + alpha_v * occ_n, 0.0, 1.0))

                wf = float(max(0.0, self.cfg.update.vcr_free_weight))
                wo = float(max(0.0, self.cfg.update.vcr_occ_weight))
                wr = float(max(0.0, self.cfg.update.vcr_res_weight))
                wv = float(max(0.0, self.cfg.update.vcr_vis_weight))
                ws_v = max(1e-6, wf + wo + wr + wv)
                v_obs = float(
                    np.clip(
                        (wf * np.clip(c.vcr_free, 0.0, 1.0) + wo * (1.0 - c.vcr_occ) + wr * res_n + wv * vis_n) / ws_v,
                        0.0,
                        1.0,
                    )
                )
                c.vcr_score = float(np.clip((1.0 - alpha_v) * c.vcr_score + alpha_v * v_obs, 0.0, 1.0))
                v_on = float(np.clip(self.cfg.update.vcr_on_thresh, 0.05, 0.95))
                v_off = float(np.clip(self.cfg.update.vcr_off_thresh, 0.01, 0.90))
                if v_off > v_on:
                    v_off = max(0.01, 0.85 * v_on)
                # Hysteresis by score damping around thresholds.
                if c.vcr_score >= v_on:
                    c.vcr_score = float(min(1.0, c.vcr_score + 0.05))
                elif c.vcr_score <= v_off:
                    c.vcr_score = float(max(0.0, c.vcr_score - 0.05))
                if surf > 1.8 * free and rho_now > 0.9:
                    c.vcr_score *= 0.90
            else:
                c.vcr_score *= 0.98

            # Structural decoupling dynamic state:
            # dynamic suppression state is maintained independently from geometry debias.
            if self.cfg.update.enable_evidence:
                surf = float(max(1e-6, c.surf_evidence))
                free = float(max(0.0, c.free_evidence))
                conflict_dyn = float(4.0 * free * surf / max(1e-6, (free + surf) * (free + surf)))
                vis_dyn = float(np.clip(c.visibility_contradiction, 0.0, 1.0))
                res_dyn = float(np.clip(c.residual_evidence, 0.0, 1.0))
                osc_dyn = float(np.clip(c.rho_osc / max(1e-6, self.cfg.update.rho_osc_ref), 0.0, 1.0))
                wc_dyn = float(np.clip(self.cfg.update.dyn_state_conflict_weight, 0.0, 1.0))
                wv_dyn = float(np.clip(self.cfg.update.dyn_state_visibility_weight, 0.0, 1.0))
                wr_dyn = float(np.clip(self.cfg.update.dyn_state_residual_weight, 0.0, 1.0))
                wo_dyn = float(np.clip(self.cfg.update.dyn_state_osc_weight, 0.0, 1.0))
                ws_dyn = max(1e-6, wc_dyn + wv_dyn + wr_dyn + wo_dyn)
                dyn_obs = float((wc_dyn * conflict_dyn + wv_dyn * vis_dyn + wr_dyn * res_dyn + wo_dyn * osc_dyn) / ws_dyn)
                if bool(self.cfg.update.vcr_enable):
                    dyn_obs = float(np.clip(max(dyn_obs, 0.65 * dyn_obs + 0.35 * np.clip(c.vcr_score, 0.0, 1.0)), 0.0, 1.0))
                if bool(self.cfg.update.dyn_channel_enable) and c.phi_dyn_w > 1e-8 and c.phi_geo_w > 1e-8:
                    div_ref = float(max(1e-6, self.cfg.update.dyn_channel_div_ref))
                    div_w = float(np.clip(self.cfg.update.dyn_channel_div_weight, 0.0, 1.0))
                    div_n = float(np.clip(abs(float(c.phi_dyn) - float(c.phi_geo)) / div_ref, 0.0, 1.5))
                    dyn_obs = float(np.clip((1.0 - div_w) * dyn_obs + div_w * max(dyn_obs, div_n), 0.0, 1.0))
                alpha_dyn = float(np.clip(self.cfg.update.dyn_state_alpha, 0.01, 0.8))
                c.dyn_prob = float(np.clip((1.0 - alpha_dyn) * c.dyn_prob + alpha_dyn * dyn_obs, 0.0, 1.0))
                if surf > 1.8 * free and rho_now > 0.85:
                    c.dyn_prob *= 0.90
                if bool(self.cfg.update.zdyn_enable):
                    free_ratio = float(free / surf)
                    free_ratio_ref = float(max(1e-6, self.cfg.update.zdyn_free_ratio_ref))
                    free_n = float(np.clip(free_ratio / free_ratio_ref, 0.0, 1.5))
                    wz_c = float(np.clip(self.cfg.update.zdyn_conflict_weight, 0.0, 1.0))
                    wz_v = float(np.clip(self.cfg.update.zdyn_visibility_weight, 0.0, 1.0))
                    wz_r = float(np.clip(self.cfg.update.zdyn_residual_weight, 0.0, 1.0))
                    wz_o = float(np.clip(self.cfg.update.zdyn_osc_weight, 0.0, 1.0))
                    wz_f = float(np.clip(self.cfg.update.zdyn_free_ratio_weight, 0.0, 1.0))
                    wz_s = max(1e-6, wz_c + wz_v + wz_r + wz_o + wz_f)
                    z_obs = float(
                        np.clip(
                            (wz_c * conflict_dyn + wz_v * vis_dyn + wz_r * res_dyn + wz_o * osc_dyn + wz_f * free_n) / wz_s,
                            0.0,
                            1.0,
                        )
                    )
                    alpha_up = float(np.clip(self.cfg.update.zdyn_alpha_up, 0.01, 0.95))
                    alpha_down = float(np.clip(self.cfg.update.zdyn_alpha_down, 0.01, 0.95))
                    alpha_z = alpha_up if z_obs >= c.z_dyn else alpha_down
                    c.z_dyn = float(np.clip((1.0 - alpha_z) * c.z_dyn + alpha_z * z_obs, 0.0, 1.0))
                    if surf > 1.8 * free and rho_now > 0.90:
                        c.z_dyn *= 0.88
            else:
                c.dyn_prob *= 0.98
                if bool(self.cfg.update.zdyn_enable):
                    c.z_dyn *= float(np.clip(self.cfg.update.zdyn_decay, 0.80, 1.0))

            if bool(self.cfg.update.dual_state_enable):
                surf = float(max(1e-6, c.surf_evidence))
                free = float(max(0.0, c.free_evidence))
                occ_frac = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
                contradiction = float(np.clip(c.stcg_score, 0.0, 1.0))
                vis_contra = float(np.clip(c.visibility_contradiction, 0.0, 1.0))
                residual_n = float(np.clip(c.residual_evidence, 0.0, 1.0))
                dyn_n = float(np.clip(c.dyn_prob, 0.0, 1.0))
                p_obs = float(np.clip(0.60 * occ_frac + 0.22 * (1.0 - contradiction) + 0.18 * (1.0 - residual_n), 0.0, 1.0))
                p_obs = float(np.clip(p_obs - 0.25 * vis_contra - 0.18 * dyn_n, 0.0, 1.0))
                if bool(self.cfg.update.ptdsf_enable):
                    rho_s = float(max(0.0, getattr(c, 'rho_static', 0.0)))
                    rho_t = float(max(0.0, getattr(c, 'rho_transient', 0.0)))
                    split_ratio = float(np.clip(rho_s / max(1e-6, rho_s + rho_t), 0.0, 1.0)) if (rho_s + rho_t) > 1e-8 else p_obs
                    age_ref = float(max(1.0, self.cfg.update.ptdsf_commit_age_ref))
                    commit_n = float(np.clip(c.ptdsf_commit_age / age_ref, 0.0, 1.0))
                    rollback_n = float(np.clip(c.ptdsf_rollback_age / age_ref, 0.0, 1.0))
                    persistent_obs = float(
                        np.clip(
                            0.34 * split_ratio
                            + 0.22 * occ_frac
                            + 0.20 * commit_n
                            + 0.14 * (1.0 - contradiction)
                            + 0.10 * (1.0 - dyn_n)
                            - 0.18 * rollback_n,
                            0.0,
                            1.0,
                        )
                    )
                    blend = float(np.clip(self.cfg.update.ptdsf_static_blend, 0.0, 1.0))
                    p_obs = float(np.clip((1.0 - blend) * p_obs + blend * persistent_obs, 0.0, 1.0))
                    p_obs = float(np.clip(p_obs + self.cfg.update.ptdsf_commit_bonus * commit_n, 0.0, 1.0))
                    p_obs = float(np.clip(p_obs - self.cfg.update.ptdsf_rollback_bonus * rollback_n, 0.0, 1.0))
                alpha_s = float(np.clip(self.cfg.update.dual_state_static_ema, 0.01, 0.8))
                c.p_static = float(np.clip((1.0 - alpha_s) * c.p_static + alpha_s * p_obs, 0.0, 1.0))

                static_rho = float(self.cfg.update.dual_state_static_protect_rho)
                static_ratio = float(self.cfg.update.dual_state_static_protect_ratio)
                if rho_now >= static_rho and c.surf_evidence > static_ratio * c.free_evidence:
                    c.p_static = float(max(c.p_static, 0.72))

                commit_th = float(np.clip(self.cfg.update.dual_state_commit_thresh, 0.0, 1.0))
                rollback_th = float(np.clip(self.cfg.update.dual_state_rollback_thresh, 0.0, 1.0))
                commit_gain = float(np.clip(self.cfg.update.dual_state_commit_gain, 0.0, 1.0))
                rollback_gain = float(np.clip(self.cfg.update.dual_state_rollback_gain, 0.0, 1.0))

                if c.p_static >= commit_th and c.phi_transient_w > 1e-6:
                    w_move = float(commit_gain * c.phi_transient_w)
                    if w_move > 1e-8:
                        w_new = float(c.phi_static_w + w_move)
                        c.phi_static = float((c.phi_static_w * c.phi_static + w_move * c.phi_transient) / max(1e-9, w_new))
                        c.phi_static_w = float(min(5000.0, w_new))
                        c.phi_transient_w = float(max(0.0, c.phi_transient_w - w_move))
                    rho_move = float(commit_gain * max(0.0, getattr(c, 'rho_transient', 0.0)))
                    if rho_move > 1e-8:
                        c.rho_static = float(max(0.0, getattr(c, 'rho_static', 0.0)) + rho_move)
                        c.rho_transient = float(max(0.0, getattr(c, 'rho_transient', 0.0) - rho_move))
                elif c.p_static <= rollback_th and c.phi_static_w > 1e-6:
                    w_move = float(rollback_gain * c.phi_static_w)
                    if w_move > 1e-8:
                        w_new = float(c.phi_transient_w + w_move)
                        c.phi_transient = float((c.phi_transient_w * c.phi_transient + w_move * c.phi_static) / max(1e-9, w_new))
                        c.phi_transient_w = float(min(5000.0, w_new))
                        c.phi_static_w = float(max(0.0, c.phi_static_w - w_move))
                    rho_move = float(rollback_gain * max(0.0, getattr(c, 'rho_static', 0.0)))
                    if rho_move > 1e-8:
                        c.rho_transient = float(max(0.0, getattr(c, 'rho_transient', 0.0)) + rho_move)
                        c.rho_static = float(max(0.0, getattr(c, 'rho_static', 0.0) - rho_move))

                voxel_map._sync_legacy_channels(c)
                self._update_omhs_state(voxel_map, c)

            # Short-term contradiction memory:
            # fast transient dynamic cue used only by suppression gate.
            if bool(self.cfg.update.stmem_enable) and self.cfg.update.enable_evidence:
                surf = float(max(1e-6, c.surf_evidence))
                free = float(max(0.0, c.free_evidence))
                conflict = float(4.0 * free * surf / max(1e-6, (free + surf) * (free + surf)))
                vis_n = float(np.clip(c.visibility_contradiction, 0.0, 1.0))
                clear_n = float(np.clip(c.clear_hits / 2.0, 0.0, 1.0))
                residual_n = float(np.clip(c.residual_evidence, 0.0, 1.0))
                free_ratio = float(free / surf)
                free_ratio_ref = float(max(1e-6, self.cfg.update.stmem_free_ratio_ref))
                free_boost = float(np.clip(free_ratio / free_ratio_ref, 0.0, 2.0))
                rho_n = float(np.clip(rho_now / max(1e-6, self.cfg.update.stmem_rho_ref), 0.0, 1.0))
                low_rho = float(1.0 - rho_n)

                wc = float(np.clip(self.cfg.update.stmem_conflict_weight, 0.0, 1.0))
                wv = float(np.clip(self.cfg.update.stmem_visibility_weight, 0.0, 1.0))
                wcl = float(np.clip(self.cfg.update.stmem_clear_weight, 0.0, 1.0))
                wr = float(np.clip(self.cfg.update.stmem_residual_weight, 0.0, 1.0))
                ws = max(1e-6, wc + wv + wcl + wr)
                st_obs = float((wc * conflict + wv * vis_n + wcl * clear_n + wr * residual_n) / ws)
                st_obs = float(np.clip(st_obs * (0.65 + 0.35 * low_rho) + 0.10 * np.clip(free_boost - 1.0, 0.0, 1.0), 0.0, 1.0))
                alpha = float(np.clip(self.cfg.update.stmem_alpha, 0.01, 0.8))
                c.st_mem = float(np.clip((1.0 - alpha) * c.st_mem + alpha * st_obs, 0.0, 1.0))
                if surf > 1.8 * free and rho_now > 0.9:
                    c.st_mem *= 0.90
            else:
                c.st_mem *= float(np.clip(self.cfg.update.stmem_decay, 0.70, 1.0))
            if bool(self.cfg.update.zdyn_enable):
                # Explicit dynamic latent drives suppression channel only.
                c.d_score = float(np.clip(max(c.z_dyn, 0.35 * c.st_mem), 0.0, 1.0))
            else:
                if bool(self.cfg.update.vcr_enable):
                    c.d_score = float(np.clip(max(c.dyn_prob, 0.45 * c.vcr_score), 0.0, 1.0))
                else:
                    c.d_score = float(np.clip(c.dyn_prob, 0.0, 1.0))

    def _local_zero_crossing_debias(
        self,
        voxel_map: VoxelHashMap3D,
        touched: Iterable[VoxelIndex],
        frame_id: int,
    ) -> None:
        if not bool(self.cfg.update.lzcd_enable):
            return
        interval = max(1, int(self.cfg.update.lzcd_interval))
        if (int(frame_id) % interval) != 0:
            return

        radius = max(1, int(self.cfg.update.lzcd_radius_cells))
        min_n = max(3, int(self.cfg.update.lzcd_min_neighbors))
        min_w = float(max(0.0, self.cfg.update.lzcd_min_phi_w))
        min_rho = float(max(0.0, self.cfg.update.lzcd_min_rho))
        max_d = float(np.clip(self.cfg.update.lzcd_max_dscore, 0.0, 1.0))
        cos_min = float(np.clip(self.cfg.update.lzcd_normal_cos_min, 0.0, 1.0))
        phi_gate = float(max(1e-4, self.cfg.update.lzcd_neighbor_phi_gate))
        bias_alpha = float(np.clip(self.cfg.update.lzcd_bias_alpha, 0.01, 0.8))
        corr_gain = float(np.clip(self.cfg.update.lzcd_correction_gain, 0.0, 1.0))
        max_bias = float(max(1e-4, self.cfg.update.lzcd_max_bias))
        max_step = float(max(1e-4, self.cfg.update.lzcd_max_step))
        trim_q = float(np.clip(self.cfg.update.lzcd_trim_quantile, 0.55, 1.0))
        use_geo = bool(self.cfg.update.lzcd_use_geo_channel)
        dual_enable = bool(self.cfg.update.dual_state_enable)
        rho_ref = float(max(1e-6, self.cfg.update.dual_state_static_protect_rho))
        global_alpha = float(np.clip(0.5 * bias_alpha, 0.02, 0.35))
        local_mix = 0.82
        global_mix = 0.18
        min_improve = 0.995
        solver_iters = int(max(1, self.cfg.update.lzcd_solver_iters))
        solver_lambda = float(max(0.0, self.cfg.update.lzcd_solver_lambda_smooth))
        solver_step = float(np.clip(self.cfg.update.lzcd_solver_step, 0.10, 1.0))
        solver_tol = float(max(1e-6, self.cfg.update.lzcd_solver_tol))
        residual_anchor_w = float(np.clip(self.cfg.update.lzcd_residual_anchor_weight, 0.0, 0.95))
        residual_hit_ref = float(max(1.0, self.cfg.update.lzcd_residual_hit_ref))
        affine_enable = bool(self.cfg.update.lzcd_affine_enable)
        affine_mix = float(np.clip(self.cfg.update.lzcd_affine_mix, 0.0, 1.0))
        affine_slope_min = float(min(self.cfg.update.lzcd_affine_slope_min, self.cfg.update.lzcd_affine_slope_max))
        affine_slope_max = float(max(self.cfg.update.lzcd_affine_slope_min, self.cfg.update.lzcd_affine_slope_max))
        affine_min_samples = int(max(4, self.cfg.update.lzcd_affine_min_samples))

        touched_ext: Set[VoxelIndex] = set()
        for idx in touched:
            touched_ext.add(idx)
            touched_ext.update(voxel_map.neighbor_indices(idx, radius))

        records = []
        global_vals: List[float] = []
        global_w: List[float] = []
        for idx in touched_ext:
            c = voxel_map.get_cell(idx)
            if c is None:
                continue
            if c.rho < min_rho:
                continue
            if float(np.clip(c.dyn_prob, 0.0, 1.0)) > max_d:
                continue
            if use_geo and c.phi_geo_w >= min_w:
                phi_i = float(c.phi_geo)
                w_i = float(c.phi_geo_w)
            elif dual_enable and c.phi_static_w >= min_w:
                phi_i = float(c.phi_static)
                w_i = float(c.phi_static_w)
            else:
                phi_i = float(c.phi)
                w_i = float(c.phi_w)
            if w_i < min_w:
                continue
            n_i = self._normalize(np.asarray(c.g_mean, dtype=float))
            if float(np.linalg.norm(n_i)) < 1e-8:
                continue
            center_i = voxel_map.index_to_center(idx)
            est_vals: List[float] = []
            est_w: List[float] = []
            est_delta: List[float] = []
            est_phi: List[float] = []
            for nidx in voxel_map.neighbor_indices(idx, radius):
                if nidx == idx:
                    continue
                cj = voxel_map.get_cell(nidx)
                if cj is None:
                    continue
                if use_geo and cj.phi_geo_w >= min_w:
                    phi_j = float(cj.phi_geo)
                    w_j = float(cj.phi_geo_w)
                elif dual_enable and cj.phi_static_w >= min_w:
                    phi_j = float(cj.phi_static)
                    w_j = float(cj.phi_static_w)
                else:
                    phi_j = float(cj.phi)
                    w_j = float(cj.phi_w)
                if w_j < min_w:
                    continue
                if abs(phi_j) > phi_gate:
                    continue
                n_j = self._normalize(np.asarray(cj.g_mean, dtype=float))
                if float(np.linalg.norm(n_j)) < 1e-8:
                    continue
                cos_ij = float(abs(np.dot(n_i, n_j)))
                if cos_ij < cos_min:
                    continue
                center_j = voxel_map.index_to_center(nidx)
                delta_ij = float(np.dot(n_i, center_j - center_i))
                phi_est = float(phi_j - delta_ij)
                # Confidence uses neighbor weight/rho and normal consistency.
                w = float(min(w_j, 5.0) * (0.2 + 0.8 * cos_ij) * (0.35 + 0.65 * np.clip(cj.rho, 0.0, 1.0)))
                est_vals.append(phi_est)
                est_w.append(w)
                est_delta.append(delta_ij)
                est_phi.append(phi_j)
            if len(est_vals) < min_n:
                continue
            vals = np.asarray(est_vals, dtype=float)
            ws = np.asarray(est_w, dtype=float)
            deltas = np.asarray(est_delta, dtype=float)
            phi_obs = np.asarray(est_phi, dtype=float)
            phi_ref0 = self._weighted_median(vals, ws)
            abs_res = np.abs(vals - phi_ref0)
            if trim_q < 0.999:
                thr = float(np.quantile(abs_res, trim_q))
                mask = abs_res <= max(1e-6, thr)
                if int(np.count_nonzero(mask)) >= min_n:
                    vals = vals[mask]
                    ws = ws[mask]
                    abs_res = abs_res[mask]
                    deltas = deltas[mask]
                    phi_obs = phi_obs[mask]
            # Huber-like attenuation for outliers.
            scale = float(max(1e-4, 1.4826 * np.median(abs_res)))
            huber = 1.0 / (1.0 + (abs_res / scale) ** 2)
            ws = ws * huber
            phi_ref_med = self._weighted_median(vals, ws)

            # LZCD affine local debias:
            # robustly fit phi ~= a * <n, delta_x> + b and use intercept b as
            # local zero-level estimate. This addresses local signed-distance bias
            # without coupling to dynamic suppression states.
            phi_ref = float(phi_ref_med)
            slope_n = 1.0
            if affine_enable and vals.size >= affine_min_samples:
                x = np.column_stack((deltas, np.ones_like(deltas)))
                y = phi_obs
                wfit = np.clip(ws, 1e-8, None)
                wx = x * wfit[:, None]
                # Mild slope prior around 1.0 to avoid degenerate fits in tiny neighborhoods.
                lam = 0.03
                a = wx.T @ x + np.array([[lam, 0.0], [0.0, 1e-6]], dtype=float)
                b = wx.T @ y + np.array([lam * 1.0, 0.0], dtype=float)
                try:
                    sol = np.linalg.solve(a, b)
                except np.linalg.LinAlgError:
                    sol = np.array([1.0, phi_ref_med], dtype=float)
                slope = float(np.clip(sol[0], affine_slope_min, affine_slope_max))
                intercept = float(np.clip(sol[1], -max_bias, max_bias))
                slope_dev = abs(slope - 1.0)
                slope_n = float(np.clip(1.0 - slope_dev / max(1e-6, (affine_slope_max - affine_slope_min)), 0.0, 1.0))
                # Static-confidence gated affine mixing:
                # reduce affine influence in dynamic / low-confidence cells to avoid
                # over-correction in high-motion regions (Bonn-like dynamics).
                dyn_n = float(np.clip(c.dyn_prob, 0.0, 1.0))
                rho_n_aff = float(np.clip(c.rho / rho_ref, 0.0, 1.2))
                static_conf = float(np.clip(0.55 * rho_n_aff + 0.45 * (1.0 - dyn_n), 0.15, 1.0))
                mix_eff = float(affine_mix * (0.35 + 0.65 * slope_n) * static_conf)
                phi_ref = float(np.clip((1.0 - mix_eff) * phi_ref_med + mix_eff * intercept, -max_bias, max_bias))
            bias_local = float(np.clip(phi_i - phi_ref, -max_bias, max_bias))
            # Residual anchor term (association signed residual history on source voxels).
            # This is geometry-only and does not consume dynamic scores.
            res_hits_n = float(np.clip(c.geo_res_hits / residual_hit_ref, 0.0, 1.0))
            res_anchor = float(np.clip(c.geo_res_ema, -max_bias, max_bias))
            wa = float(np.clip(residual_anchor_w * res_hits_n, 0.0, 0.95))
            bias_obs = float(np.clip((1.0 - wa) * bias_local + wa * res_anchor, -max_bias, max_bias))
            rho_n = float(np.clip(c.rho / rho_ref, 0.0, 1.2))
            w_n = float(np.clip(w_i / max(1e-6, 2.0 * min_w), 0.0, 1.2))
            geo_conf = float(np.clip((0.55 * w_n + 0.45 * rho_n) * (0.82 + 0.18 * slope_n), 0.25, 1.0))
            use_geo_cell = bool(use_geo and c.phi_geo_w >= min_w)
            dual_static_cell = bool(dual_enable and c.phi_static_w >= min_w)
            rec = {
                "idx": idx,
                "cell": c,
                "phi_i": phi_i,
                "vals": vals,
                "ws": ws,
                "bias_obs": bias_obs,
                "geo_conf": geo_conf,
                "use_geo_cell": use_geo_cell,
                "dual_static_cell": dual_static_cell,
                "n_i": n_i,
            }
            records.append(rec)
            global_vals.append(float(bias_obs))
            global_w.append(float(geo_conf * np.clip(np.sum(ws), 0.1, 20.0)))

        if not records:
            return

        # Runtime-aware candidate capping: keep highest-confidence geometry cells.
        max_candidates = int(max(0, self.cfg.update.lzcd_max_candidates))
        if max_candidates > 0 and len(records) > max_candidates:
            scores = np.asarray(
                [
                    float(
                        r["geo_conf"]
                        * np.clip(np.sum(r["ws"]), 0.2, 24.0)
                        * (0.85 + 0.15 * np.clip(r["cell"].geo_res_hits / residual_hit_ref, 0.0, 1.0))
                    )
                    for r in records
                ],
                dtype=float,
            )
            keep = np.argpartition(scores, -max_candidates)[-max_candidates:]
            keep = keep[np.argsort(scores[keep])[::-1]]
            records = [records[int(i)] for i in keep]

        idx_to_record = {r["idx"]: i for i, r in enumerate(records)}

        # Geometry-only convergence solver:
        # solve local debias on a voxel graph with data term + smoothness term.
        cur = np.asarray([float(r["bias_obs"]) for r in records], dtype=float)
        nxt = cur.copy()
        for _ in range(solver_iters):
            max_delta = 0.0
            for i, rec in enumerate(records):
                idx = rec["idx"]
                n_i = rec["n_i"]
                s_num = 0.0
                s_den = 0.0
                for nidx in voxel_map.neighbor_indices(idx, 1):
                    j = idx_to_record.get(nidx)
                    if j is None or j == i:
                        continue
                    n_j = records[j]["n_i"]
                    cos_ij = float(abs(np.dot(n_i, n_j)))
                    if cos_ij < cos_min:
                        continue
                    w_ij = float(0.25 + 0.75 * cos_ij)
                    s_num += w_ij * float(cur[j])
                    s_den += w_ij
                b_data = float(rec["bias_obs"])
                w_data = float(0.45 + 0.55 * rec["geo_conf"])
                b_smooth = b_data if s_den <= 1e-9 else float(s_num / s_den)
                target = float((w_data * b_data + solver_lambda * s_den * b_smooth) / max(1e-9, w_data + solver_lambda * s_den))
                b_new = float(np.clip(cur[i] + solver_step * (target - cur[i]), -max_bias, max_bias))
                nxt[i] = b_new
                max_delta = max(max_delta, abs(b_new - float(cur[i])))
            cur[:] = nxt
            if max_delta < solver_tol:
                break

        # Geometry-only global bias estimate; no dynamic terms involved.
        gb_obs = self._weighted_median(cur, np.asarray(global_w, dtype=float))
        self.geo_bias_global = float(
            np.clip((1.0 - global_alpha) * self.geo_bias_global + global_alpha * gb_obs, -max_bias, max_bias)
        )

        for i, rec in enumerate(records):
            c = rec["cell"]
            phi_i = float(rec["phi_i"])
            vals = rec["vals"]
            ws = rec["ws"]
            geo_conf = float(rec["geo_conf"])
            use_geo_cell = bool(rec["use_geo_cell"])
            dual_static_cell = bool(rec["dual_static_cell"])
            bias_obs = float(cur[i])
            if use_geo_cell:
                c.phi_geo_bias = float((1.0 - bias_alpha) * c.phi_geo_bias + bias_alpha * bias_obs)
                local_bias = float(c.phi_geo_bias)
            else:
                c.phi_bias = float((1.0 - bias_alpha) * c.phi_bias + bias_alpha * bias_obs)
                local_bias = float(c.phi_bias)

            bias_mix = float(np.clip(local_mix * local_bias + global_mix * self.geo_bias_global, -max_bias, max_bias))
            corr_eff = float(corr_gain * (0.70 + 0.30 * geo_conf))
            step_eff = float(max_step * (0.70 + 0.30 * geo_conf))
            corr_base = float(np.clip(corr_eff * bias_mix, -step_eff, step_eff))
            if abs(corr_base) <= 1e-7:
                continue

            # Trust-region backtracking using local residual energy.
            err0 = phi_i - vals
            e0 = float(np.sum(ws * np.abs(err0)))
            chosen_corr = 0.0
            for step_scale in (1.0, 0.5, 0.25):
                corr_try = float(step_scale * corr_base)
                phi_try = float(phi_i - corr_try)
                e1 = float(np.sum(ws * np.abs(phi_try - vals)))
                if e1 <= min_improve * e0:
                    chosen_corr = corr_try
                    break
            if abs(chosen_corr) <= 1e-9:
                continue

            if use_geo_cell:
                c.phi_geo = float(np.clip(c.phi_geo - chosen_corr, -0.8, 0.8))
                c.phi = float(np.clip(c.phi - 0.30 * chosen_corr, -0.8, 0.8))
            elif dual_static_cell:
                static_share = float(np.clip(0.45 + 0.25 * np.clip(c.p_static, 0.0, 1.0), 0.30, 0.78))
                c.phi_static = float(np.clip(c.phi_static - static_share * chosen_corr, -0.8, 0.8))
                c.phi_geo = float(np.clip(c.phi_geo - (0.55 + 0.20 * np.clip(c.p_static, 0.0, 1.0)) * chosen_corr, -0.8, 0.8))
                voxel_map._sync_legacy_channels(c)
            else:
                c.phi = float(np.clip(c.phi - chosen_corr, -0.8, 0.8))

    def _zero_crossing_bias_field(
        self,
        voxel_map: VoxelHashMap3D,
        touched: Iterable[VoxelIndex],
    ) -> None:
        if not bool(self.cfg.update.zcbf_enable):
            return
        block_size = int(max(1, self.cfg.update.zcbf_block_size_cells))
        min_rho = float(max(0.0, self.cfg.update.zcbf_min_rho))
        min_w = float(max(0.0, self.cfg.update.zcbf_min_phi_w))
        max_d = float(np.clip(self.cfg.update.zcbf_max_dscore, 0.0, 1.0))
        alpha = float(np.clip(self.cfg.update.zcbf_alpha, 0.01, 0.8))
        trim_q = float(np.clip(self.cfg.update.zcbf_trim_quantile, 0.55, 1.0))
        gain = float(np.clip(self.cfg.update.zcbf_apply_gain, 0.0, 1.0))
        max_bias = float(max(1e-5, self.cfg.update.zcbf_max_bias))

        touched_ext: Set[VoxelIndex] = set()
        for idx in touched:
            touched_ext.add(idx)
            touched_ext.update(voxel_map.neighbor_indices(idx, 1))

        blocks: dict[tuple[int, int, int], list[tuple[float, float, float, VoxelCell3D]]] = {}
        rho_ref_local = float(max(min_rho, self.cfg.update.zcbf_static_rho_ref))
        for idx in touched_ext:
            c = voxel_map.get_cell(idx)
            if c is None or c.rho < min_rho or float(np.clip(c.d_score, 0.0, 1.0)) > max_d:
                continue
            bias_obs = float(c.phi_geo_bias if c.phi_geo_w >= min_w else c.phi_bias)
            w_raw = float(c.phi_geo_w if c.phi_geo_w >= min_w else c.phi_static_w if c.phi_static_w >= min_w else c.phi_w)
            if w_raw < min_w:
                continue
            ptdsf_stats = voxel_map._ptdsf_state_stats(c) if hasattr(voxel_map, '_ptdsf_state_stats') else {"static_conf": 0.0, "dominance": 0.0, "rho_static": 0.0}
            persist_n = float(np.clip(0.55 * float(ptdsf_stats.get('static_conf', 0.0)) + 0.45 * float(ptdsf_stats.get('dominance', 0.0)), 0.0, 1.0))
            rho_n = float(np.clip(max(c.rho, float(ptdsf_stats.get('rho_static', 0.0))) / max(1e-6, rho_ref_local), 0.0, 1.5))
            obs_conf = float(np.clip(0.45 * persist_n + 0.35 * min(1.0, rho_n) + 0.20 * (1.0 - float(np.clip(c.d_score, 0.0, 1.0))), 0.05, 1.0))
            w_obs = float(w_raw * obs_conf)
            bx = (idx[0] // block_size, idx[1] // block_size, idx[2] // block_size)
            blocks.setdefault(bx, []).append((bias_obs, w_obs, obs_conf, c))

        if not blocks:
            return

        block_bias: dict[tuple[int, int, int], tuple[float, float]] = {}
        for bidx, vals in blocks.items():
            arr = np.asarray([v for v, _, _, _ in vals], dtype=float)
            ws = np.asarray([w for _, w, _, _ in vals], dtype=float)
            confs = np.asarray([cf for _, _, cf, _ in vals], dtype=float)
            if arr.size == 0:
                continue
            if arr.size >= 4:
                q = float(np.quantile(np.abs(arr), trim_q))
                keep = np.abs(arr) <= max(q, 1e-6)
                if np.any(keep):
                    arr = arr[keep]
                    ws = ws[keep]
                    confs = confs[keep]
            b = self._weighted_median(arr, ws)
            conf_mass = float(np.sum(ws) / max(1.0, 4.0 * min_w))
            conf_norm = float(np.mean(confs)) if confs.size else 0.0
            conf = float(np.clip(conf_mass * (0.35 + 0.65 * conf_norm), 0.0, 1.0))
            block_bias[bidx] = (float(np.clip(b, -max_bias, max_bias)), conf)

        # Confidence-normalized propagation:
        # keep high-confidence blocks anchored while letting nearby static-consistent
        # blocks absorb a smoothed bias field instead of a one-hop median only.
        smooth_bias = dict(block_bias)
        for _ in range(2):
            next_bias: dict[tuple[int, int, int], tuple[float, float]] = {}
            for bidx, (b0, c0) in smooth_bias.items():
                bx, by, bz = bidx
                num = 1.35 * max(1e-6, c0) * b0
                den = 1.35 * max(1e-6, c0)
                conf_acc = c0
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            nb = (bx + dx, by + dy, bz + dz)
                            if nb == bidx or nb not in smooth_bias:
                                continue
                            bj, cj = smooth_bias[nb]
                            dist = max(abs(dx), abs(dy), abs(dz))
                            base_w = 0.70 if dist == 1 else 0.40
                            compat = float(np.exp(-abs(float(bj) - float(b0)) / max(1e-6, 0.50 * max_bias)))
                            w_nb = float(base_w * max(1e-6, cj) * (0.35 + 0.65 * compat))
                            num += w_nb * bj
                            den += w_nb
                            conf_acc += 0.25 * cj
                b_hat = float(np.clip(num / max(1e-9, den), -max_bias, max_bias))
                c_hat = float(np.clip(conf_acc / max(1.0, 1.0 + 0.25 * 26.0), 0.0, 1.0))
                next_bias[bidx] = (b_hat, c_hat)
            smooth_bias = next_bias

        rho_ref = float(max(1e-6, self.cfg.update.zcbf_static_rho_ref))
        for idx in touched_ext:
            c = voxel_map.get_cell(idx)
            if c is None:
                continue
            bidx = (idx[0] // block_size, idx[1] // block_size, idx[2] // block_size)
            item = smooth_bias.get(bidx)
            if item is None:
                continue
            b_s, conf = item
            ptdsf_stats = voxel_map._ptdsf_state_stats(c) if hasattr(voxel_map, '_ptdsf_state_stats') else {"static_conf": 0.0, "dominance": 0.0}
            rho_n = float(np.clip(max(c.rho, getattr(c, 'rho_static', 0.0)) / rho_ref, 0.0, 1.0))
            persist_n = float(np.clip(0.55 * float(ptdsf_stats.get('static_conf', 0.0)) + 0.45 * float(ptdsf_stats.get('dominance', 0.0)), 0.0, 1.0))
            conf_eff = float(np.clip(conf * (0.35 + 0.35 * rho_n + 0.30 * persist_n), 0.0, 1.0))
            c.zcbf_bias = float(np.clip((1.0 - alpha) * c.zcbf_bias + alpha * b_s, -max_bias, max_bias))
            c.zcbf_bias_conf = float(np.clip((1.0 - alpha) * c.zcbf_bias_conf + alpha * conf_eff, 0.0, 1.0))
            corr = float(np.clip(gain * c.zcbf_bias * c.zcbf_bias_conf, -max_bias, max_bias))
            if abs(corr) <= 1e-7:
                continue
            if c.phi_geo_w > min_w:
                c.phi_geo = float(np.clip(c.phi_geo - corr, -0.8, 0.8))
            if bool(self.cfg.update.dual_state_enable) and c.phi_static_w > min_w:
                static_gain = float(np.clip(0.75 + 0.20 * persist_n, 0.75, 0.95))
                c.phi_static = float(np.clip(c.phi_static - static_gain * corr, -0.8, 0.8))
                voxel_map._sync_legacy_channels(c)

    def _raycast_clear(
        self,
        voxel_map: VoxelHashMap3D,
        accepted: List[AssocMeasurement3D],
        touched: Set[VoxelIndex],
        sensor_origin: np.ndarray | None,
    ) -> None:
        if sensor_origin is None:
            return
        gain = float(max(0.0, self.cfg.update.raycast_clear_gain))
        if gain <= 1e-9 or not accepted:
            return
        origin = np.asarray(sensor_origin, dtype=float).reshape(3)
        step = float(max(0.5 * voxel_map.voxel_size, self.cfg.update.raycast_step_scale * voxel_map.voxel_size))
        end_margin = float(max(step, self.cfg.update.raycast_end_margin))
        max_rays = max(1, int(self.cfg.update.raycast_max_rays))
        stride = max(1, len(accepted) // max_rays)
        rho_max = float(self.cfg.update.raycast_rho_max)
        phiw_max = float(self.cfg.update.raycast_phiw_max)
        dyn_boost = float(np.clip(self.cfg.update.raycast_dyn_boost, 0.0, 1.0))

        for m in accepted[::stride]:
            p = np.asarray(m.point_world, dtype=float).reshape(3)
            v = p - origin
            dist = float(np.linalg.norm(v))
            if dist <= (end_margin + step):
                continue
            d = v / max(1e-9, dist)
            # Endpoint confidence is used as an occlusion-aware attenuator:
            # high endpoint rho usually indicates a stable background surface.
            endpoint = voxel_map.get_cell(m.voxel_index)
            endpoint_rho = 0.0 if endpoint is None else float(endpoint.rho)
            endpoint_conf = float(np.clip(endpoint_rho / max(1e-6, rho_max), 0.0, 1.0))
            s = step
            while s < dist - end_margin:
                x = origin + s * d
                idx = voxel_map.world_to_index(x)
                c = voxel_map.get_cell(idx)
                if c is not None and (c.rho <= rho_max or c.phi_w <= phiw_max or c.dyn_prob >= 0.2):
                    # Free-space consistency gate: only clear strongly when the voxel has
                    # persistent free-space evidence across frames.
                    free_ratio = float(c.free_evidence / max(1e-6, c.surf_evidence))
                    consistent_free = bool((free_ratio >= 1.1 and c.free_evidence >= 0.25) or c.clear_hits >= 1.2)
                    dyn = float(np.clip(c.dyn_prob, 0.0, 1.0))
                    if bool(self.cfg.update.dccm_enable):
                        alpha_d = float(np.clip(self.cfg.update.dccm_alpha, 0.01, 0.8))
                        c.dccm_free = float(np.clip((1.0 - alpha_d) * c.dccm_free + alpha_d * 1.0, 0.0, 1.0))
                        c.dccm_rear = float(np.clip((1.0 - alpha_d) * c.dccm_rear + alpha_d * endpoint_conf, 0.0, 1.0))
                        age_gain = float(max(0.0, self.cfg.update.dccm_age_gain))
                        c.dccm_age = float(min(20.0, c.dccm_age + age_gain * (1.0 if consistent_free else 0.5)))
                        touched.add(idx)
                        surf_n = float(np.clip(c.surf_evidence / max(1e-6, c.surf_evidence + c.free_evidence), 0.0, 1.0))
                        rho_n = float(np.clip(max(c.rho, getattr(c, 'rho_static', 0.0)) / max(1e-6, self.cfg.update.dual_state_static_protect_rho), 0.0, 1.0))
                        age_n = float(np.clip(c.dccm_age / max(1e-6, self.cfg.update.ptdsf_commit_age_ref), 0.0, 1.0))
                        score = (
                            float(self.cfg.update.dccm_free_weight) * float(np.clip(c.dccm_free, 0.0, 1.0))
                            + float(self.cfg.update.dccm_rear_weight) * float(np.clip(c.dccm_rear, 0.0, 1.0))
                            + float(self.cfg.update.dccm_age_weight) * age_n
                            - float(self.cfg.update.dccm_surface_weight) * surf_n
                            - float(self.cfg.update.dccm_rho_weight) * rho_n
                        )
                        c.dccm_commit = float(np.clip((1.0 - alpha_d) * c.dccm_commit + alpha_d * self._sigmoid(2.6 * (score - 0.15)), 0.0, 1.0))
                    if (not consistent_free) and dyn < 0.35 and float(getattr(c, 'dccm_commit', 0.0)) < self.cfg.update.dccm_commit_thresh:
                        s += step
                        continue

                    # Occlusion-aware attenuation:
                    # if endpoint is very confident (likely a background wall), reduce
                    # along-ray clearing to avoid over-erasing temporarily occluded regions.
                    occ_att = float(1.0 - 0.55 * endpoint_conf)
                    if endpoint_conf > 0.6 and c.surf_evidence > c.free_evidence:
                        occ_att *= 0.6

                    clear_base = gain * (0.45 + 0.55 * dyn)
                    if bool(self.cfg.update.dccm_enable):
                        commit_n = float(np.clip(getattr(c, 'dccm_commit', 0.0), 0.0, 1.0))
                        clear_base *= float(0.35 + 0.65 * commit_n)
                    if consistent_free:
                        clear = clear_base * 1.15 * occ_att
                    else:
                        clear = clear_base * 0.55 * occ_att
                    clear = float(np.clip(clear, 0.0, 0.85))
                    if clear <= 1e-4:
                        s += step
                        continue

                    # Soft-decay with floors, never hard-delete to preserve recoverability.
                    rho_floor = float(0.02 + 0.04 * endpoint_conf)
                    phiw_floor = float(0.02 + 0.06 * endpoint_conf)
                    if bool(self.cfg.update.dual_state_enable):
                        # DCCM only acts on transient/competitive branch.
                        trans_clear = float(np.clip(clear * (0.75 + 0.25 * float(np.clip(getattr(c, 'dccm_commit', 0.0), 0.0, 1.0))), 0.0, 0.90))
                        c.phi_transient_w = max(phiw_floor, c.phi_transient_w * (1.0 - trans_clear))
                        c.phi_dyn_w = max(phiw_floor, c.phi_dyn_w * (1.0 - 0.85 * trans_clear))
                        if hasattr(c, 'rho_transient'):
                            c.rho_transient = max(rho_floor, c.rho_transient * (1.0 - trans_clear))
                        if hasattr(c, 'rho_static'):
                            c.rho_static = max(0.0, c.rho_static * (1.0 - 0.08 * clear * max(0.0, 1.0 - endpoint_conf)))
                        voxel_map._sync_legacy_channels(c)
                        c.rho = max(rho_floor, max(float(getattr(c, 'rho_static', 0.0)), 0.0) + 0.20 * max(float(getattr(c, 'rho_transient', 0.0)), 0.0))
                    else:
                        c.phi_w = max(phiw_floor, c.phi_w * (1.0 - clear))
                        c.rho = max(rho_floor, c.rho * (1.0 - 0.75 * clear))
                    c.free_evidence += 0.8 * clear
                    c.free_hit_ema = float(np.clip(0.92 * c.free_hit_ema + 0.08 * 1.0, 0.0, 1.0))
                    c.occ_hit_ema = float(np.clip(0.96 * c.occ_hit_ema, 0.0, 1.0))
                    c.visibility_contradiction = float(np.clip(c.free_hit_ema - c.occ_hit_ema, 0.0, 1.0))
                    c.residual_evidence = float(np.clip(c.residual_evidence + 0.6 * clear, 0.0, 1.0))
                    c.dyn_prob = float(np.clip(c.dyn_prob + dyn_boost * clear, 0.0, 1.0))
                    c.d_score = float(np.clip(c.dyn_prob, 0.0, 1.0))
                    c.clear_hits += 1.0
                    touched.add(idx)
                s += step

    def _apply_rbi_reintegration(self, voxel_map: VoxelHashMap3D, touched: Iterable[VoxelIndex]) -> None:
        if not bool(self.cfg.update.rbi_enable):
            return
        min_w = float(max(1e-6, self.cfg.update.rbi_min_weight))
        static_p = float(np.clip(self.cfg.update.rbi_commit_static_p, 0.0, 1.0))
        dyn_gate = float(np.clip(self.cfg.update.rbi_dyn_gate, 0.0, 1.0))
        gain = float(np.clip(self.cfg.update.rbi_recover_gain, 0.0, 1.0))
        step_max = float(max(1e-4, self.cfg.update.rbi_max_step))
        dual_enable = bool(self.cfg.update.dual_state_enable)

        touched_ext: Set[VoxelIndex] = set()
        for idx in touched:
            touched_ext.add(idx)
            touched_ext.update(voxel_map.neighbor_indices(idx, 1))

        for idx in touched_ext:
            c = voxel_map.get_cell(idx)
            if c is None:
                continue
            if float(c.rbi_sum_w) < min_w:
                continue
            if float(c.rbi_dyn_ema) > dyn_gate:
                continue
            if dual_enable and float(np.clip(c.p_static, 0.0, 1.0)) < static_p:
                continue
            phi_buf = float(c.rbi_sum_phi / max(1e-6, c.rbi_sum_w))
            if dual_enable:
                target = float(c.phi_static)
                delta_raw = float(phi_buf - target)
                delta = float(np.clip(gain * delta_raw, -step_max, step_max))
                if abs(delta) <= 1e-9:
                    continue
                c.phi_static = float(np.clip(c.phi_static + delta, -0.8, 0.8))
                c.phi_geo = float(np.clip(c.phi_geo + 0.55 * delta, -0.8, 0.8))
                c.phi_static_w = float(min(5000.0, c.phi_static_w + 0.25 * min_w))
                voxel_map._sync_legacy_channels(c)
            else:
                delta_raw = float(phi_buf - c.phi)
                delta = float(np.clip(gain * delta_raw, -step_max, step_max))
                if abs(delta) <= 1e-9:
                    continue
                c.phi = float(np.clip(c.phi + delta, -0.8, 0.8))
                c.phi_w = float(min(5000.0, c.phi_w + 0.25 * min_w))
            # Consume a fraction of buffer after commit to avoid double counting.
            c.rbi_sum_phi *= 0.82
            c.rbi_sum_w *= 0.82

    def _phi_gradient(self, voxel_map: VoxelHashMap3D, idx: VoxelIndex) -> np.ndarray:
        vs = voxel_map.voxel_size
        ix, iy, iz = idx
        gx = (voxel_map.get_phi((ix + 1, iy, iz)) - voxel_map.get_phi((ix - 1, iy, iz))) / (2.0 * vs)
        gy = (voxel_map.get_phi((ix, iy + 1, iz)) - voxel_map.get_phi((ix, iy - 1, iz))) / (2.0 * vs)
        gz = (voxel_map.get_phi((ix, iy, iz + 1)) - voxel_map.get_phi((ix, iy, iz - 1))) / (2.0 * vs)
        return np.array([gx, gy, gz], dtype=float)

    def _g_divergence(self, voxel_map: VoxelHashMap3D, idx: VoxelIndex) -> float:
        vs = voxel_map.voxel_size
        ix, iy, iz = idx
        gx_p = voxel_map.get_cell((ix + 1, iy, iz))
        gx_n = voxel_map.get_cell((ix - 1, iy, iz))
        gy_p = voxel_map.get_cell((ix, iy + 1, iz))
        gy_n = voxel_map.get_cell((ix, iy - 1, iz))
        gz_p = voxel_map.get_cell((ix, iy, iz + 1))
        gz_n = voxel_map.get_cell((ix, iy, iz - 1))
        dgx = ((gx_p.g_mean[0] if gx_p is not None else 0.0) - (gx_n.g_mean[0] if gx_n is not None else 0.0)) / (2.0 * vs)
        dgy = ((gy_p.g_mean[1] if gy_p is not None else 0.0) - (gy_n.g_mean[1] if gy_n is not None else 0.0)) / (2.0 * vs)
        dgz = ((gz_p.g_mean[2] if gz_p is not None else 0.0) - (gz_n.g_mean[2] if gz_n is not None else 0.0)) / (2.0 * vs)
        return float(dgx + dgy + dgz)

    def _laplacian_phi(self, voxel_map: VoxelHashMap3D, idx: VoxelIndex) -> float:
        vs2 = voxel_map.voxel_size * voxel_map.voxel_size
        ix, iy, iz = idx
        c = voxel_map.get_phi((ix, iy, iz))
        lap = (
            voxel_map.get_phi((ix + 1, iy, iz))
            + voxel_map.get_phi((ix - 1, iy, iz))
            + voxel_map.get_phi((ix, iy + 1, iz))
            + voxel_map.get_phi((ix, iy - 1, iz))
            + voxel_map.get_phi((ix, iy, iz + 1))
            + voxel_map.get_phi((ix, iy, iz - 1))
            - 6.0 * c
        ) / max(1e-9, vs2)
        return float(lap)

    def _poisson_refine(self, voxel_map: VoxelHashMap3D, touched: Iterable[VoxelIndex]) -> None:
        iters = int(max(0, self.cfg.update.poisson_iters))
        if iters <= 0:
            return
        touched_list = list(set(touched))
        if not touched_list:
            return

        lr = float(self.cfg.update.poisson_lr)
        eik = float(self.cfg.update.eikonal_lambda)
        for _ in range(iters):
            phi_updates = {}
            for idx in touched_list:
                cell = voxel_map.get_cell(idx)
                if cell is None or cell.phi_w <= 1e-6:
                    continue
                div_g = self._g_divergence(voxel_map, idx)
                lap = self._laplacian_phi(voxel_map, idx)
                grad_phi = self._phi_gradient(voxel_map, idx)
                grad_norm = float(np.linalg.norm(grad_phi))
                eik_term = (grad_norm - 1.0)
                dphi = lr * (div_g - lap) - 0.2 * lr * eik * eik_term
                phi_updates[idx] = float(np.clip(cell.phi + dphi, -0.8, 0.8))
            for idx, val in phi_updates.items():
                c = voxel_map.get_cell(idx)
                if c is not None:
                    if bool(self.cfg.update.dual_state_enable):
                        c.phi_static = float(val)
                        voxel_map._sync_legacy_channels(c)
                    else:
                        c.phi = val

    def update(
        self,
        voxel_map: VoxelHashMap3D,
        accepted: List[AssocMeasurement3D],
        rejected: List[AssocMeasurement3D],
        sensor_origin: np.ndarray | None = None,
        frame_id: int = 0,
        pose_cov: np.ndarray | None = None,
    ) -> dict:
        touched: Set[VoxelIndex] = set()
        for m in accepted:
            m.dynamic_prob = self._estimate_dynamic_prob(voxel_map, m, pose_cov=pose_cov)
            self._integrate_measurement(voxel_map, m, touched, frame_id=frame_id)
        self._raycast_clear(voxel_map, accepted, touched, sensor_origin)
        # Frontier activation: unmatched points become growth hints for next frames.
        frontier_boost = float(max(0.0, self.cfg.update.frontier_boost))
        for m in rejected:
            for nidx in voxel_map.neighbor_indices(m.voxel_index, 1):
                c = voxel_map.get_or_create(nidx)
                c.frontier_score = float(min(10.0, c.frontier_score + frontier_boost))
        self._refresh_evidence_gradient(voxel_map, touched)
        self._local_zero_crossing_debias(voxel_map, touched, frame_id=frame_id)
        self._zero_crossing_bias_field(voxel_map, touched)
        self._apply_rbi_reintegration(voxel_map, touched)
        self._poisson_refine(voxel_map, touched)
        vcr_vals: List[float] = []
        dyn_vals: List[float] = []
        dccm_vals: List[float] = []
        zcbf_vals: List[float] = []
        for idx in touched:
            c = voxel_map.get_cell(idx)
            if c is None:
                continue
            vcr_vals.append(float(np.clip(getattr(c, "vcr_score", 0.0), 0.0, 1.0)))
            dyn_vals.append(float(np.clip(c.d_score, 0.0, 1.0)))
            dccm_vals.append(float(np.clip(getattr(c, "dccm_commit", 0.0), 0.0, 1.0)))
            zcbf_vals.append(float(abs(getattr(c, "zcbf_bias", 0.0))))
        return {
            "accepted": float(len(accepted)),
            "frontier_count": float(len(rejected)),
            "touched_voxels": float(len(touched)),
            "active_voxels": float(len(voxel_map)),
            "mean_vcr_score": float(np.mean(vcr_vals)) if vcr_vals else 0.0,
            "mean_d_score": float(np.mean(dyn_vals)) if dyn_vals else 0.0,
            "mean_dccm_commit": float(np.mean(dccm_vals)) if dccm_vals else 0.0,
            "mean_abs_zcbf_bias": float(np.mean(zcbf_vals)) if zcbf_vals else 0.0,
        }
