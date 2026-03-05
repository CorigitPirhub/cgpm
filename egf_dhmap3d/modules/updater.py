from __future__ import annotations

from typing import Iterable, List, Set

import numpy as np

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.core.voxel_hash import VoxelHashMap3D, VoxelIndex
from egf_dhmap3d.modules.associator import AssocMeasurement3D


class Updater3D:
    def __init__(self, cfg: EGF3DConfig):
        self.cfg = cfg

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
        if measurement.seed:
            weight_assoc *= 0.55
            trunc_eff = max(min_trunc, 0.06)
            neighbor_iter = [measurement.voxel_index]
        else:
            trunc_eff = max(min_trunc, trunc * trunc_scale)
            neighbor_iter = voxel_map.neighbor_indices_for_point(p, radius_m=trunc_eff)
        surf_band = float(np.clip(self.cfg.update.surf_band_ratio, 0.1, 0.9)) * trunc_eff

        for nidx in neighbor_iter:
            center = voxel_map.index_to_center(nidx)
            rel = center - p
            d_signed = 0.0 if measurement.seed else float(np.dot(rel, n))
            if abs(d_signed) > trunc_eff:
                continue

            cell = voxel_map.get_or_create(nidx)
            dist2 = float(np.dot(rel, rel))
            if self.cfg.update.enable_evidence:
                rho_w = np.exp(-0.5 * dist2 / (rho_sigma * rho_sigma))
                cell.rho += float(rho_w)
            else:
                cell.rho = 1.0

            w_obs = float(np.exp(-0.5 * (d_signed / max(1e-6, trunc_eff)) ** 2) * weight_assoc)
            w_new = cell.phi_w + w_obs
            if w_new <= 1e-9:
                continue
            cell.phi = float((cell.phi_w * cell.phi + w_obs * d_signed) / w_new)
            cell.phi_w = float(min(5000.0, w_new))
            # Geometry-only channel: integrate near-surface observations only,
            # decoupled from dynamic suppression / free-space contradiction updates.
            geo_band = float(max(0.6 * voxel_map.voxel_size, 0.35 * trunc_eff))
            if abs(d_signed) <= geo_band:
                w_geo = float(w_obs * np.exp(-0.5 * (d_signed / max(1e-6, geo_band)) ** 2))
                if measurement.seed:
                    w_geo *= 0.55
                w_geo_new = cell.phi_geo_w + w_geo
                if w_geo_new > 1e-9:
                    cell.phi_geo = float((cell.phi_geo_w * cell.phi_geo + w_geo * d_signed) / w_geo_new)
                    cell.phi_geo_w = float(min(5000.0, w_geo_new))
            # Update local dynamic evidence (surface vs free-space observation consistency).
            if self.cfg.update.enable_evidence and abs(d_signed) <= surf_band:
                cell.surf_evidence += w_obs
            else:
                if self.cfg.update.enable_evidence:
                    free_w = w_obs * float(np.clip(abs(d_signed) / max(1e-6, trunc_eff), 0.0, 1.0))
                    cell.free_evidence += free_w
            # Residual-driven dynamic cue: high association inconsistency raises local dynamicity.
            d2_ref = float(max(1e-6, self.cfg.update.dyn_d2_ref))
            res_obs = float(np.clip(measurement.d2 / d2_ref, 0.0, 1.0))
            if measurement.seed:
                res_obs *= 0.5
            res_alpha = float(np.clip(self.cfg.update.dyn_score_alpha, 0.01, 0.8))
            cell.residual_evidence = float((1.0 - res_alpha) * cell.residual_evidence + res_alpha * res_obs)

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
                wc = float(np.clip(self.cfg.update.stcg_conflict_weight, 0.0, 1.0))
                wr = float(np.clip(self.cfg.update.stcg_residual_weight, 0.0, 1.0))
                wo = float(np.clip(self.cfg.update.stcg_osc_weight, 0.0, 1.0))
                ws = max(1e-6, wc + wr + wo)
                stcg_obs = float((wc * conflict + wr * residual_n + wo * osc_n) / ws)
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
                if c.surf_evidence > 1.8 * c.free_evidence and rho_now > 0.9:
                    c.stcg_score *= 0.88
            else:
                c.stcg_score *= 0.98
                c.stcg_active *= 0.95

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

        touched_ext: Set[VoxelIndex] = set()
        for idx in touched:
            touched_ext.add(idx)
            touched_ext.update(voxel_map.neighbor_indices(idx, radius))

        for idx in touched_ext:
            c = voxel_map.get_cell(idx)
            if c is None:
                continue
            if c.rho < min_rho or c.d_score > max_d:
                continue
            if use_geo and c.phi_geo_w >= min_w:
                phi_i = float(c.phi_geo)
                w_i = float(c.phi_geo_w)
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
            for nidx in voxel_map.neighbor_indices(idx, radius):
                if nidx == idx:
                    continue
                cj = voxel_map.get_cell(nidx)
                if cj is None:
                    continue
                if use_geo and cj.phi_geo_w >= min_w:
                    phi_j = float(cj.phi_geo)
                    w_j = float(cj.phi_geo_w)
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
                phi_est = float(phi_j - np.dot(n_i, center_j - center_i))
                # Confidence uses neighbor weight/rho and normal consistency.
                w = float(min(w_j, 5.0) * (0.2 + 0.8 * cos_ij) * (0.35 + 0.65 * np.clip(cj.rho, 0.0, 1.0)))
                est_vals.append(phi_est)
                est_w.append(w)
            if len(est_vals) < min_n:
                continue
            vals = np.asarray(est_vals, dtype=float)
            ws = np.asarray(est_w, dtype=float)
            phi_ref0 = self._weighted_median(vals, ws)
            abs_res = np.abs(vals - phi_ref0)
            if trim_q < 0.999:
                thr = float(np.quantile(abs_res, trim_q))
                mask = abs_res <= max(1e-6, thr)
                if int(np.count_nonzero(mask)) >= min_n:
                    vals = vals[mask]
                    ws = ws[mask]
                    abs_res = abs_res[mask]
            # Huber-like attenuation for outliers.
            scale = float(max(1e-4, 1.4826 * np.median(abs_res)))
            huber = 1.0 / (1.0 + (abs_res / scale) ** 2)
            ws = ws * huber
            phi_ref = self._weighted_median(vals, ws)
            bias_obs = float(np.clip(phi_i - phi_ref, -max_bias, max_bias))
            if use_geo and c.phi_geo_w >= min_w:
                c.phi_geo_bias = float((1.0 - bias_alpha) * c.phi_geo_bias + bias_alpha * bias_obs)
                corr = float(np.clip(corr_gain * c.phi_geo_bias, -max_step, max_step))
            else:
                c.phi_bias = float((1.0 - bias_alpha) * c.phi_bias + bias_alpha * bias_obs)
                corr = float(np.clip(corr_gain * c.phi_bias, -max_step, max_step))
            if abs(corr) <= 1e-7:
                continue
            if use_geo and c.phi_geo_w >= min_w:
                c.phi_geo = float(np.clip(c.phi_geo - corr, -0.8, 0.8))
                # Keep projection channel partially aligned with corrected geometry.
                c.phi = float(np.clip(c.phi - 0.35 * corr, -0.8, 0.8))
            else:
                c.phi = float(np.clip(c.phi - corr, -0.8, 0.8))

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
                if c is not None and (c.rho <= rho_max or c.phi_w <= phiw_max or c.d_score >= 0.2):
                    # Free-space consistency gate: only clear strongly when the voxel has
                    # persistent free-space evidence across frames.
                    free_ratio = float(c.free_evidence / max(1e-6, c.surf_evidence))
                    consistent_free = bool((free_ratio >= 1.1 and c.free_evidence >= 0.25) or c.clear_hits >= 1.2)
                    dyn = float(np.clip(c.d_score, 0.0, 1.0))
                    if (not consistent_free) and dyn < 0.35:
                        s += step
                        continue

                    # Occlusion-aware attenuation:
                    # if endpoint is very confident (likely a background wall), reduce
                    # along-ray clearing to avoid over-erasing temporarily occluded regions.
                    occ_att = float(1.0 - 0.55 * endpoint_conf)
                    if endpoint_conf > 0.6 and c.surf_evidence > c.free_evidence:
                        occ_att *= 0.6

                    clear_base = gain * (0.45 + 0.55 * dyn)
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
                    c.phi_w = max(phiw_floor, c.phi_w * (1.0 - clear))
                    c.rho = max(rho_floor, c.rho * (1.0 - 0.75 * clear))
                    c.free_evidence += 0.8 * clear
                    c.residual_evidence = float(np.clip(c.residual_evidence + 0.6 * clear, 0.0, 1.0))
                    c.d_score = float(np.clip(c.d_score + dyn_boost * clear, 0.0, 1.0))
                    c.clear_hits += 1.0
                    touched.add(idx)
                s += step

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
                    c.phi = val

    def update(
        self,
        voxel_map: VoxelHashMap3D,
        accepted: List[AssocMeasurement3D],
        rejected: List[AssocMeasurement3D],
        sensor_origin: np.ndarray | None = None,
        frame_id: int = 0,
    ) -> dict:
        touched: Set[VoxelIndex] = set()
        for m in accepted:
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
        self._poisson_refine(voxel_map, touched)
        return {
            "accepted": float(len(accepted)),
            "frontier_count": float(len(rejected)),
            "touched_voxels": float(len(touched)),
            "active_voxels": float(len(voxel_map)),
        }
