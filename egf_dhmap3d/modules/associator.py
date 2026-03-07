from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.core.voxel_hash import VoxelHashMap3D, VoxelIndex


@dataclass
class AssocMeasurement3D:
    point_world: np.ndarray
    normal_world: np.ndarray
    projected_world: np.ndarray
    voxel_index: VoxelIndex
    source_index: VoxelIndex
    d2: float
    sigma_d: float
    sigma_n: float
    phi: float
    seed: bool = False
    frontier: bool = False
    dynamic_prob: float = 0.0
    assoc_risk: float = 0.0
    sensor_origin: np.ndarray | None = None
    seed_blocked: bool = False
    pfv_penalty: float = 0.0


class Associator3D:
    def __init__(self, cfg: EGF3DConfig):
        self.cfg = cfg

    @staticmethod
    def _huber(val: float, delta: float) -> float:
        a = abs(val)
        if a <= delta:
            return val
        return float(np.sign(val) * (delta + np.sqrt(max(0.0, 2.0 * delta * (a - delta)))))

    def _contra_risk(self, cell) -> float:
        if cell is None:
            return 0.0
        st = float(np.clip(getattr(cell, "st_mem", 0.0), 0.0, 1.0))
        vis = float(np.clip(getattr(cell, "visibility_contradiction", 0.0), 0.0, 1.0))
        res = float(np.clip(getattr(cell, "residual_evidence", 0.0), 0.0, 1.0))
        surf = float(max(1e-6, getattr(cell, "surf_evidence", 0.0)))
        free = float(max(0.0, getattr(cell, "free_evidence", 0.0)))
        free_ratio = float(free / surf)
        p_static = float(np.clip(getattr(cell, "p_static", 0.5), 0.0, 1.0))
        rho = float(max(0.0, getattr(cell, "rho", 0.0)))

        w_st = float(np.clip(self.cfg.assoc.contra_stmem_weight, 0.0, 1.0))
        w_vis = float(np.clip(self.cfg.assoc.contra_visibility_weight, 0.0, 1.0))
        w_res = float(np.clip(self.cfg.assoc.contra_residual_weight, 0.0, 1.0))
        ws = max(1e-6, w_st + w_vis + w_res)
        base = float((w_st * st + w_vis * vis + w_res * res) / ws)

        free_ref = float(max(1e-6, self.cfg.assoc.contra_free_ratio_ref))
        free_n = float(np.clip(free_ratio / free_ref, 0.0, 2.0))
        rho_ref = float(max(1e-6, self.cfg.assoc.contra_rho_ref))
        rho_n = float(np.clip(rho / rho_ref, 0.0, 1.0))
        static_guard = float(np.clip(self.cfg.assoc.contra_static_guard, 0.0, 1.0))
        rho_guard = float(np.clip(self.cfg.assoc.contra_rho_guard, 0.0, 1.0))

        risk = float(
            np.clip(
                base
                * (0.65 + 0.35 * free_n)
                * (1.0 - static_guard * p_static)
                * (1.0 - rho_guard * rho_n),
                0.0,
                1.0,
            )
        )
        return risk


    def _static_support(self, voxel_map: VoxelHashMap3D, cell) -> float:
        if cell is None:
            return 0.0
        rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
        surf = float(max(1e-6, getattr(cell, 'surf_evidence', 0.0)))
        free = float(max(0.0, getattr(cell, 'free_evidence', 0.0)))
        occ = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
        rho_s = float(np.clip(getattr(cell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0))
        rho_bg = float(np.clip(getattr(cell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0))
        p_static = float(np.clip(getattr(cell, 'p_static', 0.5), 0.0, 1.0))
        obl = float(voxel_map._obl_conf(cell)) if hasattr(voxel_map, '_obl_conf') else 0.0
        return float(np.clip(max(0.30 * occ + 0.25 * p_static + 0.20 * rho_s + 0.20 * rho_bg + 0.15 * obl, max(occ, p_static, rho_s, rho_bg, obl)), 0.0, 1.0))

    def _pfv_penalty(self, voxel_map: VoxelHashMap3D, cell, near_cell, point_world: np.ndarray, sensor_origin: np.ndarray | None = None) -> tuple[float, bool]:
        if not bool(getattr(self.cfg.assoc, 'pfv_gate_enable', True)):
            return 0.0, False
        pfv = 0.0
        if cell is not None and hasattr(voxel_map, '_pfv_conf'):
            pfv = max(pfv, float(voxel_map._pfv_conf(cell)))
        if near_cell is not None and hasattr(voxel_map, '_pfv_conf'):
            pfv = max(pfv, float(voxel_map._pfv_conf(near_cell)))
        if pfv <= 1e-6:
            return 0.0, False
        support = max(self._static_support(voxel_map, cell), self._static_support(voxel_map, near_cell))
        rho_ref = float(max(1e-6, self.cfg.assoc.contra_rho_ref))
        rho_bg = float(np.clip(max(getattr(cell, 'rho_bg', 0.0) if cell is not None else 0.0, getattr(near_cell, 'rho_bg', 0.0) if near_cell is not None else 0.0) / rho_ref, 0.0, 1.0))
        view_align = 0.0
        if sensor_origin is not None:
            v = np.asarray(point_world, dtype=float).reshape(3) - np.asarray(sensor_origin, dtype=float).reshape(3)
            d = float(np.linalg.norm(v))
            if d > 1e-9:
                n = v / d
                if cell is not None and float(np.linalg.norm(getattr(cell, 'g_mean', np.zeros(3)))) > 1e-9:
                    g = np.asarray(cell.g_mean, dtype=float)
                    g = g / max(1e-9, float(np.linalg.norm(g)))
                    view_align = float(np.clip(abs(np.dot(g, -n)), 0.0, 1.0))
        guard = float(np.clip(getattr(self.cfg.assoc, 'pfv_static_guard', 0.84), 0.0, 1.0))
        rho_guard = float(np.clip(getattr(self.cfg.assoc, 'pfv_bg_rho_guard', 0.82), 0.0, 1.0))
        view_w = float(np.clip(getattr(self.cfg.assoc, 'pfv_view_align_weight', 0.25), 0.0, 1.0))
        penalty = float(np.clip(pfv * (1.0 - guard * support) * (1.0 - rho_guard * rho_bg) * (1.0 + view_w * view_align), 0.0, 1.0))
        seed_block = bool(pfv >= float(np.clip(getattr(self.cfg.assoc, 'pfv_seed_block_on', 0.42), 0.0, 1.0)) and support < 0.72)
        return penalty, seed_block

    def _find_surface_source(self, voxel_map: VoxelHashMap3D, seed_idx: VoxelIndex) -> VoxelIndex | None:
        radius = max(0, int(self.cfg.assoc.search_radius_cells))
        best_idx = None
        best_score = None
        strict_w = float(self.cfg.assoc.strict_surface_weight)
        seed_center = voxel_map.index_to_center(seed_idx)
        for idx in voxel_map.neighbor_indices(seed_idx, radius):
            cell = voxel_map.get_cell(idx)
            if cell is None:
                continue
            if cell.phi_w < strict_w:
                continue
            c = voxel_map.index_to_center(idx)
            dist = float(np.linalg.norm(c - seed_center))
            score = abs(float(cell.phi)) + 0.35 * dist
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _project_and_score(
        self,
        voxel_map: VoxelHashMap3D,
        point_world: np.ndarray,
        normal_world: np.ndarray,
        sensor_origin: np.ndarray | None = None,
    ) -> AssocMeasurement3D | None:
        idx = voxel_map.world_to_index(point_world)
        source_idx = self._find_surface_source(voxel_map, idx)
        cell = voxel_map.get_cell(source_idx) if source_idx is not None else None
        near_cell = voxel_map.get_cell(idx)
        normal_world = np.asarray(normal_world, dtype=float)
        normal_world = normal_world / np.clip(np.linalg.norm(normal_world), 1e-9, None)

        frontier_flag = False
        if near_cell is not None and near_cell.frontier_score >= self.cfg.assoc.frontier_activate_thresh:
            frontier_flag = True

        if cell is None or cell.phi_w <= 1e-6:
            sigma_n_seed = self.cfg.assoc.sigma_n0 * (1.15 if frontier_flag else 1.0)
            pfv_penalty, seed_block = self._pfv_penalty(voxel_map, cell, near_cell, point_world, sensor_origin=sensor_origin)
            return AssocMeasurement3D(
                point_world=point_world,
                normal_world=normal_world,
                projected_world=point_world.copy(),
                voxel_index=idx,
                source_index=idx,
                d2=0.0,
                sigma_d=self.cfg.assoc.sigma_d0,
                sigma_n=sigma_n_seed,
                phi=0.0,
                seed=True,
                frontier=frontier_flag,
                assoc_risk=float(pfv_penalty),
                seed_blocked=bool(seed_block),
                pfv_penalty=float(pfv_penalty),
                sensor_origin=None if sensor_origin is None else np.asarray(sensor_origin, dtype=float).reshape(3),
            )

        g = np.asarray(cell.g_mean, dtype=float)
        g_norm = float(np.linalg.norm(g))
        if g_norm < 1e-8:
            g = normal_world.copy()
            g_norm = float(np.linalg.norm(g))
        if g_norm < 1e-8:
            return None
        n_pred = g / g_norm

        phi = float(cell.phi)
        projected_world = point_world - phi * g / (g_norm * g_norm + 1e-9)
        r_d = float(phi)
        if self.cfg.assoc.use_normal_residual:
            r_n_raw = float(1.0 - np.clip(np.dot(normal_world, n_pred), -1.0, 1.0))
            r_n = self._huber(r_n_raw, delta=max(1e-4, self.cfg.assoc.huber_delta_n))
        else:
            r_n_raw = 0.0
            r_n = 0.0

        sigma_unc = float(np.trace(cell.g_cov))
        if self.cfg.assoc.use_evidence_in_noise:
            evidence_grad = float(np.linalg.norm(cell.c_rho))
            sigma_d = self.cfg.assoc.sigma_d0 / (1.0 + self.cfg.assoc.alpha_evidence * evidence_grad)
            sigma_d += self.cfg.assoc.beta_uncert * sigma_unc
            sigma_n = self.cfg.assoc.sigma_n0 / (1.0 + 0.5 * self.cfg.assoc.alpha_evidence * evidence_grad)
            sigma_n += 0.5 * self.cfg.assoc.beta_uncert * sigma_unc
        else:
            sigma_d = self.cfg.assoc.sigma_d0 + self.cfg.assoc.beta_uncert * sigma_unc
            sigma_n = self.cfg.assoc.sigma_n0 + 0.5 * self.cfg.assoc.beta_uncert * sigma_unc

        # Heteroscedastic association noise:
        # degrade noisy observations (grazing/deep/inconsistent normals),
        # while mildly rewarding high-quality incidence to improve geometric accuracy.
        if bool(self.cfg.assoc.hetero_enable):
            if sensor_origin is None:
                v = np.asarray(point_world, dtype=float)
            else:
                v = np.asarray(point_world, dtype=float) - np.asarray(sensor_origin, dtype=float).reshape(3)
            v_norm = float(np.linalg.norm(v))
            if v_norm > 1e-9:
                view_dir = v / v_norm
            else:
                view_dir = np.array([0.0, 0.0, 1.0], dtype=float)
            # Use absolute incidence because surface orientation can be flipped.
            cos_inc = float(np.clip(abs(np.dot(n_pred, -view_dir)), 0.0, 1.0))
            inc_ref = float(np.clip(self.cfg.assoc.hetero_inc_ref_cos, 1e-3, 1.0))
            depth_ref = float(max(1e-3, self.cfg.assoc.hetero_depth_ref_m))
            normal_ref = float(max(1e-4, self.cfg.assoc.hetero_normal_ref))
            inc_pen = float(np.clip((inc_ref - cos_inc) / inc_ref, 0.0, 1.0))
            depth_pen = float(np.clip(v_norm / depth_ref, 0.0, 2.0))
            normal_pen = float(np.clip(r_n_raw / normal_ref, 0.0, 2.0))
            scale_raw = float(
                1.0
                + self.cfg.assoc.hetero_k_inc * inc_pen
                + self.cfg.assoc.hetero_k_depth * depth_pen
                + self.cfg.assoc.hetero_k_normal * normal_pen
            )
            good_cos = float(np.clip(self.cfg.assoc.hetero_good_cos, 0.0, 1.0))
            good_bonus = float(np.clip(self.cfg.assoc.hetero_good_bonus, 0.0, 0.9))
            if cos_inc > good_cos and good_cos < 0.999:
                q = (cos_inc - good_cos) / max(1e-3, 1.0 - good_cos)
                bonus = float(np.clip(1.0 - good_bonus * q, 0.5, 1.0))
            else:
                bonus = 1.0
            d_min = float(max(1e-3, self.cfg.assoc.hetero_sigma_d_min_scale))
            d_max = float(max(d_min, self.cfg.assoc.hetero_sigma_d_max_scale))
            n_min = float(max(1e-3, self.cfg.assoc.hetero_sigma_n_min_scale))
            n_max = float(max(n_min, self.cfg.assoc.hetero_sigma_n_max_scale))
            scale_d = float(np.clip(scale_raw * bonus, d_min, d_max))
            scale_n = float(np.clip(scale_raw * (0.95 * bonus), n_min, n_max))
            sigma_d *= scale_d
            sigma_n *= scale_n
        sigma_d = max(1e-4, float(sigma_d))
        sigma_n = max(1e-4, float(sigma_n))

        d2 = (r_d * r_d) / (sigma_d * sigma_d) + (r_n * r_n) / (sigma_n * sigma_n)
        assoc_risk = 0.0
        if bool(self.cfg.assoc.contra_gate_enable):
            assoc_risk = self._contra_risk(cell)
            boost_max = float(max(1.0, self.cfg.assoc.contra_d2_boost_max))
            d2 *= float(1.0 + assoc_risk * (boost_max - 1.0))
        pfv_penalty, _seed_block = self._pfv_penalty(voxel_map, cell, near_cell, point_world, sensor_origin=sensor_origin)
        # PFAG v2: treat PFV as admissibility prior rather than a generic residual inflator.
        # Keep a very mild cost bump for bookkeeping, but do not hard-reject mature matches here.
        if pfv_penalty > float(np.clip(getattr(self.cfg.assoc, 'pfv_gate_off', 0.18), 0.0, 1.0)):
            d2 *= float(1.0 + 0.20 * pfv_penalty)
            assoc_risk = float(max(assoc_risk, 0.65 * pfv_penalty))
        return AssocMeasurement3D(
            point_world=point_world,
            normal_world=normal_world,
            projected_world=projected_world,
            voxel_index=idx,
            source_index=source_idx if source_idx is not None else idx,
            d2=float(d2),
            sigma_d=sigma_d,
            sigma_n=sigma_n,
            phi=phi,
            seed=False,
            frontier=frontier_flag,
            assoc_risk=float(assoc_risk),
            pfv_penalty=float(pfv_penalty),
            sensor_origin=None if sensor_origin is None else np.asarray(sensor_origin, dtype=float).reshape(3),
        )

    def associate(
        self,
        voxel_map: VoxelHashMap3D,
        points_world: np.ndarray,
        normals_world: np.ndarray,
        sensor_origin: np.ndarray | None = None,
    ) -> Tuple[List[AssocMeasurement3D], List[AssocMeasurement3D], Dict[str, float]]:
        accepted: List[AssocMeasurement3D] = []
        rejected: List[AssocMeasurement3D] = []
        d2_vals: List[float] = []
        pfv_blocks = 0

        gate = float(self.cfg.assoc.gate_threshold)
        relax_gate = float(self.cfg.assoc.relax_gate_scale * gate)
        for p, n in zip(points_world, normals_world):
            m = self._project_and_score(voxel_map, p, n, sensor_origin=sensor_origin)
            if m is None:
                continue
            if m.seed:
                if bool(m.seed_blocked):
                    rejected.append(m)
                    pfv_blocks += 1
                    continue
                accepted.append(m)
                continue
            local_gate = gate
            if m.frontier:
                local_gate *= float(self.cfg.assoc.frontier_relax_boost)
            if m.d2 < gate:
                accepted.append(m)
                d2_vals.append(float(m.d2))
                continue
            if m.d2 < max(relax_gate, local_gate):
                m.d2 = float(min(m.d2, local_gate))
                accepted.append(m)
                d2_vals.append(float(m.d2))
                continue
            # Fallback: unresolved/low-support voxels are force-seeded to avoid
            # cold-start holes. This can be disabled in conservative static mode.
            if bool(self.cfg.assoc.seed_fallback_enable):
                near = voxel_map.get_cell(m.voxel_index)
                low_support = near is None or near.phi_w < float(self.cfg.assoc.seed_fallback_low_support_scale) * self.cfg.assoc.strict_surface_weight
                frontier_hot = (
                    near is not None
                    and near.frontier_score >= float(self.cfg.assoc.seed_fallback_frontier_scale) * self.cfg.assoc.frontier_activate_thresh
                )
                if low_support or frontier_hot:
                    if m.pfv_penalty >= float(np.clip(getattr(self.cfg.assoc, 'pfv_seed_block_on', 0.42), 0.0, 1.0)):
                        rejected.append(m)
                        pfv_blocks += 1
                        continue
                    m.seed = True
                    m.frontier = True
                    m.d2 = float(local_gate)
                    accepted.append(m)
                    continue
            rejected.append(m)

        total = max(1, points_world.shape[0])
        stats = {
            "assoc_ratio": float(len(accepted) / total),
            "mean_d2": float(np.mean(d2_vals)) if d2_vals else 0.0,
            "accepted": float(len(accepted)),
            "rejected": float(len(rejected)),
            "pfv_blocked": float(pfv_blocks),
        }
        return accepted, rejected, stats
