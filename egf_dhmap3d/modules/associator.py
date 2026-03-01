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


class Associator3D:
    def __init__(self, cfg: EGF3DConfig):
        self.cfg = cfg

    @staticmethod
    def _huber(val: float, delta: float) -> float:
        a = abs(val)
        if a <= delta:
            return val
        return float(np.sign(val) * (delta + np.sqrt(max(0.0, 2.0 * delta * (a - delta)))))

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
        sigma_d = max(1e-4, float(sigma_d))
        sigma_n = max(1e-4, float(sigma_n))

        d2 = (r_d * r_d) / (sigma_d * sigma_d) + (r_n * r_n) / (sigma_n * sigma_n)
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
        )

    def associate(
        self,
        voxel_map: VoxelHashMap3D,
        points_world: np.ndarray,
        normals_world: np.ndarray,
    ) -> Tuple[List[AssocMeasurement3D], List[AssocMeasurement3D], Dict[str, float]]:
        accepted: List[AssocMeasurement3D] = []
        rejected: List[AssocMeasurement3D] = []
        d2_vals: List[float] = []

        gate = float(self.cfg.assoc.gate_threshold)
        relax_gate = float(self.cfg.assoc.relax_gate_scale * gate)
        for p, n in zip(points_world, normals_world):
            m = self._project_and_score(voxel_map, p, n)
            if m is None:
                continue
            if m.seed:
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
        }
        return accepted, rejected, stats
