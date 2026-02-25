from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

from .config import EGF3DConfig
from .types import VoxelCell3D

VoxelIndex = Tuple[int, int, int]


class VoxelHashMap3D:
    def __init__(self, cfg: EGF3DConfig):
        self.cfg = cfg
        self.voxel_size = cfg.map3d.voxel_size
        self.cells: Dict[VoxelIndex, VoxelCell3D] = {}

    def world_to_index(self, point: np.ndarray) -> VoxelIndex:
        p = np.asarray(point, dtype=float)
        idx = np.floor(p / self.voxel_size).astype(int)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def index_to_center(self, index: VoxelIndex) -> np.ndarray:
        idx = np.array(index, dtype=float)
        return (idx + 0.5) * self.voxel_size

    def get_cell(self, index: VoxelIndex) -> VoxelCell3D | None:
        return self.cells.get(index)

    def get_or_create(self, index: VoxelIndex) -> VoxelCell3D:
        cell = self.cells.get(index)
        if cell is not None:
            return cell
        cell = VoxelCell3D(
            phi=0.0,
            phi_w=0.0,
            rho=0.0,
            g_mean=np.zeros(3, dtype=float),
            g_cov=np.eye(3, dtype=float) * self.cfg.map3d.init_grad_cov,
            c_rho=np.zeros(3, dtype=float),
            d_score=0.0,
            surf_evidence=0.0,
            free_evidence=0.0,
            residual_evidence=0.0,
            rho_prev=0.0,
            rho_osc=0.0,
            clear_hits=0.0,
            frontier_score=0.0,
        )
        self.cells[index] = cell
        return cell

    def get_rho(self, index: VoxelIndex) -> float:
        cell = self.cells.get(index)
        return 0.0 if cell is None else float(cell.rho)

    def get_phi(self, index: VoxelIndex) -> float:
        cell = self.cells.get(index)
        return 0.0 if cell is None else float(cell.phi)

    def neighbor_indices(self, index: VoxelIndex, radius_cells: int) -> Iterator[VoxelIndex]:
        ix, iy, iz = index
        for dz in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    yield ix + dx, iy + dy, iz + dz

    def neighbor_indices_for_point(self, point: np.ndarray, radius_m: float) -> Iterator[VoxelIndex]:
        center_idx = self.world_to_index(point)
        radius_cells = max(1, int(np.ceil(radius_m / self.voxel_size)))
        yield from self.neighbor_indices(center_idx, radius_cells)

    def iter_cells(self) -> Iterable[Tuple[VoxelIndex, VoxelCell3D]]:
        return self.cells.items()

    def decay_fields(self, mode: str = "local", global_dynamic_score: float = 0.0, dyn_forget_gain: float = 0.0) -> None:
        cfg = self.cfg.map3d
        ucfg = self.cfg.update
        mode = str(mode).lower()
        dyn_gain = max(0.0, float(dyn_forget_gain))
        glob_dyn = float(np.clip(global_dynamic_score, 0.0, 1.0))
        to_delete: List[VoxelIndex] = []
        for idx, cell in self.cells.items():
            local_dyn = float(np.clip(cell.d_score, 0.0, 1.0))
            if dyn_gain <= 1e-9 or mode == "off":
                dyn_coeff = 0.0
            elif mode == "global":
                dyn_coeff = glob_dyn
            else:
                dyn_coeff = local_dyn

            rho_decay_eff = max(0.82, cfg.rho_decay - 0.25 * dyn_gain * dyn_coeff)
            phi_decay_eff = max(0.72, cfg.phi_w_decay - 0.45 * dyn_gain * dyn_coeff)
            cell.rho *= rho_decay_eff
            cell.phi_w *= phi_decay_eff
            cell.g_cov *= cfg.cov_inflation
            cell.g_cov = np.clip(cell.g_cov, cfg.min_cov, cfg.max_cov)
            cell.surf_evidence *= ucfg.evidence_decay
            cell.free_evidence *= ucfg.evidence_decay
            cell.residual_evidence *= ucfg.evidence_decay
            cell.clear_hits *= 0.98
            cell.frontier_score *= ucfg.frontier_decay
            if self.cfg.update.enable_evidence:
                stale = bool(cell.phi_w < 1e-3 and cell.rho < 1e-4 and cell.frontier_score < 0.05)
            else:
                stale = bool(cell.phi_w < 1e-3 and cell.frontier_score < 0.05)
            if stale:
                to_delete.append(idx)
        for idx in to_delete:
            self.cells.pop(idx, None)

    def extract_surface_points(
        self,
        phi_thresh: float,
        rho_thresh: float,
        min_weight: float,
        max_d_score: float = 1.0,
        max_free_ratio: float = 1e9,
        prune_free_min: float = 1e9,
        prune_residual_min: float = 1e9,
        max_clear_hits: float = 1e9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pts: List[np.ndarray] = []
        nrm: List[np.ndarray] = []
        for idx, cell in self.cells.items():
            if abs(cell.phi) > phi_thresh:
                continue
            if cell.rho < rho_thresh:
                continue
            if cell.phi_w < min_weight:
                continue
            if cell.d_score > max_d_score:
                continue
            free_ratio = float(cell.free_evidence / max(1e-6, cell.surf_evidence))
            if free_ratio > max_free_ratio:
                continue
            if cell.free_evidence >= prune_free_min and cell.residual_evidence >= prune_residual_min:
                continue
            if cell.clear_hits > max_clear_hits:
                continue
            g = np.asarray(cell.g_mean, dtype=float)
            gn = np.linalg.norm(g)
            if gn < 1e-7:
                continue
            pts.append(self.index_to_center(idx))
            nrm.append(g / gn)
        if not pts:
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
        return np.asarray(pts, dtype=float), np.asarray(nrm, dtype=float)

    def __len__(self) -> int:
        return len(self.cells)

    def export_voxel_arrays(self, min_phi_w: float = 0.1):
        centers = []
        d_scores = []
        rho_vals = []
        phi_w_vals = []
        surf_vals = []
        free_vals = []
        residual_vals = []
        clear_vals = []
        for idx, cell in self.cells.items():
            if cell.phi_w < min_phi_w:
                continue
            centers.append(self.index_to_center(idx))
            d_scores.append(float(cell.d_score))
            rho_vals.append(float(cell.rho))
            phi_w_vals.append(float(cell.phi_w))
            surf_vals.append(float(cell.surf_evidence))
            free_vals.append(float(cell.free_evidence))
            residual_vals.append(float(cell.residual_evidence))
            clear_vals.append(float(cell.clear_hits))
        if not centers:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float),
            )
        return (
            np.asarray(centers, dtype=float),
            np.asarray(d_scores, dtype=float),
            np.asarray(rho_vals, dtype=float),
            np.asarray(phi_w_vals, dtype=float),
            np.asarray(surf_vals, dtype=float),
            np.asarray(free_vals, dtype=float),
            np.asarray(residual_vals, dtype=float),
            np.asarray(clear_vals, dtype=float),
        )
