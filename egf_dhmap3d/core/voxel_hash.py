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
        self._neighbor_offset_cache: Dict[int, List[VoxelIndex]] = {}

    def _offsets_for_radius(self, radius_cells: int) -> List[VoxelIndex]:
        r = max(0, int(radius_cells))
        cached = self._neighbor_offset_cache.get(r)
        if cached is not None:
            return cached
        offsets: List[VoxelIndex] = []
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    offsets.append((dx, dy, dz))
        self._neighbor_offset_cache[r] = offsets
        return offsets

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
        for dx, dy, dz in self._offsets_for_radius(radius_cells):
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

            # Dynamic-selective exponential forgetting:
            # keep static cells near base decay, but aggressively down-weight dynamic cells.
            forget_strength = float(max(0.0, dyn_gain * dyn_coeff))
            rho_decay_eff = float(np.clip(cfg.rho_decay * np.exp(-1.8 * forget_strength), 0.55, 1.0))
            phi_decay_eff = float(np.clip(cfg.phi_w_decay * np.exp(-2.4 * forget_strength), 0.45, 1.0))
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
        current_step: int = 0,
        max_age_frames: int = 1_000_000_000,
        max_d_score: float = 1.0,
        max_free_ratio: float = 1e9,
        prune_free_min: float = 1e9,
        prune_residual_min: float = 1e9,
        max_clear_hits: float = 1e9,
        use_zero_crossing: bool = True,
        zero_crossing_max_offset: float = 0.06,
        zero_crossing_phi_gate: float = 0.05,
        consistency_enable: bool = False,
        consistency_radius: int = 1,
        consistency_min_neighbors: int = 4,
        consistency_normal_cos: float = 0.55,
        consistency_phi_diff: float = 0.04,
        snef_local_enable: bool = False,
        snef_block_size_cells: int = 8,
        snef_dscore_quantile: float = 0.80,
        snef_dscore_margin: float = 0.05,
        snef_free_ratio_quantile: float = 0.85,
        snef_free_ratio_margin: float = 0.10,
        snef_abs_phi_quantile: float = 1.0,
        snef_abs_phi_margin: float = 0.0,
        snef_min_keep_per_block: int = 16,
        snef_min_keep_ratio_per_block: float = 0.0,
        snef_min_candidates_per_block: int = 10,
        snef_anchor_rho_quantile: float = 0.90,
        snef_anchor_dscore_quantile: float = 0.25,
        snef_anchor_min_per_block: int = 2,
        two_stage_enable: bool = False,
        two_stage_geom_margin: float = 0.02,
        two_stage_dynamic_dscore_quantile: float = 0.70,
        two_stage_dynamic_free_quantile: float = 0.70,
        two_stage_dynamic_rho_quantile: float = 0.40,
        two_stage_dynamic_rho_margin: float = 0.0,
        two_stage_dynamic_require_low_rho: bool = True,
        adaptive_enable: bool = False,
        adaptive_rho_ref: float = 2.0,
        adaptive_phi_min_scale: float = 0.55,
        adaptive_phi_max_scale: float = 1.15,
        adaptive_min_weight_gain: float = 0.8,
        adaptive_free_ratio_gain: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        candidates: List[Tuple[VoxelIndex, VoxelCell3D, np.ndarray, np.ndarray, float]] = []
        geom_margin = float(max(0.0, two_stage_geom_margin))
        for idx, cell in self.cells.items():
            phi_thr = float(phi_thresh)
            min_w = float(min_weight)
            max_free = float(max_free_ratio)
            if adaptive_enable:
                # Bonn adaptive extraction:
                # high-rho + low-dscore cells get looser phi gate;
                # high-dscore cells get stricter phi/free gates + higher min weight.
                rho_ref = float(max(1e-6, adaptive_rho_ref))
                rho_n = float(np.clip(cell.rho / rho_ref, 0.0, 2.0))
                d_n = float(np.clip(cell.d_score, 0.0, 1.0))
                static_conf = float(np.clip(0.5 * rho_n + 0.5 * (1.0 - d_n), 0.0, 1.0))
                phi_scale = float(
                    np.clip(
                        adaptive_phi_min_scale + (adaptive_phi_max_scale - adaptive_phi_min_scale) * static_conf,
                        0.25,
                        2.0,
                    )
                )
                phi_thr = max(1e-4, float(phi_thresh) * phi_scale)
                min_w = max(0.0, float(min_weight) * (1.0 + adaptive_min_weight_gain * d_n))
                max_free = max(1e-6, float(max_free_ratio) * (1.0 - adaptive_free_ratio_gain * d_n))
            if two_stage_enable:
                phi_thr = max(phi_thr, phi_thr + geom_margin)
                min_w = max(0.0, min_w * 0.85)
                max_free = max(max_free, float(max_free_ratio) * 1.15)

            if abs(cell.phi) > phi_thr:
                continue
            if cell.rho < rho_thresh:
                continue
            if cell.phi_w < min_w:
                continue
            if cell.d_score > max_d_score:
                continue
            if (int(current_step) - int(cell.last_seen)) > int(max_age_frames):
                continue
            free_ratio = float(cell.free_evidence / max(1e-6, cell.surf_evidence))
            if free_ratio > max_free:
                continue
            if cell.free_evidence >= prune_free_min and cell.residual_evidence >= prune_residual_min:
                continue
            if cell.clear_hits > max_clear_hits:
                continue
            g = np.asarray(cell.g_mean, dtype=float)
            gn = np.linalg.norm(g)
            if gn < 1e-7:
                continue
            candidates.append((idx, cell, self.index_to_center(idx), g / gn, free_ratio))
        if not candidates:
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

        accepted_idx: set[VoxelIndex]
        if not consistency_enable:
            accepted_idx = {idx for idx, _, _, _, _ in candidates}
        else:
            cand_map = {idx: (cell, n) for idx, cell, _, n, _ in candidates}
            accepted_idx = set()
            r = max(1, int(consistency_radius))
            min_n = max(0, int(consistency_min_neighbors))
            cos_th = float(np.clip(consistency_normal_cos, 0.0, 1.0))
            phi_th = float(max(1e-4, consistency_phi_diff))
            if two_stage_enable:
                min_n = max(0, min_n - 2)
                cos_th = max(0.0, cos_th - 0.10)
                phi_th = max(1e-4, phi_th + geom_margin)
            for idx, cell, _, n_i, _ in candidates:
                consistent = 0
                for nidx in self.neighbor_indices(idx, r):
                    if nidx == idx:
                        continue
                    other = cand_map.get(nidx)
                    if other is None:
                        continue
                    c_j, n_j = other
                    if abs(float(c_j.phi) - float(cell.phi)) > phi_th:
                        continue
                    if abs(float(np.dot(n_i, n_j))) < cos_th:
                        continue
                    consistent += 1
                if consistent >= min_n:
                    accepted_idx.add(idx)

        if accepted_idx and bool(snef_local_enable):
            block_size = max(1, int(snef_block_size_cells))
            q_d = float(np.clip(snef_dscore_quantile, 0.0, 1.0))
            q_f = float(np.clip(snef_free_ratio_quantile, 0.0, 1.0))
            q_p = float(np.clip(snef_abs_phi_quantile, 0.0, 1.0))
            m_d = float(max(0.0, snef_dscore_margin))
            m_f = float(max(0.0, snef_free_ratio_margin))
            m_p = float(max(0.0, snef_abs_phi_margin))
            min_keep = max(1, int(snef_min_keep_per_block))
            keep_ratio = float(np.clip(snef_min_keep_ratio_per_block, 0.0, 1.0))
            min_candidates = max(1, int(snef_min_candidates_per_block))
            anchor_rho_q = float(np.clip(snef_anchor_rho_quantile, 0.0, 1.0))
            anchor_d_q = float(np.clip(snef_anchor_dscore_quantile, 0.0, 1.0))
            anchor_min = max(0, int(snef_anchor_min_per_block))

            groups: Dict[VoxelIndex, List[Tuple[VoxelIndex, VoxelCell3D, float]]] = {}
            for idx, cell, _, _, free_ratio in candidates:
                if idx not in accepted_idx:
                    continue
                bidx = (idx[0] // block_size, idx[1] // block_size, idx[2] // block_size)
                groups.setdefault(bidx, []).append((idx, cell, float(max(0.0, free_ratio))))

            snef_keep: set[VoxelIndex] = set()
            dyn_block_d_thr = 1.0
            dyn_block_f_thr = 1.0
            dyn_block_rho_thr = 0.0
            dyn_block_osc_thr = 1.0
            if two_stage_enable and groups:
                all_d_vals = np.asarray(
                    [float(np.clip(cell.d_score, 0.0, 1.0)) for entries in groups.values() for _, cell, _ in entries],
                    dtype=float,
                )
                all_f_vals = np.asarray([float(fr) for entries in groups.values() for _, _, fr in entries], dtype=float)
                all_rho_vals = np.asarray(
                    [float(max(0.0, cell.rho)) for entries in groups.values() for _, cell, _ in entries],
                    dtype=float,
                )
                all_osc_vals = np.asarray(
                    [float(max(0.0, cell.rho_osc)) for entries in groups.values() for _, cell, _ in entries],
                    dtype=float,
                )
                if all_d_vals.size > 0:
                    dyn_block_d_thr = float(np.quantile(all_d_vals, float(np.clip(two_stage_dynamic_dscore_quantile, 0.0, 1.0))))
                if all_f_vals.size > 0:
                    dyn_block_f_thr = float(np.quantile(all_f_vals, float(np.clip(two_stage_dynamic_free_quantile, 0.0, 1.0))))
                if all_rho_vals.size > 0:
                    dyn_block_rho_thr = float(
                        np.quantile(all_rho_vals, float(np.clip(two_stage_dynamic_rho_quantile, 0.0, 1.0)))
                    ) + float(max(0.0, two_stage_dynamic_rho_margin))
                if all_osc_vals.size > 0:
                    dyn_block_osc_thr = float(np.quantile(all_osc_vals, 0.75))

            for entries in groups.values():
                n_entries = len(entries)
                if n_entries <= min_candidates:
                    snef_keep.update([idx for idx, _, _ in entries])
                    continue

                d_vals = np.asarray([float(np.clip(c.d_score, 0.0, 1.0)) for _, c, _ in entries], dtype=float)
                f_vals = np.asarray([float(fr) for _, _, fr in entries], dtype=float)
                p_vals = np.asarray([abs(float(c.phi)) for _, c, _ in entries], dtype=float)
                rho_vals = np.asarray([float(max(0.0, c.rho)) for _, c, _ in entries], dtype=float)
                osc_vals = np.asarray([float(max(0.0, c.rho_osc)) for _, c, _ in entries], dtype=float)

                block_dyn = False
                if two_stage_enable:
                    hi_dyn = bool((float(np.median(d_vals)) >= dyn_block_d_thr) or (float(np.median(f_vals)) >= dyn_block_f_thr))
                    low_rho = bool(float(np.median(rho_vals)) <= dyn_block_rho_thr)
                    high_osc = bool(float(np.median(osc_vals)) >= dyn_block_osc_thr)
                    if bool(two_stage_dynamic_require_low_rho):
                        # Volatility-aware gating: dynamic block is either low-rho
                        # or high-rho but strongly oscillatory in evidence over time.
                        block_dyn = bool(hi_dyn and (low_rho or high_osc))
                    else:
                        block_dyn = bool(hi_dyn or low_rho or high_osc)
                    if not block_dyn:
                        snef_keep.update([idx for idx, _, _ in entries])
                        continue

                d_thr = float(np.quantile(d_vals, q_d) + m_d)
                f_thr = float(np.quantile(f_vals, q_f) + m_f)
                p_thr = float(np.quantile(p_vals, q_p) + m_p)
                hard_mask = (d_vals <= d_thr) & (f_vals <= f_thr) & (p_vals <= p_thr)
                anchor_mask = np.zeros((n_entries,), dtype=bool)

                if rho_vals.size > 0 and anchor_min > 0:
                    rho_thr = float(np.quantile(rho_vals, anchor_rho_q))
                    d_anchor_thr = float(np.quantile(d_vals, anchor_d_q))
                    anchor_mask = (rho_vals >= rho_thr) & (d_vals <= d_anchor_thr)
                    anchor_ids = set(np.flatnonzero(anchor_mask).tolist())
                    if len(anchor_ids) < anchor_min:
                        # Fill anchors by prioritizing high-rho + low-dscore cells.
                        anchor_rank = np.argsort(-rho_vals + 0.35 * d_vals)
                        for i in anchor_rank.tolist():
                            anchor_ids.add(int(i))
                            if len(anchor_ids) >= anchor_min:
                                break
                else:
                    anchor_ids = set()

                if two_stage_enable and block_dyn:
                    # Dynamic blocks use adaptive keep-ratio to increase denoising aggressiveness
                    # while preserving static anchors and minimum support.
                    d_med = float(np.median(d_vals))
                    f_med = float(np.median(f_vals))
                    rho_med = float(np.median(rho_vals))
                    osc_med = float(np.median(osc_vals))
                    d_ref = max(1e-6, dyn_block_d_thr)
                    f_ref = max(1e-6, dyn_block_f_thr)
                    rho_ref_med = max(1e-6, dyn_block_rho_thr)
                    osc_ref_med = max(1e-6, dyn_block_osc_thr)
                    dyn_strength = float(
                        np.clip(
                            0.55 * (d_med / d_ref)
                            + 0.25 * (f_med / f_ref)
                            + 0.20 * (osc_med / osc_ref_med)
                            - 0.25 * (rho_med / rho_ref_med),
                            0.0,
                            1.5,
                        )
                    )
                    # Keep-ratio attenuation is moderated to avoid over-pruning stable support.
                    keep_ratio_eff = float(max(0.26, keep_ratio * (1.0 - 0.30 * min(1.0, dyn_strength))))
                else:
                    keep_ratio_eff = keep_ratio

                min_keep_target = max(min_keep, int(np.ceil(keep_ratio_eff * float(n_entries))), len(anchor_ids))

                rho_ref = float(max(1e-6, np.quantile(rho_vals, 0.95)))
                osc_ref = float(max(1e-6, np.quantile(osc_vals, 0.75)))
                d_norm = d_vals / max(1e-6, d_thr)
                f_norm = f_vals / max(1e-6, f_thr)
                p_norm = p_vals / max(1e-6, p_thr)
                rho_norm = np.clip(rho_vals / rho_ref, 0.0, 2.0)
                osc_norm = np.clip(osc_vals / osc_ref, 0.0, 3.0)
                # Soft ranking:
                # lower risk => more likely static/stable surface sample to keep.
                risk = 0.40 * d_norm + 0.22 * f_norm + 0.20 * p_norm + 0.18 * osc_norm - 0.20 * rho_norm
                # Prefer points that pass geometric hard gate and static anchors.
                risk = risk - 0.08 * hard_mask.astype(float) - 0.20 * anchor_mask.astype(float)
                # In dynamic blocks, penalize over-threshold candidates further.
                if two_stage_enable and block_dyn:
                    risk = risk + 0.10 * np.clip(d_norm - 1.0, 0.0, 2.0) + 0.05 * np.clip(f_norm - 1.0, 0.0, 2.0)

                order = np.argsort(risk)
                k = min(n_entries, max(min_keep_target, len(anchor_ids)))
                keep_ids = set(order[:k].tolist())
                keep_ids.update(anchor_ids)
                if two_stage_enable and block_dyn:
                    # Geometry skeleton preservation in dynamic blocks:
                    # keep a subset of low-|phi| hard-gated points so completeness/accuracy
                    # does not collapse when dynamic suppression is strong.
                    hard_ids = np.flatnonzero(hard_mask)
                    if hard_ids.size > 0:
                        hard_keep_n = max(4, int(np.ceil(0.30 * float(hard_ids.size))))
                        hard_keep_n = min(int(hard_ids.size), hard_keep_n)
                        hard_phi_order = hard_ids[np.argsort(p_vals[hard_ids])]
                        keep_ids.update(int(i) for i in hard_phi_order[:hard_keep_n].tolist())

                for i in sorted(keep_ids):
                    snef_keep.add(entries[i][0])

            if snef_keep:
                accepted_idx = snef_keep

        if not accepted_idx:
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

        pts: List[np.ndarray] = []
        nrm: List[np.ndarray] = []
        max_off = float(max(0.0, zero_crossing_max_offset))
        phi_gate = float(max(1e-4, zero_crossing_phi_gate))
        for idx, cell, center, n_i, _ in candidates:
            if idx not in accepted_idx:
                continue
            p = center
            if use_zero_crossing and abs(float(cell.phi)) <= phi_gate:
                off = -float(cell.phi) * n_i
                off_norm = float(np.linalg.norm(off))
                if max_off > 0.0 and off_norm > max_off:
                    off = off * (max_off / max(off_norm, 1e-9))
                p = center + off
            pts.append(p)
            nrm.append(n_i)

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
