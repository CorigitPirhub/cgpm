from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

from .config import EGF3DConfig
from .types import VoxelCell3D
from experiments.p10.cgcc import cgcc_conf as p10_cgcc_conf
from experiments.p10.cmct import cmct_conf as p10_cmct_conf
from experiments.p10.csr_xmap import counterfactual_static_readout as p10_counterfactual_static_readout
from experiments.p10.csr_xmap import dynamic_exclusion_readout as p10_dynamic_exclusion_readout
from experiments.p10.obl import obl_conf as p10_obl_conf
from experiments.p10.otv import otv_conf as p10_otv_conf
from experiments.p10.otv import otv_surface_conf as p10_otv_surface_conf
from experiments.p10.pfv import cross_map_fg_conf as p10_cross_map_fg_conf
from experiments.p10.pfv import pfv_cluster_conf as p10_pfv_cluster_conf
from experiments.p10.pfv import pfv_conf as p10_pfv_conf
from experiments.p10.pfv import pfv_exclusive_conf as p10_pfv_exclusive_conf
from experiments.p10.ptdsf import persistent_surface_readout as p10_persistent_surface_readout
from experiments.p10.ptdsf import ptdsf_state_stats as p10_ptdsf_state_stats
from experiments.p10.rps_admission import decay_protect_factors as p10_decay_protect_factors
from experiments.p10.rps_admission import rear_state_support as p10_rear_state_support
from experiments.p10.bg_manifold import manifold_state_components as p10_manifold_state_components
from experiments.p10.rps_selectivity import filter_rear_records as p10_filter_rear_records
from experiments.p10.rps_selectivity import rear_selectivity_components as p10_rear_selectivity_components
from experiments.p10.xmem import xmem_clear_conf as p10_xmem_clear_conf
from experiments.p10.xmem import xmem_conf as p10_xmem_conf

VoxelIndex = Tuple[int, int, int]


class VoxelHashMap3D:
    def __init__(self, cfg: EGF3DConfig):
        self.cfg = cfg
        self.voxel_size = cfg.map3d.voxel_size
        self.cells: Dict[VoxelIndex, VoxelCell3D] = {}
        self._neighbor_offset_cache: Dict[int, List[VoxelIndex]] = {}
        self.last_extract_stats: Dict[str, float] = {}

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
            phi_static=0.0,
            phi_static_w=0.0,
            phi_transient=0.0,
            phi_transient_w=0.0,
            p_static=0.5,
            phi_geo=0.0,
            phi_geo_w=0.0,
            phi_dyn=0.0,
            phi_dyn_w=0.0,
            rho=0.0,
            g_mean=np.zeros(3, dtype=float),
            g_cov=np.eye(3, dtype=float) * self.cfg.map3d.init_grad_cov,
            c_rho=np.zeros(3, dtype=float),
            d_score=0.0,
            dyn_prob=0.0,
            z_dyn=0.0,
            st_mem=0.0,
            surf_evidence=0.0,
            free_evidence=0.0,
            residual_evidence=0.0,
            phi_geo_bias=0.0,
            geo_res_ema=0.0,
            geo_res_hits=0.0,
            stcg_active=0.0,
            rho_prev=0.0,
            rho_osc=0.0,
            free_hit_ema=0.0,
            occ_hit_ema=0.0,
            visibility_contradiction=0.0,
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

    def _otv_conf(self, cell: VoxelCell3D) -> float:
        return p10_otv_conf(self, cell)


    def _otv_surface_conf(self, cell: VoxelCell3D) -> float:
        return p10_otv_surface_conf(self, cell)


    def _xmem_clear_conf(self, cell: VoxelCell3D) -> float:
        return p10_xmem_clear_conf(self, cell)


    def _xmem_conf(self, cell: VoxelCell3D) -> float:
        return p10_xmem_conf(self, cell)


    def _obl_conf(self, cell: VoxelCell3D) -> float:
        return p10_obl_conf(self, cell)


    def _cmct_conf(self, cell: VoxelCell3D) -> float:
        return p10_cmct_conf(self, cell)


    def _cgcc_conf(self, cell: VoxelCell3D) -> float:
        return p10_cgcc_conf(self, cell)


    def _pfv_conf(self, cell: VoxelCell3D) -> float:
        return p10_pfv_conf(self, cell)


    def _pfv_cluster_conf(self, idx: VoxelIndex) -> float:
        return p10_pfv_cluster_conf(self, idx)


    def _pfv_exclusive_conf(self, cell: VoxelCell3D) -> float:
        return p10_pfv_exclusive_conf(self, cell)


    def _cross_map_fg_conf(self, foreground_map: 'VoxelHashMap3D', idx: VoxelIndex) -> float:
        return p10_cross_map_fg_conf(self, foreground_map, idx)


    def _ptdsf_state_stats(self, cell: VoxelCell3D) -> Dict[str, float]:
        return p10_ptdsf_state_stats(self, cell)


    def _persistent_surface_readout(self, cell: VoxelCell3D) -> Tuple[float, float, float, Dict[str, float]] | None:
        return p10_persistent_surface_readout(self, cell)


    def _counterfactual_static_readout(self, cell: VoxelCell3D, geo_blend: float, geo_agree_min: float) -> Tuple[float, float, float, Dict[str, float]] | None:
        return p10_counterfactual_static_readout(self, cell, geo_blend, geo_agree_min)


    def _dynamic_exclusion_readout(self, cell: VoxelCell3D) -> Tuple[float, float, float, Dict[str, float]] | None:
        return p10_dynamic_exclusion_readout(self, cell)


    def _sync_legacy_channels(self, cell: VoxelCell3D) -> None:
        # Keep legacy phi/phi_w aligned with dual-state representation so
        # existing gradient/poisson paths remain compatible.
        ws = float(max(0.0, cell.phi_static_w))
        wt = float(max(0.0, cell.phi_transient_w))
        wr = float(max(0.0, getattr(cell, "phi_rear_w", 0.0)))
        wp = float(max(0.0, getattr(cell, "phi_spg_w", 0.0)))
        if ws <= 1e-12 and wt <= 1e-12 and wr <= 1e-12:
            return
        if bool(self.cfg.update.ptdsf_enable):
            prev_ctx = getattr(self, '_ptdsf_context', None)
            self._ptdsf_context = 'sync'
            try:
                persistent_read = self._persistent_surface_readout(cell)
            finally:
                if prev_ctx is None:
                    try:
                        delattr(self, '_ptdsf_context')
                    except AttributeError:
                        pass
                else:
                    self._ptdsf_context = prev_ctx
            if persistent_read is not None:
                phi_p, w_p, bias_p, stats = persistent_read
                persist_n = float(np.clip(0.55 * float(stats.get("static_conf", 0.0)) + 0.45 * float(stats.get("dominance", 0.0)), 0.0, 1.0))
                bias_conf = float(np.clip(getattr(cell, "zcbf_bias_conf", 0.0), 0.0, 1.0))
                bias_gain = float(np.clip(0.20 + 0.45 * persist_n, 0.20, 0.65))
                phi_sync = float(np.clip(phi_p - bias_gain * bias_conf * bias_p, -0.8, 0.8))
                cell.phi = phi_sync
                cell.phi_w = float(min(5000.0, w_p))
                return
        ps = float(np.clip(cell.p_static, 0.0, 1.0))
        rho_s = float(max(0.0, getattr(cell, 'rho_static', 0.0)))
        rho_t = float(max(0.0, getattr(cell, 'rho_transient', 0.0)))
        rho_sum = float(rho_s + rho_t)
        if rho_sum > 1e-8:
            ps = float(np.clip(0.72 * ps + 0.28 * (rho_s / rho_sum), 0.0, 1.0))
        if ws <= 1e-12:
            cell.phi = float(cell.phi_transient)
            cell.phi_w = float(min(5000.0, wt))
            return
        if wt <= 1e-12:
            cell.phi = float(cell.phi_static)
            cell.phi_w = float(min(5000.0, ws))
            return
        w_eff_s = float(max(1e-6, ps * ws))
        w_eff_t = float(max(1e-6, (1.0 - ps) * wt))
        w_sum = w_eff_s + w_eff_t
        phi_legacy = float((w_eff_s * cell.phi_static + w_eff_t * cell.phi_transient) / max(1e-9, w_sum))

        wg = float(max(0.0, cell.phi_geo_w))
        wd = float(max(0.0, cell.phi_dyn_w))
        if wg > 1e-8 or wd > 1e-8:
            denom = float(max(1e-6, wg + wd + w_sum))
            geo_rel = float(np.clip(wg / denom, 0.0, 1.0))
            dyn_rel = float(np.clip(wd / max(1e-6, wg + wd), 0.0, 1.0))
            k_geo = float(np.clip(0.20 * geo_rel + 0.16 * ps + 0.12 * (1.0 - dyn_rel), 0.0, 0.42))
            k_dyn = float(np.clip(0.18 * dyn_rel * (1.0 - ps), 0.0, 0.22))
            k_base = float(max(0.0, 1.0 - k_geo - k_dyn))
            phi_blend = float(k_base * phi_legacy + k_geo * float(cell.phi_geo) + k_dyn * float(cell.phi_dyn))
            cell.phi = float(np.clip(phi_blend, -0.8, 0.8))
        else:
            cell.phi = float(np.clip(phi_legacy, -0.8, 0.8))
        cell.phi_w = float(min(5000.0, w_sum))

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

    def decay_fields(
        self,
        mode: str = "local",
        global_dynamic_score: float = 0.0,
        dyn_forget_gain: float = 0.0,
        decay_steps: int = 1,
    ) -> None:
        cfg = self.cfg.map3d
        ucfg = self.cfg.update
        mode = str(mode).lower()
        dyn_gain = max(0.0, float(dyn_forget_gain))
        glob_dyn = float(np.clip(global_dynamic_score, 0.0, 1.0))
        steps = max(1, int(decay_steps))
        evidence_decay_pow = float(ucfg.evidence_decay ** steps)
        frontier_decay_pow = float(ucfg.frontier_decay ** steps)
        cov_inflation_pow = float(cfg.cov_inflation ** steps)
        clear_decay_pow = float(0.98 ** steps)
        stmem_decay_pow = float(np.clip(ucfg.stmem_decay, 0.7, 1.0) ** steps)
        geo_decay_pow = float(np.clip(cfg.phi_geo_w_decay, 0.55, 1.0) ** steps)
        dual_enable = bool(ucfg.dual_state_enable)
        dual_static_mult = float(max(0.2, ucfg.dual_state_static_decay_mult))
        dual_trans_mult = float(max(0.2, ucfg.dual_state_transient_decay_mult))
        to_delete: List[VoxelIndex] = []

        global_decay_only = bool(dyn_gain <= 1e-9 or mode == "off" or mode == "global")
        if global_decay_only:
            if dyn_gain <= 1e-9 or mode == "off":
                dyn_coeff_const = 0.0
            else:
                dyn_coeff_const = glob_dyn
            forget_strength = float(max(0.0, dyn_gain * dyn_coeff_const))
            rho_decay_eff = float(np.clip(cfg.rho_decay * np.exp(-1.8 * forget_strength), 0.55, 1.0))
            phi_decay_eff = float(np.clip(cfg.phi_w_decay * np.exp(-2.4 * forget_strength), 0.45, 1.0))
            rho_decay_pow = float(rho_decay_eff ** steps)
            phi_decay_pow = float(phi_decay_eff ** steps)
            for idx, cell in self.cells.items():
                cell.rho *= rho_decay_pow
                if dual_enable:
                    static_decay_eff = float(np.clip(cfg.phi_w_decay * np.exp(-0.8 * dual_static_mult * forget_strength), 0.55, 1.0))
                    trans_decay_eff = float(np.clip(cfg.phi_w_decay * np.exp(-2.4 * dual_trans_mult * forget_strength), 0.35, 1.0))
                    cell.phi_static_w *= float(static_decay_eff ** steps)
                    cell.phi_transient_w *= float(trans_decay_eff ** steps)
                    self._sync_legacy_channels(cell)
                else:
                    cell.phi_w *= phi_decay_pow
                cell.phi_geo_w *= geo_decay_pow
                cell.phi_dyn_w *= float(np.clip(cfg.phi_w_decay * np.exp(-2.1 * forget_strength), 0.30, 1.0) ** steps)
                cell.g_cov *= cov_inflation_pow
                cell.g_cov = np.clip(cell.g_cov, cfg.min_cov, cfg.max_cov)
                cell.surf_evidence *= evidence_decay_pow
                cell.free_evidence *= evidence_decay_pow
                cell.residual_evidence *= evidence_decay_pow
                cell.free_hit_ema *= evidence_decay_pow
                cell.occ_hit_ema *= evidence_decay_pow
                cell.visibility_contradiction *= evidence_decay_pow
                cell.clear_hits *= clear_decay_pow
                cell.st_mem *= stmem_decay_pow
                cell.rho_static *= evidence_decay_pow
                cell.rho_transient *= evidence_decay_pow
                cell.ptdsf_commit_age *= float(0.96**steps)
                cell.ptdsf_rollback_age *= float(0.96**steps)
                cell.zcbf_bias_conf *= float(0.985**steps)
                cell.dccm_free *= evidence_decay_pow
                cell.dccm_surface *= evidence_decay_pow
                cell.dccm_rear *= evidence_decay_pow
                cell.dccm_age *= float(np.clip(ucfg.dccm_age_decay, 0.70, 1.0) ** steps)
                cell.dccm_commit *= evidence_decay_pow
                cell.omhs_front_conf *= evidence_decay_pow
                cell.omhs_rear_conf *= evidence_decay_pow
                cell.omhs_gap *= clear_decay_pow
                cell.omhs_active *= evidence_decay_pow
                cell.wod_front_conf *= evidence_decay_pow
                cell.wod_rear_conf *= evidence_decay_pow
                cell.wod_shell_conf *= evidence_decay_pow
                rear_evidence_decay = evidence_decay_pow
                rear_geo_decay = geo_decay_pow
                if bool(getattr(ucfg, 'rps_rear_state_protect_enable', False)) and float(max(0.0, getattr(cell, 'phi_rear_w', 0.0))) > 1e-12:
                    support = float(p10_rear_state_support(self, cell))
                    factors = p10_decay_protect_factors(ucfg, support)
                    rear_evidence_decay = float(np.clip(evidence_decay_pow ** factors['exponent_scale'], 0.0, 1.0))
                    rear_geo_decay = float(np.clip(geo_decay_pow ** (0.70 * factors['exponent_scale'] + 0.30), 0.0, 1.0))
                cell.rho_rear *= rear_evidence_decay
                cell.phi_rear_w *= rear_geo_decay
                cell.phi_otv_w *= float(np.clip(getattr(ucfg, 'otv_decay', 0.96), 0.80, 1.0) ** steps)
                cell.rho_otv *= evidence_decay_pow
                cell.otv_score *= evidence_decay_pow
                cell.otv_age *= float(np.clip(getattr(ucfg, 'otv_decay', 0.96), 0.80, 1.0) ** steps)
                cell.otv_active *= float(np.clip(getattr(ucfg, 'otv_decay', 0.96), 0.80, 1.0) ** steps)
                cell.rho_bg_cand *= float(np.clip(getattr(ucfg, 'pfv_bg_candidate_decay', 0.94), 0.70, 0.999) ** steps)
                cell.phi_bg_cand_w *= float(np.clip(getattr(ucfg, 'pfv_bg_candidate_decay', 0.94), 0.70, 0.999) ** steps)
                cell.rho_bg_stable *= float(np.clip(getattr(ucfg, 'rps_bg_manifold_decay', 0.992), 0.80, 0.9999) ** steps)
                cell.phi_bg_memory_w *= float(np.clip(getattr(ucfg, 'rps_bg_manifold_decay', 0.992), 0.80, 0.9999) ** steps)
                cell.bg_visible_mem *= float(np.clip(getattr(ucfg, 'rps_bg_manifold_mem_decay', 0.996), 0.80, 0.9999) ** steps)
                cell.bg_obstruction_mem *= float(np.clip(getattr(ucfg, 'rps_bg_manifold_mem_decay', 0.996), 0.80, 0.9999) ** steps)
                cell.bg_cand_score *= evidence_decay_pow
                cell.bg_cand_age *= float(np.clip(getattr(ucfg, 'pfv_bg_candidate_decay', 0.94), 0.70, 0.999) ** steps)
                cell.bg_cand_active *= float(np.clip(getattr(ucfg, 'pfv_bg_candidate_decay', 0.94), 0.70, 0.999) ** steps)
                bank_decay = float(np.clip(getattr(ucfg, 'tri_map_delay_bank_decay', 0.96), 0.70, 0.999) ** steps)
                cell.phi_delayed_bank_w *= bank_decay
                cell.rho_delayed_bank *= bank_decay
                cell.g_delayed_bank_w *= bank_decay
                cell.delayed_bank_conf *= evidence_decay_pow
                cell.delayed_bank_age *= bank_decay
                cell.delayed_bank_active *= bank_decay
                cell.rho_rear_cand *= float(np.clip(ucfg.rps_candidate_decay, 0.70, 0.999) ** steps)
                cell.phi_rear_cand_w *= float(np.clip(ucfg.rps_candidate_decay, 0.70, 0.999) ** steps)
                cell.rps_commit_score *= evidence_decay_pow
                cell.rps_commit_age *= float(0.96**steps)
                rear_active_decay = float(np.clip(ucfg.rps_active_decay, 0.80, 0.999) ** steps)
                if bool(getattr(ucfg, 'rps_rear_state_protect_enable', False)) and float(max(0.0, getattr(cell, 'phi_rear_w', 0.0))) > 1e-12:
                    support = float(p10_rear_state_support(self, cell))
                    factors = p10_decay_protect_factors(ucfg, support)
                    rear_active_decay = float(np.clip(rear_active_decay ** factors['exponent_scale'], 0.0, 1.0))
                    cell.rps_active = float(max(float(getattr(cell, 'rps_active', 0.0)) * rear_active_decay, factors['active_floor']))
                else:
                    cell.rps_active *= rear_active_decay
                cell.geo_res_ema *= float(0.995**steps)
                cell.geo_res_hits *= float(0.98**steps)
                cell.frontier_score *= frontier_decay_pow
                if self.cfg.update.enable_evidence:
                    stale = bool(
                        cell.phi_w < 1e-3
                        and cell.phi_geo_w < 1e-3
                        and cell.phi_dyn_w < 1e-3
                        and cell.phi_rear_w < 1e-3
                        and cell.phi_rear_cand_w < 1e-3
                        and cell.phi_otv_w < 1e-3
                        and cell.phi_bg_cand_w < 1e-3
                        and cell.rho < 1e-4
                        and cell.frontier_score < 0.05
                    )
                else:
                    stale = bool(cell.phi_w < 1e-3 and cell.phi_geo_w < 1e-3 and cell.phi_dyn_w < 1e-3 and cell.phi_rear_w < 1e-3 and cell.phi_rear_cand_w < 1e-3 and cell.phi_otv_w < 1e-3 and cell.frontier_score < 0.05)
                if stale:
                    to_delete.append(idx)
        else:
            for idx, cell in self.cells.items():
                local_dyn = float(np.clip(cell.d_score, 0.0, 1.0))
                dyn_coeff = local_dyn
                # Dynamic-selective exponential forgetting:
                # keep static cells near base decay, but aggressively down-weight dynamic cells.
                forget_strength = float(max(0.0, dyn_gain * dyn_coeff))
                rho_decay_eff = float(np.clip(cfg.rho_decay * np.exp(-1.8 * forget_strength), 0.55, 1.0))
                phi_decay_eff = float(np.clip(cfg.phi_w_decay * np.exp(-2.4 * forget_strength), 0.45, 1.0))
                cell.rho *= float(rho_decay_eff ** steps)
                if dual_enable:
                    static_decay_eff = float(np.clip(cfg.phi_w_decay * np.exp(-0.8 * dual_static_mult * forget_strength), 0.55, 1.0))
                    trans_decay_eff = float(np.clip(cfg.phi_w_decay * np.exp(-2.4 * dual_trans_mult * forget_strength), 0.30, 1.0))
                    cell.phi_static_w *= float(static_decay_eff ** steps)
                    cell.phi_transient_w *= float(trans_decay_eff ** steps)
                    self._sync_legacy_channels(cell)
                else:
                    cell.phi_w *= float(phi_decay_eff ** steps)
                cell.phi_geo_w *= geo_decay_pow
                cell.phi_dyn_w *= float(np.clip(cfg.phi_w_decay * np.exp(-2.1 * forget_strength), 0.28, 1.0) ** steps)
                cell.g_cov *= cov_inflation_pow
                cell.g_cov = np.clip(cell.g_cov, cfg.min_cov, cfg.max_cov)
                cell.surf_evidence *= evidence_decay_pow
                cell.free_evidence *= evidence_decay_pow
                cell.residual_evidence *= evidence_decay_pow
                cell.free_hit_ema *= evidence_decay_pow
                cell.occ_hit_ema *= evidence_decay_pow
                cell.visibility_contradiction *= evidence_decay_pow
                cell.clear_hits *= clear_decay_pow
                cell.st_mem *= stmem_decay_pow
                cell.rho_static *= evidence_decay_pow
                cell.rho_transient *= evidence_decay_pow
                cell.ptdsf_commit_age *= float(0.96**steps)
                cell.ptdsf_rollback_age *= float(0.96**steps)
                cell.zcbf_bias_conf *= float(0.985**steps)
                cell.dccm_free *= evidence_decay_pow
                cell.dccm_surface *= evidence_decay_pow
                cell.dccm_rear *= evidence_decay_pow
                cell.dccm_age *= float(np.clip(ucfg.dccm_age_decay, 0.70, 1.0) ** steps)
                cell.dccm_commit *= evidence_decay_pow
                cell.omhs_front_conf *= evidence_decay_pow
                cell.omhs_rear_conf *= evidence_decay_pow
                cell.omhs_gap *= clear_decay_pow
                cell.omhs_active *= evidence_decay_pow
                cell.wod_front_conf *= evidence_decay_pow
                cell.wod_rear_conf *= evidence_decay_pow
                cell.wod_shell_conf *= evidence_decay_pow
                rear_evidence_decay = evidence_decay_pow
                rear_geo_decay = geo_decay_pow
                if bool(getattr(ucfg, 'rps_rear_state_protect_enable', False)) and float(max(0.0, getattr(cell, 'phi_rear_w', 0.0))) > 1e-12:
                    support = float(p10_rear_state_support(self, cell))
                    factors = p10_decay_protect_factors(ucfg, support)
                    rear_evidence_decay = float(np.clip(evidence_decay_pow ** factors['exponent_scale'], 0.0, 1.0))
                    rear_geo_decay = float(np.clip(geo_decay_pow ** (0.70 * factors['exponent_scale'] + 0.30), 0.0, 1.0))
                cell.rho_rear *= rear_evidence_decay
                cell.phi_rear_w *= rear_geo_decay
                cell.phi_otv_w *= float(np.clip(getattr(ucfg, 'otv_decay', 0.96), 0.80, 1.0) ** steps)
                cell.rho_otv *= evidence_decay_pow
                cell.otv_score *= evidence_decay_pow
                cell.otv_age *= float(np.clip(getattr(ucfg, 'otv_decay', 0.96), 0.80, 1.0) ** steps)
                cell.otv_active *= float(np.clip(getattr(ucfg, 'otv_decay', 0.96), 0.80, 1.0) ** steps)
                cell.rho_bg_cand *= float(np.clip(getattr(ucfg, 'pfv_bg_candidate_decay', 0.94), 0.70, 0.999) ** steps)
                cell.phi_bg_cand_w *= float(np.clip(getattr(ucfg, 'pfv_bg_candidate_decay', 0.94), 0.70, 0.999) ** steps)
                cell.rho_bg_stable *= float(np.clip(getattr(ucfg, 'rps_bg_manifold_decay', 0.992), 0.80, 0.9999) ** steps)
                cell.phi_bg_memory_w *= float(np.clip(getattr(ucfg, 'rps_bg_manifold_decay', 0.992), 0.80, 0.9999) ** steps)
                cell.bg_visible_mem *= float(np.clip(getattr(ucfg, 'rps_bg_manifold_mem_decay', 0.996), 0.80, 0.9999) ** steps)
                cell.bg_obstruction_mem *= float(np.clip(getattr(ucfg, 'rps_bg_manifold_mem_decay', 0.996), 0.80, 0.9999) ** steps)
                cell.bg_cand_score *= evidence_decay_pow
                cell.bg_cand_age *= float(np.clip(getattr(ucfg, 'pfv_bg_candidate_decay', 0.94), 0.70, 0.999) ** steps)
                cell.bg_cand_active *= float(np.clip(getattr(ucfg, 'pfv_bg_candidate_decay', 0.94), 0.70, 0.999) ** steps)
                bank_decay = float(np.clip(getattr(ucfg, 'tri_map_delay_bank_decay', 0.96), 0.70, 0.999) ** steps)
                cell.phi_delayed_bank_w *= bank_decay
                cell.rho_delayed_bank *= bank_decay
                cell.g_delayed_bank_w *= bank_decay
                cell.delayed_bank_conf *= evidence_decay_pow
                cell.delayed_bank_age *= bank_decay
                cell.delayed_bank_active *= bank_decay
                cell.rho_rear_cand *= float(np.clip(ucfg.rps_candidate_decay, 0.70, 0.999) ** steps)
                cell.phi_rear_cand_w *= float(np.clip(ucfg.rps_candidate_decay, 0.70, 0.999) ** steps)
                cell.rps_commit_score *= evidence_decay_pow
                cell.rps_commit_age *= float(0.96**steps)
                rear_active_decay = float(np.clip(ucfg.rps_active_decay, 0.80, 0.999) ** steps)
                if bool(getattr(ucfg, 'rps_rear_state_protect_enable', False)) and float(max(0.0, getattr(cell, 'phi_rear_w', 0.0))) > 1e-12:
                    support = float(p10_rear_state_support(self, cell))
                    factors = p10_decay_protect_factors(ucfg, support)
                    rear_active_decay = float(np.clip(rear_active_decay ** factors['exponent_scale'], 0.0, 1.0))
                    cell.rps_active = float(max(float(getattr(cell, 'rps_active', 0.0)) * rear_active_decay, factors['active_floor']))
                else:
                    cell.rps_active *= rear_active_decay
                cell.geo_res_ema *= float(0.995**steps)
                cell.geo_res_hits *= float(0.98**steps)
                cell.frontier_score *= frontier_decay_pow
                if self.cfg.update.enable_evidence:
                    stale = bool(
                        cell.phi_w < 1e-3
                        and cell.phi_geo_w < 1e-3
                        and cell.phi_dyn_w < 1e-3
                        and cell.phi_rear_w < 1e-3
                        and cell.phi_rear_cand_w < 1e-3
                        and cell.phi_bg_cand_w < 1e-3
                        and cell.rho < 1e-4
                        and cell.frontier_score < 0.05
                    )
                else:
                    stale = bool(cell.phi_w < 1e-3 and cell.phi_geo_w < 1e-3 and cell.phi_dyn_w < 1e-3 and cell.phi_rear_w < 1e-3 and cell.phi_rear_cand_w < 1e-3 and cell.frontier_score < 0.05)
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
        point_bias_along_normal_m: float = 0.0,
        use_phi_geo_channel: bool = False,
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
        lzcd_apply_in_extraction: bool = False,
        lzcd_bias_scale: float = 1.0,
        ptdsf_persistent_only_enable: bool = False,
        ptdsf_persistent_min_rho: float = 0.15,
        ptdsf_static_rho_weight: float = 0.35,
        zcbf_apply_in_extraction: bool = False,
        zcbf_bias_scale: float = 1.0,
        stcg_enable: bool = False,
        dccm_enable: bool = False,
        dccm_commit_weight: float = 0.30,
        dccm_static_guard: float = 0.65,
        dccm_drop_gain: float = 0.22,
        stcg_min_score: float = 0.35,
        stcg_rho_ref: float = 1.8,
        stcg_free_shrink: float = 0.45,
        stcg_phi_shrink: float = 0.25,
        stcg_dscore_shrink: float = 0.30,
        stcg_weight_gain: float = 0.50,
        stcg_static_protect: float = 0.70,
        use_dual_static_channel: bool = False,
        dual_p_static_min: float = 0.0,
        structural_decouple_enable: bool = True,
        decouple_min_geo_weight_ratio: float = 0.35,
        decouple_dyn_drop_thresh: float = 0.78,
        decouple_dyn_rho_guard: float = 1.2,
        decouple_dyn_free_ratio_thresh: float = 1.10,
        decouple_channel_div_enable: bool = False,
        decouple_channel_div_thresh: float = 0.04,
        decouple_channel_div_weight: float = 0.35,
        decouple_channel_div_static_guard: float = 0.70,
        decouple_stmem_enable: bool = True,
        decouple_stmem_weight: float = 0.55,
        decouple_stmem_static_guard: float = 0.55,
        decouple_stmem_rho_guard: float = 0.45,
        decouple_stmem_free_shrink: float = 0.12,
        dual_layer_extract_enable: bool = False,
        dual_layer_geo_min_weight_ratio: float = 0.30,
        dual_layer_dyn_use_zdyn: bool = True,
        dual_layer_dyn_prob_weight: float = 0.38,
        dual_layer_dyn_stmem_weight: float = 0.22,
        dual_layer_dyn_contra_weight: float = 0.20,
        dual_layer_dyn_transient_weight: float = 0.20,
        dual_layer_dyn_phi_div_weight: float = 0.16,
        dual_layer_dyn_phi_ratio_weight: float = 0.10,
        dual_layer_dyn_phi_div_ref: float = 0.04,
        dual_layer_dyn_use_phi_dyn: bool = True,
        dual_layer_compete_enable: bool = True,
        dual_layer_compete_margin: float = 0.08,
        dual_layer_compete_geo_weight: float = 0.62,
        dual_layer_compete_dyn_mix_weight: float = 0.55,
        dual_layer_compete_dyn_conf_weight: float = 0.25,
        dual_layer_dyn_drop_thresh: float = 0.72,
        dual_layer_dyn_free_ratio_min: float = 0.90,
        dual_layer_static_anchor_rho: float = 0.90,
        dual_layer_static_anchor_p: float = 0.70,
        dual_layer_static_anchor_ratio: float = 1.70,
        csr_enable: bool = False,
        csr_min_score: float = 0.38,
        csr_geo_blend: float = 0.18,
        csr_geo_agree_min: float = 0.70,
        xmap_enable: bool = False,
        xmap_dyn_min_score: float = 0.52,
        xmap_static_min_score: float = 0.42,
        xmap_sep_ref_vox: float = 0.90,
        omhs_enable: bool = False,
        dual_map_background_only: bool = False,
        foreground_map: 'VoxelHashMap3D | None' = None,
        ebcut_enable: bool = False,
        ebcut_energy_thresh: float = 0.58,
        ebcut_w_phi: float = 0.30,
        ebcut_w_dyn: float = 0.35,
        ebcut_w_free: float = 0.20,
        ebcut_w_conf: float = 0.15,
        ebcut_w_smooth: float = 0.10,
        ebcut_smooth_radius: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        candidates: List[Tuple[VoxelIndex, VoxelCell3D, np.ndarray, np.ndarray, float, float]] = []
        candidate_map: Dict[VoxelIndex, Tuple[VoxelCell3D, float, bool, bool, float, float, float, str]] = {}
        ebcut_rejects = 0
        prefilter_candidates = 0
        xmap_rescue_count = 0
        xmap_front_drop_count = 0
        rear_density_drops = 0
        geom_margin = float(max(0.0, two_stage_geom_margin))
        decouple = bool(structural_decouple_enable)
        dual_layer = bool(dual_layer_extract_enable)
        persistent_only = bool(ptdsf_persistent_only_enable)
        omhs = bool(omhs_enable)
        dual_geo_min_ratio = float(np.clip(dual_layer_geo_min_weight_ratio, 0.0, 1.0))
        prev_extract_ctx = getattr(self, '_ptdsf_context', None)
        self._ptdsf_context = 'extract'
        for idx, cell in self.cells.items():
            _read_stats = {}
            phi_thr = float(phi_thresh)
            min_w = float(min_weight)
            max_d_eff = float(max_d_score)
            max_free = float(max_free_ratio)
            geo_min_ratio = float(max(decouple_min_geo_weight_ratio, dual_geo_min_ratio if dual_layer else 0.0))
            geo_min_w = float(max(1e-6, geo_min_ratio * max(1e-6, min_w)))
            rho_stat = float(max(0.0, getattr(cell, 'rho_static', 0.0)))
            rho_trans = float(max(0.0, getattr(cell, 'rho_transient', 0.0)))
            rho_split_sum = float(rho_stat + rho_trans)
            ptdsf_ratio = float(np.clip(rho_stat / max(1e-6, rho_split_sum), 0.0, 1.0)) if rho_split_sum > 1e-8 else float(np.clip(cell.p_static, 0.0, 1.0))
            ptdsf_rho_conf = float(np.clip(rho_stat / max(1e-6, ptdsf_persistent_min_rho), 0.0, 1.5)) if persistent_only else 0.0
            ptdsf_stats = self._ptdsf_state_stats(cell)
            ptdsf_dom = float(ptdsf_stats['dominance'])
            persistent_read = self._persistent_surface_readout(cell) if bool(self.cfg.update.ptdsf_enable) else None
            omhs_front_conf = float(np.clip(getattr(cell, "omhs_front_conf", 0.0), 0.0, 1.0)) if omhs else 0.0
            omhs_rear_conf = float(np.clip(getattr(cell, "omhs_rear_conf", 0.0), 0.0, 1.0)) if omhs else 0.0
            omhs_active_score = float(np.clip(getattr(cell, "omhs_active", 0.0), 0.0, 1.0)) if omhs else 0.0
            omhs_active = bool(
                omhs
                and omhs_active_score >= 0.40
                and max(omhs_front_conf, omhs_rear_conf) >= 0.18
            )
            omhs_rear_keep = False
            if decouple:
                if bool(dual_map_background_only) and float(getattr(cell, 'phi_bg_w', 0.0)) > 1e-8:
                    bg_conf = float(self._obl_conf(cell))
                    cmct_conf = float(self._cmct_conf(cell))
                    bg_rho = float(max(0.0, getattr(cell, 'rho_bg', 0.0)))
                    if persistent_only and max(bg_rho, rho_stat, float(cell.rho)) < 0.60 * ptdsf_persistent_min_rho:
                        continue
                    fg_cross_conf = float(self._cross_map_fg_conf(foreground_map, idx)) if (foreground_map is not None) else 0.0
                    cgcc_conf = float(self._cgcc_conf(cell))
                    pfv_cluster = float(self._pfv_cluster_conf(idx))
                    pfv_conf = float(max(self._pfv_conf(cell), float(getattr(self.cfg.update, 'pfv_cluster_weight', 0.35)) * pfv_cluster))
                    pfv_exclusive = float(self._pfv_exclusive_conf(cell)) if bool(getattr(self.cfg.update, 'pfv_exclusive_enable', False)) else 0.0
                    if cmct_conf >= 0.40 and bg_conf < 0.60 and bg_rho < 0.90 * ptdsf_persistent_min_rho:
                        continue
                    if fg_cross_conf >= 0.38 and bg_conf < 0.72 and bg_rho < 1.20 * ptdsf_persistent_min_rho:
                        continue
                    if cgcc_conf >= 0.34 and bg_conf < 0.78 and bg_rho < 1.30 * ptdsf_persistent_min_rho:
                        continue
                    if pfv_conf >= max(float(getattr(self.cfg.update, 'pfv_bank_extract_thresh', 0.48)), float(getattr(self.cfg.update, 'pfv_extract_thresh', 0.36))) and bg_conf < 0.88 and bg_rho < 1.50 * ptdsf_persistent_min_rho:
                        continue
                    if pfv_exclusive >= float(getattr(self.cfg.update, 'pfv_exclusive_extract_thresh', 0.52)) and bg_conf < 0.92 and bg_rho < 1.80 * ptdsf_persistent_min_rho:
                        continue
                    if bg_conf >= 0.12 or bg_rho >= 0.50 * ptdsf_persistent_min_rho:
                        phi_eff = float(getattr(cell, 'phi_bg', 0.0))
                        weight_eff = float(getattr(cell, 'phi_bg_w', 0.0))
                        bias_eff = float(getattr(cell, 'zcbf_bias', 0.0))
                    elif persistent_read is not None and (persistent_only or dual_layer or ptdsf_dom >= 0.45):
                        phi_eff, weight_eff, bias_eff, _read_stats = persistent_read
                        if persistent_only and max(rho_stat, float(cell.rho)) < ptdsf_persistent_min_rho:
                            continue
                        if weight_eff < max(1e-6, 0.75 * geo_min_w):
                            continue
                    elif float(cell.phi_geo_w) >= geo_min_w:
                        phi_eff = float(cell.phi_geo)
                        weight_eff = float(cell.phi_geo_w)
                        bias_eff = float(cell.phi_geo_bias)
                    else:
                        phi_eff = float(cell.phi)
                        weight_eff = float(cell.phi_w)
                        bias_eff = float(cell.phi_bias)
                elif persistent_read is not None and (persistent_only or dual_layer or ptdsf_dom >= 0.45):
                    phi_eff, weight_eff, bias_eff, _read_stats = persistent_read
                    if persistent_only and max(rho_stat, float(cell.rho)) < ptdsf_persistent_min_rho:
                        continue
                    if weight_eff < max(1e-6, 0.75 * geo_min_w):
                        continue
                elif float(cell.phi_geo_w) >= geo_min_w:
                    phi_eff = float(cell.phi_geo)
                    weight_eff = float(cell.phi_geo_w)
                    bias_eff = float(cell.phi_geo_bias)
                elif bool(use_dual_static_channel) and (float(cell.phi_static_w) > 1e-8 or float(cell.phi_w) > 1e-8):
                    w_s = float(max(0.0, cell.phi_static_w))
                    w_g = float(max(0.0, cell.phi_geo_w))
                    w_l = float(max(0.0, cell.phi_w))
                    w_es = 0.82 * w_s + 0.18 * w_g
                    w_el = 0.15 * w_l
                    w_sum = float(w_es + w_el)
                    if w_sum <= 1e-9:
                        continue
                    phi_eff = float((w_es * cell.phi_static + w_el * cell.phi) / w_sum)
                    weight_eff = float(min(5000.0, w_sum))
                    bias_eff = float((w_es * float(getattr(cell, 'zcbf_bias', 0.0)) + w_el * cell.phi_bias) / max(1e-9, w_sum))
                elif bool(use_phi_geo_channel) and float(cell.phi_geo_w) > 1e-8:
                    phi_eff = float(cell.phi_geo)
                    weight_eff = float(cell.phi_geo_w)
                    bias_eff = float(cell.phi_geo_bias)
                else:
                    phi_eff = float(cell.phi)
                    weight_eff = float(cell.phi_w)
                    bias_eff = float(cell.phi_bias)
                if omhs_active and bool(self.cfg.update.ptdsf_enable) and float(cell.phi_static_w) > 1e-8:
                    ws_h = float(max(0.0, cell.phi_static_w))
                    wg_h = float(max(0.0, cell.phi_geo_w))
                    if wg_h > 1e-8:
                        div_ref_h = float(max(1e-6, 2.0 * self.voxel_size))
                        div_h = float(abs(float(cell.phi_geo) - float(cell.phi_static)))
                        geo_agree_h = float(np.exp(-0.5 * (div_h / div_ref_h) * (div_h / div_ref_h)))
                    else:
                        geo_agree_h = 0.85
                    w_s_h = float(ws_h * (1.0 + 0.45 * omhs_rear_conf))
                    w_g_h = float(wg_h * (0.20 + 0.40 * omhs_rear_conf) * (0.30 + 0.70 * geo_agree_h))
                    w_h = float(w_s_h + w_g_h)
                    if w_h > 1e-8:
                        phi_eff = float((w_s_h * float(cell.phi_static) + w_g_h * float(cell.phi_geo)) / w_h)
                        weight_eff = float(max(weight_eff, min(5000.0, w_h)))
                        bias_eff = float((w_s_h * float(getattr(cell, 'zcbf_bias', 0.0)) + w_g_h * (float(getattr(cell, 'zcbf_bias', 0.0)) + float(cell.phi_geo_bias))) / max(1e-9, w_h))
                        phi_thr = float(max(phi_thr, float(phi_thresh) * (1.05 + 0.15 * omhs_rear_conf)))
                        omhs_rear_keep = bool(omhs_rear_conf >= 0.48 and omhs_front_conf >= 0.28 and ptdsf_dom >= 0.40)
            elif bool(use_dual_static_channel) and (
                float(cell.phi_static_w) > 1e-8 or float(cell.phi_geo_w) > 1e-8 or float(cell.phi_w) > 1e-8
            ):
                # SNEF-3D local (v2): adaptive mixed extraction.
                # Instead of hard dropping low p_static voxels, combine static/geo/legacy
                # channels with p_static-dependent weights to preserve completeness.
                p_s = float(np.clip(max(cell.p_static, ptdsf_ratio if persistent_only else 0.0), 0.0, 1.0))
                p_ref = float(max(1e-6, dual_p_static_min))
                # Use thresholded static confidence instead of p_s/p_ref saturation.
                static_ratio = float(np.clip((p_s - p_ref) / max(1e-6, 1.0 - p_ref), 0.0, 1.0))
                dyn_relax = float(np.clip(1.0 - 0.7 * cell.d_score, 0.15, 1.0))
                rho_relax = float(np.clip(cell.rho / max(1e-6, stcg_rho_ref), 0.15, 1.5))
                static_ratio = float(np.clip(static_ratio * (0.65 + 0.35 * rho_relax) * dyn_relax, 0.0, 1.0))

                w_s = float(max(0.0, cell.phi_static_w))
                w_g = float(max(0.0, cell.phi_geo_w))
                w_l = float(max(0.0, cell.phi_w))
                # Keep a minimum auxiliary budget for geo/legacy channels so completeness
                # does not collapse when p_static is high but static channel is sparse.
                aux_min = float(np.clip(0.18 + 0.22 * (1.0 - np.clip(rho_relax, 0.0, 1.0)) + 0.10 * cell.d_score, 0.18, 0.55))
                aux_budget = float(max(1.0 - static_ratio, aux_min))
                blend_geo = float(aux_budget * np.clip(0.62 + 0.18 * rho_relax - 0.25 * cell.d_score, 0.20, 0.88))
                blend_leg = float(aux_budget * np.clip(0.34 + 0.18 * (1.0 - cell.d_score), 0.10, 0.62))
                w_es = static_ratio * w_s
                w_eg = blend_geo * w_g
                w_el = blend_leg * w_l
                w_sum = float(w_es + w_eg + w_el)
                if w_sum <= 1e-9:
                    continue
                phi_eff = float((w_es * cell.phi_static + w_eg * cell.phi_geo + w_el * cell.phi) / w_sum)
                weight_eff = float(min(5000.0, w_sum))
                bias_eff = float((w_eg * cell.phi_geo_bias + (w_es + w_el) * cell.phi_bias) / max(1e-9, w_sum))
            elif bool(use_phi_geo_channel) and float(cell.phi_geo_w) > 1e-8:
                phi_eff = float(cell.phi_geo)
                weight_eff = float(cell.phi_geo_w)
                bias_eff = float(cell.phi_geo_bias)
            else:
                phi_eff = float(cell.phi)
                weight_eff = float(cell.phi_w)
                bias_eff = float(cell.phi_bias)
            xmap_rescue = False
            csr_score = 0.0
            xmap_score = 0.0
            xmap_sep = 0.0
            if bool(csr_enable) or bool(xmap_enable):
                csr_read = self._counterfactual_static_readout(
                    cell,
                    geo_blend=float(csr_geo_blend),
                    geo_agree_min=float(csr_geo_agree_min),
                ) if bool(csr_enable) else None
                xmap_read = self._dynamic_exclusion_readout(cell) if bool(xmap_enable) else None
                if csr_read is not None:
                    csr_phi, csr_w, csr_bias, csr_stats = csr_read
                    csr_score = float(np.clip(csr_stats.get("score", 0.0), 0.0, 1.0))
                if xmap_read is not None:
                    xmap_phi, _xmap_w, xmap_score, _xmap_stats = xmap_read
                    ref_phi = float(csr_phi) if csr_read is not None else float(phi_eff)
                    sep_ref = float(max(1e-6, max(0.50 * self.voxel_size, float(xmap_sep_ref_vox) * self.voxel_size)))
                    xmap_sep = float(np.clip(abs(float(xmap_phi) - ref_phi) / sep_ref, 0.0, 2.0))
                    sign_split = bool(float(xmap_phi) * ref_phi <= 0.0)
                    xmap_strong = bool(
                        float(xmap_score) >= float(xmap_dyn_min_score)
                        and (xmap_sep >= 1.0 or (sign_split and xmap_sep >= 0.55))
                    )
                    if xmap_strong:
                        static_min = float(max(float(csr_min_score), float(xmap_static_min_score)))
                        if csr_read is not None and float(csr_score) >= static_min and float(csr_w) >= max(1e-6, 0.60 * geo_min_w):
                            phi_eff = float(csr_phi)
                            weight_eff = float(max(weight_eff, csr_w))
                            bias_eff = float(csr_bias)
                            xmap_rescue = True
                            xmap_rescue_count += 1
                        else:
                            xmap_front_drop_count += 1
                            continue
            total_bias = 0.0
            if bool(lzcd_apply_in_extraction):
                total_bias += float(lzcd_bias_scale) * bias_eff
            if bool(zcbf_apply_in_extraction):
                total_bias += float(zcbf_bias_scale) * float(getattr(cell, 'zcbf_bias', 0.0)) * float(np.clip(getattr(cell, 'zcbf_bias_conf', 0.0), 0.0, 1.0))
            if abs(total_bias) > 1e-12:
                phi_eff = float(phi_eff - total_bias)
            if adaptive_enable:
                # Bonn adaptive extraction:
                # high-rho + low-dscore cells get looser phi gate;
                # high-dscore cells get stricter phi/free gates + higher min weight.
                rho_ref = float(max(1e-6, adaptive_rho_ref))
                rho_n = float(np.clip(cell.rho / rho_ref, 0.0, 2.0))
                d_raw = float(np.clip(cell.dyn_prob if decouple else cell.d_score, 0.0, 1.0))
                # In explicit dual-layer mode, keep geometry candidate gates
                # mostly static-driven; dynamic suppression is handled in Stage-D.
                d_n = float(d_raw if (not dual_layer) else 0.35 * d_raw)
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
            if stcg_enable:
                stcg = float(np.clip(cell.stcg_score, 0.0, 1.0))
                if float(cell.stcg_active) >= 0.5:
                    # Hysteresis-held active state keeps suppression stable.
                    stcg = max(stcg, float(stcg_min_score) + 0.08)
                vis_c = float(np.clip(cell.visibility_contradiction, 0.0, 1.0))
                res_c = float(np.clip(cell.residual_evidence, 0.0, 1.0))
                clear_c = float(np.clip(cell.clear_hits / 1.8, 0.0, 1.0))
                stcg_mix = float(np.clip(0.52 * stcg + 0.20 * vis_c + 0.16 * res_c + 0.12 * clear_c, 0.0, 1.0))
                rho_n = float(np.clip(cell.rho / max(1e-6, stcg_rho_ref), 0.0, 2.0))
                static_factor = float(
                    np.clip(
                        rho_n
                        * stcg_static_protect
                        * (0.55 + 0.45 * np.clip(cell.p_static, 0.0, 1.0))
                        * (1.0 - 0.25 * np.clip((cell.dyn_prob if decouple else cell.d_score), 0.0, 1.0)),
                        0.0,
                        1.0,
                    )
                )
                dyn_act = float(np.clip(stcg_mix - float(stcg_min_score), 0.0, 1.0) * (1.0 - static_factor))
                if dyn_act > 1e-6:
                    if not dual_layer:
                        phi_thr = max(1e-4, phi_thr * (1.0 - float(stcg_phi_shrink) * dyn_act))
                        max_free = max(1e-6, max_free * (1.0 - float(stcg_free_shrink) * dyn_act))
                        min_w = max(0.0, min_w * (1.0 + float(stcg_weight_gain) * dyn_act))
                        max_d_eff = max(0.0, max_d_eff - float(stcg_dscore_shrink) * dyn_act)
                        weight_eff = float(max(0.0, weight_eff * (1.0 - 0.12 * dyn_act)))
            if two_stage_enable:
                phi_thr = max(phi_thr, phi_thr + geom_margin)
                min_w = max(0.0, min_w * 0.85)
                max_free = max(max_free, float(max_free_ratio) * 1.15)

            if abs(phi_eff) > phi_thr:
                continue
            if cell.rho < rho_thresh:
                continue
            if weight_eff < min_w:
                continue
            if (int(current_step) - int(cell.last_seen)) > int(max_age_frames):
                continue
            free_ratio = float(cell.free_evidence / max(1e-6, cell.surf_evidence))
            dyn_src = float(cell.dyn_prob if decouple else cell.d_score)
            dyn_src = float(max(dyn_src, self._xmem_conf(cell), self._otv_conf(cell), self._otv_surface_conf(cell)))
            if dual_layer and bool(dual_layer_dyn_use_zdyn):
                dyn_src = float(max(dyn_src, getattr(cell, "z_dyn", 0.0)))
            dyn_val = float(np.clip(dyn_src, 0.0, 1.0))
            dyn_gate_eval = dyn_val
            st_eff = 0.0
            if decouple:
                contra = float(
                    np.clip(
                        0.55 * cell.stcg_score + 0.25 * cell.visibility_contradiction + 0.20 * cell.residual_evidence,
                        0.0,
                        1.0,
                    )
                )
                dyn_gate = float(max(dyn_val, contra))
                dyn_gate_eval = dyn_gate
                # Cross-channel contradiction gating:
                # if geometry channel and legacy/static channel disagree, raise dynamic gate
                # without changing geometry-channel point placement.
                if bool(decouple_channel_div_enable) and float(cell.phi_geo_w) > 1e-8:
                    if float(cell.phi_static_w) > 1e-8:
                        phi_ref = float(cell.phi_static)
                    else:
                        phi_ref = float(cell.phi)
                    div = float(abs(float(cell.phi_geo) - phi_ref))
                    div_n = float(np.clip(div / max(1e-6, float(decouple_channel_div_thresh)), 0.0, 2.0))
                    static_guard = float(np.clip(cell.p_static, 0.0, 1.0))
                    div_boost = float(
                        np.clip(
                            float(decouple_channel_div_weight)
                            * div_n
                            * (1.0 - float(decouple_channel_div_static_guard) * static_guard),
                            0.0,
                            1.0,
                        )
                    )
                    dyn_gate = float(max(dyn_gate, div_boost))
                    if div_boost > 1e-6:
                        max_free = float(max(1e-6, max_free * (1.0 - 0.25 * div_boost)))
                # Short-term contradiction memory gate:
                # suppression-only transient cue; does not modify geometry channels.
                if bool(decouple_stmem_enable):
                    st_n = float(np.clip(cell.st_mem, 0.0, 1.0))
                    if st_n > 1e-8:
                        p_static_n = float(np.clip(cell.p_static, 0.0, 1.0))
                        rho_n = float(np.clip(cell.rho / max(1e-6, stcg_rho_ref), 0.0, 1.0))
                        st_eff = float(
                            np.clip(
                                st_n
                                * (1.0 - float(decouple_stmem_static_guard) * p_static_n)
                                * (1.0 - float(decouple_stmem_rho_guard) * rho_n),
                                0.0,
                                1.0,
                            )
                        )
                    dyn_gate = float(max(dyn_gate, float(decouple_stmem_weight) * st_eff))
                    if st_eff > 1e-8:
                        max_free = float(max(1e-6, max_free * (1.0 - float(decouple_stmem_free_shrink) * st_eff)))
                if not dual_layer:
                    if (
                        dyn_gate >= float(decouple_dyn_drop_thresh)
                        and float(cell.rho) <= float(decouple_dyn_rho_guard)
                        and free_ratio >= float(decouple_dyn_free_ratio_thresh)
                    ):
                        continue
                # Strong contradiction bypass:
                # allow short-memory dynamic suppressor to prune persistent ghost
                # candidates even when rho is moderately high, while still requiring
                # free-space support and high contradiction confidence.
                if bool(decouple_stmem_enable) and (not dual_layer):
                    stmem_drop = bool(
                        st_eff >= 0.72
                        and free_ratio >= max(0.8, 0.8 * float(decouple_dyn_free_ratio_thresh))
                        and float(cell.rho) <= 1.35 * float(decouple_dyn_rho_guard)
                    )
                    if stmem_drop:
                        continue
            if (not dual_layer) and free_ratio > max_free:
                continue
            if cell.free_evidence >= prune_free_min and cell.residual_evidence >= prune_residual_min:
                continue
            if cell.clear_hits > max_clear_hits:
                continue
            if (not dual_layer) and dyn_val > max_d_eff:
                continue
            prefilter_candidates += 1
            if bool(ebcut_enable):
                phi_n = float(np.clip(abs(float(phi_eff)) / max(1e-6, phi_thr), 0.0, 1.5))
                dyn_n = float(np.clip(dyn_gate_eval, 0.0, 1.0))
                free_ref = float(max(1.0, max_free if max_free < 1e8 else 2.0))
                free_n = float(np.clip(free_ratio / max(1e-6, free_ref), 0.0, 1.5))
                rho_ref = float(max(1.0, rho_thresh + 1.0))
                conf_n = float(1.0 - np.clip(float(cell.rho) / rho_ref, 0.0, 1.0))
                smooth_n = 0.0
                sm_r = int(max(0, ebcut_smooth_radius))
                if sm_r > 0 and float(ebcut_w_smooth) > 1e-9:
                    diffs: List[float] = []
                    for nidx in self.neighbor_indices(idx, sm_r):
                        if nidx == idx:
                            continue
                        cn = self.get_cell(nidx)
                        if cn is None:
                            continue
                        if decouple and float(cn.phi_geo_w) >= geo_min_w:
                            phi_nb = float(cn.phi_geo)
                        elif bool(use_phi_geo_channel) and float(cn.phi_geo_w) > 1e-8:
                            phi_nb = float(cn.phi_geo)
                        else:
                            phi_nb = float(cn.phi)
                        diffs.append(abs(float(phi_eff) - phi_nb))
                    if diffs:
                        smooth_n = float(np.clip(float(np.mean(diffs)) / max(1e-6, 1.5 * phi_thr), 0.0, 1.5))
                wp = float(max(0.0, ebcut_w_phi))
                wd = float(max(0.0, ebcut_w_dyn))
                wf = float(max(0.0, ebcut_w_free))
                wc = float(max(0.0, ebcut_w_conf))
                ws = float(max(0.0, ebcut_w_smooth))
                wsum = max(1e-6, wp + wd + wf + wc + ws)
                energy = float((wp * phi_n + wd * dyn_n + wf * free_n + wc * conf_n + ws * smooth_n) / wsum)
                if energy > float(max(0.0, ebcut_energy_thresh)):
                    ebcut_rejects += 1
                    continue
            g = np.asarray(cell.g_mean, dtype=float)
            gn = np.linalg.norm(g)
            if gn < 1e-7:
                continue
            center = self.index_to_center(idx)
            if abs(float(point_bias_along_normal_m)) > 1e-9:
                center = center + float(point_bias_along_normal_m) * (g / gn)
            candidates.append((idx, cell, center, g / gn, free_ratio, phi_eff))
            bank_selected_tag = str(_read_stats.get("bank_selected", "unknown")) if (persistent_read is not None and "_read_stats" in locals()) else "unknown"
            candidate_map[idx] = (
                cell,
                free_ratio,
                omhs_rear_keep,
                xmap_rescue,
                float(csr_score),
                float(xmap_score),
                float(xmap_sep),
                bank_selected_tag,
                float(_read_stats.get("front_score", 0.0)) if "_read_stats" in locals() else 0.0,
                float(_read_stats.get("rear_score", 0.0)) if "_read_stats" in locals() else 0.0,
                float(_read_stats.get("rear_gap", 0.0)) if "_read_stats" in locals() else 0.0,
                float(_read_stats.get("rear_sep", 0.0)) if "_read_stats" in locals() else 0.0,
            )
        if not candidates:
            self.last_extract_stats = {
                "candidates_prefilter": float(prefilter_candidates),
                "candidates_after_filter": 0.0,
                "ebcut_rejects": float(ebcut_rejects),
                "ebcut_reject_ratio": float(ebcut_rejects / max(1, prefilter_candidates)),
                "xmap_rescues": float(xmap_rescue_count),
                "xmap_front_drops": float(xmap_front_drop_count),
                "accepted_points": 0.0,
                "rear_density_drops": float(rear_density_drops),
                "rear_density_drops": float(rear_density_drops),
        }
            if prev_extract_ctx is None:
                try:
                    delattr(self, '_ptdsf_context')
                except AttributeError:
                    pass
            else:
                self._ptdsf_context = prev_extract_ctx
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

        accepted_idx: set[VoxelIndex]
        if not consistency_enable:
            accepted_idx = {idx for idx, _, _, _, _, _ in candidates}
        else:
            cand_map = {idx: (cell, n, phi_eff) for idx, cell, _, n, _, phi_eff in candidates}
            accepted_idx = set()
            r = max(1, int(consistency_radius))
            min_n = max(0, int(consistency_min_neighbors))
            cos_th = float(np.clip(consistency_normal_cos, 0.0, 1.0))
            phi_th = float(max(1e-4, consistency_phi_diff))
            if two_stage_enable:
                min_n = max(0, min_n - 2)
                cos_th = max(0.0, cos_th - 0.10)
                phi_th = max(1e-4, phi_th + geom_margin)
            for idx, cell, _, n_i, _, phi_i_eff in candidates:
                consistent = 0
                for nidx in self.neighbor_indices(idx, r):
                    if nidx == idx:
                        continue
                    other = cand_map.get(nidx)
                    if other is None:
                        continue
                    _c_j, n_j, phi_j_eff = other
                    if abs(float(phi_j_eff) - float(phi_i_eff)) > phi_th:
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

            groups: Dict[VoxelIndex, List[Tuple[VoxelIndex, VoxelCell3D, float, float]]] = {}
            for idx, cell, _, _, free_ratio, phi_eff in candidates:
                if idx not in accepted_idx:
                    continue
                bidx = (idx[0] // block_size, idx[1] // block_size, idx[2] // block_size)
                groups.setdefault(bidx, []).append((idx, cell, float(max(0.0, free_ratio)), float(phi_eff)))

            snef_keep: set[VoxelIndex] = set()
            dyn_block_d_thr = 1.0
            dyn_block_f_thr = 1.0
            dyn_block_rho_thr = 0.0
            dyn_block_osc_thr = 1.0
            if two_stage_enable and groups:
                all_d_vals = np.asarray(
                    [
                        float(np.clip((cell.dyn_prob if decouple else cell.d_score), 0.0, 1.0))
                        for entries in groups.values()
                        for _, cell, _, _ in entries
                    ],
                    dtype=float,
                )
                all_f_vals = np.asarray([float(fr) for entries in groups.values() for _, _, fr, _ in entries], dtype=float)
                all_rho_vals = np.asarray(
                    [float(max(0.0, cell.rho)) for entries in groups.values() for _, cell, _, _ in entries],
                    dtype=float,
                )
                all_osc_vals = np.asarray(
                    [float(max(0.0, cell.rho_osc)) for entries in groups.values() for _, cell, _, _ in entries],
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
                    snef_keep.update([idx for idx, _, _, _ in entries])
                    continue

                d_vals = np.asarray(
                    [float(np.clip((c.dyn_prob if decouple else c.d_score), 0.0, 1.0)) for _, c, _, _ in entries],
                    dtype=float,
                )
                f_vals = np.asarray([float(fr) for _, _, fr, _ in entries], dtype=float)
                p_vals = np.asarray([abs(float(phi_eff)) for _, _, _, phi_eff in entries], dtype=float)
                rho_vals = np.asarray([float(max(0.0, c.rho)) for _, c, _, _ in entries], dtype=float)
                osc_vals = np.asarray([float(max(0.0, c.rho_osc)) for _, c, _, _ in entries], dtype=float)

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
                        snef_keep.update([idx for idx, _, _, _ in entries])
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

        if accepted_idx and dual_layer:
            w_dyn = float(max(0.0, dual_layer_dyn_prob_weight))
            w_st = float(max(0.0, dual_layer_dyn_stmem_weight))
            w_contra = float(max(0.0, dual_layer_dyn_contra_weight))
            w_trans = float(max(0.0, dual_layer_dyn_transient_weight))
            w_phi_div = float(max(0.0, dual_layer_dyn_phi_div_weight))
            w_phi_ratio = float(max(0.0, dual_layer_dyn_phi_ratio_weight))
            ws = max(1e-6, w_dyn + w_st + w_contra + w_trans + w_phi_div + w_phi_ratio)
            drop_th = float(np.clip(dual_layer_dyn_drop_thresh, 0.0, 1.2))
            free_min = float(max(0.0, dual_layer_dyn_free_ratio_min))
            anchor_rho = float(max(0.0, dual_layer_static_anchor_rho))
            anchor_p = float(np.clip(dual_layer_static_anchor_p, 0.0, 1.0))
            anchor_ratio = float(max(1e-6, dual_layer_static_anchor_ratio))
            phi_div_ref = float(max(1e-6, dual_layer_dyn_phi_div_ref))
            comp_margin = float(max(0.0, dual_layer_compete_margin))
            comp_geo_w = float(np.clip(dual_layer_compete_geo_weight, 0.0, 1.0))
            comp_dyn_mix_w = float(np.clip(dual_layer_compete_dyn_mix_weight, 0.0, 1.0))
            comp_dyn_conf_w = float(np.clip(dual_layer_compete_dyn_conf_weight, 0.0, 1.0))
            comp_dyn_rest_w = float(max(0.0, 1.0 - comp_dyn_mix_w - comp_dyn_conf_w))
            dyn_env_list: List[float] = []
            for idx in accepted_idx:
                info = candidate_map.get(idx)
                if info is None:
                    continue
                cell = info[0]
                dyn_primary = float(np.clip(max(cell.dyn_prob, self._xmem_conf(cell), self._otv_conf(cell), self._otv_surface_conf(cell)), 0.0, 1.0))
                if bool(dual_layer_dyn_use_zdyn):
                    dyn_primary = float(np.clip(max(dyn_primary, getattr(cell, "z_dyn", 0.0)), 0.0, 1.0))
                st_n = float(np.clip(cell.st_mem, 0.0, 1.0))
                contra = float(
                    np.clip(
                        0.55 * cell.stcg_score + 0.25 * cell.visibility_contradiction + 0.20 * cell.residual_evidence,
                        0.0,
                        1.0,
                    )
                )
                dyn_env_list.append(float(np.clip(0.52 * dyn_primary + 0.24 * st_n + 0.24 * contra, 0.0, 1.0)))
            if dyn_env_list:
                global_dyn_level = float(np.clip(np.quantile(np.asarray(dyn_env_list, dtype=float), 0.70), 0.0, 1.0))
            else:
                global_dyn_level = 0.0
            keep_idx: set[VoxelIndex] = set()
            for idx in accepted_idx:
                info = candidate_map.get(idx)
                if info is None:
                    continue
                cell, free_ratio, omhs_keep, xmap_rescue, csr_score, xmap_score, xmap_sep, bank_selected_tag = info[:8]
                omhs_front_conf = float(np.clip(getattr(cell, "omhs_front_conf", 0.0), 0.0, 1.0)) if omhs else 0.0
                omhs_rear_conf = float(np.clip(getattr(cell, "omhs_rear_conf", 0.0), 0.0, 1.0)) if omhs else 0.0
                omhs_active_score = float(np.clip(getattr(cell, "omhs_active", 0.0), 0.0, 1.0)) if omhs else 0.0
                omhs_active = bool(
                    omhs
                    and omhs_active_score >= 0.40
                    and max(omhs_front_conf, omhs_rear_conf) >= 0.18
                )
                dyn_primary = float(np.clip(max(cell.dyn_prob, self._xmem_conf(cell), self._otv_conf(cell), self._otv_surface_conf(cell)), 0.0, 1.0))
                if bool(dual_layer_dyn_use_zdyn):
                    dyn_primary = float(np.clip(max(dyn_primary, getattr(cell, "z_dyn", 0.0)), 0.0, 1.0))
                st_n = float(np.clip(cell.st_mem, 0.0, 1.0))
                contra = float(
                    np.clip(
                        0.55 * cell.stcg_score + 0.25 * cell.visibility_contradiction + 0.20 * cell.residual_evidence,
                        0.0,
                        1.0,
                    )
                )
                ptdsf_stats = self._ptdsf_state_stats(cell)
                otv_geom = float(np.clip(self._otv_surface_conf(cell), 0.0, 1.0))
                rho_stat = float(ptdsf_stats['rho_static'])
                ptdsf_ratio = float(ptdsf_stats['split_ratio'])
                ptdsf_dom = float(ptdsf_stats['dominance'])
                ptdsf_static_conf = float(ptdsf_stats['static_conf'])
                ws_stat = float(max(0.0, cell.phi_static_w))
                wt_trans = float(max(0.0, cell.phi_transient_w))
                trans_ratio = float(wt_trans / max(1e-6, ws_stat + wt_trans))
                phi_dyn_div = 0.0
                phi_dyn_ratio = 0.0
                if bool(dual_layer_dyn_use_phi_dyn):
                    wd = float(max(0.0, cell.phi_dyn_w))
                    wg = float(max(0.0, cell.phi_geo_w))
                    if wd > 1e-8:
                        phi_dyn_ratio = float(np.clip(wd / max(1e-6, wd + wg + ws_stat), 0.0, 1.0))
                    if wd > 1e-8 and wg > 1e-8:
                        phi_dyn_div = float(np.clip(abs(float(cell.phi_dyn) - float(cell.phi_geo)) / phi_div_ref, 0.0, 1.5))
                dyn_mix = float(
                    np.clip(
                        (
                            w_dyn * dyn_primary
                            + w_st * st_n
                            + w_contra * contra
                            + w_trans * trans_ratio
                            + w_phi_div * phi_dyn_div
                            + w_phi_ratio * phi_dyn_ratio
                        )
                        / ws,
                        0.0,
                        1.0,
                    )
                )
                surf = float(max(1e-6, cell.surf_evidence))
                free = float(max(0.0, cell.free_evidence))
                static_occ = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
                rho_n = float(np.clip(float(max(cell.rho, rho_stat)) / max(1e-6, anchor_rho), 0.0, 1.5))
                static_conf = float(
                    np.clip(
                        0.38 * ptdsf_static_conf
                        + 0.22 * static_occ
                        + 0.18 * min(1.0, rho_n)
                        + 0.12 * np.clip(cell.p_static, 0.0, 1.0)
                        + float(np.clip(ptdsf_static_rho_weight, 0.0, 1.0)) * 0.10 * np.clip(ptdsf_ratio, 0.0, 1.0),
                        0.0,
                        1.0,
                    )
                )
                dccm_commit = float(np.clip(getattr(cell, 'dccm_commit', 0.0), 0.0, 1.0))
                dccm_transient_push = 0.0
                if bool(dccm_enable):
                    dccm_transient_push = float(
                        np.clip(
                            dccm_commit * max(0.0, 1.0 - dccm_static_guard * max(static_conf, ptdsf_dom)),
                            0.0,
                            1.0,
                        )
                    )
                dyn_ctx = float(np.clip(0.55 * dyn_mix + 0.45 * global_dyn_level, 0.0, 1.0))
                drop_scale = float(np.clip(1.0 + 0.26 * static_conf - 0.30 * dyn_ctx, 0.65, 1.30))
                drop_th_eff = float(np.clip(drop_th * drop_scale, max(0.30, 0.55 * drop_th), min(1.05, 1.20 * drop_th + 0.06)))
                xmap_static_anchor = bool(
                    xmap_rescue
                    and max(static_conf, ptdsf_dom, ptdsf_ratio, float(csr_score)) >= max(float(csr_min_score), float(xmap_static_min_score))
                    and max(float(cell.rho), rho_stat) >= 0.40 * max(1e-6, anchor_rho)
                    and (float(xmap_score) >= 0.85 * float(xmap_dyn_min_score) or float(xmap_sep) >= 0.60)
                )
                if xmap_static_anchor:
                    keep_idx.add(idx)
                    continue
                xmem_conf = float(self._xmem_conf(cell))
                xmem_clear_conf = float(self._xmem_clear_conf(cell))
                xmem_exclude = bool(
                    (
                        xmem_conf >= max(0.70, 0.95 * drop_th_eff)
                        and max(static_conf, ptdsf_dom, ptdsf_ratio) < 0.64
                        and float(free_ratio) >= max(0.60, 0.75 * free_min)
                    )
                    or (
                        xmem_clear_conf >= 0.40
                        and max(static_conf, ptdsf_dom, ptdsf_ratio) < 0.72
                        and float(surf) <= max(1.10 * float(free), 0.06)
                    )
                )
                if xmem_exclude:
                    continue
                pfv_exclusive_conf = float(self._pfv_exclusive_conf(cell)) if bool(getattr(self.cfg.update, 'pfv_exclusive_enable', False)) else 0.0
                pfv_exclusive_compete = bool(
                    pfv_exclusive_conf >= max(float(getattr(self.cfg.update, 'pfv_exclusive_extract_thresh', 0.30)), 0.80 * drop_th_eff)
                    and float(free_ratio) >= max(float(getattr(self.cfg.update, 'pfv_exclusive_free_ratio_min', 0.45)), 0.70 * free_min)
                    and max(float(xmap_score), float(csr_score)) < 0.65
                )
                static_anchor = bool(
                    max(float(cell.rho), rho_stat) >= anchor_rho
                    and max(ptdsf_static_conf, float(cell.p_static), ptdsf_ratio) >= anchor_p
                    and surf >= anchor_ratio * free
                )
                if static_anchor and not pfv_exclusive_compete:
                    keep_idx.add(idx)
                    continue
                omhs_rear_anchor = bool(
                    omhs
                    and (omhs_keep or (omhs_active and omhs_rear_conf >= 0.50 and omhs_front_conf >= 0.24))
                    and max(static_conf, ptdsf_dom, ptdsf_ratio) >= 0.42
                    and rho_stat >= 0.5 * max(1e-6, ptdsf_persistent_min_rho)
                )
                if omhs_rear_anchor and not pfv_exclusive_compete:
                    keep_idx.add(idx)
                    continue
                pfv_exclusive_veto = bool(
                    pfv_exclusive_conf >= max(float(getattr(self.cfg.update, 'pfv_exclusive_extract_thresh', 0.30)), 0.80 * drop_th_eff)
                    and max(static_conf, ptdsf_dom, ptdsf_ratio, float(csr_score), float(getattr(cell, 'p_static', 0.0))) < float(getattr(self.cfg.update, 'pfv_exclusive_anchor_guard', 0.92))
                    and max(float(cell.rho), rho_stat, float(getattr(cell, 'rho_bg', 0.0))) < float(getattr(self.cfg.update, 'pfv_exclusive_rho_guard', 1.20)) * max(1e-6, anchor_rho)
                    and float(free_ratio) >= max(float(getattr(self.cfg.update, 'pfv_exclusive_free_ratio_min', 0.45)), 0.70 * free_min)
                )
                if pfv_exclusive_veto:
                    continue
                otv_sep = 0.0
                if otv_geom > 1e-6:
                    otv_sep = float(
                        np.clip(
                            abs(float(getattr(cell, 'phi_otv', 0.0)) - float(phi_eff))
                            / max(1e-6, max(0.5 * self.voxel_size, float(getattr(self.cfg.update, 'otv_sep_ref_vox', 0.90)) * self.voxel_size)),
                            0.0,
                            1.5,
                        )
                    )
                otv_exclude = bool(
                    otv_geom >= max(0.55, 0.95 * drop_th_eff)
                    and otv_sep >= 0.35
                    and max(static_conf, ptdsf_dom, ptdsf_ratio) < 0.72
                )
                if otv_exclude:
                    continue
                if float(free_ratio) < free_min:
                    keep_idx.add(idx)
                    continue
                if bool(dccm_enable):
                    transient_veto = bool(
                        max(dccm_commit, float(dccm_commit_weight) * dccm_transient_push) >= max(0.55, 0.9 * drop_th_eff)
                        and ptdsf_dom < 0.55
                        and (trans_ratio >= 0.45 or float(free_ratio) >= 0.85 * free_min)
                    )
                    if transient_veto:
                        continue
                if bool(dual_layer_compete_enable):
                    wd = float(max(0.0, cell.phi_dyn_w))
                    wg = float(max(0.0, cell.phi_geo_w))
                    geo_conf = float(np.clip(wg / max(1e-6, wg + wd), 0.0, 1.0))
                    dyn_conf = float(np.clip(wd / max(1e-6, wg + wd), 0.0, 1.0))
                    phi_dyn_adv = 0.0
                    if wd > 1e-8 and wg > 1e-8:
                        # positive when dynamic channel dominates local signed distance.
                        phi_dyn_adv = float(np.clip((abs(float(cell.phi_dyn)) - abs(float(cell.phi_geo))) / phi_div_ref, -1.0, 1.0))
                        phi_dyn_adv = float(np.clip(0.5 + 0.5 * phi_dyn_adv, 0.0, 1.0))
                    geo_score = float(np.clip(comp_geo_w * geo_conf + (1.0 - comp_geo_w) * (1.0 - phi_dyn_div), 0.0, 1.0))
                    dyn_score = float(
                        np.clip(
                            comp_dyn_mix_w * dyn_mix
                            + comp_dyn_conf_w * dyn_conf
                            + comp_dyn_rest_w * phi_dyn_adv
                            + (float(dccm_commit_weight) * dccm_transient_push if bool(dccm_enable) else 0.0),
                            0.0,
                            1.0,
                        )
                    )
                    # Adaptive competition margin:
                    # increase margin in static/stable areas and reduce it under strong
                    # dynamic contradictions to suppress ghost with lower collateral damage.
                    margin_scale = float(np.clip(1.0 + 0.28 * static_conf - 0.32 * dyn_ctx, 0.55, 1.55))
                    comp_margin_eff = float(max(0.0, comp_margin * margin_scale))
                    if dyn_score > geo_score + comp_margin_eff:
                        continue
                # Posterior gate without explicit competition.
                if dyn_mix < drop_th_eff:
                    keep_idx.add(idx)
            accepted_idx = keep_idx

        if not accepted_idx:
            self.last_extract_stats = {
                "candidates_prefilter": float(prefilter_candidates),
                "candidates_after_filter": float(len(candidates)),
                "ebcut_rejects": float(ebcut_rejects),
                "ebcut_reject_ratio": float(ebcut_rejects / max(1, prefilter_candidates)),
                "xmap_rescues": float(xmap_rescue_count),
                "xmap_front_drops": float(xmap_front_drop_count),
                "accepted_points": 0.0,
                "rear_density_drops": float(rear_density_drops),
                "rear_density_drops": float(rear_density_drops),
        }
            if prev_extract_ctx is None:
                try:
                    delattr(self, '_ptdsf_context')
                except AttributeError:
                    pass
            else:
                self._ptdsf_context = prev_extract_ctx
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

        pts: List[np.ndarray] = []
        nrm: List[np.ndarray] = []
        rear_pts: List[np.ndarray] = []
        rear_nrm: List[np.ndarray] = []
        rear_feature_rows: List[Dict[str, float]] = []
        front_pts: List[np.ndarray] = []
        front_nrm: List[np.ndarray] = []
        bg_pts: List[np.ndarray] = []
        bg_nrm: List[np.ndarray] = []
        rear_records: List[tuple[VoxelIndex, np.ndarray, np.ndarray, Dict[str, float]]] = []
        max_off = float(max(0.0, zero_crossing_max_offset))
        phi_gate = float(max(1e-4, zero_crossing_phi_gate))
        rear_selectivity_enabled = bool(getattr(self.cfg.update, 'rps_rear_selectivity_enable', False))
        rear_selectivity_stats = {
            'rear_selectivity_pre_count': 0.0,
            'rear_selectivity_kept_count': 0.0,
            'rear_selectivity_drop_count': 0.0,
            'rear_selectivity_topk_drop_count': 0.0,
            'rear_selectivity_score_sum': 0.0,
            'rear_selectivity_risk_sum': 0.0,
            'rear_selectivity_pre_front_score_sum': 0.0,
            'rear_selectivity_pre_front_residual_sum': 0.0,
            'rear_selectivity_pre_occlusion_order_sum': 0.0,
            'rear_selectivity_pre_local_conflict_sum': 0.0,
            'rear_selectivity_pre_dynamic_trail_sum': 0.0,
            'rear_selectivity_pre_dyn_risk_sum': 0.0,
            'rear_selectivity_front_score_sum': 0.0,
            'rear_selectivity_rear_score_sum': 0.0,
            'rear_selectivity_gap_sum': 0.0,
            'rear_selectivity_competition_sum': 0.0,
            'rear_selectivity_occlusion_order_sum': 0.0,
            'rear_selectivity_occluder_protect_sum': 0.0,
            'rear_selectivity_local_conflict_sum': 0.0,
            'rear_selectivity_front_residual_sum': 0.0,
            'rear_selectivity_dynamic_trail_sum': 0.0,
            'rear_selectivity_pre_history_anchor_sum': 0.0,
            'rear_selectivity_pre_surface_anchor_sum': 0.0,
            'rear_selectivity_pre_surface_distance_sum': 0.0,
            'rear_selectivity_pre_dynamic_shell_sum': 0.0,
            'rear_selectivity_dyn_risk_sum': 0.0,
            'rear_selectivity_history_anchor_sum': 0.0,
            'rear_selectivity_surface_anchor_sum': 0.0,
            'rear_selectivity_surface_distance_sum': 0.0,
            'rear_selectivity_dynamic_shell_sum': 0.0,
            'rear_selectivity_pre_penetration_sum': 0.0,
            'rear_selectivity_pre_penetration_free_span_sum': 0.0,
            'rear_selectivity_pre_observation_count_sum': 0.0,
            'rear_selectivity_pre_observation_support_sum': 0.0,
            'rear_selectivity_pre_static_coherence_sum': 0.0,
            'rear_selectivity_penetration_sum': 0.0,
            'rear_selectivity_penetration_free_span_sum': 0.0,
            'rear_selectivity_observation_count_sum': 0.0,
            'rear_selectivity_observation_support_sum': 0.0,
            'rear_selectivity_static_coherence_sum': 0.0,
            'rear_selectivity_pre_topology_thickness_sum': 0.0,
            'rear_selectivity_pre_normal_consistency_sum': 0.0,
            'rear_selectivity_pre_ray_convergence_sum': 0.0,
            'rear_selectivity_topology_thickness_sum': 0.0,
            'rear_selectivity_normal_consistency_sum': 0.0,
            'rear_selectivity_ray_convergence_sum': 0.0,
        }
        for idx, _cell, center, n_i, _free_ratio, phi_eff in candidates:
            if idx not in accepted_idx:
                continue
            info = candidate_map.get(idx)
            bank_selected_tag = str(info[7]) if info is not None and len(info) >= 8 else "unknown"
            p = center
            if use_zero_crossing and abs(float(phi_eff)) <= phi_gate:
                off = -float(phi_eff) * n_i
                off_norm = float(np.linalg.norm(off))
                if max_off > 0.0 and off_norm > max_off:
                    off = off * (max_off / max(off_norm, 1e-9))
                p = center + off
            if bank_selected_tag.startswith("rear") and bool(getattr(self.cfg.update, 'rps_rear_density_gate_enable', False)):
                radius = max(1, int(getattr(self.cfg.update, 'rps_rear_density_radius_cells', 1)))
                min_nb = max(0, int(getattr(self.cfg.update, 'rps_rear_density_min_neighbors', 2)))
                support_min = float(np.clip(getattr(self.cfg.update, 'rps_rear_density_support_min', 0.45), 0.0, 1.0))
                rear_neighbors = 0
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nidx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                            if nidx not in accepted_idx:
                                continue
                            ninfo = candidate_map.get(nidx)
                            if ninfo is not None and len(ninfo) >= 8 and str(ninfo[7]).startswith("rear"):
                                rear_neighbors += 1
                rear_support = float(p10_rear_state_support(self, _cell))
                if rear_neighbors < min_nb and rear_support < support_min:
                    rear_density_drops += 1
                    continue
            if bank_selected_tag.startswith("rear") and bool(getattr(self.cfg.update, 'rps_rear_hybrid_filter_enable', False)):
                bridge_min = float(np.clip(getattr(self.cfg.update, 'rps_rear_hybrid_bridge_support_min', 0.20), 0.0, 1.0))
                dyn_max = float(np.clip(getattr(self.cfg.update, 'rps_rear_hybrid_dyn_max', 0.22), 0.0, 1.0))
                manifold_min = float(np.clip(getattr(self.cfg.update, 'rps_rear_hybrid_manifold_min', 0.25), 0.0, 1.0))
                manifold = p10_manifold_state_components(_cell, self.cfg)
                bridge_support = float(max(manifold.get('dense_support', 0.0), manifold.get('visible', 0.0)))
                dyn_risk = float(max(float(np.clip(getattr(_cell, 'dyn_prob', 0.0), 0.0, 1.0)), float(np.clip(getattr(_cell, 'z_dyn', 0.0), 0.0, 1.0)), float(np.clip(getattr(_cell, 'wod_front_conf', 0.0), 0.0, 1.0))))
                if bridge_support < bridge_min and (dyn_risk > dyn_max or float(manifold.get('visible', 0.0)) < manifold_min):
                    rear_density_drops += 1
                    continue
            if bank_selected_tag.startswith("rear") and rear_selectivity_enabled:
                comps = p10_rear_selectivity_components(
                    self,
                    idx=idx,
                    cell=_cell,
                    point=p,
                    normal=n_i,
                    accepted_idx=accepted_idx,
                    candidate_map=candidate_map,
                )
                rear_records.append((idx, p, n_i, comps))
                continue
            pts.append(p)
            nrm.append(n_i)
            if bank_selected_tag.startswith("rear"):
                rear_pts.append(p)
                rear_nrm.append(n_i)
            elif bank_selected_tag.startswith("background"):
                bg_pts.append(p)
                bg_nrm.append(n_i)
            else:
                front_pts.append(p)
                front_nrm.append(n_i)

        if rear_selectivity_enabled and rear_records:
            kept_records, rear_selectivity_stats = p10_filter_rear_records(self, rear_records)
            for _idx, p, n_i, _comps in kept_records:
                pts.append(p)
                nrm.append(n_i)
                rear_pts.append(p)
                rear_nrm.append(n_i)
                rear_feature_rows.append({
                    'x': float(p[0]),
                    'y': float(p[1]),
                    'z': float(p[2]),
                    'nx': float(n_i[0]),
                    'ny': float(n_i[1]),
                    'nz': float(n_i[2]),
                    'front_score': float(_comps.get('front_score', 0.0)),
                    'rear_score': float(_comps.get('rear_score', 0.0)),
                    'rear_gap': float(_comps.get('rear_gap', 0.0)),
                    'competition': float(_comps.get('competition', 0.0)),
                    'history_anchor': float(_comps.get('history_anchor', 0.0)),
                    'surface_anchor': float(_comps.get('surface_anchor', 0.0)),
                    'surface_distance': float(_comps.get('surface_distance', 0.0)),
                    'dynamic_shell': float(_comps.get('dynamic_shell', 0.0)),
                    'penetration_score': float(_comps.get('penetration_score', 0.0)),
                    'penetration_free_span': float(_comps.get('penetration_free_span', 0.0)),
                    'topology_thickness': float(_comps.get('topology_thickness', 0.0)),
                    'observation_count': float(_comps.get('observation_count', 0.0)),
                    'observation_support': float(_comps.get('observation_support', 0.0)),
                    'static_coherence': float(_comps.get('static_coherence', 0.0)),
                    'normal_consistency': float(_comps.get('normal_consistency', 0.0)),
                    'ray_convergence': float(_comps.get('ray_convergence', 0.0)),
                    'score': float(_comps.get('score', 0.0)),
                    'risk': float(_comps.get('risk', 0.0)),
                })

        self.last_extract_bank_points = {
            "rear_points": np.asarray(rear_pts, dtype=float) if rear_pts else np.zeros((0, 3), dtype=float),
            "rear_normals": np.asarray(rear_nrm, dtype=float) if rear_nrm else np.zeros((0, 3), dtype=float),
            "rear_feature_rows": rear_feature_rows,
            "front_points": np.asarray(front_pts, dtype=float) if front_pts else np.zeros((0, 3), dtype=float),
            "front_normals": np.asarray(front_nrm, dtype=float) if front_nrm else np.zeros((0, 3), dtype=float),
            "background_points": np.asarray(bg_pts, dtype=float) if bg_pts else np.zeros((0, 3), dtype=float),
            "background_normals": np.asarray(bg_nrm, dtype=float) if bg_nrm else np.zeros((0, 3), dtype=float),
        }

        if not pts:
            self.last_extract_stats = {
                "candidates_prefilter": float(prefilter_candidates),
                "candidates_after_filter": float(len(candidates)),
                "ebcut_rejects": float(ebcut_rejects),
                "ebcut_reject_ratio": float(ebcut_rejects / max(1, prefilter_candidates)),
                "xmap_rescues": float(xmap_rescue_count),
                "xmap_front_drops": float(xmap_front_drop_count),
                "accepted_points": 0.0,
                "rear_density_drops": float(rear_density_drops),
                "rear_selectivity_pre_count": float(rear_selectivity_stats.get('rear_selectivity_pre_count', 0.0)),
                "rear_selectivity_kept_count": float(rear_selectivity_stats.get('rear_selectivity_kept_count', 0.0)),
                "rear_selectivity_drop_count": float(rear_selectivity_stats.get('rear_selectivity_drop_count', 0.0)),
                "rear_selectivity_topk_drop_count": float(rear_selectivity_stats.get('rear_selectivity_topk_drop_count', 0.0)),
                "rear_selectivity_score_sum": float(rear_selectivity_stats.get('rear_selectivity_score_sum', 0.0)),
                "rear_selectivity_risk_sum": float(rear_selectivity_stats.get('rear_selectivity_risk_sum', 0.0)),
                "rear_selectivity_pre_front_score_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_front_score_sum', 0.0)),
                "rear_selectivity_pre_front_residual_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_front_residual_sum', 0.0)),
                "rear_selectivity_pre_occlusion_order_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_occlusion_order_sum', 0.0)),
                "rear_selectivity_pre_local_conflict_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_local_conflict_sum', 0.0)),
                "rear_selectivity_pre_dynamic_trail_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_dynamic_trail_sum', 0.0)),
                "rear_selectivity_pre_dyn_risk_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_dyn_risk_sum', 0.0)),
                "rear_selectivity_pre_history_anchor_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_history_anchor_sum', 0.0)),
                "rear_selectivity_pre_surface_anchor_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_surface_anchor_sum', 0.0)),
                "rear_selectivity_pre_surface_distance_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_surface_distance_sum', 0.0)),
                "rear_selectivity_pre_dynamic_shell_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_dynamic_shell_sum', 0.0)),
                "rear_selectivity_pre_penetration_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_penetration_sum', 0.0)),
                "rear_selectivity_pre_penetration_free_span_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_penetration_free_span_sum', 0.0)),
                "rear_selectivity_pre_observation_count_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_observation_count_sum', 0.0)),
                "rear_selectivity_pre_observation_support_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_observation_support_sum', 0.0)),
                "rear_selectivity_pre_static_coherence_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_static_coherence_sum', 0.0)),
                "rear_selectivity_pre_topology_thickness_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_topology_thickness_sum', 0.0)),
                "rear_selectivity_pre_normal_consistency_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_normal_consistency_sum', 0.0)),
                "rear_selectivity_pre_ray_convergence_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_ray_convergence_sum', 0.0)),
                "rear_selectivity_front_score_sum": float(rear_selectivity_stats.get('rear_selectivity_front_score_sum', 0.0)),
                "rear_selectivity_rear_score_sum": float(rear_selectivity_stats.get('rear_selectivity_rear_score_sum', 0.0)),
                "rear_selectivity_gap_sum": float(rear_selectivity_stats.get('rear_selectivity_gap_sum', 0.0)),
                "rear_selectivity_competition_sum": float(rear_selectivity_stats.get('rear_selectivity_competition_sum', 0.0)),
                "rear_selectivity_occlusion_order_sum": float(rear_selectivity_stats.get('rear_selectivity_occlusion_order_sum', 0.0)),
                "rear_selectivity_occluder_protect_sum": float(rear_selectivity_stats.get('rear_selectivity_occluder_protect_sum', 0.0)),
                "rear_selectivity_local_conflict_sum": float(rear_selectivity_stats.get('rear_selectivity_local_conflict_sum', 0.0)),
                "rear_selectivity_front_residual_sum": float(rear_selectivity_stats.get('rear_selectivity_front_residual_sum', 0.0)),
                "rear_selectivity_dynamic_trail_sum": float(rear_selectivity_stats.get('rear_selectivity_dynamic_trail_sum', 0.0)),
                "rear_selectivity_dyn_risk_sum": float(rear_selectivity_stats.get('rear_selectivity_dyn_risk_sum', 0.0)),
                "rear_selectivity_history_anchor_sum": float(rear_selectivity_stats.get('rear_selectivity_history_anchor_sum', 0.0)),
                "rear_selectivity_surface_anchor_sum": float(rear_selectivity_stats.get('rear_selectivity_surface_anchor_sum', 0.0)),
                "rear_selectivity_surface_distance_sum": float(rear_selectivity_stats.get('rear_selectivity_surface_distance_sum', 0.0)),
                "rear_selectivity_dynamic_shell_sum": float(rear_selectivity_stats.get('rear_selectivity_dynamic_shell_sum', 0.0)),
                "rear_selectivity_penetration_sum": float(rear_selectivity_stats.get('rear_selectivity_penetration_sum', 0.0)),
                "rear_selectivity_penetration_free_span_sum": float(rear_selectivity_stats.get('rear_selectivity_penetration_free_span_sum', 0.0)),
                "rear_selectivity_observation_count_sum": float(rear_selectivity_stats.get('rear_selectivity_observation_count_sum', 0.0)),
                "rear_selectivity_observation_support_sum": float(rear_selectivity_stats.get('rear_selectivity_observation_support_sum', 0.0)),
                "rear_selectivity_static_coherence_sum": float(rear_selectivity_stats.get('rear_selectivity_static_coherence_sum', 0.0)),
                "rear_selectivity_topology_thickness_sum": float(rear_selectivity_stats.get('rear_selectivity_topology_thickness_sum', 0.0)),
                "rear_selectivity_normal_consistency_sum": float(rear_selectivity_stats.get('rear_selectivity_normal_consistency_sum', 0.0)),
                "rear_selectivity_ray_convergence_sum": float(rear_selectivity_stats.get('rear_selectivity_ray_convergence_sum', 0.0)),
                "rear_density_drops": float(rear_density_drops),
        }
            if prev_extract_ctx is None:
                try:
                    delattr(self, '_ptdsf_context')
                except AttributeError:
                    pass
            else:
                self._ptdsf_context = prev_extract_ctx
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
        self.last_extract_stats = {
            "candidates_prefilter": float(prefilter_candidates),
            "candidates_after_filter": float(len(candidates)),
            "ebcut_rejects": float(ebcut_rejects),
            "ebcut_reject_ratio": float(ebcut_rejects / max(1, prefilter_candidates)),
            "xmap_rescues": float(xmap_rescue_count),
            "xmap_front_drops": float(xmap_front_drop_count),
            "accepted_points": float(len(pts)),
            "rear_density_drops": float(rear_density_drops),
            "rear_selectivity_pre_count": float(rear_selectivity_stats.get('rear_selectivity_pre_count', 0.0)),
            "rear_selectivity_kept_count": float(rear_selectivity_stats.get('rear_selectivity_kept_count', 0.0)),
            "rear_selectivity_drop_count": float(rear_selectivity_stats.get('rear_selectivity_drop_count', 0.0)),
            "rear_selectivity_topk_drop_count": float(rear_selectivity_stats.get('rear_selectivity_topk_drop_count', 0.0)),
            "rear_selectivity_score_sum": float(rear_selectivity_stats.get('rear_selectivity_score_sum', 0.0)),
            "rear_selectivity_risk_sum": float(rear_selectivity_stats.get('rear_selectivity_risk_sum', 0.0)),
            "rear_selectivity_pre_front_score_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_front_score_sum', 0.0)),
            "rear_selectivity_pre_front_residual_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_front_residual_sum', 0.0)),
            "rear_selectivity_pre_occlusion_order_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_occlusion_order_sum', 0.0)),
            "rear_selectivity_pre_local_conflict_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_local_conflict_sum', 0.0)),
            "rear_selectivity_pre_dynamic_trail_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_dynamic_trail_sum', 0.0)),
            "rear_selectivity_pre_dyn_risk_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_dyn_risk_sum', 0.0)),
            "rear_selectivity_pre_history_anchor_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_history_anchor_sum', 0.0)),
            "rear_selectivity_pre_surface_anchor_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_surface_anchor_sum', 0.0)),
            "rear_selectivity_pre_surface_distance_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_surface_distance_sum', 0.0)),
            "rear_selectivity_pre_dynamic_shell_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_dynamic_shell_sum', 0.0)),
            "rear_selectivity_pre_penetration_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_penetration_sum', 0.0)),
            "rear_selectivity_pre_penetration_free_span_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_penetration_free_span_sum', 0.0)),
            "rear_selectivity_pre_observation_count_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_observation_count_sum', 0.0)),
            "rear_selectivity_pre_observation_support_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_observation_support_sum', 0.0)),
            "rear_selectivity_pre_static_coherence_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_static_coherence_sum', 0.0)),
            "rear_selectivity_pre_topology_thickness_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_topology_thickness_sum', 0.0)),
            "rear_selectivity_pre_normal_consistency_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_normal_consistency_sum', 0.0)),
            "rear_selectivity_pre_ray_convergence_sum": float(rear_selectivity_stats.get('rear_selectivity_pre_ray_convergence_sum', 0.0)),
            "rear_selectivity_front_score_sum": float(rear_selectivity_stats.get('rear_selectivity_front_score_sum', 0.0)),
            "rear_selectivity_rear_score_sum": float(rear_selectivity_stats.get('rear_selectivity_rear_score_sum', 0.0)),
            "rear_selectivity_gap_sum": float(rear_selectivity_stats.get('rear_selectivity_gap_sum', 0.0)),
            "rear_selectivity_competition_sum": float(rear_selectivity_stats.get('rear_selectivity_competition_sum', 0.0)),
            "rear_selectivity_occlusion_order_sum": float(rear_selectivity_stats.get('rear_selectivity_occlusion_order_sum', 0.0)),
            "rear_selectivity_occluder_protect_sum": float(rear_selectivity_stats.get('rear_selectivity_occluder_protect_sum', 0.0)),
            "rear_selectivity_local_conflict_sum": float(rear_selectivity_stats.get('rear_selectivity_local_conflict_sum', 0.0)),
            "rear_selectivity_front_residual_sum": float(rear_selectivity_stats.get('rear_selectivity_front_residual_sum', 0.0)),
            "rear_selectivity_dynamic_trail_sum": float(rear_selectivity_stats.get('rear_selectivity_dynamic_trail_sum', 0.0)),
            "rear_selectivity_dyn_risk_sum": float(rear_selectivity_stats.get('rear_selectivity_dyn_risk_sum', 0.0)),
            "rear_selectivity_history_anchor_sum": float(rear_selectivity_stats.get('rear_selectivity_history_anchor_sum', 0.0)),
            "rear_selectivity_surface_anchor_sum": float(rear_selectivity_stats.get('rear_selectivity_surface_anchor_sum', 0.0)),
            "rear_selectivity_surface_distance_sum": float(rear_selectivity_stats.get('rear_selectivity_surface_distance_sum', 0.0)),
            "rear_selectivity_dynamic_shell_sum": float(rear_selectivity_stats.get('rear_selectivity_dynamic_shell_sum', 0.0)),
            "rear_selectivity_penetration_sum": float(rear_selectivity_stats.get('rear_selectivity_penetration_sum', 0.0)),
            "rear_selectivity_penetration_free_span_sum": float(rear_selectivity_stats.get('rear_selectivity_penetration_free_span_sum', 0.0)),
            "rear_selectivity_observation_count_sum": float(rear_selectivity_stats.get('rear_selectivity_observation_count_sum', 0.0)),
            "rear_selectivity_observation_support_sum": float(rear_selectivity_stats.get('rear_selectivity_observation_support_sum', 0.0)),
            "rear_selectivity_static_coherence_sum": float(rear_selectivity_stats.get('rear_selectivity_static_coherence_sum', 0.0)),
            "rear_selectivity_topology_thickness_sum": float(rear_selectivity_stats.get('rear_selectivity_topology_thickness_sum', 0.0)),
            "rear_selectivity_normal_consistency_sum": float(rear_selectivity_stats.get('rear_selectivity_normal_consistency_sum', 0.0)),
            "rear_selectivity_ray_convergence_sum": float(rear_selectivity_stats.get('rear_selectivity_ray_convergence_sum', 0.0)),
        }
        if prev_extract_ctx is None:
            try:
                delattr(self, '_ptdsf_context')
            except AttributeError:
                pass
        else:
            self._ptdsf_context = prev_extract_ctx
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
