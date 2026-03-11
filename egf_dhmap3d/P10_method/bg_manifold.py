from __future__ import annotations

from typing import Dict

import numpy as np


"""Stable background manifold state helpers for S2 rear-state recovery."""


def update_bg_manifold_state(
    self,
    voxel_map,
    cell,
    *,
    idx,
    view_axis,
    d_static_obs: float,
    d_bg_obs: float,
    d_geo: float,
    w_obs: float,
    static_mass: float,
    wod_rear: float,
    wod_shell: float,
    q_dyn_obs: float,
) -> None:
    if not bool(getattr(self.cfg.update, 'rps_bg_manifold_state_enable', False)):
        return

    diag = getattr(self, '_bg_manifold_diag', None)
    if diag is None:
        diag = {}
        setattr(self, '_bg_manifold_diag', diag)
    diag['calls'] = float(diag.get('calls', 0.0) + 1.0)

    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'rps_bg_manifold_rho_ref', getattr(self.cfg.update, 'rps_admission_rho_ref', 0.08))))
    weight_ref = float(max(1e-6, getattr(self.cfg.update, 'rps_bg_manifold_weight_ref', getattr(self.cfg.update, 'rps_admission_weight_ref', 0.35))))

    rho_static_n = float(np.clip(float(getattr(cell, 'rho_static', 0.0)) / rho_ref, 0.0, 1.5))
    static_w_n = float(1.0 - np.exp(-float(max(0.0, getattr(cell, 'phi_static_w', 0.0))) / weight_ref))
    rho_bg_n = float(np.clip(float(getattr(cell, 'rho_bg', 0.0)) / rho_ref, 0.0, 1.5))
    bg_w_n = float(1.0 - np.exp(-float(max(0.0, getattr(cell, 'phi_bg_w', 0.0))) / weight_ref))
    rho_bg_cand_n = float(np.clip(float(getattr(cell, 'rho_bg_cand', 0.0)) / rho_ref, 0.0, 1.5))
    bg_cand_w_n = float(1.0 - np.exp(-float(max(0.0, getattr(cell, 'phi_bg_cand_w', 0.0))) / weight_ref))
    p_static = float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0))

    visible_obs = float(np.clip(max(0.30 * p_static + 0.25 * static_mass, rho_static_n, static_w_n, 0.90 * rho_bg_n, 0.90 * bg_w_n, 0.80 * rho_bg_cand_n, 0.80 * bg_cand_w_n), 0.0, 1.0))
    obstruction_obs = float(np.clip(max(0.70 * float(np.clip(wod_rear, 0.0, 1.0)) + 0.30 * float(np.clip(wod_shell, 0.0, 1.0)), float(np.clip(getattr(cell, 'wod_rear_conf', 0.0), 0.0, 1.0)), 0.75 * float(np.clip(q_dyn_obs, 0.0, 1.0))), 0.0, 1.0))

    vis_prev = float(np.clip(getattr(cell, 'bg_visible_mem', 0.0), 0.0, 1.0))
    occ_prev = float(np.clip(getattr(cell, 'bg_obstruction_mem', 0.0), 0.0, 1.0))
    alpha_up = float(np.clip(getattr(self.cfg.update, 'rps_bg_manifold_alpha_up', 0.08), 0.001, 1.0))
    alpha_down = float(np.clip(getattr(self.cfg.update, 'rps_bg_manifold_alpha_down', 0.02), 0.001, 1.0))
    vis_alpha = alpha_up if visible_obs >= vis_prev else alpha_down
    occ_alpha = alpha_up if obstruction_obs >= occ_prev else alpha_down
    cell.bg_visible_mem = float(np.clip((1.0 - vis_alpha) * vis_prev + vis_alpha * visible_obs, 0.0, 1.0))
    cell.bg_obstruction_mem = float(np.clip((1.0 - occ_alpha) * occ_prev + occ_alpha * obstruction_obs, 0.0, 1.0))

    refs = []
    if float(getattr(cell, 'phi_bg_w', 0.0)) > 1e-12:
        refs.append((float(getattr(cell, 'phi_bg', d_bg_obs)), float(max(rho_bg_n, bg_w_n))))
    if float(getattr(cell, 'phi_bg_cand_w', 0.0)) > 1e-12:
        refs.append((float(getattr(cell, 'phi_bg_cand', d_bg_obs)), 0.75 * float(max(rho_bg_cand_n, bg_cand_w_n))))
    if float(getattr(cell, 'phi_static_w', 0.0)) > 1e-12:
        refs.append((float(getattr(cell, 'phi_static', d_static_obs)), 0.60 * float(max(rho_static_n, static_w_n, p_static))))
    if float(getattr(cell, 'phi_geo_w', 0.0)) > 1e-12:
        refs.append((float(getattr(cell, 'phi_geo', d_geo)), 0.40 * float(max(static_w_n, p_static))))
    if not refs:
        return

    manifold_phi = float(sum(v * w for v, w in refs) / max(1e-9, sum(w for _, w in refs)))
    weight_gain = float(max(0.0, getattr(self.cfg.update, 'rps_bg_manifold_weight_gain', 0.55)))
    w_mem_add = float(w_obs * weight_gain * max(cell.bg_visible_mem, 0.25 * cell.bg_obstruction_mem))
    if w_mem_add > 1e-12:
        w_prev = float(max(0.0, getattr(cell, 'phi_bg_memory_w', 0.0)))
        w_new = float(min(5000.0, w_prev + w_mem_add))
        cell.phi_bg_memory = float((w_prev * float(getattr(cell, 'phi_bg_memory', manifold_phi)) + w_mem_add * manifold_phi) / max(1e-9, w_new))
        cell.phi_bg_memory_w = w_new
        rho_alpha = float(np.clip(getattr(self.cfg.update, 'rps_bg_manifold_rho_alpha', 0.10), 0.001, 1.0))
        cell.rho_bg_stable = float(max(0.0, getattr(cell, 'rho_bg_stable', 0.0)) + rho_alpha * w_mem_add * max(cell.bg_visible_mem, 0.5))
        diag['memory_updates'] = float(diag.get('memory_updates', 0.0) + 1.0)
        diag['memory_w_sum'] = float(diag.get('memory_w_sum', 0.0) + w_mem_add)
        diag['visible_mem_sum'] = float(diag.get('visible_mem_sum', 0.0) + cell.bg_visible_mem)
        diag['obstruction_mem_sum'] = float(diag.get('obstruction_mem_sum', 0.0) + cell.bg_obstruction_mem)

    if bool(getattr(self.cfg.update, 'rps_bg_dense_state_enable', False)):
        dense_floor = float(np.clip(getattr(self.cfg.update, 'rps_bg_dense_support_floor', 0.18), 0.0, 1.0))
        dense_support_n, dense_phi_n, dense_w_n = _dense_neighbor_support(voxel_map, idx, self.cfg, manifold_phi)
        dense_base = float(np.clip(max(cell.bg_visible_mem, 0.70 * min(1.0, getattr(cell, 'rho_bg_stable', 0.0) / rho_ref)), 0.0, 1.0))
        dense_support = float(np.clip(max(dense_base, dense_floor * dense_support_n, dense_support_n), 0.0, 1.0))
        dense_w = float(np.clip(max(dense_w_n, float(getattr(cell, 'phi_bg_memory_w', 0.0)) / max(1e-6, getattr(self.cfg.update, 'rps_bg_manifold_weight_ref', 0.35))), 0.0, float(getattr(self.cfg.update, 'rps_bg_dense_max_weight', 1.0))))
        if dense_support > 1e-6:
            if dense_w_n > 1e-6:
                cell.bg_dense_phi = float((1.0 - min(0.85, dense_w_n)) * manifold_phi + min(0.85, dense_w_n) * dense_phi_n)
            else:
                cell.bg_dense_phi = manifold_phi
        dense_decay = float(np.clip(getattr(self.cfg.update, 'rps_bg_dense_decay', 0.996), 0.80, 0.9999))
        cell.bg_dense_support = float(np.clip(max(dense_support, dense_decay * float(getattr(cell, 'bg_dense_support', 0.0))), 0.0, 1.0))
        cell.bg_dense_w = float(np.clip(max(dense_w, dense_decay * float(getattr(cell, 'bg_dense_w', 0.0))), 0.0, float(getattr(self.cfg.update, 'rps_bg_dense_max_weight', 1.0))))
        diag['dense_updates'] = float(diag.get('dense_updates', 0.0) + 1.0)
        diag['dense_support_sum'] = float(diag.get('dense_support_sum', 0.0) + cell.bg_dense_support)
        diag['dense_weight_sum'] = float(diag.get('dense_weight_sum', 0.0) + cell.bg_dense_w)

    if bool(getattr(self.cfg.update, 'rps_bg_bridge_enable', False)) and view_axis is not None:
        source = manifold_state_components(cell, self.cfg)
        visible_min = float(np.clip(getattr(self.cfg.update, 'rps_bg_bridge_min_visible', 0.35), 0.0, 1.0))
        obstruction_min = float(np.clip(getattr(self.cfg.update, 'rps_bg_bridge_min_obstruction', 0.30), 0.0, 1.0))
        if source['visible'] >= visible_min and max(source['obstructed'], float(np.clip(wod_rear, 0.0, 1.0)), float(np.clip(wod_shell, 0.0, 1.0)), float(np.clip(q_dyn_obs, 0.0, 1.0))) >= obstruction_min:
            v = np.asarray(view_axis, dtype=float)
            vnorm = float(np.linalg.norm(v))
            if vnorm > 1e-9:
                v = v / vnorm
                center = voxel_map.index_to_center(idx)
                min_step = max(1, int(getattr(self.cfg.update, 'rps_bg_bridge_min_step', 1)))
                max_step = max(min_step, int(getattr(self.cfg.update, 'rps_bg_bridge_max_step', 3)))
                gain = float(max(0.0, getattr(self.cfg.update, 'rps_bg_bridge_gain', 0.65)))
                phi_blend = float(np.clip(getattr(self.cfg.update, 'rps_bg_bridge_phi_blend', 0.85), 0.0, 1.0))
                dyn_max = float(np.clip(getattr(self.cfg.update, 'rps_bg_bridge_target_dyn_max', 0.35), 0.0, 1.0))
                surf_max = float(np.clip(getattr(self.cfg.update, 'rps_bg_bridge_target_surface_max', 0.35), 0.0, 1.0))
                ghost_suppress_on = bool(getattr(self.cfg.update, 'rps_bg_bridge_ghost_suppress_enable', False))
                ghost_suppress_w = float(max(0.0, getattr(self.cfg.update, 'rps_bg_bridge_ghost_suppress_weight', 0.22)))
                relaxed_dyn_max = float(np.clip(getattr(self.cfg.update, 'rps_bg_bridge_relaxed_dyn_max', dyn_max), 0.0, 1.0))
                keep_multi = bool(getattr(self.cfg.update, 'rps_bg_bridge_keep_multi_hits', False))
                max_hits = max(1, int(getattr(self.cfg.update, 'rps_bg_bridge_max_hits_per_source', 3)))
                cone_on = bool(getattr(self.cfg.update, 'rps_bg_bridge_cone_enable', False))
                cone_radius = max(0, int(getattr(self.cfg.update, 'rps_bg_bridge_cone_radius_cells', 1)))
                cone_gain_scale = float(max(0.0, getattr(self.cfg.update, 'rps_bg_bridge_cone_gain_scale', 0.65)))
                patch_radius = max(0, int(getattr(self.cfg.update, 'rps_bg_bridge_patch_radius_cells', 0)))
                patch_gain_scale = float(max(0.0, getattr(self.cfg.update, 'rps_bg_bridge_patch_gain_scale', 0.55)))
                depth_hyp = max(0, int(getattr(self.cfg.update, 'rps_bg_bridge_depth_hypothesis_count', 0)))
                depth_step_scale = float(max(0.0, getattr(self.cfg.update, 'rps_bg_bridge_depth_step_scale', 0.50)))
                rear_synth_on = bool(getattr(self.cfg.update, 'rps_bg_bridge_rear_synth_enable', False))
                hits = 0
                visited_targets: set[tuple[int, int, int]] = set()
                for step in range(min_step, max_step + 1):
                    point = center + float(step) * voxel_map.voxel_size * v
                    candidate_targets = _bridge_candidate_targets(
                        voxel_map,
                        point,
                        v,
                        cone_on=cone_on,
                        cone_radius=cone_radius,
                        cone_gain_scale=cone_gain_scale,
                        patch_radius=patch_radius,
                        patch_gain_scale=patch_gain_scale,
                        depth_hypothesis_count=depth_hyp,
                        depth_step_scale=depth_step_scale,
                    )
                    for tidx, target_scale in candidate_targets:
                        if tidx == idx:
                            continue
                        if tidx in visited_targets:
                            continue
                        target = voxel_map.get_or_create(tidx)
                        dyn_risk = _target_dyn_risk(target)
                        surf_conf = _target_surface_conf(target, self.cfg)
                        if surf_conf > surf_max:
                            continue
                        if dyn_risk > relaxed_dyn_max and step < max_step:
                            diag['bridge_tunnel_steps'] = float(diag.get('bridge_tunnel_steps', 0.0) + 1.0)
                            continue
                        if dyn_risk > dyn_max and step < max_step and not cone_on:
                            diag['bridge_tunnel_steps'] = float(diag.get('bridge_tunnel_steps', 0.0) + 1.0)
                            continue
                        if ghost_suppress_on:
                            gain_eff = float(np.clip(gain - ghost_suppress_w * dyn_risk, 0.0, gain))
                        else:
                            gain_eff = gain
                        gain_eff *= float(np.clip(target_scale, 0.0, 1.0))
                        if gain_eff <= 1e-6:
                            continue
                        support_add, weight_new = _apply_bridge_support(target, float(source['phi']), float(source['visible']), float(source['obstructed']), gain_eff, phi_blend)
                        diag['bridge_updates'] = float(diag.get('bridge_updates', 0.0) + 1.0)
                        diag['bridge_support_sum'] = float(diag.get('bridge_support_sum', 0.0) + support_add)
                        diag['bridge_dyn_risk_sum'] = float(diag.get('bridge_dyn_risk_sum', 0.0) + dyn_risk)
                        diag['bridge_target_weight_sum'] = float(diag.get('bridge_target_weight_sum', 0.0) + weight_new)
                        if rear_synth_on:
                            synth_w, synth_rho = _apply_bridge_rear_synthesis(
                                target,
                                source_phi=float(source['phi']),
                                support_add=support_add,
                                cfg=self.cfg,
                            )
                            if synth_w > 1e-12:
                                diag['bridge_rear_synth_updates'] = float(diag.get('bridge_rear_synth_updates', 0.0) + 1.0)
                                diag['bridge_rear_synth_w_sum'] = float(diag.get('bridge_rear_synth_w_sum', 0.0) + synth_w)
                                diag['bridge_rear_synth_rho_sum'] = float(diag.get('bridge_rear_synth_rho_sum', 0.0) + synth_rho)
                        hits += 1
                        visited_targets.add(tidx)
                        if not keep_multi or hits >= max_hits:
                            break
                    if not keep_multi or hits >= max_hits:
                        break





def _surface_signal(cell, cfg) -> tuple[float, float]:
    weight_ref = float(max(1e-6, getattr(cfg.update, 'rps_bg_manifold_weight_ref', getattr(cfg.update, 'rps_admission_weight_ref', 0.35))))
    vals = []
    wts = []
    for phi_attr, w_attr, scale in [
        ('phi_bg', 'phi_bg_w', 1.00),
        ('phi_geo', 'phi_geo_w', 0.95),
        ('phi_static', 'phi_static_w', 0.75),
        ('phi_bg_memory', 'phi_bg_memory_w', 0.85),
    ]:
        w_raw = float(max(0.0, getattr(cell, w_attr, 0.0)))
        if w_raw <= 1e-12:
            continue
        w_n = float((1.0 - np.exp(-w_raw / weight_ref)) * scale)
        vals.append(float(getattr(cell, phi_attr, 0.0)))
        wts.append(w_n)
    if not wts:
        return 0.0, 0.0
    conf = float(max(wts))
    phi = float(sum(v * w for v, w in zip(vals, wts)) / max(1e-9, sum(wts)))
    return conf, phi


def _tangent_weight(voxel_map, idx, nidx, cell, cfg) -> float:
    if not bool(getattr(cfg.update, 'rps_bg_surface_tangent_enable', False)):
        return 1.0
    tangent_weight = float(np.clip(getattr(cfg.update, 'rps_bg_surface_tangent_weight', 0.65), 0.0, 1.0))
    tangent_floor = float(np.clip(getattr(cfg.update, 'rps_bg_surface_tangent_floor', 0.15), 0.0, 1.0))
    grad = np.asarray(getattr(cell, 'g_mean', np.zeros(3, dtype=float)), dtype=float)
    gnorm = float(np.linalg.norm(grad))
    if gnorm <= 1e-9:
        return 1.0
    direction = voxel_map.index_to_center(nidx) - voxel_map.index_to_center(idx)
    dnorm = float(np.linalg.norm(direction))
    if dnorm <= 1e-9:
        return 1.0
    direction /= dnorm
    grad /= gnorm
    tangent = float(np.sqrt(max(0.0, 1.0 - float(np.dot(direction, grad)) ** 2)))
    return float(np.clip((1.0 - tangent_weight) + tangent_weight * tangent, tangent_floor, 1.0))



def _target_dyn_risk(cell) -> float:
    return float(np.clip(max(float(np.clip(getattr(cell, 'dyn_prob', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'st_mem', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'visibility_contradiction', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'wod_front_conf', 0.0), 0.0, 1.0))), 0.0, 1.0))


def _target_surface_conf(cell, cfg) -> float:
    conf, _phi = _surface_signal(cell, cfg)
    return conf


def _apply_bridge_support(target_cell, source_phi: float, source_visible: float, source_obstructed: float, gain: float, phi_blend: float) -> tuple[float, float]:
    support_add = float(np.clip(gain * source_visible * source_obstructed, 0.0, 1.0))
    prev_support = float(np.clip(getattr(target_cell, 'bg_dense_support', 0.0), 0.0, 1.0))
    target_cell.bg_dense_support = float(np.clip(max(prev_support, support_add), 0.0, 1.0))
    prev_phi = float(getattr(target_cell, 'bg_dense_phi', source_phi))
    target_cell.bg_dense_phi = float((1.0 - phi_blend) * prev_phi + phi_blend * source_phi)
    prev_w = float(max(0.0, getattr(target_cell, 'bg_dense_w', 0.0)))
    target_cell.bg_dense_w = float(np.clip(max(prev_w, support_add), 0.0, 1.0))
    return support_add, target_cell.bg_dense_w


def _bridge_candidate_targets(
    voxel_map,
    point: np.ndarray,
    view_axis: np.ndarray,
    *,
    cone_on: bool,
    cone_radius: int,
    cone_gain_scale: float,
    patch_radius: int,
    patch_gain_scale: float,
    depth_hypothesis_count: int,
    depth_step_scale: float,
) -> list[tuple[tuple[int, int, int], float]]:
    target_scales: dict[tuple[int, int, int], float] = {}
    depth_step = float(max(0.0, depth_step_scale)) * float(voxel_map.voxel_size)
    depth_offsets = range(-depth_hypothesis_count, depth_hypothesis_count + 1)
    for depth_offset in depth_offsets:
        dscale = float(np.clip(1.0 - 0.18 * abs(depth_offset), 0.45, 1.0))
        point_depth = point + float(depth_offset) * depth_step * view_axis
        base_idx = voxel_map.world_to_index(point_depth)
        _accumulate_target_scale(target_scales, base_idx, dscale)
        if cone_on and cone_radius > 0:
            for dx in range(-cone_radius, cone_radius + 1):
                for dy in range(-cone_radius, cone_radius + 1):
                    for dz in range(-cone_radius, cone_radius + 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        offset_n = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                        scale = float(np.clip(dscale * cone_gain_scale * np.exp(-0.35 * offset_n), 0.0, 1.0))
                        if scale <= 1e-6:
                            continue
                        tidx = (base_idx[0] + dx, base_idx[1] + dy, base_idx[2] + dz)
                        _accumulate_target_scale(target_scales, tidx, scale)
        if patch_radius > 0:
            for dx in range(-patch_radius, patch_radius + 1):
                for dy in range(-patch_radius, patch_radius + 1):
                    for dz in range(-patch_radius, patch_radius + 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        offset_n = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                        scale = float(np.clip(dscale * patch_gain_scale * np.exp(-0.28 * offset_n), 0.0, 1.0))
                        if scale <= 1e-6:
                            continue
                        tidx = (base_idx[0] + dx, base_idx[1] + dy, base_idx[2] + dz)
                        _accumulate_target_scale(target_scales, tidx, scale)
    return list(target_scales.items())


def _accumulate_target_scale(target_scales: dict[tuple[int, int, int], float], tidx: tuple[int, int, int], scale: float) -> None:
    prev = float(target_scales.get(tidx, 0.0))
    if scale > prev:
        target_scales[tidx] = float(scale)


def _apply_bridge_rear_synthesis(target_cell, *, source_phi: float, support_add: float, cfg) -> tuple[float, float]:
    if not bool(getattr(cfg.update, 'rps_bg_bridge_rear_synth_enable', False)):
        return 0.0, 0.0
    support_gain = float(max(0.0, getattr(cfg.update, 'rps_bg_bridge_rear_support_gain', 0.28)))
    rho_gain = float(max(0.0, getattr(cfg.update, 'rps_bg_bridge_rear_rho_gain', 0.10)))
    phi_blend = float(np.clip(getattr(cfg.update, 'rps_bg_bridge_rear_phi_blend', 0.80), 0.0, 1.0))
    score_floor = float(np.clip(getattr(cfg.update, 'rps_bg_bridge_rear_score_floor', 0.22), 0.0, 1.0))
    active_floor = float(np.clip(getattr(cfg.update, 'rps_bg_bridge_rear_active_floor', 0.52), 0.0, 1.0))
    age_floor = float(max(0.0, getattr(cfg.update, 'rps_bg_bridge_rear_age_floor', 1.0)))

    synth_w = float(np.clip(support_gain * support_add, 0.0, 1.0))
    if synth_w <= 1e-12:
        return 0.0, 0.0

    prev_rear_w = float(max(0.0, getattr(target_cell, 'phi_rear_w', 0.0)))
    new_rear_w = float(min(5000.0, prev_rear_w + synth_w))
    prev_rear_phi = float(getattr(target_cell, 'phi_rear', source_phi))
    target_cell.phi_rear = float(((1.0 - phi_blend) * prev_rear_phi) + phi_blend * source_phi) if prev_rear_w > 1e-12 else float(source_phi)
    target_cell.phi_rear_w = new_rear_w

    cand_w = float(0.65 * synth_w)
    prev_cand_w = float(max(0.0, getattr(target_cell, 'phi_rear_cand_w', 0.0)))
    new_cand_w = float(min(5000.0, prev_cand_w + cand_w))
    prev_cand_phi = float(getattr(target_cell, 'phi_rear_cand', source_phi))
    if new_cand_w > 1e-12:
        target_cell.phi_rear_cand = float((prev_cand_w * prev_cand_phi + cand_w * source_phi) / max(1e-9, new_cand_w))
        target_cell.phi_rear_cand_w = new_cand_w

    rho_add = float(rho_gain * synth_w)
    target_cell.rho_rear = float(max(0.0, getattr(target_cell, 'rho_rear', 0.0)) + rho_add)
    target_cell.rho_rear_cand = float(max(0.0, getattr(target_cell, 'rho_rear_cand', 0.0)) + 0.75 * rho_add)
    target_cell.rps_commit_score = float(max(float(np.clip(getattr(target_cell, 'rps_commit_score', 0.0), 0.0, 1.0)), float(np.clip(score_floor + 0.45 * support_add, 0.0, 1.0))))
    target_cell.rps_active = float(max(float(np.clip(getattr(target_cell, 'rps_active', 0.0), 0.0, 1.0)), float(np.clip(active_floor + 0.25 * support_add, 0.0, 1.0))))
    target_cell.rps_commit_age = float(max(float(getattr(target_cell, 'rps_commit_age', 0.0)), age_floor))
    return synth_w, rho_add


def _dense_neighbor_support(voxel_map, idx, cfg, ref_phi: float) -> tuple[float, float, float]:
    radius = max(1, int(getattr(cfg.update, 'rps_bg_dense_neighbor_radius', 1)))
    neighbor_weight = float(max(0.0, getattr(cfg.update, 'rps_bg_dense_neighbor_weight', 0.55)))
    geom_weight = float(max(0.0, getattr(cfg.update, 'rps_bg_dense_geometry_weight', 0.30)))
    cons_ref = float(max(voxel_map.voxel_size, getattr(cfg.update, 'rps_consistency_ref', 0.03)))
    surface_on = bool(getattr(cfg.update, 'rps_bg_surface_constrained_enable', False))
    surface_min_conf = float(np.clip(getattr(cfg.update, 'rps_bg_surface_min_conf', 0.12), 0.0, 1.0))
    surface_agree_weight = float(np.clip(getattr(cfg.update, 'rps_bg_surface_agree_weight', 0.40), 0.0, 1.0))
    vals = []
    wts = []
    supports = []
    ccell = voxel_map.get_cell(idx)
    src_surface_conf, src_surface_phi = _surface_signal(ccell, cfg) if ccell is not None else (0.0, ref_phi)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nidx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                ncell = voxel_map.get_cell(nidx)
                if ncell is None:
                    continue
                comp = manifold_state_components(ncell, cfg)
                if comp['weight'] <= 1e-12:
                    continue
                phi_n = float(comp['phi'])
                agree = float(np.exp(-0.5 * ((phi_n - ref_phi) / cons_ref) ** 2))
                support_n = float(np.clip((1.0 - geom_weight) * comp['visible'] + geom_weight * agree, 0.0, 1.0))
                if surface_on:
                    nbr_surface_conf, nbr_surface_phi = _surface_signal(ncell, cfg)
                    if src_surface_conf < surface_min_conf or nbr_surface_conf < surface_min_conf:
                        continue
                    surf_agree = float(np.exp(-0.5 * ((nbr_surface_phi - src_surface_phi) / cons_ref) ** 2))
                    support_n = float(np.clip((1.0 - surface_agree_weight) * support_n + surface_agree_weight * surf_agree, 0.0, 1.0))
                    support_n *= _tangent_weight(voxel_map, idx, nidx, ccell, cfg)
                if support_n <= 1e-6:
                    continue
                vals.append(phi_n)
                wts.append(max(1e-6, neighbor_weight * support_n))
                supports.append(support_n)
    if not wts:
        return 0.0, ref_phi, 0.0
    support = float(np.clip(np.mean(np.asarray(supports, dtype=float)), 0.0, 1.0))
    phi = float(sum(v * w for v, w in zip(vals, wts)) / max(1e-9, sum(wts)))
    weight = float(min(1.0, sum(wts) / max(1.0, len(wts))))
    return support, phi, weight

def manifold_state_components(cell, cfg) -> Dict[str, float]:
    rho_ref = float(max(1e-6, getattr(cfg.update, 'rps_bg_manifold_rho_ref', getattr(cfg.update, 'rps_admission_rho_ref', 0.08))))
    weight_ref = float(max(1e-6, getattr(cfg.update, 'rps_bg_manifold_weight_ref', getattr(cfg.update, 'rps_admission_weight_ref', 0.35))))
    rho_n = float(np.clip(float(getattr(cell, 'rho_bg_stable', 0.0)) / rho_ref, 0.0, 1.5))
    w_n = float(1.0 - np.exp(-float(max(0.0, getattr(cell, 'phi_bg_memory_w', 0.0))) / weight_ref))
    visible = float(np.clip(max(float(np.clip(getattr(cell, 'bg_visible_mem', 0.0), 0.0, 1.0)), rho_n, w_n), 0.0, 1.0))
    obstructed = float(np.clip(getattr(cell, 'bg_obstruction_mem', 0.0), 0.0, 1.0))
    dense_support = float(np.clip(getattr(cell, 'bg_dense_support', 0.0), 0.0, 1.0))
    dense_weight = float(max(0.0, getattr(cell, 'bg_dense_w', 0.0)))
    phi = float(getattr(cell, 'phi_bg_memory', 0.0))
    if dense_weight > 1e-12:
        phi = float(getattr(cell, 'bg_dense_phi', phi))
    visible = float(np.clip(max(visible, dense_support), 0.0, 1.0))
    return {
        'visible': visible,
        'obstructed': obstructed,
        'rho_n': rho_n,
        'w_n': w_n,
        'phi': phi,
        'weight': float(max(float(getattr(cell, 'phi_bg_memory_w', 0.0)), dense_weight)),
        'dense_support': dense_support,
        'dense_weight': dense_weight,
    }
