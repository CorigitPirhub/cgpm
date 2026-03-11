from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from egf_dhmap3d.P10_method.bg_manifold import manifold_state_components
from egf_dhmap3d.P10_method.rps_admission import rear_state_support_components


RearRecord = Tuple[tuple[int, int, int], np.ndarray, np.ndarray, Dict[str, float]]


def rear_selectivity_components(
    self,
    *,
    idx: tuple[int, int, int],
    cell,
    point: np.ndarray,
    normal: np.ndarray,
    accepted_idx: set[tuple[int, int, int]],
    candidate_map: Dict[tuple[int, int, int], tuple],
) -> Dict[str, float]:
    cfg = self.cfg.update
    support_pack = rear_state_support_components(self, cell)
    info = candidate_map.get(idx, ())
    front_score = float(np.clip(info[8], 0.0, 1.0)) if len(info) >= 9 else 0.0
    rear_score = float(np.clip(info[9], 0.0, 1.0)) if len(info) >= 10 else 0.0
    rear_gap = float(max(0.0, info[10])) if len(info) >= 11 else 0.0
    rear_sep = float(np.clip(info[11], 0.0, 1.0)) if len(info) >= 12 else 0.0
    manifold = manifold_state_components(cell, self.cfg) if bool(getattr(cfg, 'rps_bg_manifold_state_enable', False)) else {
        'visible': 0.0,
        'obstructed': 0.0,
        'dense_support': 0.0,
        'phi': float(getattr(cell, 'phi_rear', 0.0)),
        'weight': 0.0,
    }
    bridge_support = float(max(manifold.get('dense_support', 0.0), manifold.get('visible', 0.0)))
    front_risk = float(np.clip(max(
        float(np.clip(getattr(cell, 'wod_front_conf', 0.0), 0.0, 1.0)),
        float(np.clip(getattr(cell, 'wod_shell_conf', 0.0), 0.0, 1.0)),
        float(np.clip(getattr(cell, 'visibility_contradiction', 0.0), 0.0, 1.0)),
        float(np.clip(getattr(cell, 'residual_evidence', 0.0), 0.0, 1.0)),
        float(np.clip(getattr(cell, 'st_mem', 0.0), 0.0, 1.0)),
    ), 0.0, 1.0))

    radius = max(1, int(getattr(cfg, 'rps_rear_selectivity_density_radius_cells', 1)))
    density_ref = max(1, int(getattr(cfg, 'rps_rear_selectivity_density_ref', 8)))
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
                if ninfo is not None and len(ninfo) >= 8 and str(ninfo[7]).startswith('rear'):
                    rear_neighbors += 1
    local_density = float(np.clip(rear_neighbors / max(1.0, float(density_ref)), 0.0, 1.0))

    support = float(np.clip(support_pack.get('support', 0.0), 0.0, 1.0))
    history_bg = float(np.clip(support_pack.get('history_bg', 0.0), 0.0, 1.0))
    static_support = float(np.clip(support_pack.get('static_support', 0.0), 0.0, 1.0))
    geom_score = float(np.clip(support_pack.get('geom_score', 0.0), 0.0, 1.0))
    dyn_risk = float(np.clip(support_pack.get('dyn_risk', 0.0), 0.0, 1.0))
    ghost_like = float(np.clip(support_pack.get('ghost_like', 0.0), 0.0, 1.0))
    ptdsf_stats = self._ptdsf_state_stats(cell)
    commit_n = float(np.clip(ptdsf_stats.get('commit_n', 0.0), 0.0, 1.0))
    rollback_n = float(np.clip(ptdsf_stats.get('rollback_n', 0.0), 0.0, 1.0))
    xmem_conf = float(np.clip(self._xmem_conf(cell), 0.0, 1.0))
    xmem_clear_conf = float(np.clip(self._xmem_clear_conf(cell), 0.0, 1.0))
    otv_conf = float(np.clip(self._otv_conf(cell), 0.0, 1.0))
    otv_geom = float(np.clip(self._otv_surface_conf(cell), 0.0, 1.0))
    dccm_commit = float(np.clip(getattr(cell, 'dccm_commit', 0.0), 0.0, 1.0))
    age_ref = float(max(1.0, getattr(cfg, 'ptdsf_commit_age_ref', 3.0)))
    xmem_age_n = float(np.clip(getattr(cell, 'xmem_age', 0.0) / max(1.0, getattr(cfg, 'xmem_age_ref', 1.0)), 0.0, 1.0))
    otv_age_n = float(np.clip(getattr(cell, 'otv_age', 0.0) / max(1.0, getattr(cfg, 'otv_age_ref', 1.0)), 0.0, 1.0))
    dccm_age_n = float(np.clip(getattr(cell, 'dccm_age', 0.0) / age_ref, 0.0, 1.0))
    rho_ref = float(max(1e-6, getattr(cfg, 'dual_state_static_protect_rho', 0.90)))
    weight_ref = float(max(1e-6, getattr(cfg, 'rps_admission_weight_ref', 0.35)))
    history_anchor = _history_anchor_score(cell, rho_ref=rho_ref, weight_ref=weight_ref)
    comp_alpha = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_competition_alpha', 0.80)))
    competition = float(np.clip(rear_score - comp_alpha * front_score, -1.0, 1.0))
    gap_min = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_gap_min', 0.018)))
    gap_max = float(max(gap_min + 1e-6, getattr(cfg, 'rps_rear_selectivity_gap_max', 0.090)))
    if rear_gap <= 1e-9:
        gap_validity = 0.0
    elif rear_gap < gap_min:
        gap_validity = float(np.clip(rear_gap / max(1e-6, gap_min), 0.0, 1.0))
    elif rear_gap > gap_max:
        gap_validity = float(np.clip(np.exp(-(rear_gap - gap_max) / max(1e-6, gap_max - gap_min)), 0.0, 1.0))
    else:
        gap_validity = 1.0
    competition_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_competition_floor', -0.02), -1.0, 1.0))
    comp_ok = float(np.clip((competition - competition_floor) / max(1e-6, 1.0 - competition_floor), 0.0, 1.0))
    history_like = float(np.clip(max(history_bg, static_support, bridge_support), 0.0, 1.0))
    surface_distance, surface_anchor = _surface_anchor_score(
        cell,
        dist_ref=float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_surface_distance_ref', 0.05))),
    )
    front_newness = float(np.clip(
        max(
            xmem_conf * (1.0 - xmem_age_n),
            otv_conf * (1.0 - otv_age_n),
            dccm_commit * (1.0 - dccm_age_n),
            rollback_n * (0.45 + 0.55 * front_score),
        ),
        0.0,
        1.0,
    ))
    occlusion_order = float(np.clip(
        history_like
        * (0.45 * front_newness + 0.30 * rollback_n + 0.25 * float(np.clip(getattr(cell, 'wod_front_conf', 0.0), 0.0, 1.0)))
        * (0.35 + 0.65 * max(comp_ok, rear_score)),
        0.0,
        1.0,
    ))
    front_residual = float(np.clip(
        max(
            front_score,
            front_risk,
            xmem_conf,
            xmem_clear_conf,
            otv_conf,
            otv_geom,
            float(np.clip(getattr(cell, 'visibility_contradiction', 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cell, 'st_mem', 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cell, 'residual_evidence', 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cell, 'dyn_prob', 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0)),
        )
        * (0.55 + 0.45 * max(xmem_age_n, otv_age_n, dccm_age_n))
        * (1.0 - 0.35 * occlusion_order),
        0.0,
        1.5,
    ))
    occluder_protect = float(np.clip(
        history_like
        * max(
            float(np.clip(getattr(cell, 'wod_front_conf', 0.0), 0.0, 1.0)),
            xmem_conf * (1.0 - 0.45 * xmem_age_n),
            otv_conf * (1.0 - 0.45 * otv_age_n),
            float(np.clip(getattr(cell, 'dyn_prob', 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0)),
        )
        * max(rear_score, comp_ok, 0.5 * gap_validity)
        * (0.45 + 0.55 * front_newness),
        0.0,
        1.0,
    ))
    dynamic_shell = float(np.clip(
        max(
            float(np.clip(getattr(cell, 'dyn_prob', 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cell, 'wod_front_conf', 0.0), 0.0, 1.0)),
            xmem_conf,
            otv_conf,
        )
        * np.exp(-rear_gap / max(1e-6, float(getattr(cfg, 'rps_rear_selectivity_dynamic_shell_gap_ref', 0.05))))
        * (0.55 + 0.45 * front_score),
        0.0,
        1.5,
    ))
    local_conflict = _local_conflict_score(
        self,
        idx=idx,
        rear_gap=rear_gap,
        accepted_idx=accepted_idx,
        candidate_map=candidate_map,
    )
    dynamic_trail = _dynamic_trail_score(
        self,
        idx=idx,
        accepted_idx=accepted_idx,
        candidate_map=candidate_map,
    )
    observation_count, observation_support = _observation_support_score(
        cell,
        count_ref=float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_observation_count_ref', 6.0))),
    )
    static_coherence = _static_neighbor_coherence_score(
        self,
        idx=idx,
        count_ref=float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_observation_count_ref', 6.0))),
    )
    penetration_score, penetration_free_span = _ray_penetration_consistency_score(
        self,
        idx=idx,
        point=np.asarray(point, dtype=float).reshape(3),
        normal=np.asarray(normal, dtype=float).reshape(3),
        rear_gap=rear_gap,
    )
    topology_thickness = float(penetration_free_span)
    thickness_score = float(np.clip(
        topology_thickness / max(1e-6, float(getattr(cfg, 'rps_rear_selectivity_thickness_ref', 0.08))),
        0.0,
        1.0,
    ))
    normal_consistency = _front_back_normal_consistency_score(
        self,
        idx=idx,
        normal=np.asarray(normal, dtype=float).reshape(3),
    )
    ray_convergence = _ray_convergence_score(
        self,
        idx=idx,
        normal=np.asarray(normal, dtype=float).reshape(3),
        rear_gap=rear_gap,
        thickness=topology_thickness,
        accepted_idx=accepted_idx,
        candidate_map=candidate_map,
    )
    effective_conflict = float(np.clip(
        local_conflict
        * (1.0 + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_front_residual_weight', 0.0))) * front_residual)
        * (1.0 - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_occluder_relief_weight', 0.0))) * occluder_protect),
        0.0,
        1.5,
    ))
    effective_front_residual = float(np.clip(
        front_residual * (1.0 - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_occluder_relief_weight', 0.0))) * occluder_protect),
        0.0,
        1.5,
    ))

    score = float(np.clip(
        float(max(0.0, getattr(cfg, 'rps_rear_selectivity_support_weight', 0.18))) * support
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_history_weight', 0.24))) * history_bg
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_static_weight', 0.16))) * static_support
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_geom_weight', 0.22))) * geom_score
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_bridge_weight', 0.10))) * bridge_support
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_density_weight', 0.10))) * local_density
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_history_anchor_weight', 0.0))) * history_anchor
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_surface_anchor_weight', 0.0))) * surface_anchor
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_penetration_weight', 0.0))) * penetration_score
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_observation_weight', 0.0))) * observation_support
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_static_coherence_weight', 0.0))) * static_coherence
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_thickness_weight', 0.0))) * thickness_score
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_normal_consistency_weight', 0.0))) * normal_consistency
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_ray_convergence_weight', 0.0))) * ray_convergence
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_rear_score_weight', 0.28))) * rear_score
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_competition_weight', 0.34))) * comp_ok
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_gap_weight', 0.18))) * gap_validity
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_sep_weight', 0.08))) * rear_sep
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_occlusion_order_weight', 0.0))) * occlusion_order
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_occluder_protect_weight', 0.0))) * occluder_protect
        - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_dyn_weight', 0.22))) * dyn_risk
        - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_ghost_weight', 0.18))) * ghost_like
        - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_front_weight', 0.16))) * front_risk,
        0.0,
        1.0,
    ))
    risk = float(np.clip(
        float(max(0.0, getattr(cfg, 'rps_rear_selectivity_dyn_weight', 0.22))) * dyn_risk
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_ghost_weight', 0.18))) * ghost_like
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_front_weight', 0.16))) * front_risk
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_front_score_weight', 0.28))) * front_score
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_geom_risk_weight', 0.22))) * (1.0 - geom_score)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_history_risk_weight', 0.16))) * (1.0 - history_bg)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_density_risk_weight', 0.10))) * (1.0 - local_density)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_gap_risk_weight', 0.18))) * (1.0 - gap_validity)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_surface_anchor_risk_weight', 0.0))) * (1.0 - surface_anchor)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_penetration_risk_weight', 0.0))) * (1.0 - penetration_score)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_observation_risk_weight', 0.0))) * (1.0 - observation_support)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_thickness_risk_weight', 0.0))) * (1.0 - thickness_score)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_local_conflict_weight', 0.0))) * effective_conflict
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_front_residual_weight', 0.0))) * effective_front_residual
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_occlusion_order_risk_weight', 0.0))) * (1.0 - occlusion_order)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_dynamic_trail_weight', 0.0))) * dynamic_trail * (1.0 - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_dynamic_trail_relief_weight', 0.0))) * occluder_protect)
        + float(max(0.0, getattr(cfg, 'rps_rear_selectivity_dynamic_shell_weight', 0.0))) * dynamic_shell
        - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_bridge_relief_weight', 0.10))) * bridge_support
        - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_history_anchor_relief_weight', 0.0))) * history_anchor
        - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_static_coherence_relief_weight', 0.0))) * static_coherence
        - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_normal_consistency_relief_weight', 0.0))) * normal_consistency
        - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_ray_convergence_relief_weight', 0.0))) * ray_convergence
        - float(max(0.0, getattr(cfg, 'rps_rear_selectivity_static_relief_weight', 0.08))) * static_support,
        0.0,
        1.5,
    ))

    return {
        'support': support,
        'history_bg': history_bg,
        'static_support': static_support,
        'geom_score': geom_score,
        'bridge_support': bridge_support,
        'local_density': local_density,
        'history_anchor': history_anchor,
        'surface_distance': surface_distance,
        'surface_anchor': surface_anchor,
        'penetration_score': penetration_score,
        'penetration_free_span': penetration_free_span,
        'topology_thickness': topology_thickness,
        'thickness_score': thickness_score,
        'observation_count': observation_count,
        'observation_support': observation_support,
        'static_coherence': static_coherence,
        'normal_consistency': normal_consistency,
        'ray_convergence': ray_convergence,
        'front_score': front_score,
        'rear_score': rear_score,
        'rear_gap': rear_gap,
        'rear_sep': rear_sep,
        'competition': competition,
        'gap_validity': gap_validity,
        'occlusion_order': occlusion_order,
        'occluder_protect': occluder_protect,
        'dynamic_shell': dynamic_shell,
        'local_conflict': effective_conflict,
        'front_residual': effective_front_residual,
        'dynamic_trail': dynamic_trail,
        'dyn_risk': dyn_risk,
        'ghost_like': ghost_like,
        'front_risk': front_risk,
        'score': score,
        'risk': risk,
    }


def filter_rear_records(self, records: List[RearRecord]) -> tuple[List[RearRecord], Dict[str, float]]:
    cfg = self.cfg.update
    if not bool(getattr(cfg, 'rps_rear_selectivity_enable', False)):
        return records, {
            'rear_selectivity_pre_count': float(len(records)),
            'rear_selectivity_kept_count': float(len(records)),
            'rear_selectivity_drop_count': 0.0,
            'rear_selectivity_topk_drop_count': 0.0,
            'rear_selectivity_score_sum': float(sum(r[3]['score'] for r in records)),
            'rear_selectivity_risk_sum': float(sum(r[3]['risk'] for r in records)),
            'rear_selectivity_pre_front_score_sum': float(sum(r[3]['front_score'] for r in records)),
            'rear_selectivity_pre_front_residual_sum': float(sum(r[3]['front_residual'] for r in records)),
            'rear_selectivity_pre_occlusion_order_sum': float(sum(r[3]['occlusion_order'] for r in records)),
            'rear_selectivity_pre_local_conflict_sum': float(sum(r[3]['local_conflict'] for r in records)),
            'rear_selectivity_pre_dynamic_trail_sum': float(sum(r[3]['dynamic_trail'] for r in records)),
            'rear_selectivity_pre_dyn_risk_sum': float(sum(r[3]['dyn_risk'] for r in records)),
            'rear_selectivity_pre_history_anchor_sum': float(sum(r[3]['history_anchor'] for r in records)),
            'rear_selectivity_pre_surface_anchor_sum': float(sum(r[3]['surface_anchor'] for r in records)),
            'rear_selectivity_pre_surface_distance_sum': float(sum(r[3]['surface_distance'] for r in records)),
            'rear_selectivity_pre_dynamic_shell_sum': float(sum(r[3]['dynamic_shell'] for r in records)),
            'rear_selectivity_pre_penetration_sum': float(sum(r[3]['penetration_score'] for r in records)),
            'rear_selectivity_pre_penetration_free_span_sum': float(sum(r[3]['penetration_free_span'] for r in records)),
            'rear_selectivity_pre_observation_count_sum': float(sum(r[3]['observation_count'] for r in records)),
            'rear_selectivity_pre_observation_support_sum': float(sum(r[3]['observation_support'] for r in records)),
            'rear_selectivity_pre_static_coherence_sum': float(sum(r[3]['static_coherence'] for r in records)),
            'rear_selectivity_pre_topology_thickness_sum': float(sum(r[3]['topology_thickness'] for r in records)),
            'rear_selectivity_pre_normal_consistency_sum': float(sum(r[3]['normal_consistency'] for r in records)),
            'rear_selectivity_pre_ray_convergence_sum': float(sum(r[3]['ray_convergence'] for r in records)),
        }

    score_min = float(np.clip(getattr(cfg, 'rps_rear_selectivity_score_min', 0.46), 0.0, 1.0))
    risk_max = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_risk_max', 0.45)))
    geom_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_geom_floor', 0.48), 0.0, 1.0))
    history_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_history_floor', 0.36), 0.0, 1.0))
    bridge_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_bridge_floor', 0.12), 0.0, 1.0))
    competition_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_competition_floor', -0.02), -1.0, 1.0))
    gap_valid_min = float(np.clip(getattr(cfg, 'rps_rear_selectivity_gap_valid_min', 0.28), 0.0, 1.0))
    front_score_max = float(np.clip(getattr(cfg, 'rps_rear_selectivity_front_score_max', 0.92), 0.0, 1.0))
    occlusion_order_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_occlusion_order_floor', 0.0), 0.0, 1.0))
    local_conflict_max = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_local_conflict_max', 1.5)))
    front_residual_max = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_front_residual_max', 1.5)))
    occluder_protect_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_occluder_protect_floor', 0.0), 0.0, 1.0))
    dynamic_trail_max = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_dynamic_trail_max', 1.5)))
    history_anchor_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_history_anchor_floor', 0.0), 0.0, 1.0))
    surface_anchor_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_surface_anchor_floor', 0.0), 0.0, 1.0))
    dynamic_shell_max = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_dynamic_shell_max', 1.5)))
    penetration_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_penetration_floor', 0.0), 0.0, 1.0))
    observation_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_observation_floor', 0.0), 0.0, 1.0))
    observation_min_count = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_observation_min_count', 0.0)))
    unobserved_veto = bool(getattr(cfg, 'rps_rear_selectivity_unobserved_veto_enable', False))
    static_coherence_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_static_coherence_floor', 0.0), 0.0, 1.0))
    thickness_floor = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_thickness_floor', 0.0)))
    normal_consistency_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_normal_consistency_floor', 0.0), 0.0, 1.0))
    ray_convergence_floor = float(np.clip(getattr(cfg, 'rps_rear_selectivity_ray_convergence_floor', 0.0), 0.0, 1.0))
    topk = max(0, int(getattr(cfg, 'rps_rear_selectivity_topk', 0)))
    rank_risk_w = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_rank_risk_weight', 0.55)))
    rank_penetration_w = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_penetration_weight', 0.0)))
    rank_observation_w = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_observation_weight', 0.0)))
    rank_coherence_w = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_static_coherence_weight', 0.0)))
    rank_thickness_w = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_thickness_weight', 0.0)))
    rank_normal_consistency_w = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_normal_consistency_weight', 0.0)))
    rank_convergence_w = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_ray_convergence_weight', 0.0)))

    kept: List[RearRecord] = []
    drop_count = 0
    for record in records:
        comp = record[3]
        history_like = max(comp['history_bg'], comp['static_support'])
        history_ok = bool(history_like >= history_floor or comp['bridge_support'] >= bridge_floor)
        geom_ok = bool(comp['geom_score'] >= geom_floor)
        score_ok = bool(comp['score'] >= score_min)
        risk_ok = bool(comp['risk'] <= risk_max)
        competition_ok = bool(comp['competition'] >= competition_floor)
        gap_ok = bool(comp['gap_validity'] >= gap_valid_min)
        front_ok = bool(comp['front_score'] <= front_score_max)
        occlusion_ok = bool(comp['occlusion_order'] >= occlusion_order_floor or comp['competition'] >= competition_floor + 0.05)
        conflict_ok = bool(comp['local_conflict'] <= local_conflict_max)
        residual_ok = bool(comp['front_residual'] <= front_residual_max)
        protect_ok = bool(comp['occluder_protect'] >= occluder_protect_floor or comp['competition'] >= competition_floor + 0.02)
        trail_ok = bool(comp['dynamic_trail'] <= dynamic_trail_max or comp['occluder_protect'] >= occluder_protect_floor + 0.05)
        history_anchor_ok = bool(comp['history_anchor'] >= history_anchor_floor or comp['history_bg'] >= history_floor)
        surface_anchor_ok = bool(comp['surface_anchor'] >= surface_anchor_floor or comp['bridge_support'] >= bridge_floor)
        shell_ok = bool(comp['dynamic_shell'] <= dynamic_shell_max or comp['surface_anchor'] >= surface_anchor_floor + 0.05)
        penetration_ok = bool(comp['penetration_score'] >= penetration_floor or comp['occluder_protect'] >= occluder_protect_floor + 0.05)
        observation_ok = bool(comp['observation_support'] >= observation_floor or comp['static_coherence'] >= static_coherence_floor or comp['bridge_support'] >= bridge_floor)
        observation_count_ok = bool((not unobserved_veto) or comp['observation_count'] >= observation_min_count or comp['static_coherence'] >= static_coherence_floor + 0.05)
        static_coherence_ok = bool(comp['static_coherence'] >= static_coherence_floor or comp['observation_support'] >= observation_floor + 0.05)
        thickness_ok = bool(comp['topology_thickness'] >= thickness_floor or comp['penetration_score'] >= penetration_floor + 0.05)
        normal_ok = bool(comp['normal_consistency'] >= normal_consistency_floor or comp['static_coherence'] >= static_coherence_floor + 0.05)
        convergence_ok = bool(comp['ray_convergence'] >= ray_convergence_floor or comp['static_coherence'] >= static_coherence_floor + 0.05)
        if history_ok and geom_ok and score_ok and risk_ok and competition_ok and gap_ok and front_ok and occlusion_ok and conflict_ok and residual_ok and protect_ok and trail_ok and history_anchor_ok and surface_anchor_ok and shell_ok and penetration_ok and observation_ok and observation_count_ok and static_coherence_ok and thickness_ok and normal_ok and convergence_ok:
            kept.append(record)
        else:
            drop_count += 1

    topk_drop_count = 0
    if topk > 0 and len(kept) > topk:
        kept = sorted(
            kept,
            key=lambda record: (
                record[3]['score']
                + 0.40 * record[3]['history_anchor']
                + 0.35 * record[3]['surface_anchor']
                + 0.40 * rank_penetration_w * record[3]['penetration_score']
                + 0.32 * rank_observation_w * record[3]['observation_support']
                + 0.28 * rank_coherence_w * record[3]['static_coherence']
                + 0.22 * rank_thickness_w * record[3]['thickness_score']
                + 0.24 * rank_normal_consistency_w * record[3]['normal_consistency']
                + 0.28 * rank_convergence_w * record[3]['ray_convergence']
                + 0.50 * record[3]['occlusion_order']
                + 0.35 * record[3]['occluder_protect']
                + 0.45 * record[3]['competition']
                + 0.25 * record[3]['gap_validity']
                + 0.20 * record[3]['rear_score']
                - 0.30 * record[3]['local_conflict']
                - 0.30 * record[3]['front_residual']
                - 0.25 * record[3]['dynamic_trail']
                - 0.25 * record[3]['dynamic_shell']
                - rank_risk_w * record[3]['risk'],
                record[3]['history_bg'],
                record[3]['geom_score'],
                record[3]['bridge_support'],
                record[3]['local_density'],
            ),
            reverse=True,
        )
        topk_drop_count = len(kept) - topk
        kept = kept[:topk]

    return kept, {
        'rear_selectivity_pre_count': float(len(records)),
        'rear_selectivity_kept_count': float(len(kept)),
        'rear_selectivity_drop_count': float(drop_count),
        'rear_selectivity_topk_drop_count': float(topk_drop_count),
        'rear_selectivity_score_sum': float(sum(r[3]['score'] for r in kept)),
        'rear_selectivity_risk_sum': float(sum(r[3]['risk'] for r in kept)),
        'rear_selectivity_pre_front_score_sum': float(sum(r[3]['front_score'] for r in records)),
        'rear_selectivity_pre_front_residual_sum': float(sum(r[3]['front_residual'] for r in records)),
        'rear_selectivity_pre_occlusion_order_sum': float(sum(r[3]['occlusion_order'] for r in records)),
        'rear_selectivity_pre_local_conflict_sum': float(sum(r[3]['local_conflict'] for r in records)),
        'rear_selectivity_pre_dynamic_trail_sum': float(sum(r[3]['dynamic_trail'] for r in records)),
        'rear_selectivity_pre_dyn_risk_sum': float(sum(r[3]['dyn_risk'] for r in records)),
        'rear_selectivity_pre_history_anchor_sum': float(sum(r[3]['history_anchor'] for r in records)),
        'rear_selectivity_pre_surface_anchor_sum': float(sum(r[3]['surface_anchor'] for r in records)),
        'rear_selectivity_pre_surface_distance_sum': float(sum(r[3]['surface_distance'] for r in records)),
        'rear_selectivity_pre_dynamic_shell_sum': float(sum(r[3]['dynamic_shell'] for r in records)),
        'rear_selectivity_pre_penetration_sum': float(sum(r[3]['penetration_score'] for r in records)),
        'rear_selectivity_pre_penetration_free_span_sum': float(sum(r[3]['penetration_free_span'] for r in records)),
        'rear_selectivity_pre_observation_count_sum': float(sum(r[3]['observation_count'] for r in records)),
        'rear_selectivity_pre_observation_support_sum': float(sum(r[3]['observation_support'] for r in records)),
        'rear_selectivity_pre_static_coherence_sum': float(sum(r[3]['static_coherence'] for r in records)),
        'rear_selectivity_pre_topology_thickness_sum': float(sum(r[3]['topology_thickness'] for r in records)),
        'rear_selectivity_pre_normal_consistency_sum': float(sum(r[3]['normal_consistency'] for r in records)),
        'rear_selectivity_pre_ray_convergence_sum': float(sum(r[3]['ray_convergence'] for r in records)),
        'rear_selectivity_front_score_sum': float(sum(r[3]['front_score'] for r in kept)),
        'rear_selectivity_rear_score_sum': float(sum(r[3]['rear_score'] for r in kept)),
        'rear_selectivity_gap_sum': float(sum(r[3]['rear_gap'] for r in kept)),
        'rear_selectivity_competition_sum': float(sum(r[3]['competition'] for r in kept)),
        'rear_selectivity_occlusion_order_sum': float(sum(r[3]['occlusion_order'] for r in kept)),
        'rear_selectivity_occluder_protect_sum': float(sum(r[3]['occluder_protect'] for r in kept)),
        'rear_selectivity_local_conflict_sum': float(sum(r[3]['local_conflict'] for r in kept)),
        'rear_selectivity_front_residual_sum': float(sum(r[3]['front_residual'] for r in kept)),
        'rear_selectivity_dynamic_trail_sum': float(sum(r[3]['dynamic_trail'] for r in kept)),
        'rear_selectivity_dyn_risk_sum': float(sum(r[3]['dyn_risk'] for r in kept)),
        'rear_selectivity_history_anchor_sum': float(sum(r[3]['history_anchor'] for r in kept)),
        'rear_selectivity_surface_anchor_sum': float(sum(r[3]['surface_anchor'] for r in kept)),
        'rear_selectivity_surface_distance_sum': float(sum(r[3]['surface_distance'] for r in kept)),
        'rear_selectivity_dynamic_shell_sum': float(sum(r[3]['dynamic_shell'] for r in kept)),
        'rear_selectivity_penetration_sum': float(sum(r[3]['penetration_score'] for r in kept)),
        'rear_selectivity_penetration_free_span_sum': float(sum(r[3]['penetration_free_span'] for r in kept)),
        'rear_selectivity_observation_count_sum': float(sum(r[3]['observation_count'] for r in kept)),
        'rear_selectivity_observation_support_sum': float(sum(r[3]['observation_support'] for r in kept)),
        'rear_selectivity_static_coherence_sum': float(sum(r[3]['static_coherence'] for r in kept)),
        'rear_selectivity_topology_thickness_sum': float(sum(r[3]['topology_thickness'] for r in kept)),
        'rear_selectivity_normal_consistency_sum': float(sum(r[3]['normal_consistency'] for r in kept)),
        'rear_selectivity_ray_convergence_sum': float(sum(r[3]['ray_convergence'] for r in kept)),
    }


def _history_anchor_score(cell, *, rho_ref: float, weight_ref: float) -> float:
    rho_static_n = float(np.clip(getattr(cell, 'rho_static', 0.0) / rho_ref, 0.0, 1.5))
    rho_bg_n = float(np.clip(getattr(cell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.5))
    rho_stable_n = float(np.clip(getattr(cell, 'rho_bg_stable', 0.0) / rho_ref, 0.0, 1.5))
    w_static_n = float(1.0 - np.exp(-max(0.0, getattr(cell, 'phi_static_w', 0.0)) / weight_ref))
    w_geo_n = float(1.0 - np.exp(-max(0.0, getattr(cell, 'phi_geo_w', 0.0)) / weight_ref))
    w_bg_n = float(1.0 - np.exp(-max(0.0, getattr(cell, 'phi_bg_w', 0.0)) / weight_ref))
    w_mem_n = float(1.0 - np.exp(-max(0.0, getattr(cell, 'phi_bg_memory_w', 0.0)) / weight_ref))
    visible_mem = float(np.clip(getattr(cell, 'bg_visible_mem', 0.0), 0.0, 1.0))
    return float(np.clip(
        0.22 * min(1.0, rho_static_n)
        + 0.16 * min(1.0, rho_bg_n)
        + 0.16 * min(1.0, rho_stable_n)
        + 0.14 * w_static_n
        + 0.10 * w_geo_n
        + 0.10 * w_bg_n
        + 0.06 * w_mem_n
        + 0.06 * visible_mem,
        0.0,
        1.0,
    ))


def _surface_anchor_score(cell, *, dist_ref: float) -> tuple[float, float]:
    refs: List[float] = []
    if float(getattr(cell, 'phi_static_w', 0.0)) > 1e-12:
        refs.append(abs(float(getattr(cell, 'phi_static', 0.0))))
    if float(getattr(cell, 'phi_geo_w', 0.0)) > 1e-12:
        refs.append(abs(float(getattr(cell, 'phi_geo', 0.0))))
    if float(getattr(cell, 'phi_bg_w', 0.0)) > 1e-12:
        refs.append(abs(float(getattr(cell, 'phi_bg', 0.0))))
    if float(getattr(cell, 'phi_bg_memory_w', 0.0)) > 1e-12:
        refs.append(abs(float(getattr(cell, 'phi_bg_memory', 0.0))))
    if not refs:
        return float(dist_ref), 0.0
    surface_distance = float(min(refs))
    surface_anchor = float(np.clip(np.exp(-0.5 * (surface_distance / max(1e-6, dist_ref)) ** 2), 0.0, 1.0))
    return surface_distance, surface_anchor


def _observation_support_score(cell, *, count_ref: float) -> tuple[float, float]:
    obs_count = float(max(0.0, getattr(cell, 'observation_count', 0.0)))
    obs_support = float(np.clip(1.0 - np.exp(-obs_count / max(1e-6, count_ref)), 0.0, 1.0))
    return obs_count, obs_support


def _static_neighbor_coherence_score(
    self,
    *,
    idx: tuple[int, int, int],
    count_ref: float,
) -> float:
    cfg = self.cfg.update
    radius = max(1, int(getattr(cfg, 'rps_rear_selectivity_static_coherence_radius_cells', 1)))
    min_w = float(max(0.0, getattr(cfg, 'rps_rear_selectivity_static_neighbor_min_weight', 0.20)))
    dyn_max = float(np.clip(getattr(cfg, 'rps_rear_selectivity_static_neighbor_dyn_max', 0.35), 0.0, 1.0))
    coh_ref = float(np.clip(getattr(cfg, 'rps_rear_selectivity_static_coherence_ref', 0.35), 1e-6, 1.0))
    rho_ref = float(max(1e-6, getattr(cfg, 'dual_state_static_protect_rho', 0.90)))
    accum = 0.0
    wsum = 0.0
    center = self.index_to_center(idx)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nidx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                ncell = self.get_cell(nidx)
                if ncell is None:
                    continue
                dyn_like = float(np.clip(max(
                    float(np.clip(getattr(ncell, 'dyn_prob', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'z_dyn', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'wod_front_conf', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'wod_shell_conf', 0.0), 0.0, 1.0)),
                ), 0.0, 1.0))
                if dyn_like > dyn_max:
                    continue
                obs_n = _observation_support_score(ncell, count_ref=count_ref)[1]
                weight_like = float(np.clip(1.0 - np.exp(-max(
                    float(getattr(ncell, 'phi_static_w', 0.0)),
                    float(getattr(ncell, 'phi_geo_w', 0.0)),
                    float(getattr(ncell, 'phi_bg_w', 0.0)),
                ) / max(1e-6, min_w)), 0.0, 1.0))
                if obs_n <= 1e-6 and weight_like <= 1e-6:
                    continue
                static_like = float(np.clip(max(
                    float(np.clip(getattr(ncell, 'p_static', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
                ), 0.0, 1.0))
                support = static_like * max(obs_n, weight_like) * (1.0 - dyn_like)
                dist = float(np.linalg.norm(self.index_to_center(nidx) - center) / max(1e-6, self.voxel_size))
                prox = float(np.exp(-0.5 * (dist / max(1.0, float(radius))) ** 2))
                accum += prox * support
                wsum += prox
    if wsum <= 1e-9:
        return 0.0
    return float(np.clip(accum / max(1e-9, coh_ref * wsum), 0.0, 1.0))


def _front_back_normal_consistency_score(
    self,
    *,
    idx: tuple[int, int, int],
    normal: np.ndarray,
) -> float:
    cfg = self.cfg.update
    radius = max(1, int(getattr(cfg, 'rps_rear_selectivity_normal_consistency_radius_cells', 1)))
    dyn_max = float(np.clip(getattr(cfg, 'rps_rear_selectivity_normal_consistency_dyn_max', 0.35), 0.0, 1.0))
    rho_ref = float(max(1e-6, getattr(cfg, 'dual_state_static_protect_rho', 0.90)))
    target = np.asarray(normal, dtype=float).reshape(3)
    target_n = float(np.linalg.norm(target))
    if target_n < 1e-8:
        return 0.0
    target = target / target_n
    accum = 0.0
    wsum = 0.0
    center = self.index_to_center(idx)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nidx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                ncell = self.get_cell(nidx)
                if ncell is None:
                    continue
                dyn_like = float(np.clip(max(
                    float(np.clip(getattr(ncell, 'dyn_prob', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'z_dyn', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'wod_front_conf', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'wod_shell_conf', 0.0), 0.0, 1.0)),
                ), 0.0, 1.0))
                if dyn_like > dyn_max:
                    continue
                g = np.asarray(getattr(ncell, 'g_mean', np.zeros(3, dtype=float)), dtype=float).reshape(3)
                g_n = float(np.linalg.norm(g))
                if g_n < 1e-8:
                    continue
                g = g / g_n
                static_like = float(np.clip(max(
                    float(np.clip(getattr(ncell, 'p_static', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
                ), 0.0, 1.0))
                if static_like <= 1e-6:
                    continue
                obs_support = _observation_support_score(
                    ncell,
                    count_ref=float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_observation_count_ref', 6.0))),
                )[1]
                weight_like = max(
                    obs_support,
                    float(np.clip(1.0 - np.exp(-max(
                        float(getattr(ncell, 'phi_static_w', 0.0)),
                        float(getattr(ncell, 'phi_geo_w', 0.0)),
                        float(getattr(ncell, 'phi_bg_w', 0.0)),
                    ) / max(1e-6, getattr(cfg, 'rps_rear_selectivity_static_neighbor_min_weight', 0.20))), 0.0, 1.0)),
                )
                cos = float(np.clip(abs(np.dot(target, g)), 0.0, 1.0))
                dist = float(np.linalg.norm(self.index_to_center(nidx) - center) / max(1e-6, self.voxel_size))
                prox = float(np.exp(-0.5 * (dist / max(1.0, float(radius))) ** 2))
                w = prox * static_like * weight_like
                accum += w * cos
                wsum += w
    if wsum <= 1e-9:
        return 0.0
    return float(np.clip(accum / wsum, 0.0, 1.0))


def _ray_convergence_score(
    self,
    *,
    idx: tuple[int, int, int],
    normal: np.ndarray,
    rear_gap: float,
    thickness: float,
    accepted_idx: set[tuple[int, int, int]],
    candidate_map: Dict[tuple[int, int, int], tuple],
) -> float:
    cfg = self.cfg.update
    radius = max(1, int(getattr(cfg, 'rps_rear_selectivity_ray_convergence_radius_cells', 1)))
    gap_ref = float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_ray_convergence_gap_ref', 0.06)))
    thickness_ref = float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_ray_convergence_thickness_ref', 0.08)))
    normal_cos_min = float(np.clip(getattr(cfg, 'rps_rear_selectivity_ray_convergence_normal_cos', 0.75), 0.0, 1.0))
    conv_ref = float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_ray_convergence_ref', 2.0)))
    target = np.asarray(normal, dtype=float).reshape(3)
    target_n = float(np.linalg.norm(target))
    if target_n < 1e-8:
        return 0.0
    target = target / target_n
    accum = 0.0
    center = self.index_to_center(idx)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nidx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                if nidx not in accepted_idx:
                    continue
                ninfo = candidate_map.get(nidx)
                if ninfo is None or len(ninfo) < 12 or not str(ninfo[7]).startswith('rear'):
                    continue
                ncell = ninfo[0]
                ng = np.asarray(getattr(ncell, 'g_mean', np.zeros(3, dtype=float)), dtype=float).reshape(3)
                ng_n = float(np.linalg.norm(ng))
                if ng_n < 1e-8:
                    continue
                ng = ng / ng_n
                cos = float(np.clip(abs(np.dot(target, ng)), 0.0, 1.0))
                if cos < normal_cos_min:
                    continue
                nrear_gap = float(max(0.0, ninfo[10]))
                gap_match = float(np.exp(-abs(nrear_gap - rear_gap) / gap_ref))
                thickness_match = float(np.exp(-abs(nrear_gap - thickness) / thickness_ref))
                dist = float(np.linalg.norm(self.index_to_center(nidx) - center) / max(1e-6, self.voxel_size))
                prox = float(np.exp(-0.5 * (dist / max(1.0, float(radius))) ** 2))
                accum += prox * gap_match * max(cos, thickness_match)
    return float(np.clip(accum / conv_ref, 0.0, 1.0))


def _ray_penetration_consistency_score(
    self,
    *,
    idx: tuple[int, int, int],
    point: np.ndarray,
    normal: np.ndarray,
    rear_gap: float,
) -> tuple[float, float]:
    cfg = self.cfg.update
    direction = np.asarray(normal, dtype=float).reshape(3)
    direction_n = float(np.linalg.norm(direction))
    if direction_n < 1e-8:
        direction = self.index_to_center(idx) - point
        direction_n = float(np.linalg.norm(direction))
    if direction_n < 1e-8:
        return 0.0, 0.0
    direction = direction / direction_n
    step = float(max(0.5 * self.voxel_size, 0.75 * self.voxel_size))
    max_steps = max(2, int(getattr(cfg, 'rps_rear_selectivity_penetration_max_steps', 10)))
    free_ref = float(max(step, getattr(cfg, 'rps_rear_selectivity_penetration_free_ref', 0.05)))
    obs_ref = float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_observation_count_ref', 6.0)))
    rho_ref = float(max(1e-6, getattr(cfg, 'dual_state_static_protect_rho', 0.90)))
    max_dist = float(max(2.0 * step, max(float(rear_gap), step) + max_steps * step))

    best_score = 0.0
    best_free_span = 0.0
    for sign in (1.0, -1.0):
        ray_dir = sign * direction
        seen_free = False
        free_span = 0.0
        front_after_free = 0.0
        solid_before_free = 0.0
        s = step
        while s <= max_dist + 1e-9:
            x = point + s * ray_dir
            c = self.get_cell(self.world_to_index(x))
            if c is None:
                free_like = 1.0
                front_like = 0.0
                static_like = 0.0
            else:
                obs_n = _observation_support_score(c, count_ref=obs_ref)[1]
                weight_like = float(np.clip(1.0 - np.exp(-max(
                    float(getattr(c, 'phi_static_w', 0.0)),
                    float(getattr(c, 'phi_geo_w', 0.0)),
                    float(getattr(c, 'phi_bg_w', 0.0)),
                    float(getattr(c, 'phi_w', 0.0)),
                ) / max(1e-6, 0.35)), 0.0, 1.0))
                front_like = float(np.clip(max(
                    float(np.clip(getattr(c, 'wod_front_conf', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(c, 'dyn_prob', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(c, 'z_dyn', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(c, 'xmem_occ', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(c, 'otv_score', 0.0), 0.0, 1.0)),
                ), 0.0, 1.0))
                static_like = float(np.clip(max(
                    float(np.clip(getattr(c, 'p_static', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(c, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
                    float(np.clip(getattr(c, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
                ) * max(obs_n, weight_like), 0.0, 1.0))
                free_ratio = float(getattr(c, 'free_evidence', 0.0) / max(1e-6, getattr(c, 'surf_evidence', 0.0)))
                free_like = float(np.clip(max(
                    1.0 - max(obs_n, weight_like),
                    float(np.clip((free_ratio - 0.55) / 0.75, 0.0, 1.0)),
                    float(np.clip(getattr(c, 'clear_hits', 0.0) / 2.0, 0.0, 1.0)),
                ) * (1.0 - 0.35 * static_like), 0.0, 1.0))
            if not seen_free:
                solid_before_free = max(solid_before_free, max(front_like, static_like))
            if free_like >= 0.55:
                seen_free = True
                free_span = s
            elif seen_free:
                front_after_free = max(front_after_free, max(front_like, 0.65 * static_like))
            s += step
        free_score = float(np.clip(free_span / max(1e-6, free_ref), 0.0, 1.0))
        solid_penalty = float(np.clip((solid_before_free - 0.65) / 0.35, 0.0, 1.0))
        score = float(np.clip(front_after_free * (0.30 + 0.70 * free_score) * (1.0 - 0.40 * solid_penalty), 0.0, 1.0))
        if score > best_score:
            best_score = score
            best_free_span = free_span
    return float(best_score), float(best_free_span)


def _local_conflict_score(
    self,
    *,
    idx: tuple[int, int, int],
    rear_gap: float,
    accepted_idx: set[tuple[int, int, int]],
    candidate_map: Dict[tuple[int, int, int], tuple],
) -> float:
    cfg = self.cfg.update
    radius = max(1, int(getattr(cfg, 'rps_rear_selectivity_conflict_radius_cells', 1)))
    front_score_min = float(np.clip(getattr(cfg, 'rps_rear_selectivity_conflict_front_score_min', 0.20), 0.0, 1.0))
    static_score_min = float(np.clip(getattr(cfg, 'rps_rear_selectivity_conflict_static_score_min', 0.35), 0.0, 1.0))
    dist_scale = float(max(0.25, getattr(cfg, 'rps_rear_selectivity_conflict_dist_scale', 1.2)))
    gap_ref = float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_conflict_gap_ref', 0.06)))
    conflict_ref = float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_conflict_ref', 1.8)))
    rho_ref = float(max(1e-6, getattr(cfg, 'dual_state_static_protect_rho', 0.90)))

    accum = 0.0
    center = self.index_to_center(idx)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nidx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                if nidx not in accepted_idx:
                    continue
                ninfo = candidate_map.get(nidx)
                if ninfo is None or len(ninfo) < 8:
                    continue
                nbank = str(ninfo[7])
                if nbank.startswith('rear'):
                    continue
                ncell = ninfo[0]
                nfront = float(np.clip(ninfo[8], 0.0, 1.0)) if len(ninfo) >= 9 else 0.0
                nstatic = float(np.clip(max(
                    float(np.clip(getattr(ncell, 'p_static', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
                ), 0.0, 1.0))
                if nfront < front_score_min and nstatic < static_score_min:
                    continue
                dist = float(np.linalg.norm(self.index_to_center(nidx) - center) / max(1e-6, self.voxel_size))
                prox = float(np.exp(-0.5 * (dist / dist_scale) ** 2))
                gap_overlap = float(np.clip(1.0 - rear_gap / gap_ref, 0.0, 1.0))
                accum += prox * max(nfront, 0.75 * nstatic) * (0.45 + 0.55 * gap_overlap)
    return float(np.clip(accum / conflict_ref, 0.0, 1.5))


def _dynamic_trail_score(
    self,
    *,
    idx: tuple[int, int, int],
    accepted_idx: set[tuple[int, int, int]],
    candidate_map: Dict[tuple[int, int, int], tuple],
) -> float:
    cfg = self.cfg.update
    radius = max(1, int(getattr(cfg, 'rps_rear_selectivity_trail_radius_cells', 1)))
    trail_ref = float(max(1e-6, getattr(cfg, 'rps_rear_selectivity_trail_ref', 2.0)))
    accum = 0.0
    center = self.index_to_center(idx)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                nidx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                ninfo = candidate_map.get(nidx)
                if ninfo is None:
                    continue
                ncell = ninfo[0]
                dyn_peak = float(np.clip(max(
                    float(np.clip(getattr(ncell, 'dyn_prob', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'z_dyn', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'xmem_occ', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'xmem_score', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'otv_score', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'dccm_commit', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'visibility_contradiction', 0.0), 0.0, 1.0)),
                    float(np.clip(getattr(ncell, 'st_mem', 0.0), 0.0, 1.0)),
                ), 0.0, 1.0))
                if dyn_peak <= 1e-6:
                    continue
                dist = float(np.linalg.norm(self.index_to_center(nidx) - center) / max(1e-6, self.voxel_size))
                prox = float(np.exp(-0.5 * (dist / max(1.0, float(radius))) ** 2))
                accum += prox * dyn_peak
    return float(np.clip(accum / trail_ref, 0.0, 1.5))
