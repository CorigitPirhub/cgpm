from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np






def route_measurements_with_delayed_bg(
    self,
    accepted: List,
) -> Tuple[List, List, Dict[str, float]]:
    if self.delayed_background_map is None or self.foreground_map is None or not bool(getattr(self.cfg.update, 'tri_map_enable', False)):
        return accepted, [], {
            'trimap_bg_only': float(len(accepted)),
            'trimap_delayed_only': 0.0,
            'trimap_dual': 0.0,
            'trimap_delay_score_mean': 0.0,
            'trimap_front_occ_mean': 0.0,
            'trimap_hybrid_mean': 0.0,
            'trimap_support_gap_mean': 0.0,
            'trimap_gap_score_mean': 0.0,
            'trimap_bg_support_mean': 0.0,
            'trimap_centered_gap_mean': 0.0,
            'trimap_norm_gap_mean': 0.0,
            'trimap_gap_bias_mean': 0.0,
            'trimap_quantile_strong_thresh': 0.0,
            'trimap_quantile_soft_thresh': 0.0,
            'trimap_quantile_strong_budget': 0.0,
            'trimap_quantile_soft_budget': 0.0,
            'trimap_quantile_strong_eligible': 0.0,
            'trimap_quantile_soft_eligible': 0.0,
            'trimap_hold_blocked': 0.0,
            'trimap_hold_mean': 0.0,
            'trimap_hysteresis_mean': 0.0,
        }

    bg_map = self.background_map
    fg_map = self.foreground_map
    front_occ_soft_on = float(np.clip(getattr(self.cfg.update, 'tri_map_front_occ_soft_on', 0.34), 0.0, 1.0))
    w_front = float(max(0.0, getattr(self.cfg.update, 'tri_map_front_occ_weight', 0.60)))
    w_occ = float(max(0.0, getattr(self.cfg.update, 'tri_map_front_occ_occ_weight', 0.20)))
    w_pfv_front = float(max(0.0, getattr(self.cfg.update, 'tri_map_front_occ_pfv_weight', 0.20)))
    support_guard_strong = float(np.clip(getattr(self.cfg.update, 'tri_map_bg_support_guard_strong', 0.65), 0.0, 1.0))
    support_guard_soft = float(np.clip(getattr(self.cfg.update, 'tri_map_bg_support_guard_soft', 0.82), 0.0, 1.0))
    bg_mix_support = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_bg_support_weight', 0.60), 0.0, 2.0))
    bg_mix_rho = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_bg_rho_weight', 0.40), 0.0, 2.0))
    gap_pfv_gain = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_pfv_gain', 0.16), 0.0, 1.0))
    gap_assoc_gain = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_assoc_gain', 0.06), 0.0, 1.0))
    gap_bg_deficit_gain = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_bg_deficit_gain', 0.08), 0.0, 1.0))
    gap_front_bonus_gain = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_front_bonus_gain', 0.06), 0.0, 1.0))
    gap_front_floor = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_front_floor', 0.06), 0.0, 1.0))
    gap_pfv_floor = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_pfv_floor', 0.04), 0.0, 1.0))
    gap_assoc_floor = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_assoc_floor', 0.08), 0.0, 1.0))
    gap_front_soft_floor = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_front_soft_floor', 0.03), 0.0, 1.0))
    bg_anchor_ratio = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_center_bg_ratio', 0.38), 0.0, 1.0))
    norm_floor = float(max(1e-6, getattr(self.cfg.update, 'tri_map_support_gap_norm_floor', 0.28)))
    zero_bias = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_zero_bias', 0.02), -0.5, 0.5))
    centered_floor_strong = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_centered_floor_strong', 0.00), -1.0, 1.0))
    centered_floor_soft = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_centered_floor_soft', -0.03), -1.0, 1.0))
    strong_q = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_quantile_strong_q', 0.997), 0.0, 1.0))
    soft_q = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_quantile_soft_q', 0.985), 0.0, 1.0))
    strong_floor = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_quantile_floor_strong', 0.06), -1.0, 1.0))
    soft_floor = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_quantile_floor_soft', 0.02), -1.0, 1.0))
    sep_margin = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_quantile_sep_margin', 0.025), 0.0, 0.5))
    strong_ratio = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_strong_budget_ratio', 0.004), 0.0, 1.0))
    soft_ratio = float(np.clip(getattr(self.cfg.update, 'tri_map_support_gap_soft_budget_ratio', 0.015), 0.0, 1.0))
    strong_min_budget = int(max(0, getattr(self.cfg.update, 'tri_map_support_gap_strong_min_budget', 2)))
    soft_min_budget = int(max(0, getattr(self.cfg.update, 'tri_map_support_gap_soft_min_budget', 8)))
    strong_max_budget = int(max(strong_min_budget, getattr(self.cfg.update, 'tri_map_support_gap_strong_max_budget', 24)))
    soft_max_budget = int(max(soft_min_budget, getattr(self.cfg.update, 'tri_map_support_gap_soft_max_budget', 96)))
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))

    bg_only: List = []
    delayed_only: List = []
    front_scores: List[float] = []
    raw_gap_scores: List[float] = []
    bg_scores: List[float] = []
    centered_scores: List[float] = []
    norm_scores: List[float] = []
    bias_scores: List[float] = []
    route_scores: List[float] = []
    records = []
    counts = {'trimap_bg_only': 0.0, 'trimap_delayed_only': 0.0, 'trimap_dual': 0.0}

    for measurement in accepted:
        idx = bg_map.world_to_index(np.asarray(measurement.point_world, dtype=float).reshape(3))
        pfv_conf = 0.0
        bg_support = 0.0
        bg_rho = 0.0
        for nidx in bg_map.neighbor_indices(idx, 1):
            cell = bg_map.get_cell(nidx)
            if cell is None:
                continue
            bg_support = max(bg_support, self._map_static_support(bg_map, cell))
            if hasattr(bg_map, '_pfv_conf'):
                pfv_conf = max(pfv_conf, float(bg_map._pfv_conf(cell)))
            if hasattr(bg_map, '_pfv_cluster_conf'):
                pfv_conf = max(
                    pfv_conf,
                    float(getattr(self.cfg.update, 'pfv_cluster_weight', 0.35)) * float(bg_map._pfv_cluster_conf(nidx)),
                )
            bg_rho = max(
                bg_rho,
                float(
                    np.clip(
                        max(getattr(cell, 'rho_bg', 0.0), getattr(cell, 'rho_static', 0.0)) / rho_ref,
                        0.0,
                        1.0,
                    )
                ),
            )

        front_hist = float(bg_map._cross_map_fg_conf(fg_map, idx)) if hasattr(bg_map, '_cross_map_fg_conf') else 0.0
        front_occ = 0.0
        for nidx in fg_map.neighbor_indices(idx, 1):
            cell = fg_map.get_cell(nidx)
            if cell is None:
                continue
            rho_s = float(max(0.0, getattr(cell, 'rho_static', 0.0)))
            rho_t = float(max(0.0, getattr(cell, 'rho_transient', 0.0)))
            split = (
                float(np.clip(rho_t / max(1e-6, rho_s + rho_t), 0.0, 1.0))
                if (rho_s + rho_t) > 1e-8
                else float(np.clip(getattr(cell, 'dyn_prob', 0.0), 0.0, 1.0))
            )
            surf = float(max(1e-6, getattr(cell, 'surf_evidence', 0.0)))
            free = float(max(0.0, getattr(cell, 'free_evidence', 0.0)))
            occ = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
            pfv_local = float(fg_map._pfv_conf(cell)) if hasattr(fg_map, '_pfv_conf') else 0.0
            occ_score = float(
                np.clip(
                    (
                        w_front * max(front_hist, split, getattr(cell, 'wod_front_conf', 0.0), getattr(cell, 'wod_shell_conf', 0.0))
                        + w_occ * occ
                        + w_pfv_front * pfv_local
                    )
                    / max(1e-6, w_front + w_occ + w_pfv_front),
                    0.0,
                    1.0,
                )
            )
            front_occ = max(front_occ, occ_score)

        assoc_term = float(np.clip(getattr(measurement, 'assoc_risk', 0.0), 0.0, 1.0))
        bg_mix = float(
            np.clip(
                (bg_mix_support * bg_support + bg_mix_rho * bg_rho) / max(1e-6, bg_mix_support + bg_mix_rho),
                0.0,
                1.0,
            )
        )
        bg_deficit = float(np.clip(1.0 - bg_mix, 0.0, 1.0))
        raw_gap = float(front_occ - bg_mix)
        bg_anchor = float(np.clip(bg_anchor_ratio * bg_mix, 0.0, 1.0))
        centered_gap = float(front_occ - bg_anchor)
        norm_denom = float(max(norm_floor, front_occ + bg_anchor + 0.5 * bg_deficit))
        norm_gap = float(np.clip(centered_gap / norm_denom, -1.0, 1.0))
        gap_bias = float(
            zero_bias
            + gap_pfv_gain * max(0.0, pfv_conf - gap_pfv_floor)
            + gap_assoc_gain * max(0.0, assoc_term - gap_assoc_floor)
            + gap_bg_deficit_gain * bg_deficit
            + gap_front_bonus_gain * max(0.0, front_occ - gap_front_floor)
        )
        route_score = float(np.clip(norm_gap + gap_bias, -1.0, 1.0))

        front_scores.append(front_occ)
        bg_scores.append(bg_mix)
        raw_gap_scores.append(raw_gap)
        centered_scores.append(centered_gap)
        norm_scores.append(norm_gap)
        bias_scores.append(gap_bias)
        route_scores.append(route_score)

        strong_candidate = bool(
            centered_gap >= centered_floor_strong
            and front_occ >= gap_front_floor
            and (pfv_conf >= gap_pfv_floor or assoc_term >= gap_assoc_floor)
            and (bg_deficit >= 0.18 or bg_support <= max(support_guard_strong, 0.72))
        )
        soft_candidate = bool(
            centered_gap >= centered_floor_soft
            and front_occ >= max(gap_front_soft_floor, min(front_occ_soft_on, gap_front_floor))
            and (bg_deficit >= 0.10 or bg_support <= max(support_guard_soft, 0.88))
        )
        records.append({
            'measurement': measurement,
            'route_score': route_score,
            'front_occ': front_occ,
            'raw_gap': raw_gap,
            'centered_gap': centered_gap,
            'norm_gap': norm_gap,
            'gap_bias': gap_bias,
            'bg_support_mix': bg_mix,
            'strong_candidate': strong_candidate,
            'soft_candidate': soft_candidate,
        })

    total = len(records)
    strong_thresh = 0.0
    soft_thresh = 0.0
    strong_budget = 0
    soft_budget = 0
    soft_eligible = [i for i, rec in enumerate(records) if rec['soft_candidate']]
    strong_eligible = []

    selected_strong = set()
    selected_soft = set()

    if soft_eligible:
        soft_scores = np.asarray([records[i]['route_score'] for i in soft_eligible], dtype=float)
        soft_thresh = float(max(soft_floor, np.quantile(soft_scores, soft_q)))
        soft_budget = int(min(soft_max_budget, max(soft_min_budget, int(np.ceil(soft_ratio * max(1, total))))))
        for idx in sorted(soft_eligible, key=lambda i: records[i]['route_score'], reverse=True):
            if len(selected_soft) >= soft_budget:
                break
            if records[idx]['route_score'] >= soft_thresh:
                selected_soft.add(idx)

    strong_eligible = [i for i in selected_soft if records[i]['front_occ'] >= max(gap_front_soft_floor, 0.5 * gap_front_floor)]
    if strong_eligible:
        strong_scores = np.asarray([records[i]['route_score'] for i in strong_eligible], dtype=float)
        strong_target = float(max(strong_floor, np.quantile(strong_scores, strong_q), soft_thresh + sep_margin))
        strong_thresh = float(min(float(np.max(strong_scores)), strong_target))
        strong_budget = int(min(strong_max_budget, max(strong_min_budget, int(np.ceil(strong_ratio * max(1, len(selected_soft)))))))
        for idx in sorted(strong_eligible, key=lambda i: records[i]['route_score'], reverse=True):
            if len(selected_strong) >= strong_budget:
                break
            if records[idx]['route_score'] >= strong_thresh:
                selected_strong.add(idx)
        selected_soft.difference_update(selected_strong)

    hold_base = float(max(0.0, getattr(self.cfg.update, 'tri_map_escalation_hold_frames', 3.0)))
    hold_cap = float(max(hold_base, getattr(self.cfg.update, 'tri_map_escalation_hold_max_frames', 6.0)))
    for idx, rec in enumerate(records):
        measurement = rec['measurement']
        if idx in selected_strong:
            rel = max(0.0, rec['route_score'] - soft_thresh)
            hold_frames = float(min(hold_cap, hold_base + 6.0 * rel))
            measurement.tri_map_escalated = True
            measurement.tri_map_hold_frames = hold_frames
            measurement.tri_map_route_score = float(rec['route_score'])
            delayed_only.append(measurement)
            counts['trimap_delayed_only'] += 1.0
        elif idx in selected_soft:
            bg_only.append(measurement)
            delayed_only.append(measurement)
            counts['trimap_dual'] += 1.0
        else:
            bg_only.append(measurement)
            counts['trimap_bg_only'] += 1.0

    counts['trimap_delay_score_mean'] = float(np.mean(route_scores)) if route_scores else 0.0
    counts['trimap_front_occ_mean'] = float(np.mean(front_scores)) if front_scores else 0.0
    counts['trimap_hybrid_mean'] = float(np.mean(route_scores)) if route_scores else 0.0
    counts['trimap_support_gap_mean'] = float(np.mean(raw_gap_scores)) if raw_gap_scores else 0.0
    counts['trimap_gap_score_mean'] = float(np.mean(route_scores)) if route_scores else 0.0
    counts['trimap_bg_support_mean'] = float(np.mean(bg_scores)) if bg_scores else 0.0
    counts['trimap_centered_gap_mean'] = float(np.mean(centered_scores)) if centered_scores else 0.0
    counts['trimap_norm_gap_mean'] = float(np.mean(norm_scores)) if norm_scores else 0.0
    counts['trimap_gap_bias_mean'] = float(np.mean(bias_scores)) if bias_scores else 0.0
    counts['trimap_quantile_strong_thresh'] = float(strong_thresh)
    counts['trimap_quantile_soft_thresh'] = float(soft_thresh)
    counts['trimap_quantile_strong_budget'] = float(strong_budget)
    counts['trimap_quantile_soft_budget'] = float(soft_budget)
    counts['trimap_quantile_strong_eligible'] = float(len(strong_eligible))
    counts['trimap_quantile_soft_eligible'] = float(len(soft_eligible))
    return bg_only, delayed_only, counts


def promote_delayed_background_map(self) -> Dict[str, float]:
    if self.delayed_background_map is None or self.foreground_map is None or not bool(getattr(self.cfg.update, 'tri_map_enable', False)):
        return {'trimap_promoted': 0.0, 'trimap_promote_mean': 0.0, 'trimap_rescue_mean': 0.0}

    delayed_map = self.delayed_background_map
    committed_map = self.background_map
    fg_map = self.foreground_map
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    promote_on = float(np.clip(getattr(self.cfg.update, 'tri_map_promote_on', 0.72), 0.0, 1.0))
    fg_guard = float(np.clip(getattr(self.cfg.update, 'tri_map_promote_fg_guard', 0.28), 0.0, 1.0))
    blend = float(np.clip(getattr(self.cfg.update, 'tri_map_promote_blend', 0.40), 0.0, 1.0))
    rescue_enable = bool(getattr(self.cfg.update, 'tri_map_promotion_rescue_enable', False))
    hole_weight = float(np.clip(getattr(self.cfg.update, 'tri_map_promotion_hole_weight', 0.35), 0.0, 1.0))
    promote_min_on = float(np.clip(getattr(self.cfg.update, 'tri_map_promotion_min_on', 0.55), 0.0, 1.0))

    promoted = 0.0
    scores: List[float] = []
    rescue_scores: List[float] = []
    hold_scores: List[float] = []
    hysteresis_scores: List[float] = []
    hold_blocked = 0.0
    hysteresis_decay = float(np.clip(getattr(self.cfg.update, 'tri_map_escalation_hysteresis_decay', 0.85), 0.0, 1.0))
    promote_bonus = float(np.clip(getattr(self.cfg.update, 'tri_map_escalation_promote_bonus', 0.12), 0.0, 0.5))
    blend_suppress = float(np.clip(getattr(self.cfg.update, 'tri_map_escalation_blend_suppress', 0.35), 0.0, 0.95))
    for idx, cell in delayed_map.iter_cells():
        w_bg = float(max(0.0, getattr(cell, 'phi_bg_w', 0.0)))
        w_static = float(max(0.0, getattr(cell, 'phi_static_w', 0.0)))
        if max(w_bg, w_static) <= 1e-8:
            continue
        hold_frames = float(max(0.0, getattr(cell, 'trimap_hold_frames', 0.0)))
        hysteresis = float(max(0.0, getattr(cell, 'trimap_hysteresis', 0.0)))
        hold_scores.append(hold_frames)
        hysteresis_scores.append(hysteresis)
        surf = float(max(1e-6, getattr(cell, 'surf_evidence', 0.0)))
        free = float(max(0.0, getattr(cell, 'free_evidence', 0.0)))
        occ = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
        support = float(np.clip(max(
            occ,
            float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0)),
            float(np.clip(getattr(cell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
            float(np.clip(getattr(cell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
        ), 0.0, 1.0))
        fg_conf = float(committed_map._cross_map_fg_conf(fg_map, idx)) if hasattr(committed_map, '_cross_map_fg_conf') else 0.0
        local_commit = 0.0
        for nidx in committed_map.neighbor_indices(idx, 1):
            cc = committed_map.get_cell(nidx)
            if cc is None:
                continue
            local_commit = max(local_commit, self._map_static_support(committed_map, cc)) if hasattr(self, '_map_static_support') else max(local_commit, 0.0)
        hole_score = float(np.clip(1.0 - local_commit, 0.0, 1.0))
        rescue_scores.append(hole_score)
        scores.append(support)
        promote_thresh = promote_on
        blend_eff = blend
        if rescue_enable:
            promote_thresh = float(max(promote_min_on, promote_on - hole_weight * hole_score))
            blend_eff = float(np.clip(blend * (1.0 + 0.8 * hole_score), 0.0, 0.80))
        if hold_frames > 1e-6:
            cell.trimap_hold_frames = float(max(0.0, hold_frames - 1.0))
            cell.trimap_hysteresis = float(max(hysteresis, 1.0))
            hold_blocked += 1.0
            continue
        if hysteresis > 1e-6:
            promote_thresh = float(np.clip(promote_thresh + promote_bonus * hysteresis, 0.0, 0.98))
            blend_eff = float(np.clip(blend_eff * (1.0 - blend_suppress * hysteresis), 0.0, 0.80))
            cell.trimap_hysteresis = float(hysteresis_decay * hysteresis)
        if support < promote_thresh or fg_conf > fg_guard:
            continue
        dst = committed_map.get_or_create(idx)
        if w_bg > 1e-12:
            w_commit = float(blend_eff * w_bg)
            new_w = float(max(0.0, getattr(dst, 'phi_bg_w', 0.0)) + w_commit)
            if new_w > 1e-12:
                dst.phi_bg = float((float(getattr(dst, 'phi_bg_w', 0.0)) * float(getattr(dst, 'phi_bg', 0.0)) + w_commit * float(getattr(cell, 'phi_bg', 0.0))) / new_w)
                dst.phi_bg_w = float(min(5000.0, new_w))
                dst.rho_bg = float(float(getattr(dst, 'rho_bg', 0.0)) + blend_eff * float(getattr(cell, 'rho_bg', 0.0)))
            cell.phi_bg_w = float((1.0 - blend_eff) * w_bg)
            cell.rho_bg = float((1.0 - blend_eff) * float(getattr(cell, 'rho_bg', 0.0)))
        if w_static > 1e-12:
            w_commit = float(blend_eff * w_static)
            new_w = float(max(0.0, getattr(dst, 'phi_static_w', 0.0)) + w_commit)
            if new_w > 1e-12:
                dst.phi_static = float((float(getattr(dst, 'phi_static_w', 0.0)) * float(getattr(dst, 'phi_static', 0.0)) + w_commit * float(getattr(cell, 'phi_static', 0.0))) / new_w)
                dst.phi_static_w = float(min(5000.0, new_w))
                dst.rho_static = float(float(getattr(dst, 'rho_static', 0.0)) + blend_eff * float(getattr(cell, 'rho_static', 0.0)))
            cell.phi_static_w = float((1.0 - blend_eff) * w_static)
            cell.rho_static = float((1.0 - blend_eff) * float(getattr(cell, 'rho_static', 0.0)))
        committed_map._sync_legacy_channels(dst)
        delayed_map._sync_legacy_channels(cell)
        cell.trimap_hysteresis = float(hysteresis_decay * float(getattr(cell, 'trimap_hysteresis', 0.0)))
        cell.trimap_escalated = float(max(0.0, float(getattr(cell, 'trimap_escalated', 0.0)) * 0.5))
        promoted += 1.0
    return {
        'trimap_promoted': promoted,
        'trimap_promote_mean': float(np.mean(scores)) if scores else 0.0,
        'trimap_rescue_mean': float(np.mean(rescue_scores)) if rescue_scores else 0.0,
        'trimap_hold_blocked': hold_blocked,
        'trimap_hold_mean': float(np.mean(hold_scores)) if hold_scores else 0.0,
        'trimap_hysteresis_mean': float(np.mean(hysteresis_scores)) if hysteresis_scores else 0.0,
    }





def extract_delayed_surface_bank(
    self,
    extract_kwargs: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    delayed_map = self.delayed_background_map
    if delayed_map is None or not bool(getattr(self.cfg.update, 'tri_map_delayed_bank_enable', False)):
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float), {
            'trimap_delayed_bank_points': 0.0,
            'trimap_delayed_bank_conf_mean': 0.0,
            'trimap_delayed_bank_residency_mean': 0.0,
        }

    phi_thresh = float(extract_kwargs.get('phi_thresh', 0.8))
    min_weight = float(extract_kwargs.get('min_weight', 0.0))
    current_step = int(extract_kwargs.get('current_step', 0))
    max_age_frames = int(extract_kwargs.get('max_age_frames', 1_000_000_000))
    use_zero_crossing = bool(extract_kwargs.get('use_zero_crossing', True))
    zero_crossing_max_offset = float(extract_kwargs.get('zero_crossing_max_offset', 0.06))
    zero_crossing_phi_gate = float(extract_kwargs.get('zero_crossing_phi_gate', 0.05))

    radius = int(max(1, getattr(self.cfg.update, 'tri_map_delayed_bank_radius_cells', 1)))
    support_on = float(np.clip(getattr(self.cfg.update, 'tri_map_delayed_bank_support_on', 0.40), 0.0, 1.0))
    route_on = float(np.clip(getattr(self.cfg.update, 'tri_map_delayed_bank_route_on', 0.01), -1.0, 1.0))
    residency_on = float(np.clip(getattr(self.cfg.update, 'tri_map_delayed_bank_residency_on', 0.10), 0.0, 1.0))
    phi_scale = float(max(0.1, getattr(self.cfg.update, 'tri_map_delayed_bank_phi_thresh_scale', 1.15)))
    min_weight_scale = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_bank_min_weight_scale', 0.60)))
    w_phi_static = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_bank_phi_static_weight', 0.50)))
    w_phi_bg = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_bank_phi_bg_weight', 0.35)))
    w_phi_geo = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_bank_phi_geo_weight', 0.15)))
    w_support = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_bank_support_weight', 0.45)))
    w_route = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_bank_route_weight', 0.25)))
    w_res = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_bank_residency_weight', 0.30)))
    normal_blend = float(np.clip(getattr(self.cfg.update, 'tri_map_delayed_bank_normal_blend', 0.80), 0.0, 1.0))
    max_offset = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_bank_max_offset_vox', 1.50))) * float(delayed_map.voxel_size)
    neighbor_min = int(max(1, getattr(self.cfg.update, 'tri_map_delayed_bank_neighbor_min', 2)))
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    hold_ref = float(max(1.0, getattr(self.cfg.update, 'tri_map_escalation_hold_max_frames', 6.0)))

    points = []
    normals = []
    confs = []
    residencies = []

    for idx, cell in delayed_map.iter_cells():
        if (current_step - int(getattr(cell, 'last_seen', current_step))) > max_age_frames:
            continue
        bank_w = float(max(0.0, getattr(cell, 'phi_delayed_bank_w', 0.0)))
        bank_conf = float(np.clip(getattr(cell, 'delayed_bank_conf', 0.0), 0.0, 1.0))
        bank_active = float(np.clip(getattr(cell, 'delayed_bank_active', 0.0), 0.0, 1.0))
        bank_normal_w = float(max(0.0, getattr(cell, 'g_delayed_bank_w', 0.0)))
        ws = float(max(0.0, getattr(cell, 'phi_static_w', 0.0)))
        wb = float(max(0.0, getattr(cell, 'phi_bg_w', 0.0)))
        wg = float(max(0.0, getattr(cell, 'phi_geo_w', 0.0)))
        use_bank = bool(bank_w >= max(min_weight_scale * min_weight, float(getattr(self.cfg.update, 'tri_map_delay_bank_min_weight', 0.20))) and bank_conf >= float(getattr(self.cfg.update, 'tri_map_delay_bank_conf_on', 0.35)) and bank_normal_w > 1e-8)
        weight_eff = float(bank_w if use_bank else (w_phi_static * ws + w_phi_bg * wb + w_phi_geo * wg))
        if weight_eff < min_weight_scale * min_weight:
            continue
        support = float(np.clip(max(
            self._map_static_support(delayed_map, cell),
            float(np.clip(getattr(cell, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
            float(np.clip(getattr(cell, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
        ), 0.0, 1.0))
        route_score = float(getattr(cell, 'trimap_route_score', 0.0))
        residency = float(np.clip(max(
            float(getattr(cell, 'trimap_hold_frames', 0.0)) / hold_ref,
            float(getattr(cell, 'trimap_hysteresis', 0.0)),
            float(getattr(cell, 'trimap_escalated', 0.0)),
        ), 0.0, 1.0))
        if support < support_on or route_score < route_on or residency < residency_on:
            continue
        if use_bank:
            residency = max(residency, bank_active)
        bank_g = np.asarray(getattr(cell, 'g_delayed_bank', np.zeros(3, dtype=float)), dtype=float).reshape(3)
        bank_g_w = float(max(0.0, getattr(cell, 'g_delayed_bank_w', 0.0)))
        g = bank_g if bank_g_w > 1e-8 else np.asarray(getattr(cell, 'g_mean', np.zeros(3, dtype=float)), dtype=float).reshape(3)
        gn = float(np.linalg.norm(g))
        if gn <= 1e-8:
            continue
        n0 = g / gn

        phi_vals = []
        normals_local = []
        weights = []
        for nidx in delayed_map.neighbor_indices(idx, radius):
            c = delayed_map.get_cell(nidx)
            if c is None:
                continue
            bank_g_n = np.asarray(getattr(c, 'g_delayed_bank', np.zeros(3, dtype=float)), dtype=float).reshape(3)
            bank_g_w_n = float(max(0.0, getattr(c, 'g_delayed_bank_w', 0.0)))
            g_n = bank_g_n if bank_g_w_n > 1e-8 else np.asarray(getattr(c, 'g_mean', np.zeros(3, dtype=float)), dtype=float).reshape(3)
            gnn = float(np.linalg.norm(g_n))
            if gnn <= 1e-8:
                continue
            g_n = g_n / gnn
            bank_w_n = float(max(0.0, getattr(c, 'phi_delayed_bank_w', 0.0)))
            bank_conf_n = float(np.clip(getattr(c, 'delayed_bank_conf', 0.0), 0.0, 1.0))
            bank_normal_w_n = float(max(0.0, getattr(c, 'g_delayed_bank_w', 0.0)))
            ws_n = float(max(0.0, getattr(c, 'phi_static_w', 0.0)))
            wb_n = float(max(0.0, getattr(c, 'phi_bg_w', 0.0)))
            wg_n = float(max(0.0, getattr(c, 'phi_geo_w', 0.0)))
            phi_num = 0.0
            phi_den = 0.0
            if bank_w_n >= max(min_weight_scale * min_weight, float(getattr(self.cfg.update, 'tri_map_delay_bank_min_weight', 0.20))) and bank_conf_n >= float(getattr(self.cfg.update, 'tri_map_delay_bank_conf_on', 0.35)) and bank_normal_w_n > 1e-8:
                phi_num += bank_w_n * float(getattr(c, 'phi_delayed_bank', 0.0))
                phi_den += bank_w_n
            else:
                if ws_n > 1e-8:
                    phi_num += w_phi_static * ws_n * float(getattr(c, 'phi_static', 0.0))
                    phi_den += w_phi_static * ws_n
                if wb_n > 1e-8:
                    phi_num += w_phi_bg * wb_n * float(getattr(c, 'phi_bg', 0.0))
                    phi_den += w_phi_bg * wb_n
                if wg_n > 1e-8:
                    phi_num += w_phi_geo * wg_n * float(getattr(c, 'phi_geo', 0.0))
                    phi_den += w_phi_geo * wg_n
            if phi_den <= 1e-8:
                continue
            support_n = float(np.clip(max(
                self._map_static_support(delayed_map, c),
                float(np.clip(getattr(c, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
                float(np.clip(getattr(c, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
            ), 0.0, 1.0))
            route_n = float(np.clip(float(getattr(c, 'trimap_route_score', 0.0)), -1.0, 1.0))
            residency_n = float(np.clip(max(float(getattr(c, 'trimap_hold_frames', 0.0)) / hold_ref, float(getattr(c, 'trimap_hysteresis', 0.0)), float(getattr(c, 'trimap_escalated', 0.0))), 0.0, 1.0))
            local_conf = float((w_support * support_n + w_route * max(0.0, route_n) + w_res * residency_n) / max(1e-6, w_support + w_route + w_res))
            grid_dist = float(np.linalg.norm(np.asarray(nidx, dtype=float) - np.asarray(idx, dtype=float)))
            w_local = float(local_conf / (1.0 + 0.5 * grid_dist * grid_dist))
            weights.append(w_local)
            phi_vals.append(float(phi_num / phi_den))
            normals_local.append(g_n)
        if len(weights) < neighbor_min:
            continue
        w_arr = np.asarray(weights, dtype=float)
        w_sum = float(np.sum(w_arr))
        if w_sum <= 1e-8:
            continue
        phi_eff = float(np.dot(w_arr, np.asarray(phi_vals, dtype=float)) / w_sum)
        if abs(phi_eff) > max(1e-4, phi_scale * phi_thresh):
            continue
        n_ref = np.sum(np.asarray(normals_local, dtype=float) * w_arr.reshape(-1, 1), axis=0)
        n_ref_norm = float(np.linalg.norm(n_ref))
        if n_ref_norm <= 1e-8:
            continue
        n_ref = n_ref / n_ref_norm
        if float(np.dot(n_ref, n0)) < 0.0:
            n_ref = -n_ref
        n_eff = (1.0 - normal_blend) * n0 + normal_blend * n_ref
        n_eff_norm = float(np.linalg.norm(n_eff))
        if n_eff_norm <= 1e-8:
            continue
        n_eff = n_eff / n_eff_norm
        center = delayed_map.index_to_center(idx)
        p = center
        if use_zero_crossing and abs(phi_eff) <= max(1e-4, zero_crossing_phi_gate):
            off = -float(phi_eff) * n_eff
            off_norm = float(np.linalg.norm(off))
            if max_offset > 0.0 and off_norm > max_offset:
                off = off * (max_offset / max(off_norm, 1e-9))
            p = center + off
        points.append(p)
        normals.append(n_eff)
        confs.append(float(np.mean(w_arr)))
        residencies.append(residency)

    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float), {
            'trimap_delayed_bank_points': 0.0,
            'trimap_delayed_bank_conf_mean': 0.0,
            'trimap_delayed_bank_residency_mean': 0.0,
        }
    return np.asarray(points, dtype=float), np.asarray(normals, dtype=float), {
        'trimap_delayed_bank_points': float(len(points)),
        'trimap_delayed_bank_conf_mean': float(np.mean(confs)) if confs else 0.0,
        'trimap_delayed_bank_residency_mean': float(np.mean(residencies)) if residencies else 0.0,
    }


def refine_delayed_export_geometry(
    self,
    point: np.ndarray,
    normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    delayed_map = self.delayed_background_map
    if delayed_map is None or not bool(getattr(self.cfg.update, 'tri_map_delayed_refine_enable', False)):
        n = np.asarray(normal, dtype=float).reshape(3)
        nn = float(np.linalg.norm(n))
        if nn > 1e-8:
            n = n / nn
        return np.asarray(point, dtype=float).reshape(3), n, {'refine_offset': 0.0, 'refine_normal_cos': 1.0, 'refine_conf': 0.0}

    idx = delayed_map.world_to_index(point)
    cell = delayed_map.get_cell(idx)
    if cell is None:
        n = np.asarray(normal, dtype=float).reshape(3)
        nn = float(np.linalg.norm(n))
        if nn > 1e-8:
            n = n / nn
        return np.asarray(point, dtype=float).reshape(3), n, {'refine_offset': 0.0, 'refine_normal_cos': 1.0, 'refine_conf': 0.0}

    radius = int(max(1, getattr(self.cfg.update, 'tri_map_delayed_refine_radius_cells', 1)))
    blend = float(np.clip(getattr(self.cfg.update, 'tri_map_delayed_refine_blend', 0.65), 0.0, 1.0))
    normal_blend = float(np.clip(getattr(self.cfg.update, 'tri_map_delayed_refine_normal_blend', 0.75), 0.0, 1.0))
    max_offset = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_refine_max_offset_vox', 1.25))) * float(delayed_map.voxel_size)
    w_support = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_refine_support_weight', 0.50)))
    w_route = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_refine_route_weight', 0.25)))
    w_res = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_refine_residency_weight', 0.25)))
    w_phi_static = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_refine_phi_static_weight', 0.45)))
    w_phi_bg = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_refine_phi_bg_weight', 0.35)))
    w_phi_geo = float(max(0.0, getattr(self.cfg.update, 'tri_map_delayed_refine_phi_geo_weight', 0.20)))
    neighbor_min = int(max(1, getattr(self.cfg.update, 'tri_map_delayed_refine_neighbor_min', 2)))
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))
    hold_ref = float(max(1.0, getattr(self.cfg.update, 'tri_map_escalation_hold_max_frames', 6.0)))

    raw_point = np.asarray(point, dtype=float).reshape(3)
    raw_normal = np.asarray(normal, dtype=float).reshape(3)
    raw_normal_norm = float(np.linalg.norm(raw_normal))
    if raw_normal_norm > 1e-8:
        raw_normal = raw_normal / raw_normal_norm

    phi_vals = []
    normal_vals = []
    weights = []
    for nidx in delayed_map.neighbor_indices(idx, radius):
        c = delayed_map.get_cell(nidx)
        if c is None:
            continue
        support = float(np.clip(max(
            self._map_static_support(delayed_map, c),
            float(np.clip(getattr(c, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
            float(np.clip(getattr(c, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
        ), 0.0, 1.0))
        route_n = float(np.clip(float(getattr(c, 'trimap_route_score', 0.0)) / max(1e-6, getattr(self.cfg.update, 'tri_map_residency_compete_route_ref', 0.08)), 0.0, 1.0))
        residency_n = float(np.clip(max(float(getattr(c, 'trimap_hold_frames', 0.0)) / hold_ref, float(getattr(c, 'trimap_hysteresis', 0.0)), float(getattr(c, 'trimap_escalated', 0.0))), 0.0, 1.0))
        conf = float((w_support * support + w_route * route_n + w_res * residency_n) / max(1e-6, w_support + w_route + w_res))
        g = np.asarray(getattr(c, 'g_mean', np.zeros(3, dtype=float)), dtype=float).reshape(3)
        gn = float(np.linalg.norm(g))
        if gn <= 1e-8:
            continue
        g = g / gn
        phi_num = 0.0
        phi_den = 0.0
        ws = float(max(0.0, getattr(c, 'phi_static_w', 0.0)))
        wb = float(max(0.0, getattr(c, 'phi_bg_w', 0.0)))
        wg = float(max(0.0, getattr(c, 'phi_geo_w', 0.0)))
        if ws > 1e-8:
            phi_num += w_phi_static * ws * float(getattr(c, 'phi_static', 0.0))
            phi_den += w_phi_static * ws
        if wb > 1e-8:
            phi_num += w_phi_bg * wb * float(getattr(c, 'phi_bg', 0.0))
            phi_den += w_phi_bg * wb
        if wg > 1e-8:
            phi_num += w_phi_geo * wg * float(getattr(c, 'phi_geo', 0.0))
            phi_den += w_phi_geo * wg
        if phi_den <= 1e-8:
            continue
        grid_dist = float(np.linalg.norm(np.asarray(nidx, dtype=float) - np.asarray(idx, dtype=float)))
        w_local = float(conf / (1.0 + 0.5 * grid_dist * grid_dist))
        weights.append(w_local)
        normal_vals.append(g)
        phi_vals.append(float(phi_num / phi_den))

    if len(weights) < neighbor_min:
        return raw_point, raw_normal, {'refine_offset': 0.0, 'refine_normal_cos': 1.0, 'refine_conf': 0.0}

    w_arr = np.asarray(weights, dtype=float)
    w_sum = float(np.sum(w_arr))
    if w_sum <= 1e-8:
        return raw_point, raw_normal, {'refine_offset': 0.0, 'refine_normal_cos': 1.0, 'refine_conf': 0.0}

    phi_ref = float(np.dot(w_arr, np.asarray(phi_vals, dtype=float)) / w_sum)
    n_ref = np.sum(np.asarray(normal_vals, dtype=float) * w_arr.reshape(-1, 1), axis=0)
    n_ref_norm = float(np.linalg.norm(n_ref))
    if n_ref_norm <= 1e-8:
        return raw_point, raw_normal, {'refine_offset': 0.0, 'refine_normal_cos': 1.0, 'refine_conf': 0.0}
    n_ref = n_ref / n_ref_norm
    if raw_normal_norm > 1e-8 and float(np.dot(n_ref, raw_normal)) < 0.0:
        n_ref = -n_ref

    center = delayed_map.index_to_center(idx)
    point_ref = center - phi_ref * n_ref
    off = point_ref - raw_point
    off_norm = float(np.linalg.norm(off))
    if max_offset > 0.0 and off_norm > max_offset:
        off = off * (max_offset / max(off_norm, 1e-9))
        point_ref = raw_point + off

    refine_conf = float(np.clip(np.mean(w_arr), 0.0, 1.0))
    blend_eff = float(np.clip(blend * (0.35 + 0.65 * refine_conf), 0.0, 1.0))
    n_blend_eff = float(np.clip(normal_blend * (0.35 + 0.65 * refine_conf), 0.0, 1.0))
    refined_point = (1.0 - blend_eff) * raw_point + blend_eff * point_ref
    refined_normal = (1.0 - n_blend_eff) * raw_normal + n_blend_eff * n_ref if raw_normal_norm > 1e-8 else n_ref
    rn = float(np.linalg.norm(refined_normal))
    if rn > 1e-8:
        refined_normal = refined_normal / rn
    normal_cos = float(np.clip(abs(np.dot(refined_normal, raw_normal)), 0.0, 1.0)) if raw_normal_norm > 1e-8 else 1.0
    refine_offset = float(np.linalg.norm(refined_point - raw_point))
    return refined_point, refined_normal, {
        'refine_offset': refine_offset,
        'refine_normal_cos': normal_cos,
        'refine_conf': refine_conf,
    }


def residency_gated_delayed_export(
    self,
    extract_kwargs: Dict[str, object],
    committed_points: np.ndarray,
    committed_normals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if self.delayed_background_map is None or self.foreground_map is None or not bool(getattr(self.cfg.update, 'tri_map_residency_export_enable', False)):
        return committed_points, committed_normals, {
            'trimap_export_participation': 0.0,
            'trimap_export_candidates': 0.0,
            'trimap_export_added': 0.0,
            'trimap_export_replaced': 0.0,
            'trimap_export_residency': 0.0,
            'trimap_export_route_mean': 0.0,
            'trimap_export_compete_mean': 0.0,
            'trimap_export_normal_cos_mean': 0.0,
            'trimap_delayed_refine_offset_mean': 0.0,
            'trimap_delayed_refine_normal_cos_mean': 1.0,
            'trimap_delayed_bank_points': float(bank_stats.get('trimap_delayed_bank_points', 0.0)),
            'trimap_delayed_bank_conf_mean': float(bank_stats.get('trimap_delayed_bank_conf_mean', 0.0)),
            'trimap_delayed_bank_residency_mean': float(bank_stats.get('trimap_delayed_bank_residency_mean', 0.0)),
        }

    delayed_points, delayed_normals, bank_stats = extract_delayed_surface_bank(self, extract_kwargs)
    if delayed_points.shape[0] == 0:
        return committed_points, committed_normals, {
            'trimap_export_participation': 1.0,
            'trimap_export_candidates': 0.0,
            'trimap_export_added': 0.0,
            'trimap_export_replaced': 0.0,
            'trimap_export_residency': 0.0,
            'trimap_export_route_mean': 0.0,
            'trimap_export_compete_mean': 0.0,
            'trimap_export_normal_cos_mean': 0.0,
            'trimap_delayed_refine_offset_mean': 0.0,
            'trimap_delayed_refine_normal_cos_mean': 1.0,
            'trimap_delayed_bank_points': float(bank_stats.get('trimap_delayed_bank_points', 0.0)),
            'trimap_delayed_bank_conf_mean': float(bank_stats.get('trimap_delayed_bank_conf_mean', 0.0)),
            'trimap_delayed_bank_residency_mean': float(bank_stats.get('trimap_delayed_bank_residency_mean', 0.0)),
        }

    hold_min = float(max(0.0, getattr(self.cfg.update, 'tri_map_residency_export_hold_min', 0.5)))
    hysteresis_min = float(max(0.0, getattr(self.cfg.update, 'tri_map_residency_export_hysteresis_min', 0.05)))
    support_on = float(np.clip(getattr(self.cfg.update, 'tri_map_residency_export_support_on', 0.45), 0.0, 1.0))
    fg_guard = float(np.clip(getattr(self.cfg.update, 'tri_map_residency_export_fg_guard', 0.65), 0.0, 1.0))
    commit_max = float(np.clip(getattr(self.cfg.update, 'tri_map_residency_export_commit_max', 0.92), 0.0, 1.0))
    route_score_min = float(np.clip(getattr(self.cfg.update, 'tri_map_residency_export_route_score_min', 0.02), -1.0, 1.0))
    radius = int(max(1, getattr(self.cfg.update, 'tri_map_residency_export_radius_cells', 1)))
    replace_enable = bool(getattr(self.cfg.update, 'tri_map_residency_replace_enable', True))
    replace_radius = int(max(1, getattr(self.cfg.update, 'tri_map_residency_replace_radius_cells', 1)))
    replace_dist = float(max(0.25, getattr(self.cfg.update, 'tri_map_residency_replace_distance_vox', 1.2))) * float(self.background_map.voxel_size)
    replace_commit_max = float(np.clip(getattr(self.cfg.update, 'tri_map_residency_replace_commit_max', 0.88), 0.0, 1.0))
    replace_support_margin = float(np.clip(getattr(self.cfg.update, 'tri_map_residency_replace_support_margin', -0.02), -1.0, 1.0))
    replace_max_per_delayed = int(max(1, getattr(self.cfg.update, 'tri_map_residency_replace_max_per_delayed', 2)))
    compete_route_ref = float(max(1e-6, getattr(self.cfg.update, 'tri_map_residency_compete_route_ref', 0.08)))
    compete_normal_cos_min = float(np.clip(getattr(self.cfg.update, 'tri_map_residency_compete_normal_cos_min', 0.25), 0.0, 1.0))
    compete_margin = float(np.clip(getattr(self.cfg.update, 'tri_map_residency_compete_margin', 0.03), -1.0, 1.0))
    w_delayed_support = float(max(0.0, getattr(self.cfg.update, 'tri_map_residency_compete_delayed_support_weight', 0.55)))
    w_route = float(max(0.0, getattr(self.cfg.update, 'tri_map_residency_compete_route_weight', 0.25)))
    w_residency = float(max(0.0, getattr(self.cfg.update, 'tri_map_residency_compete_residency_weight', 0.10)))
    w_normal = float(max(0.0, getattr(self.cfg.update, 'tri_map_residency_compete_normal_weight', 0.15)))
    w_commit = float(max(0.0, getattr(self.cfg.update, 'tri_map_residency_compete_commit_support_weight', 0.60)))
    w_distance = float(max(0.0, getattr(self.cfg.update, 'tri_map_residency_compete_distance_weight', 0.20)))
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))

    committed_points = np.asarray(committed_points, dtype=float)
    committed_normals = np.asarray(committed_normals, dtype=float)
    committed_idx = {self.background_map.world_to_index(point) for point in committed_points}
    committed_index_map: Dict[Tuple[int, int, int], List[int]] = {}
    for point_idx, point in enumerate(committed_points):
        idx = self.background_map.world_to_index(point)
        committed_index_map.setdefault(idx, []).append(point_idx)

    added_pts = []
    added_nrm = []
    route_scores = []
    compete_scores = []
    normal_scores = []
    refine_offsets = []
    refine_normal_cos = []
    remove_indices = set()
    seen = set()
    candidates = 0.0
    residency_hits = 0.0

    for point, normal in zip(delayed_points, delayed_normals):
        idx = self.delayed_background_map.world_to_index(point)
        if idx in seen:
            continue
        seen.add(idx)
        dc = self.delayed_background_map.get_cell(idx)
        if dc is None:
            continue
        candidates += 1.0
        hold_frames = float(max(0.0, getattr(dc, 'trimap_hold_frames', 0.0)))
        hysteresis = float(max(0.0, getattr(dc, 'trimap_hysteresis', 0.0)))
        residency_active = bool(hold_frames >= hold_min or hysteresis >= hysteresis_min or float(getattr(dc, 'trimap_escalated', 0.0)) > 0.0)
        if not residency_active:
            continue
        residency_hits += 1.0
        delayed_support = float(np.clip(max(
            self._map_static_support(self.delayed_background_map, dc),
            float(np.clip(getattr(dc, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
            float(np.clip(getattr(dc, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
        ), 0.0, 1.0))
        route_score = float(getattr(dc, 'trimap_route_score', 0.0))
        fg_conf = float(self.background_map._cross_map_fg_conf(self.foreground_map, idx)) if hasattr(self.background_map, '_cross_map_fg_conf') else 0.0
        commit_support = 0.0
        for nidx in self.background_map.neighbor_indices(idx, radius):
            cc = self.background_map.get_cell(nidx)
            if cc is None:
                continue
            commit_support = max(commit_support, self._map_static_support(self.background_map, cc))
        if delayed_support < support_on or fg_conf > fg_guard or route_score < route_score_min:
            continue
        if idx in committed_idx and commit_support > commit_max:
            continue

        point_refined, normal_refined, refine_stats = refine_delayed_export_geometry(self, point, normal)
        refine_offsets.append(float(refine_stats.get('refine_offset', 0.0)))
        refine_normal_cos.append(float(refine_stats.get('refine_normal_cos', 1.0)))

        if replace_enable and committed_points.shape[0] > 0:
            local_candidates = []
            delayed_normal = np.asarray(normal_refined, dtype=float).reshape(3)
            delayed_normal_norm = float(np.linalg.norm(delayed_normal))
            if delayed_normal_norm > 1e-8:
                delayed_normal = delayed_normal / delayed_normal_norm
            route_n = float(np.clip(route_score / compete_route_ref, 0.0, 1.0))
            hold_ref = float(max(1.0, getattr(self.cfg.update, 'tri_map_escalation_hold_max_frames', 6.0)))
            residency_n = float(np.clip(max(hold_frames / hold_ref, hysteresis, float(getattr(dc, 'trimap_escalated', 0.0))), 0.0, 1.0))
            delayed_score_base = float(w_delayed_support * delayed_support + w_route * route_n + w_residency * residency_n)
            for nidx in self.background_map.neighbor_indices(idx, replace_radius):
                if nidx not in committed_index_map:
                    continue
                cc = self.background_map.get_cell(nidx)
                commit_local = self._map_static_support(self.background_map, cc) if cc is not None else 0.0
                if commit_local > replace_commit_max:
                    continue
                for point_idx in committed_index_map[nidx]:
                    if point_idx in remove_indices:
                        continue
                    dist = float(np.linalg.norm(committed_points[point_idx] - point_refined))
                    if dist > replace_dist:
                        continue
                    normal_cos = 1.0
                    if committed_normals.shape[0] == committed_points.shape[0]:
                        cn = np.asarray(committed_normals[point_idx], dtype=float).reshape(3)
                        cn_norm = float(np.linalg.norm(cn))
                        if cn_norm > 1e-8 and delayed_normal_norm > 1e-8:
                            cn = cn / cn_norm
                            normal_cos = float(np.clip(abs(np.dot(cn, delayed_normal)), 0.0, 1.0))
                    if normal_cos < compete_normal_cos_min:
                        continue
                    dist_n = float(np.clip(dist / max(1e-6, replace_dist), 0.0, 1.0))
                    delayed_pair = float(delayed_score_base + w_normal * normal_cos)
                    committed_pair = float(w_commit * commit_local + w_distance * dist_n)
                    compete = float(delayed_pair - committed_pair)
                    if compete <= compete_margin:
                        continue
                    local_candidates.append((compete, -dist_n, point_idx, normal_cos))
            local_candidates.sort(reverse=True)
            for compete, _, point_idx, normal_cos in local_candidates[:replace_max_per_delayed]:
                remove_indices.add(point_idx)
                compete_scores.append(float(compete))
                normal_scores.append(float(normal_cos))

        added_pts.append(point_refined)
        added_nrm.append(normal_refined)
        route_scores.append(route_score)

    if not added_pts and not remove_indices:
        return committed_points, committed_normals, {
            'trimap_export_participation': 1.0,
            'trimap_export_candidates': candidates,
            'trimap_export_added': 0.0,
            'trimap_export_replaced': 0.0,
            'trimap_export_residency': residency_hits,
            'trimap_export_route_mean': float(np.mean(route_scores)) if route_scores else 0.0,
            'trimap_export_compete_mean': float(np.mean(compete_scores)) if compete_scores else 0.0,
            'trimap_export_normal_cos_mean': float(np.mean(normal_scores)) if normal_scores else 0.0,
            'trimap_delayed_refine_offset_mean': float(np.mean(refine_offsets)) if refine_offsets else 0.0,
            'trimap_delayed_refine_normal_cos_mean': float(np.mean(refine_normal_cos)) if refine_normal_cos else 1.0,
        }

    keep_mask = np.ones((committed_points.shape[0],), dtype=bool)
    if remove_indices:
        keep_mask[list(sorted(remove_indices))] = False
    kept_points = committed_points[keep_mask]
    kept_normals = committed_normals[keep_mask] if committed_normals.shape[0] == committed_points.shape[0] else committed_normals

    if added_pts:
        merged_points = np.concatenate([kept_points, np.asarray(added_pts, dtype=float)], axis=0)
        merged_normals = np.concatenate([kept_normals, np.asarray(added_nrm, dtype=float)], axis=0)
    else:
        merged_points = kept_points
        merged_normals = kept_normals

    return merged_points, merged_normals, {
        'trimap_export_participation': 1.0,
        'trimap_export_candidates': candidates,
        'trimap_export_added': float(len(added_pts)),
        'trimap_export_replaced': float(len(remove_indices)),
        'trimap_export_residency': residency_hits,
        'trimap_export_route_mean': float(np.mean(route_scores)) if route_scores else 0.0,
        'trimap_export_compete_mean': float(np.mean(compete_scores)) if compete_scores else 0.0,
        'trimap_export_normal_cos_mean': float(np.mean(normal_scores)) if normal_scores else 0.0,
        'trimap_delayed_refine_offset_mean': float(np.mean(refine_offsets)) if refine_offsets else 0.0,
        'trimap_delayed_refine_normal_cos_mean': float(np.mean(refine_normal_cos)) if refine_normal_cos else 1.0,
        'trimap_delayed_bank_points': float(bank_stats.get('trimap_delayed_bank_points', 0.0)),
        'trimap_delayed_bank_conf_mean': float(bank_stats.get('trimap_delayed_bank_conf_mean', 0.0)),
        'trimap_delayed_bank_residency_mean': float(bank_stats.get('trimap_delayed_bank_residency_mean', 0.0)),
    }


def hole_only_delayed_rescue(self, extract_kwargs: Dict[str, object], committed_points: np.ndarray, committed_normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if self.delayed_background_map is None or self.foreground_map is None or not bool(getattr(self.cfg.update, 'tri_map_hole_rescue_enable', False)):
        return committed_points, committed_normals, {'trimap_hole_rescue': 0.0, 'trimap_hole_added': 0.0}

    delayed_points, delayed_normals = self.delayed_background_map.extract_surface_points(**extract_kwargs)
    if delayed_points.shape[0] == 0:
        return committed_points, committed_normals, {'trimap_hole_rescue': 1.0, 'trimap_hole_added': 0.0}

    support_on = float(np.clip(getattr(self.cfg.update, 'tri_map_hole_support_on', 0.72), 0.0, 1.0))
    commit_max = float(np.clip(getattr(self.cfg.update, 'tri_map_hole_commit_max', 0.25), 0.0, 1.0))
    fg_guard = float(np.clip(getattr(self.cfg.update, 'tri_map_hole_fg_guard', 0.28), 0.0, 1.0))
    radius = int(max(1, getattr(self.cfg.update, 'tri_map_hole_radius_cells', 1)))
    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'dual_state_static_protect_rho', 0.90)))

    rescued_pts = []
    rescued_nrm = []
    seen = set()
    for point, normal in zip(delayed_points, delayed_normals):
        idx = self.background_map.world_to_index(point)
        if idx in seen:
            continue
        seen.add(idx)
        commit_support = 0.0
        for nidx in self.background_map.neighbor_indices(idx, radius):
            cc = self.background_map.get_cell(nidx)
            if cc is None:
                continue
            commit_support = max(commit_support, self._map_static_support(self.background_map, cc))
        dc = self.delayed_background_map.get_cell(self.delayed_background_map.world_to_index(point))
        if dc is None:
            continue
        delayed_support = float(np.clip(max(
            self._map_static_support(self.delayed_background_map, dc),
            float(np.clip(getattr(dc, 'rho_bg', 0.0) / rho_ref, 0.0, 1.0)),
            float(np.clip(getattr(dc, 'rho_static', 0.0) / rho_ref, 0.0, 1.0)),
        ), 0.0, 1.0))
        fg_conf = float(self.background_map._cross_map_fg_conf(self.foreground_map, idx)) if hasattr(self.background_map, '_cross_map_fg_conf') else 0.0
        if commit_support <= commit_max and delayed_support >= support_on and fg_conf <= fg_guard:
            rescued_pts.append(point)
            rescued_nrm.append(normal)

    if not rescued_pts:
        return committed_points, committed_normals, {'trimap_hole_rescue': 1.0, 'trimap_hole_added': 0.0}

    merged_points = np.concatenate([committed_points, np.asarray(rescued_pts, dtype=float)], axis=0)
    merged_normals = np.concatenate([committed_normals, np.asarray(rescued_nrm, dtype=float)], axis=0)
    return merged_points, merged_normals, {'trimap_hole_rescue': 1.0, 'trimap_hole_added': float(len(rescued_pts))}
