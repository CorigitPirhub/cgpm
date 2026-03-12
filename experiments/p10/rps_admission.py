from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.p10.bg_manifold import manifold_state_components

"""Support-based admission helpers for committed rear-bank survival before extract competition."""


def rear_state_support_components(self, cell) -> Dict[str, float]:
    cfg = self.cfg.update
    rho_ref = float(max(1e-6, getattr(cfg, 'rps_admission_rho_ref', getattr(cfg, 'rps_commit_rho_ref', 0.08))))
    weight_ref = float(max(1e-6, getattr(cfg, 'rps_admission_weight_ref', getattr(cfg, 'rps_commit_weight_ref', 0.35))))
    age_ref = float(max(1.0, getattr(cfg, 'rps_commit_age_threshold', getattr(cfg, 'rps_commit_age_ref', 2.0))))

    rear_rho = float(max(0.0, getattr(cell, 'rho_rear', 0.0)))
    rear_w = float(max(0.0, getattr(cell, 'phi_rear_w', 0.0)))
    rear_rho_n = float(np.clip(rear_rho / rho_ref, 0.0, 1.5))
    rear_w_n = float(1.0 - np.exp(-rear_w / weight_ref))

    bg_rho = float(max(0.0, getattr(cell, 'rho_bg', 0.0)))
    bg_w = float(max(0.0, getattr(cell, 'phi_bg_w', 0.0)))
    bg_rho_n = float(np.clip(bg_rho / rho_ref, 0.0, 1.5))
    bg_w_n = float(1.0 - np.exp(-bg_w / weight_ref))
    bg_support = float(np.clip(max(bg_rho_n, bg_w_n, float(np.clip(getattr(cell, 'wod_rear_conf', 0.0), 0.0, 1.0))), 0.0, 1.0))

    rho_static = float(max(0.0, getattr(cell, 'rho_static', 0.0)))
    static_w = float(max(0.0, getattr(cell, 'phi_static_w', 0.0)))
    rho_static_n = float(np.clip(rho_static / rho_ref, 0.0, 1.5))
    static_w_n = float(1.0 - np.exp(-static_w / weight_ref))
    static_support = float(np.clip(max(float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0)), rho_static_n, static_w_n), 0.0, 1.0))

    rho_bg_cand = float(max(0.0, getattr(cell, 'rho_bg_cand', 0.0)))
    bg_cand_w = float(max(0.0, getattr(cell, 'phi_bg_cand_w', 0.0)))
    rho_bg_cand_n = float(np.clip(rho_bg_cand / rho_ref, 0.0, 1.5))
    bg_cand_w_n = float(1.0 - np.exp(-bg_cand_w / weight_ref))
    history_bg = float(np.clip(max(bg_support, rho_bg_cand_n, bg_cand_w_n), 0.0, 1.0))

    score_n = float(np.clip(getattr(cell, 'rps_commit_score', 0.0), 0.0, 1.0))
    age_n = float(np.clip(float(getattr(cell, 'rps_commit_age', 0.0)) / age_ref, 0.0, 1.0))

    support = float(np.clip(0.34 * min(1.0, rear_rho_n) + 0.26 * rear_w_n + 0.16 * bg_support + 0.14 * score_n + 0.10 * age_n, 0.0, 1.0))

    cons_ref = float(max(1e-6, getattr(cfg, 'rps_consistency_ref', 0.03)))
    rear_phi = float(getattr(cell, 'phi_rear', 0.0))
    def _agree(ref_phi: float | None, ref_w: float) -> float:
        if ref_phi is None or ref_w <= 1e-12 or rear_w <= 1e-12:
            return 0.5
        return float(np.exp(-0.5 * ((rear_phi - float(ref_phi)) / cons_ref) ** 2))

    geo_agree = _agree(getattr(cell, 'phi_geo', None), float(max(0.0, getattr(cell, 'phi_geo_w', 0.0))))
    bg_agree = _agree(getattr(cell, 'phi_bg', None), bg_w)
    static_agree = _agree(getattr(cell, 'phi_static', None), static_w)
    geom_score = float(np.clip(max(geo_agree, 0.90 * bg_agree, 0.70 * static_agree), 0.0, 1.0))

    history_support = float(np.clip(0.55 * history_bg + 0.45 * static_support, 0.0, 1.0))
    history_floor = 0.0
    if bool(getattr(cfg, 'rps_space_redirect_history_enable', False)):
        hw = float(np.clip(getattr(cfg, 'rps_space_redirect_history_weight', 0.32), 0.0, 1.0))
        bgw = float(max(0.0, getattr(cfg, 'rps_space_redirect_history_bg_weight', 0.60)))
        stw = float(max(0.0, getattr(cfg, 'rps_space_redirect_history_static_weight', 0.40)))
        hsum = max(1e-6, bgw + stw)
        hist_mix = float(np.clip((bgw * history_bg + stw * static_support) / hsum, 0.0, 1.0))
        support = float(np.clip((1.0 - hw) * support + hw * hist_mix, 0.0, 1.0))
        history_floor = float(np.clip(getattr(cfg, 'rps_space_redirect_history_floor', 0.30), 0.0, 1.0))
        if hist_mix < history_floor:
            support *= float(np.clip(hist_mix / max(1e-6, history_floor), 0.05, 1.0))

    if bool(getattr(cfg, 'rps_admission_geometry_enable', False)):
        geom_w = float(max(0.0, getattr(cfg, 'rps_admission_geometry_weight', 0.25)))
        geom_floor = float(np.clip(getattr(cfg, 'rps_admission_geometry_floor', 0.40), 0.0, 1.0))
        support = float(np.clip((1.0 - geom_w) * support + geom_w * geom_score, 0.0, 1.0))
        if geom_score < geom_floor:
            support *= float(np.clip(geom_score / max(1e-6, geom_floor), 0.10, 1.0))

    dyn_risk = float(np.clip(max(float(np.clip(getattr(cell, 'dyn_prob', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'st_mem', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'visibility_contradiction', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'residual_evidence', 0.0), 0.0, 1.0))), 0.0, 1.0))
    ghost_like = float(np.clip(max(float(np.clip(getattr(cell, 'wod_shell_conf', 0.0), 0.0, 1.0)), dyn_risk), 0.0, 1.0))
    if bool(getattr(cfg, 'rps_admission_occlusion_enable', False)):
        occ_w = float(max(0.0, getattr(cfg, 'rps_admission_occlusion_weight', 0.12)))
        support = float(np.clip(support + occ_w * ghost_like * max(bg_agree, geom_score), 0.0, 1.0))

    if bool(getattr(cfg, 'rps_space_redirect_ghost_suppress_enable', False)):
        gs_w = float(max(0.0, getattr(cfg, 'rps_space_redirect_ghost_suppress_weight', 0.22)))
        support = float(np.clip(support - gs_w * dyn_risk * (1.0 - 0.55 * history_support), 0.0, 1.0))

    visual_anchor = float(np.clip(max(history_bg, static_support, score_n), 0.0, 1.0))
    if bool(getattr(cfg, 'rps_space_redirect_visual_anchor_enable', False)):
        anchor_min = float(np.clip(getattr(cfg, 'rps_space_redirect_visual_anchor_min', 0.36), 0.0, 1.0))
        anchor_w = float(max(0.0, getattr(cfg, 'rps_space_redirect_visual_anchor_weight', 0.16)))
        if visual_anchor < anchor_min:
            support *= float(np.clip(visual_anchor / max(1e-6, anchor_min), 0.05, 1.0))
        else:
            support = float(np.clip(support + anchor_w * visual_anchor, 0.0, 1.0))

    manifold_visible = 0.0
    manifold_obstructed = 0.0
    manifold_phi = float(getattr(cell, 'phi_bg_memory', 0.0))
    if bool(getattr(cfg, 'rps_bg_manifold_state_enable', False)):
        manifold = manifold_state_components(cell, self.cfg)
        manifold_visible = float(manifold['visible'])
        manifold_obstructed = float(manifold['obstructed'])
        manifold_phi = float(manifold['phi'])
        hist_w = float(np.clip(getattr(cfg, 'rps_bg_manifold_history_weight', 0.30), 0.0, 1.0))
        obs_w = float(np.clip(getattr(cfg, 'rps_bg_manifold_obstruction_weight', 0.20), 0.0, 1.0))
        visible_lo = float(np.clip(getattr(cfg, 'rps_bg_manifold_visible_lo', 0.25), 0.0, 1.0))
        visible_hi = float(np.clip(getattr(cfg, 'rps_bg_manifold_visible_hi', 0.50), 0.0, 1.0))
        support = float(np.clip((1.0 - hist_w) * support + hist_w * manifold_visible + obs_w * manifold_obstructed, 0.0, 1.0))
        if manifold_visible < visible_lo:
            support *= float(np.clip(manifold_visible / max(1e-6, visible_lo), 0.02, 1.0))
        elif manifold_visible < visible_hi:
            band = float(np.clip((manifold_visible - visible_lo) / max(1e-6, visible_hi - visible_lo), 0.0, 1.0))
            support *= float(np.clip(0.35 + 0.65 * band, 0.10, 1.0))

    return {
        'support': support,
        'history_support': history_support,
        'history_floor': history_floor,
        'static_support': static_support,
        'history_bg': history_bg,
        'geom_score': geom_score,
        'dyn_risk': dyn_risk,
        'ghost_like': ghost_like,
        'visual_anchor': visual_anchor,
        'manifold_visible': manifold_visible,
        'manifold_obstructed': manifold_obstructed,
        'manifold_phi': manifold_phi,
    }


def rear_state_support(self, cell) -> float:
    return float(rear_state_support_components(self, cell)['support'])


def rear_admission_status(self, cell, *, wr: float, soft_bank_on: bool = False) -> Dict[str, float | bool]:
    cfg = self.cfg.update
    active_raw = float(np.clip(getattr(cell, 'rps_active', 0.0), 0.0, 1.0))
    score_raw = float(np.clip(getattr(cell, 'rps_commit_score', 0.0), 0.0, 1.0))
    score_off = float(np.clip(getattr(cfg, 'rps_commit_off', getattr(cfg, 'rps_commit_release', 0.40)), 0.0, 1.0))

    support_enable = bool(getattr(cfg, 'rps_admission_support_enable', False))
    support_on = float(np.clip(getattr(cfg, 'rps_admission_support_on', 0.42), 0.0, 1.0))
    support_gain = float(max(0.0, getattr(cfg, 'rps_admission_support_gain', 0.55)))
    score_relax = float(max(0.0, getattr(cfg, 'rps_admission_score_relax', 0.10)))
    active_floor = float(np.clip(getattr(cfg, 'rps_admission_active_floor', 0.32), 0.0, 1.0))

    support_pack = rear_state_support_components(self, cell) if support_enable else {'support': 0.0, 'history_support': 0.0, 'history_floor': 0.0, 'static_support': 0.0, 'history_bg': 0.0, 'geom_score': 0.0, 'dyn_risk': 0.0, 'ghost_like': 0.0, 'visual_anchor': 0.0}
    support = float(support_pack['support'])
    protected = bool(support_enable and support >= support_on)
    active_like = float(max(active_raw, support_gain * support if protected else 0.0))
    active_gate = float(0.5 if not protected else min(0.5, max(active_floor, 0.5 - 0.25 * support)))
    score_gate = bool(score_raw >= max(0.0, score_off - (score_relax * support if protected else 0.0)))

    history_visible_ok = True
    obstruction_ok = True
    non_hole_ok = True
    if bool(getattr(cfg, 'rps_history_obstructed_gate_enable', False)):
        history_visible_ok = bool(max(float(support_pack.get('history_bg', 0.0)), float(support_pack.get('static_support', 0.0))) >= float(np.clip(getattr(cfg, 'rps_history_visible_min', 0.45), 0.0, 1.0)))
        obstruction_ok = bool(max(float(np.clip(getattr(cell, 'wod_rear_conf', 0.0), 0.0, 1.0)), float(support_pack.get('dyn_risk', 0.0)), float(np.clip(getattr(cell, 'wod_shell_conf', 0.0), 0.0, 1.0))) >= float(np.clip(getattr(cfg, 'rps_obstruction_min', 0.28), 0.0, 1.0)))
        non_hole_ok = bool(float(support_pack.get('static_support', 0.0)) >= float(np.clip(getattr(cfg, 'rps_non_hole_min', 0.30), 0.0, 1.0)))
        score_gate = bool(score_gate and history_visible_ok and obstruction_ok and non_hole_ok)

    hard_commit_on = bool(wr > 1e-12 and (active_raw >= 0.5 or (protected and active_like >= active_gate)) and score_gate)
    rear_enabled = bool((wr > 1e-12 or soft_bank_on) and (hard_commit_on or soft_bank_on))

    return {
        'support': support,
        'protected': protected,
        'active_raw': active_raw,
        'active_like': active_like,
        'active_gate': active_gate,
        'score_raw': score_raw,
        'score_gate': score_gate,
        'hard_commit_on': hard_commit_on,
        'rear_enabled': rear_enabled,
        'history_support': float(support_pack['history_support']),
        'history_bg': float(support_pack['history_bg']),
        'static_support': float(support_pack['static_support']),
        'geom_score': float(support_pack['geom_score']),
        'dyn_risk': float(support_pack['dyn_risk']),
        'ghost_like': float(support_pack['ghost_like']),
        'visual_anchor': float(support_pack['visual_anchor']),
        'history_visible_ok': bool(history_visible_ok),
        'obstruction_ok': bool(obstruction_ok),
        'non_hole_ok': bool(non_hole_ok),
    }


def update_admission_diag(self, *, context: str, wr: float, status: Dict[str, float | bool], bank_selected: str | None = None) -> None:
    diag = getattr(self, '_rps_admission_diag', None)
    if diag is None:
        diag = {}
        setattr(self, '_rps_admission_diag', diag)

    prefix = f'{context}_'
    diag[prefix + 'calls'] = float(diag.get(prefix + 'calls', 0.0) + 1.0)
    if wr > 1e-12:
        diag[prefix + 'wr_present'] = float(diag.get(prefix + 'wr_present', 0.0) + 1.0)
    if float(status['active_raw']) >= 0.5:
        diag[prefix + 'active_ready'] = float(diag.get(prefix + 'active_ready', 0.0) + 1.0)
    if bool(status['score_gate']):
        diag[prefix + 'score_ready'] = float(diag.get(prefix + 'score_ready', 0.0) + 1.0)
    if bool(status['protected']):
        diag[prefix + 'support_protected'] = float(diag.get(prefix + 'support_protected', 0.0) + 1.0)
    if bool(status['hard_commit_on']):
        diag[prefix + 'hard_commit_on'] = float(diag.get(prefix + 'hard_commit_on', 0.0) + 1.0)
    else:
        if wr > 1e-12 and float(status['active_raw']) < 0.5 and not bool(status['protected']):
            diag[prefix + 'fail_active'] = float(diag.get(prefix + 'fail_active', 0.0) + 1.0)
        if wr > 1e-12 and not bool(status['score_gate']):
            diag[prefix + 'fail_score'] = float(diag.get(prefix + 'fail_score', 0.0) + 1.0)
        if wr > 1e-12 and not bool(status.get('history_visible_ok', True)):
            diag[prefix + 'fail_history_visible'] = float(diag.get(prefix + 'fail_history_visible', 0.0) + 1.0)
        if wr > 1e-12 and not bool(status.get('obstruction_ok', True)):
            diag[prefix + 'fail_obstruction'] = float(diag.get(prefix + 'fail_obstruction', 0.0) + 1.0)
        if wr > 1e-12 and not bool(status.get('non_hole_ok', True)):
            diag[prefix + 'fail_non_hole'] = float(diag.get(prefix + 'fail_non_hole', 0.0) + 1.0)
    if bool(status['rear_enabled']):
        diag[prefix + 'rear_enabled'] = float(diag.get(prefix + 'rear_enabled', 0.0) + 1.0)
    diag[prefix + 'support_sum'] = float(diag.get(prefix + 'support_sum', 0.0) + float(status['support']))
    diag[prefix + 'active_like_sum'] = float(diag.get(prefix + 'active_like_sum', 0.0) + float(status['active_like']))
    diag[prefix + 'history_support_sum'] = float(diag.get(prefix + 'history_support_sum', 0.0) + float(status.get('history_support', 0.0)))
    diag[prefix + 'history_bg_sum'] = float(diag.get(prefix + 'history_bg_sum', 0.0) + float(status.get('history_bg', 0.0)))
    diag[prefix + 'static_support_sum'] = float(diag.get(prefix + 'static_support_sum', 0.0) + float(status.get('static_support', 0.0)))
    diag[prefix + 'geom_score_sum'] = float(diag.get(prefix + 'geom_score_sum', 0.0) + float(status.get('geom_score', 0.0)))
    diag[prefix + 'dyn_risk_sum'] = float(diag.get(prefix + 'dyn_risk_sum', 0.0) + float(status.get('dyn_risk', 0.0)))
    diag[prefix + 'ghost_like_sum'] = float(diag.get(prefix + 'ghost_like_sum', 0.0) + float(status.get('ghost_like', 0.0)))
    diag[prefix + 'visual_anchor_sum'] = float(diag.get(prefix + 'visual_anchor_sum', 0.0) + float(status.get('visual_anchor', 0.0)))
    diag[prefix + 'manifold_visible_sum'] = float(diag.get(prefix + 'manifold_visible_sum', 0.0) + float(status.get('manifold_visible', 0.0)))
    diag[prefix + 'manifold_obstructed_sum'] = float(diag.get(prefix + 'manifold_obstructed_sum', 0.0) + float(status.get('manifold_obstructed', 0.0)))
    if bank_selected:
        diag[prefix + 'bank_' + bank_selected.replace('-', '_')] = float(diag.get(prefix + 'bank_' + bank_selected.replace('-', '_'), 0.0) + 1.0)


def decay_protect_factors(ucfg, support: float) -> Dict[str, float]:
    relax = float(max(0.0, getattr(ucfg, 'rps_rear_state_decay_relax', 0.45)))
    support = float(np.clip(support, 0.0, 1.0))
    exponent_scale = float(np.clip(1.0 - relax * support, 0.20, 1.0))
    active_floor = float(np.clip(getattr(ucfg, 'rps_rear_state_active_floor', 0.28), 0.0, 1.0))
    return {
        'exponent_scale': exponent_scale,
        'active_floor': active_floor * support,
    }
