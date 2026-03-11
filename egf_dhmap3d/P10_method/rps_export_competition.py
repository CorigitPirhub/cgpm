from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


"""Helpers for rear-vs-front export competition and diagnostics."""


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _append_binned(diag: Dict[str, float], prefix: str, value: float) -> None:
    bins = [(-1e9, -0.10, 'lt_m10'), (-0.10, -0.03, 'm10_m03'), (-0.03, 0.03, 'm03_p03'), (0.03, 0.10, 'p03_p10'), (0.10, 1e9, 'ge_p10')]
    for lo, hi, tag in bins:
        if value >= lo and value < hi:
            key = f'{prefix}_{tag}'
            diag[key] = float(diag.get(key, 0.0) + 1.0)
            return


def _bg_support(cell, cfg) -> float:
    rho_ref = float(max(1e-6, getattr(cfg.update, 'rps_commit_rho_ref', getattr(cfg.update, 'dual_state_static_protect_rho', 0.90))))
    weight_ref = float(max(1e-6, getattr(cfg.update, 'rps_commit_weight_ref', 0.80)))
    bg_rho_n = float(np.clip(float(getattr(cell, 'rho_bg', 0.0)) / rho_ref, 0.0, 1.5))
    bg_w_n = float(1.0 - np.exp(-float(max(0.0, getattr(cell, 'phi_bg_w', 0.0))) / weight_ref))
    wod_rear = float(np.clip(getattr(cell, 'wod_rear_conf', 0.0), 0.0, 1.0))
    return _clip01(max(bg_rho_n, bg_w_n, wod_rear))


def decide_rps_bank_competition(
    self,
    cell,
    *,
    front_weight: float,
    front_score_eff: float,
    rear_score: float,
    rear_sep: float,
    w_r: float,
    wr: float,
    rear_phi: float,
    rear_bias: float,
    phi_p: float,
    bias_p: float,
    w_sum: float,
    rear_conf: float,
    support: float,
    commit_like: float,
    active_like: float,
    rear_agree: float,
    front_dyn: float,
    out_stats: Dict[str, float],
) -> Tuple[float, float, float, Dict[str, float]]:
    cfg = self.cfg.update
    margin = float(np.clip(getattr(cfg, 'rps_bank_margin', 0.08), 0.0, 0.5))
    rear_min = float(np.clip(getattr(cfg, 'rps_bank_rear_min_score', 0.52), 0.0, 1.0))
    sep_gate = float(np.clip(getattr(cfg, 'rps_bank_sep_gate', 0.22), 0.0, 1.0))
    bg_support = _bg_support(cell, self.cfg)
    rear_score_eff = float(
        np.clip(
            rear_score
            + float(max(0.0, getattr(cfg, 'rps_bank_bg_support_gain', 0.0))) * bg_support
            + float(max(0.0, getattr(cfg, 'rps_bank_rear_score_bias', 0.0)))
            - float(max(0.0, getattr(cfg, 'rps_bank_front_dyn_penalty_gain', 0.0))) * front_dyn,
            0.0,
            1.0,
        )
    )
    local_support = float(
        np.clip(
            float(max(0.0, getattr(cfg, 'rps_bank_soft_local_support_gain', 1.0)))
            * (
                0.38 * bg_support
                + 0.22 * rear_conf
                + 0.16 * support
                + 0.12 * commit_like
                + 0.08 * active_like
                + 0.04 * rear_agree
            ),
            0.0,
            1.0,
        )
    )
    rear_threshold = float(max(rear_min, front_score_eff + margin))
    gap = float(rear_score_eff - front_score_eff)
    hard_ready = bool(
        w_r > 1e-12
        and rear_sep >= sep_gate
        and rear_score_eff >= rear_threshold
    )

    soft_enable = bool(getattr(cfg, 'rps_bank_soft_competition_enable', False))
    soft_gap = float(max(0.0, getattr(cfg, 'rps_bank_soft_competition_gap', 0.0)))
    soft_sep_gate = float(max(0.0, sep_gate - float(max(0.0, getattr(cfg, 'rps_bank_soft_sep_relax', 0.0)))))
    soft_rear_min = float(max(0.0, rear_min - float(max(0.0, getattr(cfg, 'rps_bank_soft_rear_min_relax', 0.0)))))
    soft_support_min = float(np.clip(getattr(cfg, 'rps_bank_soft_support_min', 0.45), 0.0, 1.0))
    soft_ready = bool(
        soft_enable
        and not hard_ready
        and w_r > 1e-12
        and rear_sep >= soft_sep_gate
        and rear_score_eff >= soft_rear_min
        and rear_score_eff + soft_gap >= front_score_eff
        and local_support >= soft_support_min
    )

    context = str(getattr(self, '_ptdsf_context', '')).strip().lower()
    if context == 'extract':
        diag = getattr(self, '_rps_competition_diag', None)
        if diag is None:
            diag = {}
            setattr(self, '_rps_competition_diag', diag)
        diag['extract_considered'] = float(diag.get('extract_considered', 0.0) + 1.0)
        diag['extract_front_score_sum'] = float(diag.get('extract_front_score_sum', 0.0) + front_score_eff)
        diag['extract_rear_score_sum'] = float(diag.get('extract_rear_score_sum', 0.0) + rear_score)
        diag['extract_rear_score_eff_sum'] = float(diag.get('extract_rear_score_eff_sum', 0.0) + rear_score_eff)
        diag['extract_gap_sum'] = float(diag.get('extract_gap_sum', 0.0) + gap)
        diag['extract_sep_sum'] = float(diag.get('extract_sep_sum', 0.0) + rear_sep)
        diag['extract_bg_support_sum'] = float(diag.get('extract_bg_support_sum', 0.0) + bg_support)
        diag['extract_local_support_sum'] = float(diag.get('extract_local_support_sum', 0.0) + local_support)
        diag['extract_hard_ready'] = float(diag.get('extract_hard_ready', 0.0) + (1.0 if hard_ready else 0.0))
        diag['extract_soft_ready'] = float(diag.get('extract_soft_ready', 0.0) + (1.0 if soft_ready else 0.0))
        _append_binned(diag, 'extract_gap', gap)
        _append_binned(diag, 'extract_sep', rear_sep - sep_gate)
        _append_binned(diag, 'extract_front_score', front_score_eff - 0.5)
        _append_binned(diag, 'extract_rear_score', rear_score_eff - 0.5)
        if abs(gap) <= max(soft_gap, 0.03):
            diag['extract_close_gap'] = float(diag.get('extract_close_gap', 0.0) + 1.0)
        if not hard_ready and not soft_ready:
            if rear_sep < sep_gate:
                diag['extract_fail_sep'] = float(diag.get('extract_fail_sep', 0.0) + 1.0)
            if rear_score_eff < rear_min:
                diag['extract_fail_min'] = float(diag.get('extract_fail_min', 0.0) + 1.0)
            if rear_score_eff < front_score_eff + margin:
                diag['extract_fail_margin'] = float(diag.get('extract_fail_margin', 0.0) + 1.0)
            if soft_enable and local_support < soft_support_min:
                diag['extract_fail_soft_support'] = float(diag.get('extract_fail_soft_support', 0.0) + 1.0)
        diag['extract_wr_sum'] = float(diag.get('extract_wr_sum', 0.0) + wr)
        diag['extract_w_r_sum'] = float(diag.get('extract_w_r_sum', 0.0) + w_r)

    out_stats.update(
        {
            'rear_score_eff': rear_score_eff,
            'rear_gap': gap,
            'rear_bg_support': bg_support,
            'rear_local_support': local_support,
            'rear_hard_ready': 1.0 if hard_ready else 0.0,
            'rear_soft_ready': 1.0 if soft_ready else 0.0,
        }
    )
    if hard_ready or soft_ready:
        phi_p = rear_phi
        bias_p = rear_bias
        w_sum = float(max(w_r, 0.45 * wr))
        out_stats['bank_selected'] = 'rear' if hard_ready else 'rear_soft'
        if context == 'extract':
            diag = getattr(self, '_rps_competition_diag', None)
            if diag is not None:
                key = 'extract_rear_selected' if hard_ready else 'extract_rear_soft_selected'
                diag[key] = float(diag.get(key, 0.0) + 1.0)
    else:
        if context == 'extract':
            diag = getattr(self, '_rps_competition_diag', None)
            if diag is not None:
                diag['extract_front_kept'] = float(diag.get('extract_front_kept', 0.0) + 1.0)
    return phi_p, bias_p, w_sum, out_stats
