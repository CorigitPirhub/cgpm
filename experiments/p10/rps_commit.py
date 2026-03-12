from __future__ import annotations

import numpy as np

from experiments.p10.bg_manifold import manifold_state_components

"""RPS commit-score / activation helpers for S2 rear-bank activation experiments."""


def update_rps_commit_activation(
    self,
    voxel_map,
    cell,
    *,
    d_geo: float,
    w_obs: float,
    static_mass: float,
    wod_front: float,
    wod_rear: float,
    wod_shell: float,
    q_dyn_obs: float,
    assoc_risk: float,
) -> None:
    if not (
        bool(getattr(self.cfg.update, 'rps_enable', False))
        and bool(getattr(self.cfg.update, 'rps_hard_commit_enable', False))
        and bool(getattr(self.cfg.update, 'rps_commit_activation_enable', False))
    ):
        return

    diag = getattr(self, '_rps_commit_diag', None)
    if diag is None:
        diag = {}
        setattr(self, '_rps_commit_diag', diag)
    diag['calls'] = float(diag.get('calls', 0.0) + 1.0)

    cand_w = float(max(0.0, getattr(cell, 'phi_rear_cand_w', 0.0)))
    cand_rho = float(max(0.0, getattr(cell, 'rho_rear_cand', 0.0)))
    if cand_w <= 1e-12 or cand_rho <= 1e-12:
        cell.rps_active = float(
            np.clip(
                float(getattr(cell, 'rps_active', 0.0))
                * float(np.clip(getattr(self.cfg.update, 'rps_active_decay', 0.97), 0.0, 1.0)),
                0.0,
                1.0,
            )
        )
        diag['inactive_empty'] = float(diag.get('inactive_empty', 0.0) + 1.0)
        return

    quality_enable = bool(getattr(self.cfg.update, 'rps_commit_quality_enable', False))
    cons_ref = float(max(voxel_map.voxel_size, getattr(self.cfg.update, 'rps_consistency_ref', 0.03)))
    cand_phi = float(getattr(cell, 'phi_rear_cand', d_geo))

    static_w = float(max(0.0, getattr(cell, 'phi_static_w', 0.0)))
    geo_w = float(max(0.0, getattr(cell, 'phi_geo_w', 0.0)))
    bg_w = float(max(0.0, getattr(cell, 'phi_bg_w', 0.0)))
    rear_w_prev = float(max(0.0, getattr(cell, 'phi_rear_w', 0.0)))
    static_phi = float(getattr(cell, 'phi_static', cand_phi))
    geo_phi = float(getattr(cell, 'phi_geo', cand_phi))
    bg_phi = float(getattr(cell, 'phi_bg', cand_phi))
    rear_phi_prev = float(getattr(cell, 'phi_rear', cand_phi))

    def _agree(ref_phi: float | None, ref_w: float) -> float:
        if ref_phi is None or ref_w <= 1e-12:
            return 0.5
        return float(np.exp(-0.5 * ((cand_phi - float(ref_phi)) / cons_ref) ** 2))

    static_agree = _agree(static_phi, static_w)
    geo_agree = _agree(geo_phi, geo_w)
    bg_agree = _agree(bg_phi, bg_w)
    bank_agree = _agree(rear_phi_prev, rear_w_prev)
    geom_consistency = float(max(static_agree, geo_agree, 0.65 * bg_agree))
    if quality_enable:
        geom_consistency = float(max(geom_consistency, 0.50 * bank_agree))

    rho_ref = float(max(1e-6, getattr(self.cfg.update, 'rps_commit_rho_ref', 0.08)))
    weight_ref = float(max(1e-6, getattr(self.cfg.update, 'rps_commit_weight_ref', 0.80)))
    evidence_n = float(np.clip(cand_rho / rho_ref, 0.0, 1.5))
    weight_n = float(1.0 - np.exp(-cand_w / weight_ref))
    bg_rho_n = float(np.clip(float(getattr(cell, 'rho_bg', 0.0)) / rho_ref, 0.0, 1.5))
    bg_w_n = float(1.0 - np.exp(-bg_w / weight_ref))
    bg_support = float(np.clip(max(bg_rho_n, bg_w_n, wod_rear), 0.0, 1.0))
    static_support = float(
        np.clip(
            max(
                float(np.clip(getattr(cell, 'p_static', 0.0), 0.0, 1.0)),
                static_mass,
                float(np.clip(float(getattr(cell, 'rho_static', 0.0)) / rho_ref, 0.0, 1.0)),
                float(1.0 - np.exp(-static_w / weight_ref)),
            ),
            0.0,
            1.0,
        )
    )
    front_pen = float(
        np.clip(
            wod_front
            + 0.50 * wod_shell
            + 0.25 * float(np.clip(q_dyn_obs, 0.0, 1.0))
            + 0.15 * float(np.clip(assoc_risk, 0.0, 1.0)),
            0.0,
            1.5,
        )
    )

    w_e = float(max(0.0, getattr(self.cfg.update, 'rps_commit_evidence_weight', 0.34)))
    w_g = float(max(0.0, getattr(self.cfg.update, 'rps_commit_geometry_weight', 0.28)))
    w_b = float(max(0.0, getattr(self.cfg.update, 'rps_commit_bg_weight', 0.20)))
    w_s = float(max(0.0, getattr(self.cfg.update, 'rps_commit_static_weight', 0.18)))
    w_sum = max(1e-6, w_e + w_g + w_b + w_s)
    support_score = float(
        (
            w_e * min(1.0, evidence_n)
            + w_g * geom_consistency
            + w_b * bg_support
            + w_s * static_support
        ) / w_sum
    )

    quality_score = 0.0
    if quality_enable:
        quality_score = float(
            np.clip(
                0.26 * geom_consistency
                + 0.20 * bg_support
                + 0.16 * static_support
                + 0.16 * bank_agree
                + 0.12 * min(1.0, evidence_n)
                + 0.10 * weight_n,
                0.0,
                1.0,
            )
        )
        support_score = float(np.clip(support_score + 0.10 * quality_score + 0.06 * weight_n, 0.0, 1.0))

    penalty_gain = float(max(0.0, getattr(self.cfg.update, 'rps_commit_front_penalty', 0.22)))
    score_obs = float(np.clip(support_score - penalty_gain * front_pen, 0.0, 1.0))

    alpha = float(np.clip(getattr(self.cfg.update, 'rps_score_alpha', 0.18), 0.01, 0.95))
    prev_score = float(np.clip(getattr(cell, 'rps_commit_score', 0.0), 0.0, 1.0))
    cell.rps_commit_score = float(np.clip((1.0 - alpha) * prev_score + alpha * score_obs, 0.0, 1.0))
    diag['score_obs_sum'] = float(diag.get('score_obs_sum', 0.0) + score_obs)
    diag['score_sum'] = float(diag.get('score_sum', 0.0) + float(getattr(cell, 'rps_commit_score', 0.0)))
    if quality_enable:
        diag['quality_score_sum'] = float(diag.get('quality_score_sum', 0.0) + quality_score)
        diag['weight_norm_sum'] = float(diag.get('weight_norm_sum', 0.0) + weight_n)
        diag['bank_agree_sum'] = float(diag.get('bank_agree_sum', 0.0) + bank_agree)

    age_ref = float(max(1.0, getattr(self.cfg.update, 'rps_commit_age_threshold', getattr(self.cfg.update, 'rps_commit_age_ref', 2.0))))
    if cell.rps_commit_score >= float(np.clip(getattr(self.cfg.update, 'rps_commit_release', 0.32), 0.0, 1.0)):
        cell.rps_commit_age = float(min(20.0, float(getattr(cell, 'rps_commit_age', 0.0)) + 1.0))
    else:
        cell.rps_commit_age = float(max(0.0, 0.90 * float(getattr(cell, 'rps_commit_age', 0.0))))

    min_rho = float(max(0.0, getattr(self.cfg.update, 'rps_commit_min_cand_rho', 0.02)))
    min_w = float(max(0.0, getattr(self.cfg.update, 'rps_commit_min_cand_w', 0.08)))
    on = float(np.clip(getattr(self.cfg.update, 'rps_commit_threshold', getattr(self.cfg.update, 'rps_commit_on', 0.62)), 0.0, 1.0))
    off = float(np.clip(getattr(self.cfg.update, 'rps_commit_release', getattr(self.cfg.update, 'rps_commit_off', 0.40)), 0.0, 1.0))
    active_prev = float(np.clip(getattr(cell, 'rps_active', 0.0), 0.0, 1.0))
    active_decay = float(np.clip(getattr(self.cfg.update, 'rps_active_decay', 0.97), 0.0, 1.0))

    commit_ready = bool(
        cell.rps_commit_score >= on
        and float(getattr(cell, 'rps_commit_age', 0.0)) >= age_ref
        and cand_w >= min_w
        and cand_rho >= min_rho
    )
    if commit_ready:
        diag['commit_ready'] = float(diag.get('commit_ready', 0.0) + 1.0)
        blend = float(np.clip(getattr(self.cfg.update, 'rps_commit_blend', 0.78), 0.0, 1.0))
        commit_phi = cand_phi
        transfer = blend
        rho_commit_scale = 1.0

        if quality_enable:
            geom_blend = float(np.clip(getattr(self.cfg.update, 'rps_commit_quality_geom_blend', 0.35), 0.0, 1.0))
            sep_scale = float(max(0.0, getattr(self.cfg.update, 'rps_commit_quality_sep_scale', 0.65)))
            transfer_gain = float(max(0.0, getattr(self.cfg.update, 'rps_commit_quality_transfer_gain', 0.18)))
            rho_gain = float(max(0.0, getattr(self.cfg.update, 'rps_commit_quality_rho_gain', 0.55)))

            front_terms = []
            if static_w > 1e-12:
                front_terms.append((static_phi, static_w * (0.35 + 0.65 * static_agree)))
            if geo_w > 1e-12:
                front_terms.append((geo_phi, geo_w * (0.35 + 0.65 * geo_agree)))
            if front_terms:
                front_ref_phi = float(sum(v * w for v, w in front_terms) / max(1e-9, sum(w for _, w in front_terms)))
            else:
                front_ref_phi = float(cand_phi)

            refs = [(cand_phi, 1.0 + 0.35 * min(1.0, evidence_n) + 0.25 * weight_n)]
            if rear_w_prev > 1e-12 and bank_agree >= 0.35:
                refs.append((rear_phi_prev, 0.65 * bank_agree))
            if static_w > 1e-12 and static_agree >= 0.35:
                refs.append((static_phi, geom_blend * (0.35 + 0.65 * static_agree)))
            if geo_w > 1e-12 and geo_agree >= 0.35:
                refs.append((geo_phi, geom_blend * (0.35 + 0.65 * geo_agree)))
            if bg_w > 1e-12 and bg_support >= 0.25:
                bg_sep = abs(bg_phi - front_ref_phi)
                cand_sep = abs(cand_phi - front_ref_phi)
                bg_pref = 1.0 if bg_sep >= cand_sep else 0.70
                refs.append((bg_phi, 0.55 * geom_blend * bg_pref * (0.40 + 0.60 * bg_support)))
            commit_phi = float(sum(v * w for v, w in refs) / max(1e-9, sum(w for _, w in refs)))

            if bool(getattr(self.cfg.update, 'rps_history_manifold_enable', False)):
                visible_min = float(np.clip(getattr(self.cfg.update, 'rps_history_manifold_visible_min', 0.45), 0.0, 1.0))
                obstruction_min = float(np.clip(getattr(self.cfg.update, 'rps_history_manifold_obstruction_min', 0.28), 0.0, 1.0))
                manifold = manifold_state_components(cell, self.cfg) if bool(getattr(self.cfg.update, 'rps_bg_manifold_state_enable', False)) else None
                history_visible = max(bg_support, static_support, float(manifold['visible']) if manifold is not None else 0.0) >= visible_min
                obstruction_like = max(float(np.clip(getattr(cell, 'wod_rear_conf', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'wod_shell_conf', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'dyn_prob', 0.0), 0.0, 1.0)), float(np.clip(getattr(cell, 'z_dyn', 0.0), 0.0, 1.0)), float(manifold['obstructed']) if manifold is not None else 0.0)
                if history_visible and obstruction_like >= obstruction_min:
                    bgw = float(max(0.0, getattr(self.cfg.update, 'rps_history_manifold_bg_weight', 0.50)))
                    geow = float(max(0.0, getattr(self.cfg.update, 'rps_history_manifold_geo_weight', 0.30)))
                    stw = float(max(0.0, getattr(self.cfg.update, 'rps_history_manifold_static_weight', 0.20)))
                    manifold_refs = []
                    if manifold is not None and float(manifold['weight']) > 1e-12:
                        manifold_refs.append((float(manifold['phi']), bgw * max(float(manifold['visible']), 0.5)))
                    elif bg_w > 1e-12:
                        manifold_refs.append((bg_phi, bgw * max(bg_support, bg_agree)))
                    if geo_w > 1e-12:
                        manifold_refs.append((geo_phi, geow * max(geo_agree, 0.5)))
                    if static_w > 1e-12:
                        manifold_refs.append((static_phi, stw * max(static_support, static_agree)))
                    if manifold_refs:
                        manifold_phi = float(sum(v * w for v, w in manifold_refs) / max(1e-9, sum(w for _, w in manifold_refs)))
                        manifold_blend = float(np.clip(getattr(self.cfg.update, 'rps_history_manifold_blend', 0.75), 0.0, 1.0))
                        max_off = float(max(voxel_map.voxel_size, getattr(self.cfg.update, 'rps_history_manifold_max_offset', 0.04)))
                        projected_phi = float((1.0 - manifold_blend) * commit_phi + manifold_blend * manifold_phi)
                        delta = float(np.clip(projected_phi - manifold_phi, -max_off, max_off))
                        commit_phi = float(manifold_phi + delta)
                        diag['history_manifold_used'] = float(diag.get('history_manifold_used', 0.0) + 1.0)
                        diag['history_manifold_shift_sum'] = float(diag.get('history_manifold_shift_sum', 0.0) + abs(commit_phi - cand_phi))

            sep_ref = float(max(voxel_map.voxel_size, getattr(self.cfg.update, 'rps_bank_separation_ref', 0.04)))
            sep_floor = float(sep_scale * sep_ref)
            direction = float(np.sign(cand_phi - front_ref_phi))
            if abs(direction) < 0.5:
                alt = 0.0
                if bg_w > 1e-12:
                    alt = bg_phi - front_ref_phi
                elif rear_w_prev > 1e-12:
                    alt = rear_phi_prev - front_ref_phi
                elif geo_w > 1e-12:
                    alt = cand_phi - geo_phi
                direction = float(np.sign(alt)) if abs(alt) > 1e-9 else 1.0
            cand_sep = abs(cand_phi - front_ref_phi)
            rear_sep_prev = abs(rear_phi_prev - front_ref_phi) if rear_w_prev > 1e-12 else 0.0
            bg_sep = abs(bg_phi - front_ref_phi) if bg_w > 1e-12 else 0.0
            target_sep = float(max(cand_sep, min(sep_floor, max(cand_sep, rear_sep_prev, bg_sep))))
            if quality_score > 0.38 and abs(commit_phi - front_ref_phi) < target_sep:
                commit_phi = float(front_ref_phi + direction * target_sep)

            transfer = float(
                np.clip(
                    blend
                    + transfer_gain
                    * (
                        0.45 * quality_score
                        + 0.25 * bg_support
                        + 0.20 * bank_agree
                        + 0.10 * weight_n
                        - 0.20 * front_pen
                    ),
                    0.55,
                    0.98,
                )
            )
            rho_commit_scale = float(
                1.0
                + rho_gain
                * float(np.clip(0.55 * quality_score + 0.25 * bg_support + 0.20 * bank_agree, 0.0, 1.0))
            )
            diag['commit_shift_sum'] = float(diag.get('commit_shift_sum', 0.0) + abs(commit_phi - cand_phi))

        w_commit = float(cand_w * transfer)
        rho_commit = float(transfer * cand_rho * rho_commit_scale)
        w_bank_new = float(rear_w_prev + w_commit)
        if w_bank_new > 1e-12:
            cell.phi_rear = float((rear_w_prev * rear_phi_prev + w_commit * commit_phi) / w_bank_new)
            cell.phi_rear_w = float(min(5000.0, w_bank_new))
            cell.rho_rear = float(float(getattr(cell, 'rho_rear', 0.0)) + rho_commit)
        cell.phi_rear_cand_w = float((1.0 - transfer) * cand_w)
        cell.rho_rear_cand = float((1.0 - transfer) * cand_rho)
        cell.rps_active = 1.0
        diag['commit_activated'] = float(diag.get('commit_activated', 0.0) + 1.0)
        diag['commit_transfer_sum'] = float(diag.get('commit_transfer_sum', 0.0) + transfer)
        diag['commit_w_sum'] = float(diag.get('commit_w_sum', 0.0) + w_commit)
        diag['commit_rho_sum'] = float(diag.get('commit_rho_sum', 0.0) + rho_commit)
    elif cell.rps_commit_score < off:
        cell.rps_active = float(active_prev * active_decay)
        diag['active_decay_hits'] = float(diag.get('active_decay_hits', 0.0) + 1.0)
    else:
        if quality_enable and active_prev > 1e-6:
            hold = float(np.clip(0.15 + 0.70 * quality_score + 0.15 * bank_agree, 0.0, 1.0))
            cell.rps_active = float(max(active_prev, hold))
        else:
            cell.rps_active = active_prev
