from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `ptdsf`."""

def write_time_dual_surface_targets(
    self,
    voxel_map: VoxelHashMap3D,
    measurement: AssocMeasurement3D,
    rel: np.ndarray,
    d_signed: float,
    wod_front: float,
    wod_rear: float,
    wod_shell: float,
    q_dyn_obs: float,
) -> tuple[float, float, float]:
    if not (bool(self.cfg.update.wdsg_enable) and bool(self.cfg.update.wod_enable) and bool(self.cfg.update.dual_state_enable)):
        return d_signed, d_signed, d_signed
    n = self._normalize(np.asarray(measurement.normal_world, dtype=float))
    if float(np.linalg.norm(n)) < 1e-8:
        return d_signed, d_signed, d_signed
    _view_axis, _view_signed, proj = self._measurement_view_components(measurement, rel, n)
    proj_floor = float(max(0.05, getattr(self.cfg.update, 'wdsg_proj_floor', 0.35)))
    if abs(proj) < proj_floor:
        proj_eff = (-proj_floor if proj <= 0.0 else proj_floor)
    else:
        proj_eff = proj
    voxel = float(max(1e-6, voxel_map.voxel_size))
    max_shift_vox = float(max(0.2, getattr(self.cfg.update, 'wdsg_max_shift_vox', 1.8)))
    front_delta_vox = float(np.clip(
        getattr(self.cfg.update, 'wdsg_front_shift_vox', 0.9) * wod_front
        + 0.5 * getattr(self.cfg.update, 'wdsg_shell_shift_vox', 0.4) * wod_shell
        + 0.15 * q_dyn_obs,
        0.0,
        max_shift_vox,
    ))
    rear_delta_vox = float(np.clip(
        getattr(self.cfg.update, 'wdsg_rear_shift_vox', 1.1) * wod_rear
        + 0.75 * getattr(self.cfg.update, 'wdsg_shell_shift_vox', 0.4) * wod_shell
        + 0.12 * (1.0 - float(np.clip(q_dyn_obs, 0.0, 1.0))),
        0.0,
        max_shift_vox,
    ))
    front_delta = float(voxel * front_delta_vox)
    rear_delta = float(voxel * rear_delta_vox)
    d_front = float(d_signed + front_delta * proj_eff)
    d_rear = float(d_signed - rear_delta * proj_eff)
    front_mix = float(np.clip(getattr(self.cfg.update, 'wdsg_front_mix_gain', 0.95) * (wod_front + 0.5 * wod_shell), 0.0, 0.95))
    rear_mix = float(np.clip(getattr(self.cfg.update, 'wdsg_rear_mix_gain', 1.0) * (wod_rear + 0.5 * wod_shell), 0.0, 0.95))
    d_static = float((1.0 - rear_mix) * d_signed + rear_mix * d_rear)
    d_transient = float((1.0 - front_mix) * d_signed + front_mix * d_front)
    return d_static, d_transient, d_rear

def ptdsf_state_stats(self, cell: VoxelCell3D) -> Dict[str, float]:
    ps = float(np.clip(cell.p_static, 0.0, 1.0))
    rho_s = float(max(0.0, getattr(cell, "rho_static", 0.0)))
    rho_t = float(max(0.0, getattr(cell, "rho_transient", 0.0)))
    rho_r_raw = float(max(0.0, getattr(cell, "rho_rear", 0.0)))
    if bool(getattr(self.cfg.update, "rps_hard_commit_enable", False)) and float(np.clip(getattr(cell, "rps_active", 0.0), 0.0, 1.0)) < 0.5:
        rho_r = 0.0
    else:
        rho_r = rho_r_raw
    rho_sum = float(rho_s + rho_t)
    split_ratio = float(np.clip(rho_s / max(1e-6, rho_sum), 0.0, 1.0)) if rho_sum > 1e-8 else ps
    age_ref = float(max(1.0, self.cfg.update.ptdsf_commit_age_ref))
    commit_n = float(np.clip(getattr(cell, "ptdsf_commit_age", 0.0) / age_ref, 0.0, 1.0))
    rollback_n = float(np.clip(getattr(cell, "ptdsf_rollback_age", 0.0) / age_ref, 0.0, 1.0))
    surf = float(max(1e-6, cell.surf_evidence))
    free = float(max(0.0, cell.free_evidence))
    occ_frac = float(np.clip(surf / max(1e-6, surf + free), 0.0, 1.0))
    dyn_n = float(np.clip(max(float(cell.dyn_prob), float(getattr(cell, "z_dyn", 0.0))), 0.0, 1.0))
    otv_conf = float(max(self._otv_conf(cell), 0.85 * self._otv_surface_conf(cell)))
    xmem_conf = float(self._xmem_conf(cell))
    obl_conf = float(self._obl_conf(cell))
    cmct_conf = float(self._cmct_conf(cell))
    cgcc_conf = float(self._cgcc_conf(cell))
    pfv_conf = float(self._pfv_conf(cell))
    rear_conf = float(np.clip(rho_r / max(1e-6, getattr(self.cfg.update, "rps_rho_ref", self.cfg.update.dual_state_static_protect_rho)), 0.0, 1.5))
    static_conf = float(
        np.clip(
            0.24 * ps
            + 0.24 * split_ratio
            + 0.16 * commit_n
            + 0.10 * occ_frac
            + 0.08 * (1.0 - dyn_n)
            + 0.08 * min(1.0, rear_conf)
            + 0.16 * obl_conf
            - 0.18 * rollback_n
            - 0.08 * otv_conf
            - 0.08 * xmem_conf
            - 0.18 * cmct_conf
            - 0.22 * cgcc_conf
            - 0.26 * pfv_conf,
            0.0,
            1.0,
        )
    )
    transient_conf = float(
        np.clip(
            0.40 * (1.0 - split_ratio)
            + 0.22 * (1.0 - ps)
            + 0.18 * rollback_n
            + 0.14 * dyn_n
            + 0.14 * otv_conf
            + 0.16 * xmem_conf
            - 0.16 * obl_conf
            + 0.20 * cmct_conf
            + 0.24 * cgcc_conf
            + 0.28 * pfv_conf,
            0.0,
            1.0,
        )
    )
    dominance = float(
        np.clip(
            0.46 * static_conf
            + 0.18 * commit_n
            + 0.14 * occ_frac
            + 0.08 * min(1.0, rear_conf)
            + 0.18 * obl_conf
            - 0.32 * transient_conf
            - 0.10 * otv_conf
            - 0.12 * xmem_conf
            - 0.14 * cmct_conf,
            0.0,
            1.0,
        )
    )
    return {
        "p_static": ps,
        "rho_static": rho_s,
        "rho_transient": rho_t,
        "rho_rear": rho_r,
        "split_ratio": split_ratio,
        "commit_n": commit_n,
        "rollback_n": rollback_n,
        "occ_frac": occ_frac,
        "static_conf": static_conf,
        "transient_conf": transient_conf,
        "dominance": dominance,
        "rear_conf": float(np.clip(rear_conf, 0.0, 1.0)),
        "otv_conf": otv_conf,
        "xmem_conf": xmem_conf,
        "obl_conf": obl_conf,
        "cmct_conf": cmct_conf,
        "cgcc_conf": cgcc_conf,
        "pfv_conf": pfv_conf,
    }

def persistent_surface_readout(self, cell: VoxelCell3D) -> Tuple[float, float, float, Dict[str, float]] | None:
    stats = self._ptdsf_state_stats(cell)
    ws = float(max(0.0, cell.phi_static_w))
    wg = float(max(0.0, cell.phi_geo_w))
    wt = float(max(0.0, cell.phi_transient_w))
    wr = float(max(0.0, getattr(cell, "phi_rear_w", 0.0)))
    wp = float(max(0.0, getattr(cell, "phi_spg_w", 0.0)))
    wb = float(max(0.0, getattr(cell, "phi_bg_w", 0.0)))
    dom = float(stats["dominance"])
    split_ratio = float(stats["split_ratio"])
    commit_n = float(stats["commit_n"])
    static_conf = float(stats["static_conf"])
    transient_conf = float(stats["transient_conf"])
    rear_conf = float(np.clip(stats.get("rear_conf", 0.0), 0.0, 1.0))
    obl_conf = float(np.clip(stats.get("obl_conf", self._obl_conf(cell)), 0.0, 1.0))
    otv_conf = float(np.clip(stats.get("otv_conf", self._otv_conf(cell)), 0.0, 1.0))
    otv_geom = float(np.clip(self._otv_surface_conf(cell), 0.0, 1.0))
    omhs_on = float(np.clip(getattr(cell, "omhs_active", 0.0), 0.0, 1.0))
    omhs_front = float(np.clip(getattr(cell, "omhs_front_conf", 0.0), 0.0, 1.0))
    omhs_rear = float(np.clip(getattr(cell, "omhs_rear_conf", 0.0), 0.0, 1.0))
    if ws <= 1e-12 and wg <= 1e-12 and wr <= 1e-12 and wp <= 1e-12 and wb <= 1e-12:
        return None

    # PT-DSF persistent readout: let persistent geometry dominate the exported
    # surface and only admit transient leakage when the persistent branch is weak.
    anchor = float(np.clip(0.46 * dom + 0.22 * static_conf + 0.14 * commit_n + 0.12 * rear_conf + 0.16 * omhs_on * omhs_rear, 0.0, 1.0))
    support = float(np.clip(0.45 * split_ratio + 0.25 * static_conf + 0.15 * rear_conf + 0.15 * omhs_rear, 0.0, 1.0))
    vals: List[float] = []
    wts: List[float] = []
    bvals: List[float] = []
    bwts: List[float] = []
    geo_agree = 1.0
    if ws > 1e-12:
        w_s = float(ws * (1.05 + 0.95 * anchor + 0.28 * commit_n + 0.22 * omhs_on * omhs_rear))
        vals.append(float(cell.phi_static))
        wts.append(w_s)
        bvals.append(float(getattr(cell, "zcbf_bias", 0.0)))
        bwts.append(w_s)
    geo_export_direct = not bool(getattr(self.cfg.update, "spg_enable", False))
    if wg > 1e-12:
        if ws > 1e-12:
            div_ref = float(max(1e-6, 2.0 * self.voxel_size))
            div = float(abs(float(cell.phi_geo) - float(cell.phi_static)))
            geo_agree = float(np.exp(-0.5 * (div / div_ref) * (div / div_ref)))
        else:
            geo_agree = 0.88
        if geo_export_direct or (ws <= 1e-12 and wp <= 1e-12):
            w_g = float(wg * (0.08 + 0.72 * anchor + 0.20 * support) * (0.25 + 0.75 * geo_agree) * (1.0 + 0.12 * omhs_on * omhs_rear))
            vals.append(float(cell.phi_geo))
            wts.append(w_g)
            bvals.append(float(getattr(cell, "zcbf_bias", 0.0) + cell.phi_geo_bias))
            bwts.append(w_g)

    front_weight = float(sum(wts))
    front_phi = float(sum(w * v for w, v in zip(wts, vals)) / front_weight) if front_weight > 1e-12 else 0.0
    front_bias = float(sum(w * b for w, b in zip(bwts, bvals)) / max(1e-9, sum(bwts))) if bwts else 0.0
    front_score = float(
        np.clip(
            0.40 * anchor
            + 0.22 * static_conf
            + 0.14 * support
            + 0.08 * geo_agree
            + 0.10 * obl_conf
            + 0.06 * (1.0 - transient_conf)
            - 0.08 * otv_conf,
            0.0,
            1.0,
        )
    )
    bg_phi = float(getattr(cell, "phi_bg", 0.0))
    bg_bias = float(getattr(cell, "zcbf_bias", 0.0))
    bg_weight_base = float(0.0)
    if wb > 1e-12:
        bg_weight_base = float(wb * (0.20 + float(getattr(self.cfg.update, "obl_extract_gain", 1.10)) * np.clip(0.55 * obl_conf + 0.25 * rear_conf + 0.20 * commit_n, 0.0, 1.0)))

    front_dyn_base = float(
        np.clip(
            max(
                transient_conf,
                omhs_front,
                float(np.clip(getattr(cell, "wod_front_conf", 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, "dyn_prob", 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, "z_dyn", 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, "st_mem", 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, "visibility_contradiction", 0.0), 0.0, 1.0)),
                float(np.clip(stats.get("xmem_conf", self._xmem_conf(cell)), 0.0, 1.0)),
                otv_conf,
                otv_geom,
            ),
            0.0,
            1.0,
        )
    )
    bg_hard_select = False
    if bg_weight_base > 1e-12:
        bg_boost = float(1.0 + float(getattr(self.cfg.update, "obl_extract_gain", 1.10)) * obl_conf * (0.40 + 0.60 * front_dyn_base))
        bg_weight = float(bg_weight_base * bg_boost)
        bg_hard_select = bool(
            obl_conf >= 0.42
            and front_dyn_base >= 0.28
            and bg_weight >= max(1e-6, 0.45 * max(front_weight, 1e-6))
        )
        if bg_hard_select:
            phi_p = float(bg_phi)
            bias_p = float(bg_bias)
            w_sum = float(max(bg_weight, 0.45 * wb))
        else:
            front_keep = float(np.clip(1.0 - 0.45 * obl_conf * front_dyn_base, 0.20, 1.0))
            front_weight = float(front_weight * front_keep)
            total_w = float(front_weight + bg_weight)
            if total_w > 1e-12:
                phi_p = float((front_weight * front_phi + bg_weight * bg_phi) / total_w)
                bias_p = float((front_weight * front_bias + bg_weight * bg_bias) / total_w)
                w_sum = float(total_w)
            else:
                phi_p = float(bg_phi)
                bias_p = float(bg_bias)
                w_sum = float(bg_weight)
    else:
        phi_p = front_phi
        bias_p = front_bias
        w_sum = front_weight
    out_stats = dict(stats)
    out_stats.update(
        {
            "front_score": front_score,
            "rear_score": 0.0,
            "rear_sep": 0.0,
            "bank_selected": "background_hard" if bg_hard_select else ("background" if bg_weight_base > max(1e-12, front_weight) else "front"),
            "otv_geom": otv_geom,
            "obl_conf": obl_conf,
        }
    )

    if bool(getattr(self.cfg.update, "spg_enable", False)):
        spg_score = float(np.clip(getattr(cell, "spg_score", 0.0), 0.0, 1.0))
        spg_active = float(np.clip(getattr(cell, "spg_active", 0.0), 0.0, 1.0))
        promote_conf = float(max(spg_score, spg_active))
        provisional = float(np.clip(front_dyn_base * (1.0 - promote_conf), 0.0, 1.0))
        if provisional > 1e-6 and w_sum > 1e-12:
            keep = float(np.clip(1.0 - 0.65 * provisional, 0.18, 1.0))
            w_sum *= keep
            front_weight = w_sum
            front_score = float(np.clip(front_score * (0.70 + 0.30 * keep), 0.0, 1.0))
        if provisional >= 0.42 and ws > 1e-12 and promote_conf < 0.55:
            static_only_w = float(ws * (0.38 + 0.62 * anchor) * (1.0 - 0.25 * float(np.clip(getattr(cell, "wod_shell_conf", 0.0), 0.0, 1.0))))
            if static_only_w > 1e-12:
                phi_p = float(cell.phi_static)
                bias_p = float(getattr(cell, "zcbf_bias", 0.0))
                w_sum = float(static_only_w)
                front_weight = w_sum
                front_phi = phi_p
                front_bias = bias_p
                front_score = float(np.clip(0.60 * front_score + 0.24 * anchor + 0.16 * static_conf - 0.28 * provisional, 0.0, 1.0))
                out_stats["bank_selected"] = "front_static_fallback"
        if wp > 1e-12:
            cons_ref_spg = float(max(1e-6, getattr(self.cfg.update, "rps_consistency_ref", 0.03)))
            spg_agree = float(np.exp(-0.5 * ((float(getattr(cell, "phi_spg", 0.0)) - front_phi) / cons_ref_spg) ** 2)) if front_weight > 1e-12 else 0.88
            spg_rho_ref = float(max(1e-6, getattr(self.cfg.update, "dual_state_static_protect_rho", 0.90)))
            spg_conf = float(np.clip(float(getattr(cell, "rho_spg", 0.0)) / spg_rho_ref, 0.0, 1.5))
            w_spg = float(
                wp
                * (0.18 + float(getattr(self.cfg.update, "spg_read_weight_gain", 0.85)) * np.clip(0.40 * min(1.0, spg_conf) + 0.30 * spg_score + 0.30 * spg_agree, 0.0, 1.0))
            )
            if spg_active >= 0.5 or spg_score >= float(getattr(self.cfg.update, "spg_commit_off", 0.40)):
                cand_mix = float(
                    np.clip(
                        float(getattr(self.cfg.update, "spg_candidate_mix", 0.22)) * spg_agree * (1.0 - 0.65 * front_dyn_base),
                        0.0,
                        0.45,
                    )
                )
                w_front_eff = float(cand_mix * max(0.0, front_weight))
                spg_bias = float(getattr(cell, "zcbf_bias", 0.0) + 0.5 * cell.phi_geo_bias)
                denom = float(max(1e-9, w_spg + w_front_eff))
                phi_p = float((w_spg * float(getattr(cell, "phi_spg", 0.0)) + w_front_eff * front_phi) / denom)
                bias_p = float((w_spg * spg_bias + w_front_eff * front_bias) / denom)
                w_sum = float(denom)
                front_score = float(np.clip(0.45 * front_score + 0.35 * spg_score + 0.20 * min(1.0, spg_conf) - 0.10 * front_dyn_base, 0.0, 1.0))
                out_stats["bank_selected"] = "front_spg"
            out_stats["spg_score"] = spg_score
            out_stats["spg_active"] = spg_active
            out_stats["spg_provisional"] = provisional

    rear_enabled = bool(getattr(self.cfg.update, "rps_enable", False)) and wr > 1e-12
    if bool(getattr(self.cfg.update, "rps_hard_commit_enable", False)):
        rear_enabled = bool(
            rear_enabled
            and float(np.clip(getattr(cell, "rps_active", 0.0), 0.0, 1.0)) >= 0.5
            and float(np.clip(getattr(cell, "rps_commit_score", 0.0), 0.0, 1.0))
            >= float(getattr(self.cfg.update, "rps_commit_off", 0.40))
        )
    if rear_enabled:
        cons_ref = float(max(1e-6, getattr(self.cfg.update, "rps_consistency_ref", 0.03)))
        ref_phi = front_phi if front_weight > 1e-12 else float(cell.phi_rear)
        rear_agree = float(np.exp(-0.5 * ((float(cell.phi_rear) - ref_phi) / cons_ref) ** 2))
        front_guard = float(
            np.clip(
                1.0
                - 0.35 * omhs_front
                - 0.25 * float(np.clip(getattr(cell, "wod_front_conf", 0.0), 0.0, 1.0))
                - 0.15 * float(np.clip(getattr(cell, "wod_shell_conf", 0.0), 0.0, 1.0)),
                0.15,
                1.0,
            )
        )
        rear_mix = float(np.clip(0.55 * rear_conf + 0.25 * support + 0.20 * anchor, 0.0, 1.0))
        w_r = float(
            wr
            * (0.18 + float(getattr(self.cfg.update, "rps_read_weight_gain", 0.70)) * rear_mix)
            * (0.30 + 0.70 * rear_agree)
            * front_guard
        )
        rear_phi = float(cell.phi_rear)
        rear_bias = float(getattr(cell, "zcbf_bias", 0.0) + 0.5 * cell.phi_geo_bias)
        bank_mode = bool(
            getattr(self.cfg.update, "rps_surface_bank_enable", False)
            and getattr(self.cfg.update, "rps_hard_commit_enable", False)
        )
        front_dyn = front_dyn_base
        sep_ref = float(max(self.voxel_size, getattr(self.cfg.update, "rps_bank_separation_ref", 0.04)))
        rear_sep = float(np.clip((abs(rear_phi - ref_phi) - 0.5 * sep_ref) / max(1e-6, 1.5 * sep_ref), 0.0, 1.0))
        rear_score = float(
            np.clip(
                0.30 * rear_conf
                + 0.22 * float(np.clip(getattr(cell, "rps_commit_score", 0.0), 0.0, 1.0))
                + 0.14 * float(np.clip(getattr(cell, "rps_active", 0.0), 0.0, 1.0))
                + 0.14 * support
                + 0.10 * rear_sep
                + 0.10 * front_dyn,
                0.0,
                1.0,
            )
        )
        front_score_eff = float(
            np.clip(
                front_score
                - 0.18 * rear_sep
                - 0.18 * front_dyn
                + 0.06 * (1.0 - rear_conf),
                0.0,
                1.0,
            )
        )
        out_stats.update(
            {
                "front_score": front_score_eff,
                "rear_score": rear_score,
                "rear_sep": rear_sep,
            }
        )
        if bank_mode:
            margin = float(np.clip(getattr(self.cfg.update, "rps_bank_margin", 0.08), 0.0, 0.5))
            rear_min = float(np.clip(getattr(self.cfg.update, "rps_bank_rear_min_score", 0.52), 0.0, 1.0))
            rear_win = bool(
                (front_weight <= 1e-12 and w_r > 1e-12)
                or (
                    w_r > 1e-12
                    and rear_sep >= 0.22
                    and rear_score >= max(rear_min, front_score_eff + margin)
                )
            )
            if rear_win:
                phi_p = rear_phi
                bias_p = rear_bias
                w_sum = float(max(w_r, 0.45 * wr))
                out_stats["bank_selected"] = "rear"
            elif front_weight <= 1e-12 and w_r > 1e-12:
                phi_p = rear_phi
                bias_p = rear_bias
                w_sum = float(max(w_r, 0.45 * wr))
                out_stats["bank_selected"] = "rear_fallback"
        else:
            vals.append(rear_phi)
            wts.append(w_r)
            bvals.append(rear_bias)
            bwts.append(w_r)
            w_sum = float(sum(wts))
            if w_sum <= 1e-12:
                return None
            phi_p = float(sum(w * v for w, v in zip(wts, vals)) / w_sum)
            bias_p = float(sum(w * b for w, b in zip(bwts, bvals)) / max(1e-9, sum(bwts)))

    if w_sum <= 1e-12:
        return None
    if wt > 1e-12:
        bank_selected = str(out_stats.get("bank_selected", "front"))
        rear_support = float(np.clip(max(omhs_on * omhs_rear, rear_conf, 1.0 if bank_selected.startswith("rear") else 0.0), 0.0, 1.0))
        front_drive = float(np.clip(max(transient_conf, omhs_front, float(np.clip(getattr(cell, "wod_front_conf", 0.0), 0.0, 1.0))), 0.0, 1.0))
        if bank_selected.startswith("rear"):
            front_drive *= 0.35
        transient_guard = float(
            np.clip(
                1.0 - 0.75 * anchor - 0.20 * static_conf - 0.30 * rear_support + 0.10 * omhs_on * omhs_front * (1.0 - omhs_rear),
                0.0,
                1.0,
            )
        )
        weak_persistent = bool((ws <= 1e-12 and wg <= 1e-12 and wr <= 1e-12) or (anchor < 0.45 and support < 0.58 and rear_support < 0.35))
        disagree_case = bool(geo_agree < 0.55 and support < 0.50 and rear_support < 0.45)
        if weak_persistent or disagree_case:
            leak_cap = 0.08 if weak_persistent else 0.05
            leak_ratio = float(
                np.clip(
                    0.005 + leak_cap * transient_guard * front_drive * (1.0 - 0.65 * rear_support),
                    0.0,
                    leak_cap,
                )
            )
            if leak_ratio > 1e-6:
                w_leak = float(wt * leak_ratio)
                phi_p = float((w_sum * phi_p + w_leak * float(cell.phi_transient)) / max(1e-9, w_sum + w_leak))
                w_sum += w_leak
    return phi_p, float(min(5000.0, w_sum)), bias_p, out_stats

