from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `csr_xmap`."""

def counterfactual_static_readout(
    self,
    cell: VoxelCell3D,
    geo_blend: float,
    geo_agree_min: float,
) -> Tuple[float, float, float, Dict[str, float]] | None:
    stats = self._ptdsf_state_stats(cell)
    ws = float(max(0.0, getattr(cell, "phi_static_w", 0.0)))
    wr = float(max(0.0, getattr(cell, "phi_rear_w", 0.0)))
    wp = float(max(0.0, getattr(cell, "phi_spg_w", 0.0)))
    wg = float(max(0.0, getattr(cell, "phi_geo_w", 0.0)))
    if ws <= 1e-12 and wr <= 1e-12 and wp <= 1e-12 and wg <= 1e-12:
        return None

    static_conf = float(np.clip(stats.get("static_conf", 0.0), 0.0, 1.0))
    dom = float(np.clip(stats.get("dominance", 0.0), 0.0, 1.0))
    rear_conf = float(np.clip(stats.get("rear_conf", 0.0), 0.0, 1.0))
    split_ratio = float(np.clip(stats.get("split_ratio", getattr(cell, "p_static", 0.5)), 0.0, 1.0))
    spg_score = float(np.clip(getattr(cell, "spg_score", 0.0), 0.0, 1.0))
    spg_active = float(np.clip(getattr(cell, "spg_active", 0.0), 0.0, 1.0))
    dyn_mix = float(
        np.clip(
            max(
                float(np.clip(getattr(cell, "dyn_prob", 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, "z_dyn", 0.0), 0.0, 1.0)),
                float(np.clip(stats.get("transient_conf", 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, "omhs_front_conf", 0.0), 0.0, 1.0)),
                float(np.clip(getattr(cell, "wod_front_conf", 0.0), 0.0, 1.0)),
                self._otv_conf(cell),
                self._otv_surface_conf(cell),
            ),
            0.0,
            1.0,
        )
    )
    vals: List[float] = []
    wts: List[float] = []
    bvals: List[float] = []
    bwts: List[float] = []

    if ws > 1e-12:
        w_s = float(ws * (0.24 + 0.76 * np.clip(0.55 * static_conf + 0.45 * dom, 0.0, 1.0)))
        vals.append(float(getattr(cell, "phi_static", 0.0)))
        wts.append(w_s)
        bvals.append(float(getattr(cell, "zcbf_bias", 0.0)))
        bwts.append(w_s)

    rear_enabled = bool(getattr(self.cfg.update, "rps_enable", False)) and wr > 1e-12
    if bool(getattr(self.cfg.update, "rps_hard_commit_enable", False)):
        rear_enabled = bool(
            rear_enabled
            and float(np.clip(getattr(cell, "rps_active", 0.0), 0.0, 1.0)) >= 0.5
            and float(np.clip(getattr(cell, "rps_commit_score", 0.0), 0.0, 1.0))
            >= float(getattr(self.cfg.update, "rps_commit_off", 0.40))
        )
    if rear_enabled:
        rear_mix = float(np.clip(0.55 * rear_conf + 0.25 * dom + 0.20 * split_ratio, 0.0, 1.0))
        w_r = float(wr * (0.20 + 0.80 * rear_mix))
        vals.append(float(getattr(cell, "phi_rear", 0.0)))
        wts.append(w_r)
        bvals.append(float(getattr(cell, "zcbf_bias", 0.0) + 0.5 * getattr(cell, "phi_geo_bias", 0.0)))
        bwts.append(w_r)

    spg_rho_ref = float(max(1e-6, getattr(self.cfg.update, "dual_state_static_protect_rho", 0.90)))
    spg_conf = float(np.clip(max(spg_score, spg_active, float(getattr(cell, "rho_spg", 0.0)) / spg_rho_ref), 0.0, 1.0))
    if wp > 1e-12:
        w_p = float(wp * (0.16 + 0.84 * spg_conf) * (0.30 + 0.70 * max(static_conf, dom)))
        vals.append(float(getattr(cell, "phi_spg", 0.0)))
        wts.append(w_p)
        bvals.append(float(getattr(cell, "zcbf_bias", 0.0) + 0.5 * getattr(cell, "phi_geo_bias", 0.0)))
        bwts.append(w_p)

    geo_agree = 0.0
    if wg > 1e-12:
        if vals:
            ref_phi = float(sum(w * v for w, v in zip(wts, vals)) / max(1e-9, sum(wts)))
        elif ws > 1e-12:
            ref_phi = float(getattr(cell, "phi_static", 0.0))
        else:
            ref_phi = float(getattr(cell, "phi_geo", 0.0))
        agree_ref = float(max(1e-6, 2.0 * self.voxel_size))
        geo_agree = float(np.exp(-0.5 * ((float(getattr(cell, "phi_geo", 0.0)) - ref_phi) / agree_ref) ** 2))
        dyn_guard = float(np.clip(1.0 - 0.78 * dyn_mix, 0.08, 1.0))
        anchor = float(np.clip(max(static_conf, dom, split_ratio), 0.0, 1.0))
        if geo_agree >= float(geo_agree_min) or not vals:
            w_g = float(
                wg
                * float(np.clip(geo_blend, 0.0, 1.0))
                * (0.18 + 0.82 * anchor)
                * (0.22 + 0.78 * geo_agree)
                * dyn_guard
            )
            if w_g > 1e-12:
                vals.append(float(getattr(cell, "phi_geo", 0.0)))
                wts.append(w_g)
                bvals.append(float(getattr(cell, "zcbf_bias", 0.0) + getattr(cell, "phi_geo_bias", 0.0)))
                bwts.append(w_g)

    w_sum = float(sum(wts))
    if w_sum <= 1e-12:
        return None
    phi_cf = float(sum(w * v for w, v in zip(wts, vals)) / w_sum)
    bias_cf = float(sum(w * b for w, b in zip(bwts, bvals)) / max(1e-9, sum(bwts))) if bwts else 0.0
    score = float(
        np.clip(
            0.34 * dom
            + 0.26 * static_conf
            + 0.14 * rear_conf
            + 0.10 * spg_conf
            + 0.10 * geo_agree
            + 0.08 * split_ratio
            - 0.12 * dyn_mix,
            0.0,
            1.0,
        )
    )
    out = dict(stats)
    out.update(
        {
            "score": score,
            "geo_agree": float(np.clip(geo_agree, 0.0, 1.0)),
            "spg_conf": spg_conf,
            "dyn_mix": dyn_mix,
        }
    )
    return phi_cf, float(min(5000.0, w_sum)), bias_cf, out

def dynamic_exclusion_readout(self, cell: VoxelCell3D) -> Tuple[float, float, float, Dict[str, float]] | None:
    stats = self._ptdsf_state_stats(cell)
    wt = float(max(0.0, getattr(cell, "phi_transient_w", 0.0)))
    wd = float(max(0.0, getattr(cell, "phi_dyn_w", 0.0)))
    wo = float(max(0.0, getattr(cell, "phi_otv_w", 0.0)))
    if wt <= 1e-12 and wd <= 1e-12 and wo <= 1e-12:
        return None

    transient_conf = float(np.clip(stats.get("transient_conf", 0.0), 0.0, 1.0))
    dyn_prob = float(np.clip(getattr(cell, "dyn_prob", 0.0), 0.0, 1.0))
    z_dyn = float(np.clip(getattr(cell, "z_dyn", 0.0), 0.0, 1.0))
    omhs_front = float(np.clip(getattr(cell, "omhs_front_conf", 0.0), 0.0, 1.0))
    wod_front = float(np.clip(getattr(cell, "wod_front_conf", 0.0), 0.0, 1.0))
    otv_conf = float(np.clip(self._otv_conf(cell), 0.0, 1.0))
    otv_geom = float(np.clip(self._otv_surface_conf(cell), 0.0, 1.0))
    contra = float(
        np.clip(
            0.45 * float(np.clip(getattr(cell, "visibility_contradiction", 0.0), 0.0, 1.0))
            + 0.35 * float(np.clip(getattr(cell, "stcg_score", 0.0), 0.0, 1.0))
            + 0.20 * float(np.clip(getattr(cell, "residual_evidence", 0.0), 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    dccm_commit = float(np.clip(getattr(cell, "dccm_commit", 0.0), 0.0, 1.0))
    vals: List[float] = []
    wts: List[float] = []

    if wt > 1e-12:
        w_t = float(wt * (0.18 + 0.82 * max(transient_conf, dyn_prob, z_dyn, omhs_front, wod_front)))
        vals.append(float(getattr(cell, "phi_transient", 0.0)))
        wts.append(w_t)
    if wd > 1e-12:
        w_d = float(wd * (0.18 + 0.82 * max(dyn_prob, z_dyn, transient_conf, contra)))
        vals.append(float(getattr(cell, "phi_dyn", 0.0)))
        wts.append(w_d)
    if wo > 1e-12:
        w_o = float(wo * (0.20 + 0.80 * max(otv_conf, otv_geom, transient_conf)))
        vals.append(float(getattr(cell, "phi_otv", 0.0)))
        wts.append(w_o)

    w_sum = float(sum(wts))
    if w_sum <= 1e-12:
        return None
    phi_x = float(sum(w * v for w, v in zip(wts, vals)) / w_sum)
    score = float(
        np.clip(
            0.24 * max(transient_conf, dyn_prob)
            + 0.18 * z_dyn
            + 0.16 * otv_geom
            + 0.12 * otv_conf
            + 0.12 * omhs_front
            + 0.08 * wod_front
            + 0.06 * contra
            + 0.04 * dccm_commit,
            0.0,
            1.0,
        )
    )
    out = dict(stats)
    out.update(
        {
            "score": score,
            "contra": contra,
            "otv_geom": otv_geom,
            "front_conf": float(np.clip(max(omhs_front, wod_front), 0.0, 1.0)),
        }
    )
    return phi_x, float(min(5000.0, w_sum)), score, out

