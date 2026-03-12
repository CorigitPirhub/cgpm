from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np

"""Auto-extracted P10 method helpers for `zcbf`."""

def local_zero_crossing_debias(
    self,
    voxel_map: VoxelHashMap3D,
    touched: Iterable[VoxelIndex],
    frame_id: int,
) -> None:
    if not bool(self.cfg.update.lzcd_enable):
        return
    interval = max(1, int(self.cfg.update.lzcd_interval))
    if (int(frame_id) % interval) != 0:
        return

    radius = max(1, int(self.cfg.update.lzcd_radius_cells))
    min_n = max(3, int(self.cfg.update.lzcd_min_neighbors))
    min_w = float(max(0.0, self.cfg.update.lzcd_min_phi_w))
    min_rho = float(max(0.0, self.cfg.update.lzcd_min_rho))
    max_d = float(np.clip(self.cfg.update.lzcd_max_dscore, 0.0, 1.0))
    cos_min = float(np.clip(self.cfg.update.lzcd_normal_cos_min, 0.0, 1.0))
    phi_gate = float(max(1e-4, self.cfg.update.lzcd_neighbor_phi_gate))
    bias_alpha = float(np.clip(self.cfg.update.lzcd_bias_alpha, 0.01, 0.8))
    corr_gain = float(np.clip(self.cfg.update.lzcd_correction_gain, 0.0, 1.0))
    max_bias = float(max(1e-4, self.cfg.update.lzcd_max_bias))
    max_step = float(max(1e-4, self.cfg.update.lzcd_max_step))
    trim_q = float(np.clip(self.cfg.update.lzcd_trim_quantile, 0.55, 1.0))
    use_geo = bool(self.cfg.update.lzcd_use_geo_channel)
    dual_enable = bool(self.cfg.update.dual_state_enable)
    rho_ref = float(max(1e-6, self.cfg.update.dual_state_static_protect_rho))
    global_alpha = float(np.clip(0.5 * bias_alpha, 0.02, 0.35))
    local_mix = 0.82
    global_mix = 0.18
    min_improve = 0.995
    solver_iters = int(max(1, self.cfg.update.lzcd_solver_iters))
    solver_lambda = float(max(0.0, self.cfg.update.lzcd_solver_lambda_smooth))
    solver_step = float(np.clip(self.cfg.update.lzcd_solver_step, 0.10, 1.0))
    solver_tol = float(max(1e-6, self.cfg.update.lzcd_solver_tol))
    residual_anchor_w = float(np.clip(self.cfg.update.lzcd_residual_anchor_weight, 0.0, 0.95))
    residual_hit_ref = float(max(1.0, self.cfg.update.lzcd_residual_hit_ref))
    affine_enable = bool(self.cfg.update.lzcd_affine_enable)
    affine_mix = float(np.clip(self.cfg.update.lzcd_affine_mix, 0.0, 1.0))
    affine_slope_min = float(min(self.cfg.update.lzcd_affine_slope_min, self.cfg.update.lzcd_affine_slope_max))
    affine_slope_max = float(max(self.cfg.update.lzcd_affine_slope_min, self.cfg.update.lzcd_affine_slope_max))
    affine_min_samples = int(max(4, self.cfg.update.lzcd_affine_min_samples))

    touched_ext: Set[VoxelIndex] = set()
    for idx in touched:
        touched_ext.add(idx)
        touched_ext.update(voxel_map.neighbor_indices(idx, radius))

    records = []
    global_vals: List[float] = []
    global_w: List[float] = []
    for idx in touched_ext:
        c = voxel_map.get_cell(idx)
        if c is None:
            continue
        if c.rho < min_rho:
            continue
        if float(np.clip(c.dyn_prob, 0.0, 1.0)) > max_d:
            continue
        if use_geo and c.phi_geo_w >= min_w:
            phi_i = float(c.phi_geo)
            w_i = float(c.phi_geo_w)
        elif dual_enable and c.phi_static_w >= min_w:
            phi_i = float(c.phi_static)
            w_i = float(c.phi_static_w)
        else:
            phi_i = float(c.phi)
            w_i = float(c.phi_w)
        if w_i < min_w:
            continue
        n_i = self._normalize(np.asarray(c.g_mean, dtype=float))
        if float(np.linalg.norm(n_i)) < 1e-8:
            continue
        center_i = voxel_map.index_to_center(idx)
        est_vals: List[float] = []
        est_w: List[float] = []
        est_delta: List[float] = []
        est_phi: List[float] = []
        for nidx in voxel_map.neighbor_indices(idx, radius):
            if nidx == idx:
                continue
            cj = voxel_map.get_cell(nidx)
            if cj is None:
                continue
            if use_geo and cj.phi_geo_w >= min_w:
                phi_j = float(cj.phi_geo)
                w_j = float(cj.phi_geo_w)
            elif dual_enable and cj.phi_static_w >= min_w:
                phi_j = float(cj.phi_static)
                w_j = float(cj.phi_static_w)
            else:
                phi_j = float(cj.phi)
                w_j = float(cj.phi_w)
            if w_j < min_w:
                continue
            if abs(phi_j) > phi_gate:
                continue
            n_j = self._normalize(np.asarray(cj.g_mean, dtype=float))
            if float(np.linalg.norm(n_j)) < 1e-8:
                continue
            cos_ij = float(abs(np.dot(n_i, n_j)))
            if cos_ij < cos_min:
                continue
            center_j = voxel_map.index_to_center(nidx)
            delta_ij = float(np.dot(n_i, center_j - center_i))
            phi_est = float(phi_j - delta_ij)
            # Confidence uses neighbor weight/rho and normal consistency.
            w = float(min(w_j, 5.0) * (0.2 + 0.8 * cos_ij) * (0.35 + 0.65 * np.clip(cj.rho, 0.0, 1.0)))
            est_vals.append(phi_est)
            est_w.append(w)
            est_delta.append(delta_ij)
            est_phi.append(phi_j)
        if len(est_vals) < min_n:
            continue
        vals = np.asarray(est_vals, dtype=float)
        ws = np.asarray(est_w, dtype=float)
        deltas = np.asarray(est_delta, dtype=float)
        phi_obs = np.asarray(est_phi, dtype=float)
        phi_ref0 = self._weighted_median(vals, ws)
        abs_res = np.abs(vals - phi_ref0)
        if trim_q < 0.999:
            thr = float(np.quantile(abs_res, trim_q))
            mask = abs_res <= max(1e-6, thr)
            if int(np.count_nonzero(mask)) >= min_n:
                vals = vals[mask]
                ws = ws[mask]
                abs_res = abs_res[mask]
                deltas = deltas[mask]
                phi_obs = phi_obs[mask]
        # Huber-like attenuation for outliers.
        scale = float(max(1e-4, 1.4826 * np.median(abs_res)))
        huber = 1.0 / (1.0 + (abs_res / scale) ** 2)
        ws = ws * huber
        phi_ref_med = self._weighted_median(vals, ws)

        # LZCD affine local debias:
        # robustly fit phi ~= a * <n, delta_x> + b and use intercept b as
        # local zero-level estimate. This addresses local signed-distance bias
        # without coupling to dynamic suppression states.
        phi_ref = float(phi_ref_med)
        slope_n = 1.0
        if affine_enable and vals.size >= affine_min_samples:
            x = np.column_stack((deltas, np.ones_like(deltas)))
            y = phi_obs
            wfit = np.clip(ws, 1e-8, None)
            wx = x * wfit[:, None]
            # Mild slope prior around 1.0 to avoid degenerate fits in tiny neighborhoods.
            lam = 0.03
            a = wx.T @ x + np.array([[lam, 0.0], [0.0, 1e-6]], dtype=float)
            b = wx.T @ y + np.array([lam * 1.0, 0.0], dtype=float)
            try:
                sol = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                sol = np.array([1.0, phi_ref_med], dtype=float)
            slope = float(np.clip(sol[0], affine_slope_min, affine_slope_max))
            intercept = float(np.clip(sol[1], -max_bias, max_bias))
            slope_dev = abs(slope - 1.0)
            slope_n = float(np.clip(1.0 - slope_dev / max(1e-6, (affine_slope_max - affine_slope_min)), 0.0, 1.0))
            # Static-confidence gated affine mixing:
            # reduce affine influence in dynamic / low-confidence cells to avoid
            # over-correction in high-motion regions (Bonn-like dynamics).
            dyn_n = float(np.clip(c.dyn_prob, 0.0, 1.0))
            rho_n_aff = float(np.clip(c.rho / rho_ref, 0.0, 1.2))
            static_conf = float(np.clip(0.55 * rho_n_aff + 0.45 * (1.0 - dyn_n), 0.15, 1.0))
            mix_eff = float(affine_mix * (0.35 + 0.65 * slope_n) * static_conf)
            phi_ref = float(np.clip((1.0 - mix_eff) * phi_ref_med + mix_eff * intercept, -max_bias, max_bias))
        bias_local = float(np.clip(phi_i - phi_ref, -max_bias, max_bias))
        # Residual anchor term (association signed residual history on source voxels).
        # This is geometry-only and does not consume dynamic scores.
        res_hits_n = float(np.clip(c.geo_res_hits / residual_hit_ref, 0.0, 1.0))
        res_anchor = float(np.clip(c.geo_res_ema, -max_bias, max_bias))
        wa = float(np.clip(residual_anchor_w * res_hits_n, 0.0, 0.95))
        bias_obs = float(np.clip((1.0 - wa) * bias_local + wa * res_anchor, -max_bias, max_bias))
        rho_n = float(np.clip(c.rho / rho_ref, 0.0, 1.2))
        w_n = float(np.clip(w_i / max(1e-6, 2.0 * min_w), 0.0, 1.2))
        geo_conf = float(np.clip((0.55 * w_n + 0.45 * rho_n) * (0.82 + 0.18 * slope_n), 0.25, 1.0))
        use_geo_cell = bool(use_geo and c.phi_geo_w >= min_w)
        dual_static_cell = bool(dual_enable and c.phi_static_w >= min_w)
        rec = {
            "idx": idx,
            "cell": c,
            "phi_i": phi_i,
            "vals": vals,
            "ws": ws,
            "bias_obs": bias_obs,
            "geo_conf": geo_conf,
            "use_geo_cell": use_geo_cell,
            "dual_static_cell": dual_static_cell,
            "n_i": n_i,
        }
        records.append(rec)
        global_vals.append(float(bias_obs))
        global_w.append(float(geo_conf * np.clip(np.sum(ws), 0.1, 20.0)))

    if not records:
        return

    # Runtime-aware candidate capping: keep highest-confidence geometry cells.
    max_candidates = int(max(0, self.cfg.update.lzcd_max_candidates))
    if max_candidates > 0 and len(records) > max_candidates:
        scores = np.asarray(
            [
                float(
                    r["geo_conf"]
                    * np.clip(np.sum(r["ws"]), 0.2, 24.0)
                    * (0.85 + 0.15 * np.clip(r["cell"].geo_res_hits / residual_hit_ref, 0.0, 1.0))
                )
                for r in records
            ],
            dtype=float,
        )
        keep = np.argpartition(scores, -max_candidates)[-max_candidates:]
        keep = keep[np.argsort(scores[keep])[::-1]]
        records = [records[int(i)] for i in keep]

    idx_to_record = {r["idx"]: i for i, r in enumerate(records)}

    # Geometry-only convergence solver:
    # solve local debias on a voxel graph with data term + smoothness term.
    cur = np.asarray([float(r["bias_obs"]) for r in records], dtype=float)
    nxt = cur.copy()
    for _ in range(solver_iters):
        max_delta = 0.0
        for i, rec in enumerate(records):
            idx = rec["idx"]
            n_i = rec["n_i"]
            s_num = 0.0
            s_den = 0.0
            for nidx in voxel_map.neighbor_indices(idx, 1):
                j = idx_to_record.get(nidx)
                if j is None or j == i:
                    continue
                n_j = records[j]["n_i"]
                cos_ij = float(abs(np.dot(n_i, n_j)))
                if cos_ij < cos_min:
                    continue
                w_ij = float(0.25 + 0.75 * cos_ij)
                s_num += w_ij * float(cur[j])
                s_den += w_ij
            b_data = float(rec["bias_obs"])
            w_data = float(0.45 + 0.55 * rec["geo_conf"])
            b_smooth = b_data if s_den <= 1e-9 else float(s_num / s_den)
            target = float((w_data * b_data + solver_lambda * s_den * b_smooth) / max(1e-9, w_data + solver_lambda * s_den))
            b_new = float(np.clip(cur[i] + solver_step * (target - cur[i]), -max_bias, max_bias))
            nxt[i] = b_new
            max_delta = max(max_delta, abs(b_new - float(cur[i])))
        cur[:] = nxt
        if max_delta < solver_tol:
            break

    # Geometry-only global bias estimate; no dynamic terms involved.
    gb_obs = self._weighted_median(cur, np.asarray(global_w, dtype=float))
    self.geo_bias_global = float(
        np.clip((1.0 - global_alpha) * self.geo_bias_global + global_alpha * gb_obs, -max_bias, max_bias)
    )

    for i, rec in enumerate(records):
        c = rec["cell"]
        phi_i = float(rec["phi_i"])
        vals = rec["vals"]
        ws = rec["ws"]
        geo_conf = float(rec["geo_conf"])
        use_geo_cell = bool(rec["use_geo_cell"])
        dual_static_cell = bool(rec["dual_static_cell"])
        bias_obs = float(cur[i])
        if use_geo_cell:
            c.phi_geo_bias = float((1.0 - bias_alpha) * c.phi_geo_bias + bias_alpha * bias_obs)
            local_bias = float(c.phi_geo_bias)
        else:
            c.phi_bias = float((1.0 - bias_alpha) * c.phi_bias + bias_alpha * bias_obs)
            local_bias = float(c.phi_bias)

        bias_mix = float(np.clip(local_mix * local_bias + global_mix * self.geo_bias_global, -max_bias, max_bias))
        corr_eff = float(corr_gain * (0.70 + 0.30 * geo_conf))
        step_eff = float(max_step * (0.70 + 0.30 * geo_conf))
        corr_base = float(np.clip(corr_eff * bias_mix, -step_eff, step_eff))
        if abs(corr_base) <= 1e-7:
            continue

        # Trust-region backtracking using local residual energy.
        err0 = phi_i - vals
        e0 = float(np.sum(ws * np.abs(err0)))
        chosen_corr = 0.0
        for step_scale in (1.0, 0.5, 0.25):
            corr_try = float(step_scale * corr_base)
            phi_try = float(phi_i - corr_try)
            e1 = float(np.sum(ws * np.abs(phi_try - vals)))
            if e1 <= min_improve * e0:
                chosen_corr = corr_try
                break
        if abs(chosen_corr) <= 1e-9:
            continue

        if use_geo_cell:
            c.phi_geo = float(np.clip(c.phi_geo - chosen_corr, -0.8, 0.8))
            c.phi = float(np.clip(c.phi - 0.30 * chosen_corr, -0.8, 0.8))
        elif dual_static_cell:
            static_share = float(np.clip(0.45 + 0.25 * np.clip(c.p_static, 0.0, 1.0), 0.30, 0.78))
            c.phi_static = float(np.clip(c.phi_static - static_share * chosen_corr, -0.8, 0.8))
            c.phi_geo = float(np.clip(c.phi_geo - (0.55 + 0.20 * np.clip(c.p_static, 0.0, 1.0)) * chosen_corr, -0.8, 0.8))
            voxel_map._sync_legacy_channels(c)
        else:
            c.phi = float(np.clip(c.phi - chosen_corr, -0.8, 0.8))

def zero_crossing_bias_field(
    self,
    voxel_map: VoxelHashMap3D,
    touched: Iterable[VoxelIndex],
) -> None:
    if not bool(self.cfg.update.zcbf_enable):
        return
    block_size = int(max(1, self.cfg.update.zcbf_block_size_cells))
    min_rho = float(max(0.0, self.cfg.update.zcbf_min_rho))
    min_w = float(max(0.0, self.cfg.update.zcbf_min_phi_w))
    max_d = float(np.clip(self.cfg.update.zcbf_max_dscore, 0.0, 1.0))
    alpha = float(np.clip(self.cfg.update.zcbf_alpha, 0.01, 0.8))
    trim_q = float(np.clip(self.cfg.update.zcbf_trim_quantile, 0.55, 1.0))
    gain = float(np.clip(self.cfg.update.zcbf_apply_gain, 0.0, 1.0))
    max_bias = float(max(1e-5, self.cfg.update.zcbf_max_bias))

    touched_ext: Set[VoxelIndex] = set()
    for idx in touched:
        touched_ext.add(idx)
        touched_ext.update(voxel_map.neighbor_indices(idx, 1))

    blocks: dict[tuple[int, int, int], list[tuple[float, float, float, VoxelCell3D]]] = {}
    rho_ref_local = float(max(min_rho, self.cfg.update.zcbf_static_rho_ref))
    for idx in touched_ext:
        c = voxel_map.get_cell(idx)
        if c is None or c.rho < min_rho or float(np.clip(c.d_score, 0.0, 1.0)) > max_d:
            continue
        bias_obs = float(c.phi_geo_bias if c.phi_geo_w >= min_w else c.phi_bias)
        w_raw = float(c.phi_geo_w if c.phi_geo_w >= min_w else c.phi_static_w if c.phi_static_w >= min_w else c.phi_w)
        if w_raw < min_w:
            continue
        ptdsf_stats = voxel_map._ptdsf_state_stats(c) if hasattr(voxel_map, '_ptdsf_state_stats') else {"static_conf": 0.0, "dominance": 0.0, "rho_static": 0.0}
        persist_n = float(np.clip(0.55 * float(ptdsf_stats.get('static_conf', 0.0)) + 0.45 * float(ptdsf_stats.get('dominance', 0.0)), 0.0, 1.0))
        rho_n = float(np.clip(max(c.rho, float(ptdsf_stats.get('rho_static', 0.0))) / max(1e-6, rho_ref_local), 0.0, 1.5))
        obs_conf = float(np.clip(0.45 * persist_n + 0.35 * min(1.0, rho_n) + 0.20 * (1.0 - float(np.clip(c.d_score, 0.0, 1.0))), 0.05, 1.0))
        w_obs = float(w_raw * obs_conf)
        bx = (idx[0] // block_size, idx[1] // block_size, idx[2] // block_size)
        blocks.setdefault(bx, []).append((bias_obs, w_obs, obs_conf, c))

    if not blocks:
        return

    block_bias: dict[tuple[int, int, int], tuple[float, float]] = {}
    for bidx, vals in blocks.items():
        arr = np.asarray([v for v, _, _, _ in vals], dtype=float)
        ws = np.asarray([w for _, w, _, _ in vals], dtype=float)
        confs = np.asarray([cf for _, _, cf, _ in vals], dtype=float)
        if arr.size == 0:
            continue
        if arr.size >= 4:
            q = float(np.quantile(np.abs(arr), trim_q))
            keep = np.abs(arr) <= max(q, 1e-6)
            if np.any(keep):
                arr = arr[keep]
                ws = ws[keep]
                confs = confs[keep]
        b = self._weighted_median(arr, ws)
        conf_mass = float(np.sum(ws) / max(1.0, 4.0 * min_w))
        conf_norm = float(np.mean(confs)) if confs.size else 0.0
        conf = float(np.clip(conf_mass * (0.35 + 0.65 * conf_norm), 0.0, 1.0))
        block_bias[bidx] = (float(np.clip(b, -max_bias, max_bias)), conf)

    # Confidence-normalized propagation:
    # keep high-confidence blocks anchored while letting nearby static-consistent
    # blocks absorb a smoothed bias field instead of a one-hop median only.
    smooth_bias = dict(block_bias)
    for _ in range(2):
        next_bias: dict[tuple[int, int, int], tuple[float, float]] = {}
        for bidx, (b0, c0) in smooth_bias.items():
            bx, by, bz = bidx
            num = 1.35 * max(1e-6, c0) * b0
            den = 1.35 * max(1e-6, c0)
            conf_acc = c0
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nb = (bx + dx, by + dy, bz + dz)
                        if nb == bidx or nb not in smooth_bias:
                            continue
                        bj, cj = smooth_bias[nb]
                        dist = max(abs(dx), abs(dy), abs(dz))
                        base_w = 0.70 if dist == 1 else 0.40
                        compat = float(np.exp(-abs(float(bj) - float(b0)) / max(1e-6, 0.50 * max_bias)))
                        w_nb = float(base_w * max(1e-6, cj) * (0.35 + 0.65 * compat))
                        num += w_nb * bj
                        den += w_nb
                        conf_acc += 0.25 * cj
            b_hat = float(np.clip(num / max(1e-9, den), -max_bias, max_bias))
            c_hat = float(np.clip(conf_acc / max(1.0, 1.0 + 0.25 * 26.0), 0.0, 1.0))
            next_bias[bidx] = (b_hat, c_hat)
        smooth_bias = next_bias

    rho_ref = float(max(1e-6, self.cfg.update.zcbf_static_rho_ref))
    for idx in touched_ext:
        c = voxel_map.get_cell(idx)
        if c is None:
            continue
        bidx = (idx[0] // block_size, idx[1] // block_size, idx[2] // block_size)
        item = smooth_bias.get(bidx)
        if item is None:
            continue
        b_s, conf = item
        ptdsf_stats = voxel_map._ptdsf_state_stats(c) if hasattr(voxel_map, '_ptdsf_state_stats') else {"static_conf": 0.0, "dominance": 0.0}
        rho_n = float(np.clip(max(c.rho, getattr(c, 'rho_static', 0.0)) / rho_ref, 0.0, 1.0))
        persist_n = float(np.clip(0.55 * float(ptdsf_stats.get('static_conf', 0.0)) + 0.45 * float(ptdsf_stats.get('dominance', 0.0)), 0.0, 1.0))
        conf_eff = float(np.clip(conf * (0.35 + 0.35 * rho_n + 0.30 * persist_n), 0.0, 1.0))
        c.zcbf_bias = float(np.clip((1.0 - alpha) * c.zcbf_bias + alpha * b_s, -max_bias, max_bias))
        c.zcbf_bias_conf = float(np.clip((1.0 - alpha) * c.zcbf_bias_conf + alpha * conf_eff, 0.0, 1.0))
        corr = float(np.clip(gain * c.zcbf_bias * c.zcbf_bias_conf, -max_bias, max_bias))
        if abs(corr) <= 1e-7:
            continue
        if c.phi_geo_w > min_w:
            c.phi_geo = float(np.clip(c.phi_geo - corr, -0.8, 0.8))
        if bool(self.cfg.update.dual_state_enable) and c.phi_static_w > min_w:
            static_gain = float(np.clip(0.75 + 0.20 * persist_n, 0.75, 0.95))
            c.phi_static = float(np.clip(c.phi_static - static_gain * corr, -0.8, 0.8))
            voxel_map._sync_legacy_channels(c)

