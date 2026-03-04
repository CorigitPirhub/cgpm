#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def _to_float(v: object, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _pick_row(rows: List[Dict[str, str]], method: str) -> Dict[str, str]:
    m = method.lower().strip()
    for r in rows:
        if str(r.get("method", "")).lower().strip() == m:
            return dict(r)
    raise KeyError(f"method={method} not found in rows")


def _load_method_metrics(recon_csv: Path, dyn_csv: Path, method: str) -> Dict[str, float]:
    rec = _read_csv(recon_csv)
    dyn = _read_csv(dyn_csv)
    rr = _pick_row(rec, method)
    dd = _pick_row(dyn, method)
    return {
        "fscore": _to_float(rr.get("fscore")),
        "chamfer": _to_float(rr.get("chamfer")),
        "ghost_ratio": _to_float(dd.get("ghost_ratio")),
        "ghost_tail_ratio": _to_float(dd.get("ghost_tail_ratio")),
    }


def _plot(rows: List[Dict[str, object]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    factor_order = ["point_budget", "temporal_sparsity", "motion_pattern", "occlusion_missing"]
    colors = {"egf": "#1f77b4", "tsdf": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, factor in zip(axes_flat, factor_order):
        sub = [r for r in rows if str(r["factor"]) == factor]
        levels = sorted({int(r["level_id"]) for r in sub})
        ax2 = ax.twinx()

        for method in ["egf", "tsdf"]:
            xs: List[int] = []
            g: List[float] = []
            f: List[float] = []
            for lv in levels:
                rr = [r for r in sub if int(r["level_id"]) == lv and str(r["method"]) == method]
                if not rr:
                    continue
                xs.append(lv)
                g.append(float(rr[0]["ghost_ratio"]))
                f.append(float(rr[0]["fscore"]))
            if xs:
                ax.plot(xs, g, "-o", color=colors[method], alpha=0.95, label=f"{method}: ghost")
                ax2.plot(xs, f, "--s", color=colors[method], alpha=0.85, label=f"{method}: fscore")

        ax.set_title(factor)
        ax.set_xlabel("Stress Level (1=mild, 2=medium, 3=severe)")
        ax.set_ylabel("ghost_ratio (lower)")
        ax2.set_ylabel("fscore (higher)")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.25)

    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("P14 Stress Degradation Curves (4D): EGF vs TSDF")
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build P14 final 4D stress summary / curves / failure boundaries.")
    ap.add_argument("--out_csv", type=str, default="output/summary_tables/stress_test_summary_final.csv")
    ap.add_argument("--out_png", type=str, default="assets/stress_degradation_curves_final.png")
    ap.add_argument(
        "--out_failure_md",
        type=str,
        default="output/post_cleanup/stress_test/FAILURE_CASES_FINAL.md",
    )
    # point budget
    ap.add_argument("--pb_egf_root", type=str, default="output/post_cleanup/p7_speed_probe")
    ap.add_argument("--pb_tsdf_root", type=str, default="output/post_cleanup/p9_point_budget")
    # temporal sparsity
    ap.add_argument("--temporal_root", type=str, default="output/post_cleanup/p9_temporal_stride")
    # motion pattern
    ap.add_argument("--motion_recon_csv", type=str, default="output/summary_tables/tum_reconstruction_metrics.csv")
    ap.add_argument("--motion_dyn_csv", type=str, default="output/summary_tables/tum_dynamic_metrics.csv")
    # occlusion / missing
    ap.add_argument("--occlusion_root", type=str, default="output/post_cleanup/p14_occlusion_levels")
    args = ap.parse_args()

    out_rows: List[Dict[str, object]] = []

    # Factor A: point budget.
    pb_levels = [(1, "mpp900", 900), (2, "mpp500", 500), (3, "mpp300", 300)]
    for lv_id, lv_name, mpp in pb_levels:
        egf_payload = _load_method_metrics(
            Path(args.pb_egf_root) / lv_name / "oracle" / "tables" / "reconstruction_metrics.csv",
            Path(args.pb_egf_root) / lv_name / "oracle" / "tables" / "dynamic_metrics.csv",
            "egf",
        )
        tsdf_payload = _load_method_metrics(
            Path(args.pb_tsdf_root) / lv_name / "oracle" / "tables" / "reconstruction_metrics.csv",
            Path(args.pb_tsdf_root) / lv_name / "oracle" / "tables" / "dynamic_metrics.csv",
            "tsdf",
        )
        for method, payload in [("egf", egf_payload), ("tsdf", tsdf_payload)]:
            out_rows.append(
                {
                    "factor": "point_budget",
                    "level_id": lv_id,
                    "level_name": lv_name,
                    "level_value": mpp,
                    "method": method,
                    "fscore": payload["fscore"],
                    "chamfer": payload["chamfer"],
                    "ghost_ratio": payload["ghost_ratio"],
                    "ghost_tail_ratio": payload["ghost_tail_ratio"],
                    "source": f"{'pb_egf_root' if method=='egf' else 'pb_tsdf_root'}/{lv_name}",
                }
            )

    # Factor B: temporal sparsity.
    temporal_levels = [(1, "stride2", 2), (2, "stride3", 3), (3, "stride5", 5)]
    for lv_id, lv_name, stride in temporal_levels:
        if lv_name == "stride3":
            egf_payload = _load_method_metrics(
                Path(args.pb_egf_root) / "mpp500" / "oracle" / "tables" / "reconstruction_metrics.csv",
                Path(args.pb_egf_root) / "mpp500" / "oracle" / "tables" / "dynamic_metrics.csv",
                "egf",
            )
            tsdf_payload = _load_method_metrics(
                Path(args.pb_tsdf_root) / "mpp500" / "oracle" / "tables" / "reconstruction_metrics.csv",
                Path(args.pb_tsdf_root) / "mpp500" / "oracle" / "tables" / "dynamic_metrics.csv",
                "tsdf",
            )
            src = "pb/probe/stride3"
        else:
            egf_payload = _load_method_metrics(
                Path(args.temporal_root) / lv_name / "oracle" / "tables" / "reconstruction_metrics.csv",
                Path(args.temporal_root) / lv_name / "oracle" / "tables" / "dynamic_metrics.csv",
                "egf",
            )
            tsdf_payload = _load_method_metrics(
                Path(args.temporal_root) / lv_name / "oracle" / "tables" / "reconstruction_metrics.csv",
                Path(args.temporal_root) / lv_name / "oracle" / "tables" / "dynamic_metrics.csv",
                "tsdf",
            )
            src = f"temporal_root/{lv_name}"

        for method, payload in [("egf", egf_payload), ("tsdf", tsdf_payload)]:
            out_rows.append(
                {
                    "factor": "temporal_sparsity",
                    "level_id": lv_id,
                    "level_name": lv_name,
                    "level_value": stride,
                    "method": method,
                    "fscore": payload["fscore"],
                    "chamfer": payload["chamfer"],
                    "ghost_ratio": payload["ghost_ratio"],
                    "ghost_tail_ratio": payload["ghost_tail_ratio"],
                    "source": src,
                }
            )

    # Factor C: motion pattern.
    motion_rec = _read_csv(Path(args.motion_recon_csv))
    motion_dyn = _read_csv(Path(args.motion_dyn_csv))
    motion_lv = [
        (1, "walking_static", "rgbd_dataset_freiburg3_walking_static"),
        (2, "walking_xyz", "rgbd_dataset_freiburg3_walking_xyz"),
        (3, "walking_halfsphere", "rgbd_dataset_freiburg3_walking_halfsphere"),
    ]
    for lv_id, lv_name, seq in motion_lv:
        for method in ["egf", "tsdf"]:
            rr = next(
                r
                for r in motion_rec
                if str(r.get("sequence")) == seq
                and str(r.get("method", "")).lower() == method
                and str(r.get("protocol")) == "oracle"
            )
            dd = next(
                r
                for r in motion_dyn
                if str(r.get("sequence")) == seq
                and str(r.get("method", "")).lower() == method
                and str(r.get("protocol")) == "oracle"
            )
            out_rows.append(
                {
                    "factor": "motion_pattern",
                    "level_id": lv_id,
                    "level_name": lv_name,
                    "level_value": seq,
                    "method": method,
                    "fscore": _to_float(rr.get("fscore")),
                    "chamfer": _to_float(rr.get("chamfer")),
                    "ghost_ratio": _to_float(dd.get("ghost_ratio")),
                    "ghost_tail_ratio": _to_float(dd.get("ghost_tail_ratio")),
                    "source": "summary_tables/tum_*",
                }
            )

    # Factor D: occlusion / missing.
    occ_levels = [(1, "l1", 0.10), (2, "l2", 0.20), (3, "l3", 0.35)]
    for lv_id, lv_name, ratio in occ_levels:
        egf_payload = _load_method_metrics(
            Path(args.occlusion_root) / lv_name / "oracle" / "tables" / "reconstruction_metrics.csv",
            Path(args.occlusion_root) / lv_name / "oracle" / "tables" / "dynamic_metrics.csv",
            "egf",
        )
        tsdf_payload = _load_method_metrics(
            Path(args.occlusion_root) / lv_name / "oracle" / "tables" / "reconstruction_metrics.csv",
            Path(args.occlusion_root) / lv_name / "oracle" / "tables" / "dynamic_metrics.csv",
            "tsdf",
        )
        for method, payload in [("egf", egf_payload), ("tsdf", tsdf_payload)]:
            out_rows.append(
                {
                    "factor": "occlusion_missing",
                    "level_id": lv_id,
                    "level_name": lv_name,
                    "level_value": ratio,
                    "method": method,
                    "fscore": payload["fscore"],
                    "chamfer": payload["chamfer"],
                    "ghost_ratio": payload["ghost_ratio"],
                    "ghost_tail_ratio": payload["ghost_tail_ratio"],
                    "source": f"occlusion_root/{lv_name}",
                }
            )

    # Attach EGF-vs-TSDF reductions.
    by_factor_level: Dict[Tuple[str, int], Dict[str, Dict[str, object]]] = {}
    for r in out_rows:
        k = (str(r["factor"]), int(r["level_id"]))
        by_factor_level.setdefault(k, {})[str(r["method"])] = r
    for (_factor, _lv), pair in by_factor_level.items():
        egf = pair.get("egf")
        tsdf = pair.get("tsdf")
        if egf is None or tsdf is None:
            continue
        g_e = _to_float(egf["ghost_ratio"])
        g_t = _to_float(tsdf["ghost_ratio"])
        reduction = (g_t - g_e) / g_t if g_t > 1e-12 else float("nan")
        delta_f = _to_float(egf["fscore"]) - _to_float(tsdf["fscore"])
        for rr in [egf, tsdf]:
            rr["ghost_reduction_vs_tsdf"] = reduction
            rr["delta_fscore_egf_minus_tsdf"] = delta_f

    headers = [
        "factor",
        "level_id",
        "level_name",
        "level_value",
        "method",
        "fscore",
        "chamfer",
        "ghost_ratio",
        "ghost_tail_ratio",
        "ghost_reduction_vs_tsdf",
        "delta_fscore_egf_minus_tsdf",
        "source",
    ]
    _write_csv(Path(args.out_csv), headers, out_rows)
    _plot(out_rows, Path(args.out_png))

    # P14 gates.
    factors = ["point_budget", "temporal_sparsity", "motion_pattern", "occlusion_missing"]
    lvl2_checks: List[Tuple[str, float, float, float]] = []
    for factor in factors:
        rr = by_factor_level.get((factor, 2), {})
        egf = rr.get("egf")
        tsdf = rr.get("tsdf")
        if egf is None or tsdf is None:
            continue
        g_e = _to_float(egf["ghost_ratio"])
        g_t = _to_float(tsdf["ghost_ratio"])
        red = (g_t - g_e) / g_t if g_t > 1e-12 else float("nan")
        lvl2_checks.append((factor, red, _to_float(egf["fscore"]), _to_float(tsdf["fscore"])))
    level2_pass = len(lvl2_checks) == 4 and all(red >= 0.25 for _f, red, _fe, _ft in lvl2_checks)

    lvl3_fscore_checks: List[Tuple[str, float]] = []
    for factor in factors:
        rr = by_factor_level.get((factor, 3), {})
        egf = rr.get("egf")
        if egf is None:
            continue
        lvl3_fscore_checks.append((factor, _to_float(egf["fscore"])))
    lvl3_pass_count = sum(1 for _f, fs in lvl3_fscore_checks if fs >= 0.75)
    level3_pass = lvl3_pass_count >= 3

    failure_lines: List[str] = []
    failure_lines.append("# P14 Failure Boundaries (Final)\n")
    failure_lines.append("## Acceptance Snapshot")
    for factor, red, f_e, f_t in lvl2_checks:
        failure_lines.append(
            f"- L2 `{factor}`: ghost reduction vs TSDF = `{red:.4f}` ({red*100:.2f}%), "
            f"EGF fscore=`{f_e:.4f}`, TSDF fscore=`{f_t:.4f}`."
        )
    failure_lines.append(f"- Level-2 gate (all 4 dims >=25%): `{level2_pass}`")
    failure_lines.append(
        f"- Level-3 gate (EGF fscore>=0.75 in >=3 dims): `{level3_pass}` "
        f"(pass_dims=`{lvl3_pass_count}`/4)."
    )
    failure_lines.append("")
    failure_lines.append("## Recommended Operating Region")
    failure_lines.append("- `point_budget >= 500` (mpp500+).")
    failure_lines.append("- `stride <= 3` for temporal subsampling.")
    failure_lines.append("- `walking_xyz/static` motion regime is stable; `halfsphere` remains usable but ghost tail is higher.")
    failure_lines.append("- `occlusion_ratio <= 0.20` gives the best geometry/ghost trade-off.")
    failure_lines.append("")

    # Boundary 1
    pb3 = by_factor_level.get(("point_budget", 3), {}).get("egf")
    if pb3 is not None:
        failure_lines.append("## Boundary A: Severe Point Budget Drop")
        failure_lines.append(
            f"- Condition: `point_budget=300` (L3), EGF fscore=`{_to_float(pb3['fscore']):.4f}`, "
            f"ghost_ratio=`{_to_float(pb3['ghost_ratio']):.4f}`."
        )
        failure_lines.append("- Risk: geometric coverage decays first (precision stays acceptable but detail recall degrades).")
        failure_lines.append(
            "- Mitigation: raise per-frame point budget to >=500 or enable local adaptive refinement on high-curvature cells."
        )
        failure_lines.append("")

    # Boundary 2
    occ3 = by_factor_level.get(("occlusion_missing", 3), {}).get("egf")
    if occ3 is not None:
        failure_lines.append("## Boundary B: Severe Occlusion / Missing Depth")
        failure_lines.append(
            f"- Condition: `occlusion_ratio=0.35` (L3), EGF fscore=`{_to_float(occ3['fscore']):.4f}`, "
            f"ghost_tail_ratio=`{_to_float(occ3['ghost_tail_ratio']):.4f}`."
        )
        failure_lines.append("- Risk: tail ghost accumulation rises despite low global ghost ratio.")
        failure_lines.append(
            "- Mitigation: keep occlusion below 0.20 in deployment or add short-term occlusion-aware keyframe fusion."
        )
        failure_lines.append("")

    # Boundary 3 (optional extra)
    mp3 = by_factor_level.get(("motion_pattern", 3), {}).get("egf")
    if mp3 is not None:
        failure_lines.append("## Boundary C: High Motion Pattern (walking_halfsphere)")
        failure_lines.append(
            f"- Condition: `walking_halfsphere` (L3), EGF fscore=`{_to_float(mp3['fscore']):.4f}`, "
            f"ghost_tail_ratio=`{_to_float(mp3['ghost_tail_ratio']):.4f}`."
        )
        failure_lines.append("- Risk: aggressive camera trajectory can amplify trailing artifacts.")
        failure_lines.append(
            "- Mitigation: reduce stride to 2 and cap angular velocity in online capture presets."
        )
        failure_lines.append("")

    failure_path = Path(args.out_failure_md)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text("\n".join(failure_lines).strip() + "\n", encoding="utf-8")

    meta = {
        "out_csv": str(args.out_csv),
        "out_png": str(args.out_png),
        "out_failure_md": str(args.out_failure_md),
        "level2_checks": [
            {"factor": f, "ghost_reduction_vs_tsdf": red, "egf_fscore": fe, "tsdf_fscore": ft}
            for f, red, fe, ft in lvl2_checks
        ],
        "level3_fscore_checks": [{"factor": f, "egf_fscore": fs} for f, fs in lvl3_fscore_checks],
        "acceptance_level2_all_ge_25pct": bool(level2_pass),
        "acceptance_level3_at_least3dims_ge_0p75": bool(level3_pass),
        "acceptance_overall_p14": bool(level2_pass and level3_pass),
    }
    (failure_path.parent / "stress_meta_final.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[done] summary: {args.out_csv}")
    print(f"[done] curves: {args.out_png}")
    print(f"[done] failure cases: {args.out_failure_md}")
    print(f"[done] level2_pass={level2_pass} level3_pass={level3_pass} overall={level2_pass and level3_pass}")


if __name__ == "__main__":
    main()
