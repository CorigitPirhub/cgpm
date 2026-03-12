#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
    raise KeyError(f"method={method} not found")


def _load_pair(recon_csv: Path, dyn_csv: Path) -> Dict[str, Dict[str, float]]:
    rec = _read_csv(recon_csv)
    dyn = _read_csv(dyn_csv)
    out: Dict[str, Dict[str, float]] = {}
    for method in ["egf", "tsdf"]:
        rr = _pick_row(rec, method)
        dd = _pick_row(dyn, method)
        out[method] = {
            "fscore": _to_float(rr.get("fscore")),
            "ghost_ratio": _to_float(dd.get("ghost_ratio")),
            "ghost_tail_ratio": _to_float(dd.get("ghost_tail_ratio")),
            "chamfer": _to_float(rr.get("chamfer")),
        }
    return out


def _load_method_metrics(recon_csv: Path, dyn_csv: Path, method: str) -> Dict[str, float]:
    rec = _read_csv(recon_csv)
    dyn = _read_csv(dyn_csv)
    rr = _pick_row(rec, method)
    dd = _pick_row(dyn, method)
    return {
        "fscore": _to_float(rr.get("fscore")),
        "ghost_ratio": _to_float(dd.get("ghost_ratio")),
        "ghost_tail_ratio": _to_float(dd.get("ghost_tail_ratio")),
        "chamfer": _to_float(rr.get("chamfer")),
    }


def _plot(rows: List[Dict[str, object]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    factors = ["point_budget", "temporal_sparsity", "motion_pattern"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    colors = {"egf": "#1f77b4", "tsdf": "#d62728"}

    for ax, factor in zip(axes, factors):
        sub = [r for r in rows if str(r["factor"]) == factor]
        levels = sorted({int(r["level_id"]) for r in sub})
        ax2 = ax.twinx()
        for method in ["egf", "tsdf"]:
            xs = []
            g = []
            f = []
            for lv in levels:
                rr = [r for r in sub if int(r["level_id"]) == lv and str(r["method"]) == method]
                if not rr:
                    continue
                xs.append(lv)
                g.append(float(rr[0]["ghost_ratio"]))
                f.append(float(rr[0]["fscore"]))
            if xs:
                ax.plot(xs, g, "-o", color=colors[method], alpha=0.9, label=f"{method}: ghost")
                ax2.plot(xs, f, "--s", color=colors[method], alpha=0.8, label=f"{method}: fscore")
        ax.set_title(factor)
        ax.set_xlabel("level (1=mild, 2=medium, 3=severe)")
        ax.set_ylabel("ghost_ratio (lower)")
        ax2.set_ylabel("fscore (higher)")
        ax.grid(alpha=0.25)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("P9 Stress Degradation Curves (EGF vs TSDF)")
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build P9 stress summary/curves/failure cases from local mapping results.")
    ap.add_argument("--out_csv", type=str, default="output/summary_tables/stress_test_summary.csv")
    ap.add_argument("--out_png", type=str, default="assets/stress_degradation_curves.png")
    ap.add_argument("--out_failure_md", type=str, default="output/tmp/stress_test/FAILURE_CASES.md")
    # point-budget sources
    ap.add_argument("--pb_egf_root", type=str, default="output/tmp/p7_speed_probe")
    ap.add_argument("--pb_tsdf_root", type=str, default="output/tmp/p9_point_budget")
    # temporal sparsity sources
    ap.add_argument("--temporal_root", type=str, default="output/tmp/p9_temporal_stride")
    # motion pattern sources
    ap.add_argument("--motion_recon_csv", type=str, default="output/summary_tables/tum_reconstruction_metrics.csv")
    ap.add_argument("--motion_dyn_csv", type=str, default="output/summary_tables/tum_dynamic_metrics.csv")
    args = ap.parse_args()

    out_rows: List[Dict[str, object]] = []

    # Factor A: depth missing proxy (point budget)
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

    # Factor B: temporal sparsity (stride)
    temporal_levels = [(1, "stride2", 2), (2, "stride3", 3), (3, "stride5", 5)]
    for lv_id, lv_name, stride in temporal_levels:
        if lv_name == "stride3":
            # reuse point-budget mpp500, stride=3 from probe table (egf) and dedicated tsdf run.
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
        else:
            pair = _load_pair(
                Path(args.temporal_root) / lv_name / "oracle" / "tables" / "reconstruction_metrics.csv",
                Path(args.temporal_root) / lv_name / "oracle" / "tables" / "dynamic_metrics.csv",
            )
            egf_payload = pair["egf"]
            tsdf_payload = pair["tsdf"]
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
                    "source": f"{'pb/probe' if lv_name=='stride3' else 'temporal_root'}/{lv_name}",
                }
            )

    # Factor C: motion pattern (walking sequence as speed/dynamicity proxy)
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
                if str(r.get("sequence")) == seq and str(r.get("method", "")).lower() == method and str(r.get("protocol")) == "oracle"
            )
            dd = next(
                r
                for r in motion_dyn
                if str(r.get("sequence")) == seq and str(r.get("method", "")).lower() == method and str(r.get("protocol")) == "oracle"
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

    # Attach per-level EGF-vs-TSDF reductions.
    by_factor_level: Dict[Tuple[str, int], Dict[str, Dict[str, object]]] = {}
    for r in out_rows:
        k = (str(r["factor"]), int(r["level_id"]))
        by_factor_level.setdefault(k, {})[str(r["method"])] = r
    for (factor, lv), pair in by_factor_level.items():
        egf = pair.get("egf")
        tsdf = pair.get("tsdf")
        if egf is None or tsdf is None:
            continue
        g_e = _to_float(egf["ghost_ratio"])
        g_t = _to_float(tsdf["ghost_ratio"])
        if g_t > 1e-12:
            reduction = (g_t - g_e) / g_t
        else:
            reduction = float("nan")
        for rr in [egf, tsdf]:
            rr["ghost_reduction_vs_tsdf"] = reduction
            rr["delta_fscore_egf_minus_tsdf"] = _to_float(egf["fscore"]) - _to_float(tsdf["fscore"])

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

    # Acceptance check and failure boundaries
    lvl2_checks = []
    for factor in ["point_budget", "temporal_sparsity", "motion_pattern"]:
        rr = by_factor_level.get((factor, 2), {})
        egf = rr.get("egf")
        tsdf = rr.get("tsdf")
        if egf is None or tsdf is None:
            continue
        g_e = _to_float(egf["ghost_ratio"])
        g_t = _to_float(tsdf["ghost_ratio"])
        red = (g_t - g_e) / g_t if g_t > 1e-12 else float("nan")
        lvl2_checks.append((factor, red, _to_float(egf["fscore"]), _to_float(tsdf["fscore"])))

    failure_lines: List[str] = []
    failure_lines.append("# Stress Test Failure Cases\n")
    failure_lines.append("## Level-2 Acceptance")
    for factor, red, f_e, f_t in lvl2_checks:
        failure_lines.append(
            f"- `{factor}` level-2 ghost reduction vs TSDF: `{red:.4f}` ({red*100:.2f}%), "
            f"EGF fscore=`{f_e:.4f}`, TSDF fscore=`{f_t:.4f}`."
        )
    pass_all = all(red >= 0.25 for _factor, red, _fe, _ft in lvl2_checks) if lvl2_checks else False
    failure_lines.append(f"- Acceptance (`>=25%` reduction on all level-2 factors): `{pass_all}`")
    failure_lines.append("")

    # Two explicit boundaries.
    r_pb3 = by_factor_level.get(("point_budget", 3), {})
    if "egf" in r_pb3 and "tsdf" in r_pb3:
        egf = r_pb3["egf"]
        failure_lines.append("## Boundary A: Severe Depth Sparsity")
        failure_lines.append(
            f"- Condition: `point_budget=300` (L3). "
            f"EGF fscore=`{_to_float(egf['fscore']):.4f}`, ghost_ratio=`{_to_float(egf['ghost_ratio']):.4f}`."
        )
        failure_lines.append(
            "- Observation: ghost remains controlled, but geometric completeness degrades noticeably."
        )
        failure_lines.append(
            "- Mitigation: increase point budget to >=500 or enable adaptive voxel refinement on high-curvature regions."
        )
        failure_lines.append("")

    r_mp3 = by_factor_level.get(("motion_pattern", 3), {})
    if "egf" in r_mp3 and "tsdf" in r_mp3:
        egf = r_mp3["egf"]
        failure_lines.append("## Boundary B: High-Motion Pattern (Walking Halfsphere)")
        failure_lines.append(
            f"- Condition: `walking_halfsphere` (L3). "
            f"EGF ghost_ratio=`{_to_float(egf['ghost_ratio']):.4f}`, fscore=`{_to_float(egf['fscore']):.4f}`."
        )
        failure_lines.append(
            "- Observation: geometry remains strong, but residual ghost is still non-trivial under aggressive motion pattern."
        )
        failure_lines.append(
            "- Mitigation: tighten dynamic residual gating only for high d-score voxels to avoid global recall loss."
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
            {"factor": f, "ghost_reduction_vs_tsdf": r, "egf_fscore": fe, "tsdf_fscore": ft}
            for f, r, fe, ft in lvl2_checks
        ],
        "acceptance_level2_all_ge_25pct": bool(pass_all),
    }
    (failure_path.parent / "stress_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] summary: {args.out_csv}")
    print(f"[done] curves: {args.out_png}")
    print(f"[done] failure cases: {args.out_failure_md}")
    print(f"[done] level2_acceptance={pass_all}")


if __name__ == "__main__":
    main()
