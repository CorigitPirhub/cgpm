#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import run_benchmark as bench


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIME_PAT = re.compile(r"ELAPSED_SEC=([0-9.]+)")
RSS_PAT = re.compile(r"MAX_RSS_KB=([0-9]+)")


def _to_float(v: object, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if np.isfinite(x) else default


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError(f"empty int list: {text}")
    return out


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    if not out:
        raise ValueError(f"empty float list: {text}")
    return out


def _run_timed(cmd: Sequence[str], stdout_log: Path, time_log: Path) -> Dict[str, float]:
    wrapped = ["/usr/bin/time", "-f", "ELAPSED_SEC=%e\nMAX_RSS_KB=%M", *cmd]
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        wrapped,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_log.open("w", encoding="utf-8"),
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    txt = proc.stderr or ""
    time_log.parent.mkdir(parents=True, exist_ok=True)
    time_log.write_text(txt, encoding="utf-8")
    mt = TIME_PAT.search(txt)
    mr = RSS_PAT.search(txt)
    return {
        "returncode": float(proc.returncode),
        "elapsed_sec": float(mt.group(1)) if mt else float("nan"),
        "max_rss_kb": float(mr.group(1)) if mr else float("nan"),
    }


def _read_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"missing summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tsdf_reference(args: argparse.Namespace, max_mpp: int, out_root: Path) -> Dict[str, float]:
    tsdf_dir = out_root / "tsdf_ref"
    summary_path = tsdf_dir / "summary.json"
    time_log_path = tsdf_dir / "time.log"
    if args.force or (not summary_path.exists()) or (not time_log_path.exists()):
        cmd = [
            args.python_exe,
            "scripts/run_tsdf_baseline.py",
            "--dataset_root",
            str(args.dataset_root),
            "--sequence",
            str(args.sequence),
            "--frames",
            str(int(args.frames)),
            "--stride",
            str(int(args.stride)),
            "--max_points_per_frame",
            str(int(max_mpp)),
            "--voxel_size",
            str(float(args.voxel_size)),
            "--surface_eval_thresh",
            str(float(args.eval_thresh)),
            "--out",
            str(tsdf_dir),
            "--seed",
            str(int(args.seed)),
        ]
        timed = _run_timed(cmd, stdout_log=tsdf_dir / "stdout.log", time_log=tsdf_dir / "time.log")
        if int(timed["returncode"]) != 0:
            raise RuntimeError(f"TSDF reference failed. see {tsdf_dir / 'stdout.log'}")
    else:
        txt = time_log_path.read_text(encoding="utf-8")
        mt = TIME_PAT.search(txt)
        mr = RSS_PAT.search(txt)
        timed = {
            "returncode": 0.0,
            "elapsed_sec": float(mt.group(1)) if mt else float("nan"),
            "max_rss_kb": float(mr.group(1)) if mr else float("nan"),
        }
    summary = _read_summary(summary_path)
    elapsed = _to_float(timed.get("elapsed_sec"))
    return {
        "elapsed_sec": elapsed,
        "wall_fps": (float(args.frames) / elapsed) if elapsed > 0 else float("nan"),
        "max_rss_kb": _to_float(timed.get("max_rss_kb")),
        "fscore": _to_float(((summary.get("metrics", {}) or {}).get("fscore")), default=float("nan")),
    }


def _build_planned_configs(
    mpps: Sequence[int],
    decay_intervals: Sequence[int],
    assoc_radii: Sequence[int],
    integration_scales: Sequence[float],
    integration_min_radius_vox_list: Sequence[float],
    surface_max_free_ratio_list: Sequence[float],
) -> List[Tuple[int, int, int, float, float, float]]:
    mpps_sorted = sorted(set(int(x) for x in mpps), reverse=True)
    decays_sorted = sorted(set(int(x) for x in decay_intervals))
    radii_sorted = sorted(set(int(x) for x in assoc_radii))
    ir_sorted = sorted(set(float(x) for x in integration_scales), reverse=True)

    minrad_sorted = sorted(set(float(x) for x in integration_min_radius_vox_list))
    free_ratio_vals = sorted(set(float(x) for x in surface_max_free_ratio_list))
    default_fr = 1e9
    if not any(abs(x - default_fr) < 1e-6 for x in free_ratio_vals):
        free_ratio_vals.append(default_fr)
    # Prioritize denoising-friendly free-ratio caps for RT branch.
    free_ratio_rt = sorted([x for x in free_ratio_vals if x < 1e8])
    free_ratio_rt = free_ratio_rt + [default_fr]

    planned: List[Tuple[int, int, int, float, float, float]] = []
    anchor_minrad = 1.2 if 1.2 in minrad_sorted else minrad_sorted[0]
    anchor = (400, 1, 2, 0.45, anchor_minrad, default_fr)
    planned.append(anchor)

    # High-quality neighborhood around anchor.
    for mpp in [400, 360, 320, 280]:
        if mpp not in mpps_sorted:
            continue
        for d in [2, 3, 4]:
            if d not in decays_sorted:
                continue
            if 2 in radii_sorted and 0.45 in ir_sorted:
                planned.append((mpp, d, 2, 0.45, anchor_minrad, default_fr))

    # Real-time search branch.
    for mpp in [320, 280, 240, 220, 200, 180, 160, 140, 120]:
        if mpp not in mpps_sorted:
            continue
        for d in [3, 4, 5, 6]:
            if d not in decays_sorted:
                continue
            for r in [1, 2]:
                if r not in radii_sorted:
                    continue
                for ir in [0.35, 0.30, 0.25, 0.20]:
                    if ir not in ir_sorted:
                        continue
                    for minrad in minrad_sorted:
                        for fr in free_ratio_rt:
                            planned.append((mpp, d, r, ir, minrad, fr))

    # De-duplicate and keep order.
    uniq: List[Tuple[int, int, int, float, float, float]] = []
    seen = set()
    for c in planned:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def _run_egf_config(
    args: argparse.Namespace,
    cfg: Tuple[int, int, int, float, float, float],
    out_root: Path,
    stable_bg: np.ndarray,
    tail_points: np.ndarray,
    dynamic_region: set[Tuple[int, int, int]],
    dynamic_voxel: float,
) -> Dict[str, object]:
    mpp, decay_int, assoc_radius, ir_scale, minrad, surface_max_free_ratio = cfg
    fr_tag = "INF" if surface_max_free_ratio >= 1e8 else f"{surface_max_free_ratio:.2f}"
    cid = (
        f"mpp{mpp}_d{decay_int}_r{assoc_radius}_ir{ir_scale:.2f}"
        f"_mr{minrad:.2f}_fr{fr_tag}"
    )
    run_dir = out_root / cid
    summary_path = run_dir / "summary.json"
    timed: Dict[str, float]
    time_log_path = run_dir / "time.log"
    if args.force or (not summary_path.exists()) or (not time_log_path.exists()):
        cmd = [
            args.python_exe,
            "scripts/run_egf_3d_tum.py",
            "--dataset_root",
            str(args.dataset_root),
            "--sequence",
            str(args.sequence),
            "--frames",
            str(int(args.frames)),
            "--stride",
            str(int(args.stride)),
            "--max_points_per_frame",
            str(int(mpp)),
            "--voxel_size",
            str(float(args.voxel_size)),
            "--surface_eval_thresh",
            str(float(args.eval_thresh)),
            "--use_gt_pose",
            "--seed",
            str(int(args.seed)),
            "--poisson_iters",
            "0",
            "--sigma_n0",
            "0.26",
            "--truncation",
            "0.08",
            "--rho_decay",
            "0.998",
            "--phi_w_decay",
            "0.998",
            "--no_dynamic_forgetting",
            "--forget_mode",
            "off",
            "--raycast_clear_gain",
            "0.0",
            "--surface_phi_thresh",
            "0.8",
            "--surface_rho_thresh",
            "0.0",
            "--surface_min_weight",
            "0.0",
            "--surface_max_dscore",
            "1.0",
            "--surface_max_free_ratio",
            f"{float(surface_max_free_ratio):.6g}",
            "--surface_use_zero_crossing",
            "--surface_zero_crossing_max_offset",
            "0.06",
            "--surface_zero_crossing_phi_gate",
            "0.05",
            "--surface_adaptive_enable",
            "--integration_radius_scale",
            f"{float(ir_scale):.4f}",
            "--integration_min_radius_vox",
            f"{float(minrad):.4f}",
            "--decay_interval_frames",
            str(int(decay_int)),
            "--assoc_search_radius_cells",
            str(int(assoc_radius)),
            "--skip_mesh_export",
            "--out",
            str(run_dir),
        ]
        timed = _run_timed(cmd, stdout_log=run_dir / "stdout.log", time_log=run_dir / "time.log")
        if int(timed["returncode"]) != 0:
            raise RuntimeError(f"EGF run failed: {cid}. see {run_dir / 'stdout.log'}")
    else:
        txt = time_log_path.read_text(encoding="utf-8")
        mt = TIME_PAT.search(txt)
        mr = RSS_PAT.search(txt)
        timed = {
            "returncode": 0.0,
            "elapsed_sec": float(mt.group(1)) if mt else float("nan"),
            "max_rss_kb": float(mr.group(1)) if mr else float("nan"),
        }

    summary = _read_summary(summary_path)
    metrics = (summary.get("metrics", {}) or {})
    runtime = (summary.get("runtime", {}) or {})
    pred_points, _ = bench.load_points_with_normals(run_dir / "surface_points.ply")
    dyn = bench.compute_dynamic_metrics(
        pred_points=pred_points,
        stable_bg_points=stable_bg,
        tail_points=tail_points,
        dynamic_region=dynamic_region,
        dynamic_voxel=float(dynamic_voxel),
        ghost_thresh=float(args.ghost_thresh),
        bg_thresh=float(args.bg_thresh),
    )
    elapsed = _to_float(timed.get("elapsed_sec"))
    wall_fps = (float(args.frames) / elapsed) if elapsed > 0 else float("nan")
    row: Dict[str, object] = {
        "config_id": cid,
        "max_points_per_frame": int(mpp),
        "decay_interval_frames": int(decay_int),
        "assoc_search_radius_cells": int(assoc_radius),
        "integration_radius_scale": float(ir_scale),
        "integration_min_radius_vox": float(minrad),
        "surface_max_free_ratio": float(surface_max_free_ratio),
        "elapsed_sec": elapsed,
        "wall_fps": wall_fps,
        "max_rss_kb": _to_float(timed.get("max_rss_kb")),
        "mapping_fps": _to_float(runtime.get("mapping_fps"), default=wall_fps),
        "mapping_step_total_sec": _to_float(runtime.get("mapping_step_total_sec")),
        "stage_predict_sec": _to_float(runtime.get("stage_predict_sec")),
        "stage_associate_sec": _to_float(runtime.get("stage_associate_sec")),
        "stage_update_sec": _to_float(runtime.get("stage_update_sec")),
        "stage_map_refine_sec": _to_float(runtime.get("stage_map_refine_sec")),
        "extract_total_sec": _to_float(runtime.get("extract_total_sec")),
        "eval_recon_sec": _to_float(runtime.get("eval_recon_sec")),
        "eval_traj_sec": _to_float(runtime.get("eval_traj_sec")),
        "fscore": _to_float(metrics.get("fscore")),
        "chamfer": _to_float(metrics.get("chamfer")),
        "ghost_ratio": _to_float(dyn.get("ghost_ratio")),
        "ghost_tail_ratio": _to_float(dyn.get("ghost_tail_ratio")),
        "background_recovery": _to_float(dyn.get("background_recovery")),
        "out_dir": str(run_dir),
    }
    return row


def _plot(rows: List[Dict[str, object]], out_png: Path, anchor_id: str, rt10_id: str, rt15_id: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)
    fps = np.asarray([_to_float(r.get("mapping_fps")) for r in rows], dtype=float)
    fscore = np.asarray([_to_float(r.get("fscore")) for r in rows], dtype=float)
    ghost = np.asarray([_to_float(r.get("ghost_ratio")) for r in rows], dtype=float)
    mpp = np.asarray([_to_float(r.get("max_points_per_frame")) for r in rows], dtype=float)
    sc = ax.scatter(
        fps,
        fscore,
        c=ghost,
        s=45 + 0.16 * mpp,
        cmap="viridis_r",
        alpha=0.92,
        edgecolors="black",
        linewidths=0.4,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("ghost_ratio (lower better)")
    ax.axvline(10.0, linestyle="--", color="#d62728", linewidth=1.2, label="RT10 gate")
    ax.axvline(15.0, linestyle="--", color="#9467bd", linewidth=1.2, label="RT15 target")
    for r in rows:
        cid = str(r.get("config_id", ""))
        x = _to_float(r.get("mapping_fps"))
        y = _to_float(r.get("fscore"))
        if cid == anchor_id:
            ax.scatter([x], [y], s=230, facecolors="none", edgecolors="#1f77b4", linewidths=2.2, zorder=7)
        if cid == rt10_id:
            ax.scatter([x], [y], s=250, facecolors="none", edgecolors="#d62728", linewidths=2.4, zorder=7)
        if cid == rt15_id:
            ax.scatter([x], [y], s=250, facecolors="none", edgecolors="#9467bd", linewidths=2.4, zorder=7)
    ax.set_xlabel("Mapping FPS (step-only)")
    ax.set_ylabel("F-score")
    ax.set_title("P15 Realtime Quality-Speed Tradeoff (walking_xyz, oracle)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def _write_csv(path: Path, rows: List[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def _write_profile_md(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("rows", [])
    tsdf = payload.get("tsdf_ref", {})
    anchor = payload.get("anchor", {})
    rt10 = payload.get("rt10_selected", {})
    rt15 = payload.get("rt15_selected", {})
    lines: List[str] = []
    lines.append("# P15 Runtime Profile")
    lines.append("")
    lines.append("## Reference")
    lines.append(f"- `tsdf_wall_fps`: {float(tsdf.get('wall_fps', float('nan'))):.4f}")
    lines.append(f"- `tsdf_max_rss_kb`: {float(tsdf.get('max_rss_kb', float('nan'))):.0f}")
    lines.append("")
    lines.append("## Selected Configs")
    lines.append(f"- `anchor`: {anchor.get('config_id', 'NONE')}")
    lines.append(f"- `rt10`: {rt10.get('config_id', 'NONE')}")
    lines.append(f"- `rt15`: {rt15.get('config_id', 'NONE')}")
    lines.append("")
    lines.append("## Hotspots")
    for name, row in [("anchor", anchor), ("rt10", rt10), ("rt15", rt15)]:
        if not isinstance(row, dict) or not row:
            continue
        step = float(row.get("mapping_step_total_sec", 0.0))
        if step <= 1e-9:
            continue
        pred = float(row.get("stage_predict_sec", 0.0))
        assoc = float(row.get("stage_associate_sec", 0.0))
        upd = float(row.get("stage_update_sec", 0.0))
        lines.append(f"### {name}: {row.get('config_id', 'N/A')}")
        lines.append(f"- `mapping_fps`: {float(row.get('mapping_fps', 0.0)):.4f}")
        lines.append(f"- `predict_share`: {100.0 * pred / step:.2f}%")
        lines.append(f"- `associate_share`: {100.0 * assoc / step:.2f}%")
        lines.append(f"- `update_share`: {100.0 * upd / step:.2f}%")
        lines.append("")
    lines.append("## Notes")
    lines.append("- `mapping_fps` excludes evaluation and export overhead; it reflects per-frame mapping step runtime.")
    lines.append("- Quality constraints are computed against `anchor` using `F-score` and `ghost_ratio` deltas.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="P15 realtime closure: RT10/RT15 with quality constraints.")
    ap.add_argument("--python_exe", type=str, default="/home/zzy/anaconda3/envs/cgpm/bin/python")
    ap.add_argument("--dataset_root", type=str, default="data/tum")
    ap.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg3_walking_xyz")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--voxel_size", type=float, default=0.02)
    ap.add_argument("--eval_thresh", type=float, default=0.05)
    ap.add_argument("--ghost_thresh", type=float, default=0.08)
    ap.add_argument("--bg_thresh", type=float, default=0.05)
    ap.add_argument("--max_points_list", type=str, default="400,360,320,280,240,220,200,180,160,140,120")
    ap.add_argument("--decay_interval_list", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--assoc_radius_list", type=str, default="2,1")
    ap.add_argument("--integration_radius_list", type=str, default="0.45,0.35,0.30,0.25,0.20")
    ap.add_argument("--integration_min_radius_vox_list", type=str, default="1.2,1.0,0.8")
    ap.add_argument("--surface_max_free_ratio_list", type=str, default="1000000000,0.25,0.20,0.16,0.12")
    ap.add_argument("--max_runs", type=int, default=42)
    ap.add_argument("--min_rt10_fps", type=float, default=10.0)
    ap.add_argument("--min_rt15_fps", type=float, default=15.0)
    ap.add_argument("--max_fscore_drop", type=float, default=0.015)
    ap.add_argument("--max_ghost_rise", type=float, default=0.03)
    ap.add_argument("--max_memory_vs_tsdf", type=float, default=3.0)
    ap.add_argument("--out_root", type=str, default="output/post_cleanup/p15_realtime")
    ap.add_argument("--out_csv", type=str, default="output/summary_tables/local_mapping_efficiency_realtime.csv")
    ap.add_argument("--out_json", type=str, default="output/summary_tables/local_mapping_efficiency_realtime.json")
    ap.add_argument("--plot_png", type=str, default="assets/quality_speed_tradeoff_realtime.png")
    ap.add_argument("--profile_md", type=str, default="output/post_cleanup/p15_realtime/PROFILE.md")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    mpps = _parse_int_list(args.max_points_list)
    decay_list = _parse_int_list(args.decay_interval_list)
    assoc_list = _parse_int_list(args.assoc_radius_list)
    ir_list = _parse_float_list(args.integration_radius_list)
    minrad_list = _parse_float_list(args.integration_min_radius_vox_list)
    free_ratio_list = _parse_float_list(args.surface_max_free_ratio_list)

    seq_dir = Path(args.dataset_root) / str(args.sequence)
    if not seq_dir.exists():
        raise FileNotFoundError(f"sequence not found: {seq_dir}")
    stable_bg, tail_points, dynamic_region, dynamic_voxel = bench.build_dynamic_references(
        sequence_dir=seq_dir,
        frames=int(args.frames),
        stride=int(args.stride),
        max_points_per_frame=int(max(mpps)),
        seed=int(args.seed),
        stable_voxel=max(0.03, float(args.voxel_size)),
        stable_ratio=0.25,
        tail_frames=12,
        dynamic_voxel=max(0.05, float(2.0 * args.voxel_size)),
        min_dynamic_hits=2,
        max_dynamic_ratio=0.40,
    )

    tsdf_ref = _run_tsdf_reference(args, max_mpp=int(max(mpps)), out_root=out_root)
    plan = _build_planned_configs(mpps, decay_list, assoc_list, ir_list, minrad_list, free_ratio_list)
    if int(args.max_runs) > 0:
        plan = plan[: int(args.max_runs)]

    rows: List[Dict[str, object]] = []
    for cfg in plan:
        row = _run_egf_config(args, cfg, out_root, stable_bg, tail_points, dynamic_region, dynamic_voxel)
        row["memory_vs_tsdf"] = (
            _to_float(row.get("max_rss_kb")) / max(1e-9, float(tsdf_ref.get("max_rss_kb", float("nan"))))
            if np.isfinite(float(tsdf_ref.get("max_rss_kb", float("nan"))))
            else float("nan")
        )
        rows.append(row)
        print(
            f"[p15] {row['config_id']} map_fps={_to_float(row['mapping_fps']):.3f} "
            f"f={_to_float(row['fscore']):.4f} g={_to_float(row['ghost_ratio']):.4f}"
        )

    anchor_id = "mpp400_d1_r2_ir0.45_mr1.20_frINF"
    anchor = next((r for r in rows if str(r.get("config_id")) == anchor_id), None)
    if anchor is None:
        anchor = max(rows, key=lambda r: (_to_float(r.get("fscore")), -_to_float(r.get("ghost_ratio"))))
        anchor_id = str(anchor.get("config_id"))

    anchor_f = _to_float(anchor.get("fscore"))
    anchor_g = _to_float(anchor.get("ghost_ratio"))
    for r in rows:
        r["delta_fscore_vs_anchor"] = _to_float(r.get("fscore")) - anchor_f
        r["delta_ghost_vs_anchor"] = _to_float(r.get("ghost_ratio")) - anchor_g
        r["pass_quality"] = bool(
            _to_float(r.get("delta_fscore_vs_anchor")) >= -float(args.max_fscore_drop)
            and _to_float(r.get("delta_ghost_vs_anchor")) <= float(args.max_ghost_rise)
        )
        r["pass_memory"] = bool(_to_float(r.get("memory_vs_tsdf")) <= float(args.max_memory_vs_tsdf))
        r["pass_rt10"] = bool(
            _to_float(r.get("mapping_fps")) >= float(args.min_rt10_fps)
            and bool(r["pass_quality"])
            and bool(r["pass_memory"])
        )
        r["pass_rt15"] = bool(
            _to_float(r.get("mapping_fps")) >= float(args.min_rt15_fps)
            and bool(r["pass_quality"])
            and bool(r["pass_memory"])
        )

    rt10_candidates = [r for r in rows if bool(r.get("pass_rt10"))]
    rt15_candidates = [r for r in rows if bool(r.get("pass_rt15"))]
    rt10 = max(rt10_candidates, key=lambda r: (_to_float(r.get("mapping_fps")), _to_float(r.get("fscore"))), default={})
    rt15 = max(rt15_candidates, key=lambda r: (_to_float(r.get("mapping_fps")), _to_float(r.get("fscore"))), default={})
    rt10_id = str(rt10.get("config_id", ""))
    rt15_id = str(rt15.get("config_id", ""))

    rows.sort(key=lambda r: _to_float(r.get("mapping_fps")), reverse=True)
    headers = [
        "config_id",
        "max_points_per_frame",
        "decay_interval_frames",
        "assoc_search_radius_cells",
        "integration_radius_scale",
        "integration_min_radius_vox",
        "surface_max_free_ratio",
        "elapsed_sec",
        "wall_fps",
        "mapping_fps",
        "mapping_step_total_sec",
        "stage_predict_sec",
        "stage_associate_sec",
        "stage_update_sec",
        "stage_map_refine_sec",
        "extract_total_sec",
        "eval_recon_sec",
        "eval_traj_sec",
        "max_rss_kb",
        "memory_vs_tsdf",
        "fscore",
        "chamfer",
        "ghost_ratio",
        "ghost_tail_ratio",
        "background_recovery",
        "delta_fscore_vs_anchor",
        "delta_ghost_vs_anchor",
        "pass_quality",
        "pass_memory",
        "pass_rt10",
        "pass_rt15",
        "out_dir",
    ]
    _write_csv(Path(args.out_csv), rows, headers)
    _plot(rows, Path(args.plot_png), anchor_id=anchor_id, rt10_id=rt10_id, rt15_id=rt15_id)

    payload = {
        "constraints": {
            "min_rt10_fps": float(args.min_rt10_fps),
            "min_rt15_fps": float(args.min_rt15_fps),
            "max_fscore_drop": float(args.max_fscore_drop),
            "max_ghost_rise": float(args.max_ghost_rise),
            "max_memory_vs_tsdf": float(args.max_memory_vs_tsdf),
        },
        "tsdf_ref": tsdf_ref,
        "anchor": anchor,
        "rt10_selected": rt10,
        "rt15_selected": rt15,
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_profile_md(Path(args.profile_md), payload)
    print(f"[done] wrote {args.out_csv}")
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.plot_png}")
    print(f"[done] wrote {args.profile_md}")
    print(f"[select] anchor={anchor_id} rt10={rt10_id or 'NONE'} rt15={rt15_id or 'NONE'}")


if __name__ == "__main__":
    main()
