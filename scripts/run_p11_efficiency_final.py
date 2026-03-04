#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIME_PAT = re.compile(r"ELAPSED_SEC=([0-9.]+)")
RSS_PAT = re.compile(r"MAX_RSS_KB=([0-9]+)")


def _to_float(v: object, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if np.isfinite(x) else default


def _run_timed(cmd: Sequence[str], stdout_log: Path, time_log: Path) -> Dict[str, float]:
    wrapped = ["/usr/bin/time", "-f", "ELAPSED_SEC=%e\nMAX_RSS_KB=%M", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(
        wrapped,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_log.open("w", encoding="utf-8"),
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    t1 = time.perf_counter()
    txt = proc.stderr or ""
    time_log.parent.mkdir(parents=True, exist_ok=True)
    time_log.write_text(txt, encoding="utf-8")
    mt = TIME_PAT.search(txt)
    mr = RSS_PAT.search(txt)
    return {
        "returncode": float(proc.returncode),
        "elapsed_sec": float(mt.group(1)) if mt else float(t1 - t0),
        "max_rss_kb": float(mr.group(1)) if mr else float("nan"),
    }


def _read_single(csv_path: Path) -> Dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty csv: {csv_path}")
    return dict(rows[0])


def _write_csv(path: Path, rows: List[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def _plot(rows: List[Dict[str, object]], out_png: Path, rt_id: str, hq_id: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    fps = np.asarray([_to_float(r.get("fps")) for r in rows], dtype=float)
    fscore = np.asarray([_to_float(r.get("fscore")) for r in rows], dtype=float)
    ghost = np.asarray([_to_float(r.get("ghost_ratio")) for r in rows], dtype=float)
    mpp = np.asarray([_to_float(r.get("max_points_per_frame")) for r in rows], dtype=float)
    sc = ax.scatter(fps, fscore, c=ghost, s=40 + 0.18 * mpp, cmap="viridis_r", alpha=0.92, edgecolors="black", linewidths=0.45)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("ghost_ratio (lower better)")
    ax.axvline(2.5, linestyle="--", color="#d62728", linewidth=1.2, label="RT gate: 2.5 FPS")
    ax.axvline(1.0, linestyle="--", color="#666666", linewidth=1.1, label="HQ gate: 1.0 FPS")
    for r in rows:
        cid = str(r.get("config_id", ""))
        x = _to_float(r.get("fps"))
        y = _to_float(r.get("fscore"))
        if cid == rt_id:
            ax.scatter([x], [y], s=230, facecolors="none", edgecolors="#d62728", linewidths=2.2, zorder=6)
        if cid == hq_id:
            ax.scatter([x], [y], s=210, facecolors="none", edgecolors="#1f77b4", linewidths=2.0, zorder=6)
        ax.annotate(cid.replace("mpp", ""), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("FPS (40 frames / elapsed_sec)")
    ax.set_ylabel("F-score")
    ax.set_title("P11 Final Quality-Speed Tradeoff (walking_xyz, oracle)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="P11 final efficiency closure.")
    ap.add_argument("--dataset_root", type=str, default="data/tum")
    ap.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg3_walking_xyz")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--voxel_size", type=float, default=0.02)
    ap.add_argument("--eval_thresh", type=float, default=0.05)
    ap.add_argument("--max_points_list", type=str, default="900,700,500,380,300,250,220,200,180,150")
    ap.add_argument("--egf_poisson_iters", type=int, default=0)
    ap.add_argument("--egf_integration_radius_scale", type=float, default=0.68)
    ap.add_argument("--out_root", type=str, default="output/post_cleanup/p11_efficiency_final")
    ap.add_argument("--out_csv", type=str, default="output/summary_tables/local_mapping_efficiency_final.csv")
    ap.add_argument("--out_json", type=str, default="output/summary_tables/local_mapping_efficiency_final.json")
    ap.add_argument("--plot_png", type=str, default="assets/quality_speed_tradeoff_final.png")
    ap.add_argument("--min_hq_fps", type=float, default=1.0)
    ap.add_argument("--hq_max_fscore_drop", type=float, default=0.015)
    ap.add_argument("--min_rt_fps", type=float, default=2.5)
    ap.add_argument("--max_memory_vs_tsdf", type=float, default=3.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    py = "/home/zzy/anaconda3/envs/cgpm/bin/python"
    mpps = [int(x.strip()) for x in str(args.max_points_list).split(",") if x.strip()]
    if not mpps:
        raise ValueError("empty max_points_list")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Reference TSDF runtime/memory.
    tsdf_root = out_root / "tsdf_ref"
    tsdf_time_log = tsdf_root / "time.log"
    tsdf_out = tsdf_root / "bench"
    tsdf_recon = tsdf_out / "oracle" / "tables" / "reconstruction_metrics.csv"
    tsdf_dyn = tsdf_out / "oracle" / "tables" / "dynamic_metrics.csv"
    if args.force or (not tsdf_recon.exists()) or (not tsdf_dyn.exists()) or (not tsdf_time_log.exists()):
        tsdf_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            py,
            "scripts/run_benchmark.py",
            "--dataset_kind",
            "tum",
            "--dataset_root",
            str(args.dataset_root),
            "--protocol",
            "oracle",
            "--static_sequences",
            "",
            "--dynamic_sequences",
            str(args.sequence),
            "--methods",
            "tsdf",
            "--frames",
            str(int(args.frames)),
            "--stride",
            str(int(args.stride)),
            "--max_points_per_frame",
            str(int(max(mpps))),
            "--voxel_size",
            str(float(args.voxel_size)),
            "--eval_thresh",
            str(float(args.eval_thresh)),
            "--seed",
            str(int(args.seed)),
            "--out_root",
            str(tsdf_out),
            "--force",
        ]
        tsdf_timed = _run_timed(cmd, stdout_log=tsdf_root / "stdout.log", time_log=tsdf_time_log)
    else:
        txt = tsdf_time_log.read_text(encoding="utf-8")
        mt = TIME_PAT.search(txt)
        mr = RSS_PAT.search(txt)
        tsdf_timed = {
            "returncode": 0.0,
            "elapsed_sec": float(mt.group(1)) if mt else float("nan"),
            "max_rss_kb": float(mr.group(1)) if mr else float("nan"),
        }
    tsdf_elapsed = _to_float(tsdf_timed["elapsed_sec"])
    tsdf_fps = float(args.frames) / tsdf_elapsed if tsdf_elapsed > 0 else float("nan")
    tsdf_rss = _to_float(tsdf_timed["max_rss_kb"])

    rows: List[Dict[str, object]] = []
    for mpp in mpps:
        cid = f"mpp{mpp}"
        run_root = out_root / cid
        bench_out = run_root / "bench"
        time_log = run_root / "time.log"
        recon_csv = bench_out / "oracle" / "tables" / "reconstruction_metrics.csv"
        dyn_csv = bench_out / "oracle" / "tables" / "dynamic_metrics.csv"
        if args.force or (not recon_csv.exists()) or (not dyn_csv.exists()) or (not time_log.exists()):
            bench_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                py,
                "scripts/run_benchmark.py",
                "--dataset_kind",
                "tum",
                "--dataset_root",
                str(args.dataset_root),
                "--protocol",
                "oracle",
                "--static_sequences",
                "",
                "--dynamic_sequences",
                str(args.sequence),
                "--methods",
                "egf",
                "--frames",
                str(int(args.frames)),
                "--stride",
                str(int(args.stride)),
                "--max_points_per_frame",
                str(int(mpp)),
                "--voxel_size",
                str(float(args.voxel_size)),
                "--eval_thresh",
                str(float(args.eval_thresh)),
                "--seed",
                str(int(args.seed)),
                "--out_root",
                str(bench_out),
                "--egf_sigma_n0",
                "0.26",
                "--egf_truncation",
                "0.08",
                "--egf_dyn_forget_gain",
                "0.0",
                "--egf_raycast_clear_gain",
                "0.0",
                "--egf_surface_phi_thresh",
                "0.8",
                "--egf_surface_rho_thresh",
                "0.0",
                "--egf_surface_min_weight",
                "0.0",
                "--egf_surface_max_dscore",
                "1.0",
                "--egf_surface_max_free_ratio",
                "1000000000",
                "--egf_surface_use_zero_crossing",
                "--egf_surface_zero_crossing_max_offset",
                "0.06",
                "--egf_surface_zero_crossing_phi_gate",
                "0.05",
                "--egf_surface_adaptive_enable",
                "--egf_poisson_iters",
                str(int(max(0, args.egf_poisson_iters))),
                "--egf_integration_radius_scale",
                str(float(np.clip(args.egf_integration_radius_scale, 0.45, 1.0))),
                "--force",
            ]
            timed = _run_timed(cmd, stdout_log=run_root / "stdout.log", time_log=time_log)
        else:
            txt = time_log.read_text(encoding="utf-8")
            mt = TIME_PAT.search(txt)
            mr = RSS_PAT.search(txt)
            timed = {
                "returncode": 0.0,
                "elapsed_sec": float(mt.group(1)) if mt else float("nan"),
                "max_rss_kb": float(mr.group(1)) if mr else float("nan"),
            }
        recon = _read_single(recon_csv)
        dyn = _read_single(dyn_csv)
        elapsed = _to_float(timed["elapsed_sec"])
        fps = float(args.frames) / elapsed if elapsed > 0 else float("nan")
        max_rss = _to_float(timed["max_rss_kb"])
        rows.append(
            {
                "config_id": cid,
                "max_points_per_frame": int(mpp),
                "elapsed_sec": elapsed,
                "fps": fps,
                "max_rss_kb": max_rss,
                "memory_vs_tsdf": (max_rss / tsdf_rss) if tsdf_rss > 0 else float("nan"),
                "fscore": _to_float(recon.get("fscore")),
                "chamfer": _to_float(recon.get("chamfer")),
                "ghost_ratio": _to_float(dyn.get("ghost_ratio")),
                "ghost_tail_ratio": _to_float(dyn.get("ghost_tail_ratio")),
                "out_dir": str(bench_out),
            }
        )

    rows.sort(key=lambda r: int(r["max_points_per_frame"]), reverse=True)
    best = max(rows, key=lambda r: _to_float(r["fscore"], float("-inf")))
    best_f = _to_float(best["fscore"])
    for r in rows:
        r["delta_fscore_vs_best"] = _to_float(r["fscore"]) - best_f
        r["pass_memory"] = bool(_to_float(r["memory_vs_tsdf"]) <= float(args.max_memory_vs_tsdf))
        r["pass_hq"] = bool(_to_float(r["fps"]) >= float(args.min_hq_fps) and _to_float(r["delta_fscore_vs_best"]) >= -float(args.hq_max_fscore_drop))
        r["pass_rt"] = bool(_to_float(r["fps"]) >= float(args.min_rt_fps))

    hq_candidates = [r for r in rows if bool(r["pass_hq"]) and bool(r["pass_memory"])]
    rt_candidates = [r for r in rows if bool(r["pass_rt"]) and bool(r["pass_memory"])]
    hq = max(hq_candidates, key=lambda r: (_to_float(r["fscore"]), _to_float(r["fps"])), default=None)
    rt = max(rt_candidates, key=lambda r: (_to_float(r["fps"]), _to_float(r["fscore"])), default=None)
    hq_id = str(hq["config_id"]) if hq else ""
    rt_id = str(rt["config_id"]) if rt else ""
    for r in rows:
        r["is_hq_selected"] = str(r["config_id"]) == hq_id
        r["is_rt_selected"] = str(r["config_id"]) == rt_id

    headers = [
        "config_id",
        "max_points_per_frame",
        "elapsed_sec",
        "fps",
        "max_rss_kb",
        "memory_vs_tsdf",
        "fscore",
        "chamfer",
        "ghost_ratio",
        "ghost_tail_ratio",
        "delta_fscore_vs_best",
        "pass_memory",
        "pass_hq",
        "pass_rt",
        "is_hq_selected",
        "is_rt_selected",
        "out_dir",
    ]
    out_csv = Path(args.out_csv)
    _write_csv(out_csv, rows, headers)
    _plot(rows, Path(args.plot_png), rt_id=rt_id, hq_id=hq_id)

    payload = {
        "tsdf_ref": {
            "fps": tsdf_fps,
            "max_rss_kb": tsdf_rss,
            "out_dir": str(tsdf_out),
        },
        "best_quality_config": str(best["config_id"]),
        "hq_selected": hq_id,
        "rt_selected": rt_id,
        "constraints": {
            "min_hq_fps": float(args.min_hq_fps),
            "hq_max_fscore_drop": float(args.hq_max_fscore_drop),
            "min_rt_fps": float(args.min_rt_fps),
            "max_memory_vs_tsdf": float(args.max_memory_vs_tsdf),
        },
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.plot_png}")
    print(f"[select] hq={hq_id or 'NONE'} rt={rt_id or 'NONE'}")


if __name__ == "__main__":
    main()
