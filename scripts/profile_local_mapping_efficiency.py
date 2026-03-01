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


TIME_PAT = re.compile(r"ELAPSED_SEC=([0-9.]+)")
RSS_PAT = re.compile(r"MAX_RSS_KB=([0-9]+)")


def run_timed(cmd: Sequence[str]) -> Dict[str, float]:
    wrapped = [
        "/usr/bin/time",
        "-f",
        "ELAPSED_SEC=%e\nMAX_RSS_KB=%M",
        *cmd,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(wrapped, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    t1 = time.perf_counter()
    elapsed_wall = float(t1 - t0)
    stderr = proc.stderr or ""
    m_t = TIME_PAT.search(stderr)
    m_r = RSS_PAT.search(stderr)
    elapsed_sec = float(m_t.group(1)) if m_t else elapsed_wall
    max_rss_kb = float(m_r.group(1)) if m_r else float("nan")
    return {
        "returncode": float(proc.returncode),
        "elapsed_sec": elapsed_sec,
        "elapsed_wall_sec": elapsed_wall,
        "max_rss_kb": max_rss_kb,
    }


def write_csv(path: Path, rows: List[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile local mapping runtime/memory for EGF and baselines.")
    parser.add_argument("--dataset_root", type=str, default="data/tum")
    parser.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg3_walking_xyz")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_points_per_frame", type=int, default=2500)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", type=str, default="output/post_cleanup/efficiency_profile")
    parser.add_argument("--out_csv", type=str, default="output/summary_tables/local_mapping_efficiency.csv")
    parser.add_argument("--out_json", type=str, default="output/summary_tables/local_mapping_efficiency.json")
    args = parser.parse_args()

    py = "/home/zzy/anaconda3/envs/cgpm/bin/python"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cmds: Dict[str, List[str]] = {
        "egf": [
            py,
            "scripts/run_egf_3d_tum.py",
            "--dataset_root",
            args.dataset_root,
            "--sequence",
            args.sequence,
            "--frames",
            str(args.frames),
            "--stride",
            str(args.stride),
            "--max_points_per_frame",
            str(args.max_points_per_frame),
            "--voxel_size",
            str(args.voxel_size),
            "--surface_eval_thresh",
            str(args.eval_thresh),
            "--out",
            str(out_root / "egf"),
            "--use_gt_pose",
            "--truncation",
            "0.08",
            "--sigma_n0",
            "0.26",
            "--dynamic_forgetting",
            "--forget_mode",
            "local",
            "--dyn_forget_gain",
            "0.0",
            "--surface_phi_thresh",
            "0.8",
            "--surface_rho_thresh",
            "0.0",
            "--surface_min_weight",
            "1.0",
            "--surface_max_age_frames",
            "120",
            "--surface_max_dscore",
            "0.9",
            "--mesh_min_points",
            "100000000",
            "--seed",
            str(args.seed),
        ],
        "tsdf": [
            py,
            "scripts/run_tsdf_baseline.py",
            "--dataset_root",
            args.dataset_root,
            "--sequence",
            args.sequence,
            "--frames",
            str(args.frames),
            "--stride",
            str(args.stride),
            "--max_points_per_frame",
            str(args.max_points_per_frame),
            "--voxel_size",
            str(args.voxel_size),
            "--surface_eval_thresh",
            str(args.eval_thresh),
            "--out",
            str(out_root / "tsdf"),
            "--seed",
            str(args.seed),
        ],
        "simple_removal": [
            py,
            "scripts/run_simple_removal_baseline.py",
            "--dataset_root",
            args.dataset_root,
            "--sequence",
            args.sequence,
            "--frames",
            str(args.frames),
            "--stride",
            str(args.stride),
            "--max_points_per_frame",
            str(args.max_points_per_frame),
            "--voxel_size",
            str(args.voxel_size),
            "--surface_eval_thresh",
            str(args.eval_thresh),
            "--out",
            str(out_root / "simple_removal"),
            "--seed",
            str(args.seed),
        ],
    }

    rows: List[Dict[str, object]] = []
    for method, cmd in cmds.items():
        print(f"[profile] {method}")
        m = run_timed(cmd)
        m.update(
            {
                "method": method,
                "sequence": args.sequence,
                "frames": float(args.frames),
                "stride": float(args.stride),
                "max_points_per_frame": float(args.max_points_per_frame),
                "voxel_size": float(args.voxel_size),
                "out_dir": str(out_root / method),
            }
        )
        rows.append(m)
        print(
            f"  return={int(m['returncode'])} elapsed={m['elapsed_sec']:.3f}s "
            f"max_rss={m['max_rss_kb']:.0f}KB"
        )

    # Relative factors vs TSDF
    tsdf = next((r for r in rows if r["method"] == "tsdf"), None)
    if tsdf is not None and float(tsdf["elapsed_sec"]) > 0:
        tsdf_t = float(tsdf["elapsed_sec"])
        tsdf_m = float(tsdf["max_rss_kb"]) if float(tsdf["max_rss_kb"]) > 0 else float("nan")
        for r in rows:
            r["speed_vs_tsdf"] = float(r["elapsed_sec"]) / tsdf_t
            if tsdf_m == tsdf_m and tsdf_m > 0:
                r["memory_vs_tsdf"] = float(r["max_rss_kb"]) / tsdf_m
            else:
                r["memory_vs_tsdf"] = float("nan")
    else:
        for r in rows:
            r["speed_vs_tsdf"] = float("nan")
            r["memory_vs_tsdf"] = float("nan")

    headers = [
        "method",
        "sequence",
        "frames",
        "stride",
        "max_points_per_frame",
        "voxel_size",
        "returncode",
        "elapsed_sec",
        "elapsed_wall_sec",
        "max_rss_kb",
        "speed_vs_tsdf",
        "memory_vs_tsdf",
        "out_dir",
    ]
    write_csv(Path(args.out_csv), rows, headers)
    Path(args.out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[done] efficiency csv -> {args.out_csv}")
    print(f"[done] efficiency json -> {args.out_json}")


if __name__ == "__main__":
    main()

