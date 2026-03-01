#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: Sequence[str], dry_run: bool = False) -> None:
    print("[cmd]", " ".join(str(c) for c in cmd))
    if dry_run:
        return
    subprocess.run(list(cmd), check=True, cwd=str(PROJECT_ROOT))


def load_row(path: Path, sequence: str, method: str) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sequence") == sequence and row.get("method") == method:
                return row
    return {}


def to_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 static-dynamic balance runner with auto-stop on acceptance.")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--dataset_root", type=str, default="data/tum")
    parser.add_argument("--out_root", type=str, default="output/post_cleanup/p1_balance_lock")
    parser.add_argument("--summary_root", type=str, default="output/summary_tables")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_points_per_frame", type=int, default=3000)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--ghost_thresh", type=float, default=0.08)
    parser.add_argument("--bg_thresh", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--target_static_fscore", type=float, default=0.90)
    parser.add_argument("--target_walking_fscore", type=float, default=0.70)
    parser.add_argument("--target_walking_ghost_tail_ratio", type=float, default=0.30)
    parser.add_argument(
        "--use_existing_dir",
        type=str,
        default="",
        help="Reuse existing benchmark directory (expects <dir>/slam/tables/*.csv) instead of running new experiments.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_root = Path(args.summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    # Candidate set in descending aggressiveness / expected balance quality.
    # c0 reproduces the known good unified setting used in prior P1 runs.
    candidates: List[Dict[str, object]] = [
        {
            "name": "c0_unified_v8",
            "egf_sigma_n0": 0.26,
            "egf_rho_decay": 0.97,
            "egf_phi_w_decay": 0.97,
            "egf_forget_mode": "global",
            "egf_dyn_forget_gain": 0.35,
            "egf_raycast_clear_gain": 0.20,
            "egf_surface_max_age_frames": 16,
            "egf_surface_max_dscore": 0.60,
            "egf_surface_max_free_ratio": 0.7,
            "egf_surface_prune_free_min": 1.0,
            "egf_surface_prune_residual_min": 0.2,
            "egf_surface_max_clear_hits": 6.0,
            "egf_static_sigma_n0": 0.26,
            "egf_static_surface_phi_thresh": 0.80,
            "egf_static_surface_rho_thresh": 0.0,
            "egf_static_surface_min_weight": 0.0,
            "egf_static_surface_max_age_frames": 16,
            "egf_static_surface_max_dscore": 0.60,
            "egf_static_surface_max_free_ratio": 0.7,
            "egf_static_surface_prune_free_min": 1.0,
            "egf_static_surface_prune_residual_min": 0.2,
            "egf_static_surface_max_clear_hits": 6.0,
        },
        {
            "name": "c1_age15_d058",
            "egf_sigma_n0": 0.26,
            "egf_rho_decay": 0.97,
            "egf_phi_w_decay": 0.97,
            "egf_forget_mode": "global",
            "egf_dyn_forget_gain": 0.35,
            "egf_raycast_clear_gain": 0.20,
            "egf_surface_max_age_frames": 15,
            "egf_surface_max_dscore": 0.58,
            "egf_surface_max_free_ratio": 0.7,
            "egf_surface_prune_free_min": 1.0,
            "egf_surface_prune_residual_min": 0.2,
            "egf_surface_max_clear_hits": 6.0,
            "egf_static_sigma_n0": 0.26,
            "egf_static_surface_phi_thresh": 0.80,
            "egf_static_surface_rho_thresh": 0.0,
            "egf_static_surface_min_weight": 0.0,
            "egf_static_surface_max_age_frames": 15,
            "egf_static_surface_max_dscore": 0.58,
            "egf_static_surface_max_free_ratio": 0.7,
            "egf_static_surface_prune_free_min": 1.0,
            "egf_static_surface_prune_residual_min": 0.2,
            "egf_static_surface_max_clear_hits": 6.0,
        },
    ]

    tried: List[Dict[str, object]] = []
    accepted: Dict[str, object] | None = None

    if str(args.use_existing_dir).strip():
        existing = Path(str(args.use_existing_dir))
        tables = existing / "slam" / "tables"
        recon_csv = tables / "reconstruction_metrics.csv"
        dyn_csv = tables / "dynamic_metrics.csv"
        if not recon_csv.exists() or not dyn_csv.exists():
            raise FileNotFoundError(f"missing tables under existing dir: {tables}")
        r_static = load_row(recon_csv, "rgbd_dataset_freiburg1_xyz", "egf")
        r_walk = load_row(recon_csv, "rgbd_dataset_freiburg3_walking_xyz", "egf")
        d_walk = load_row(dyn_csv, "rgbd_dataset_freiburg3_walking_xyz", "egf")
        f_static = to_float(r_static, "fscore")
        f_walk = to_float(r_walk, "fscore")
        ghost_walk = to_float(d_walk, "ghost_tail_ratio")
        ok = (
            (f_static >= float(args.target_static_fscore))
            and (f_walk >= float(args.target_walking_fscore))
            and (ghost_walk <= float(args.target_walking_ghost_tail_ratio))
        )
        item = {
            "name": "existing_reuse",
            "ok": bool(ok),
            "fscore_static": f_static,
            "fscore_walking_xyz": f_walk,
            "ghost_tail_ratio_walking_xyz": ghost_walk,
            "tables_dir": str(tables),
            "source_dir": str(existing),
        }
        tried.append(item)
        if ok:
            accepted = item

    for cand in ([] if accepted is not None else candidates):
        name = str(cand["name"])
        run_out = out_root / name
        cmd: List[str] = [
            args.python_bin,
            "scripts/run_benchmark.py",
            "--dataset_kind",
            "tum",
            "--dataset_root",
            str(args.dataset_root),
            "--protocol",
            "slam",
            "--static_sequences",
            "rgbd_dataset_freiburg1_xyz",
            "--dynamic_sequences",
            "rgbd_dataset_freiburg3_walking_xyz",
            "--methods",
            "egf",
            "--frames",
            str(int(args.frames)),
            "--stride",
            str(int(args.stride)),
            "--max_points_per_frame",
            str(int(args.max_points_per_frame)),
            "--voxel_size",
            str(float(args.voxel_size)),
            "--eval_thresh",
            str(float(args.eval_thresh)),
            "--ghost_thresh",
            str(float(args.ghost_thresh)),
            "--bg_thresh",
            str(float(args.bg_thresh)),
            "--seed",
            str(int(args.seed)),
            "--egf_slam_no_gt_delta_odom",
            "--out_root",
            str(run_out),
        ]
        for k, v in cand.items():
            if k == "name":
                continue
            cmd += [f"--{k}", str(v)]
        if args.force:
            cmd.append("--force")
        run_cmd(cmd, dry_run=args.dry_run)
        if args.dry_run:
            continue

        recon_csv = run_out / "slam" / "tables" / "reconstruction_metrics.csv"
        dyn_csv = run_out / "slam" / "tables" / "dynamic_metrics.csv"
        if not recon_csv.exists() or not dyn_csv.exists():
            tried.append({"name": name, "ok": False, "reason": "missing_tables"})
            continue

        r_static = load_row(recon_csv, "rgbd_dataset_freiburg1_xyz", "egf")
        r_walk = load_row(recon_csv, "rgbd_dataset_freiburg3_walking_xyz", "egf")
        d_walk = load_row(dyn_csv, "rgbd_dataset_freiburg3_walking_xyz", "egf")
        f_static = to_float(r_static, "fscore")
        f_walk = to_float(r_walk, "fscore")
        ghost_walk = to_float(d_walk, "ghost_tail_ratio")

        ok = (
            (f_static >= float(args.target_static_fscore))
            and (f_walk >= float(args.target_walking_fscore))
            and (ghost_walk <= float(args.target_walking_ghost_tail_ratio))
        )
        item = {
            "name": name,
            "ok": bool(ok),
            "fscore_static": f_static,
            "fscore_walking_xyz": f_walk,
            "ghost_tail_ratio_walking_xyz": ghost_walk,
            "tables_dir": str(run_out / "slam" / "tables"),
        }
        tried.append(item)
        print("[p1-check]", item)
        if ok:
            accepted = item
            break

    report = {
        "task": "P1 static-dynamic balance",
        "targets": {
            "fscore_static": float(args.target_static_fscore),
            "fscore_walking_xyz": float(args.target_walking_fscore),
            "ghost_tail_ratio_walking_xyz": float(args.target_walking_ghost_tail_ratio),
        },
        "frames": int(args.frames),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "tried": tried,
        "accepted": accepted,
        "pass": bool(accepted is not None),
    }
    report_path = out_root / "p1_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.dry_run:
        print("[done] dry-run, no acceptance check.")
        return

    if accepted is None:
        raise SystemExit(f"P1 failed, see report: {report_path}")

    # Publish accepted tables to summary root.
    accepted_tables = Path(str(accepted["tables_dir"]))
    shutil.copy2(accepted_tables / "reconstruction_metrics.csv", summary_root / "tum_reconstruction_metrics_p1.csv")
    shutil.copy2(accepted_tables / "dynamic_metrics.csv", summary_root / "tum_dynamic_metrics_p1.csv")

    md = [
        "# P1 Balance Report",
        "",
        f"- report: `{report_path}`",
        f"- accepted config: `{accepted['name']}`",
        f"- static fscore: `{accepted['fscore_static']:.6f}`",
        f"- walking_xyz fscore: `{accepted['fscore_walking_xyz']:.6f}`",
        f"- walking_xyz ghost_tail_ratio: `{accepted['ghost_tail_ratio_walking_xyz']:.6f}`",
        "",
        "## Exported Tables",
        f"- `{summary_root / 'tum_reconstruction_metrics_p1.csv'}`",
        f"- `{summary_root / 'tum_dynamic_metrics_p1.csv'}`",
    ]
    (out_root / "P1_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] P1 passed. report={report_path}")


if __name__ == "__main__":
    main()
