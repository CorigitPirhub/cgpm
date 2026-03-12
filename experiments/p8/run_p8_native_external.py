#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TIME_PAT = re.compile(r"ELAPSED_SEC=([0-9.]+)")
RSS_PAT = re.compile(r"MAX_RSS_KB=([0-9]+)")


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


def _run_timed(cmd: Sequence[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> Dict[str, float]:
    wrapped = ["/usr/bin/time", "-f", "ELAPSED_SEC=%e\nMAX_RSS_KB=%M", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(
        wrapped,
        cwd=str(cwd),
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    t1 = time.perf_counter()
    stderr = proc.stderr or ""
    stderr_path.write_text(stderr, encoding="utf-8")
    mt = TIME_PAT.search(stderr)
    mr = RSS_PAT.search(stderr)
    return {
        "returncode": float(proc.returncode),
        "elapsed_sec": float(mt.group(1)) if mt else float(t1 - t0),
        "max_rss_kb": float(mr.group(1)) if mr else float("nan"),
    }


def _to_float(v: object, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Run P8 native external baseline suite (non-adapter placeholder).")
    ap.add_argument("--dataset_root", type=str, default="data/tum")
    ap.add_argument(
        "--dynamic_sequences",
        type=str,
        default="rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static",
    )
    ap.add_argument("--protocol", type=str, default="slam", choices=["slam", "oracle"])
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_points_per_frame", type=int, default=900)
    ap.add_argument("--voxel_size", type=float, default=0.02)
    ap.add_argument("--eval_thresh", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--methods",
        type=str,
        default="egf,tsdf,dynaslam,neural_implicit",
        help="Methods to run independently for runtime accounting.",
    )
    ap.add_argument("--out_root", type=str, default="output/tmp/p8_native_external")
    ap.add_argument(
        "--out_recon_csv",
        type=str,
        default="output/summary_tables/external_baselines_native_reconstruction.csv",
    )
    ap.add_argument(
        "--out_dynamic_csv",
        type=str,
        default="output/summary_tables/external_baselines_native_dynamic.csv",
    )
    ap.add_argument(
        "--out_runtime_csv",
        type=str,
        default="output/summary_tables/external_baselines_native_runtime.csv",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    seqs = [s.strip() for s in str(args.dynamic_sequences).split(",") if s.strip()]
    if not methods:
        raise ValueError("empty methods")
    if not seqs:
        raise ValueError("empty dynamic_sequences")

    recon_rows_all: List[Dict[str, object]] = []
    dyn_rows_all: List[Dict[str, object]] = []
    runtime_rows: List[Dict[str, object]] = []

    for method in methods:
        method_out = out_root / method
        tables_dir = method_out / str(args.protocol) / "tables"
        recon_csv = tables_dir / "reconstruction_metrics.csv"
        dyn_csv = tables_dir / "dynamic_metrics.csv"
        stdout_log = method_out / "run_stdout.log"
        stderr_log = method_out / "run_time.log"

        need_run = bool(args.force) or (not recon_csv.exists()) or (not dyn_csv.exists())
        timed: Dict[str, float]
        if need_run:
            method_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                py,
                "scripts/run_benchmark.py",
                "--dataset_kind",
                "tum",
                "--dataset_root",
                str(args.dataset_root),
                "--protocol",
                str(args.protocol),
                "--static_sequences",
                "",
                "--dynamic_sequences",
                ",".join(seqs),
                "--methods",
                method,
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
                "--seed",
                str(int(args.seed)),
                "--out_root",
                str(method_out),
                "--dynaslam_pred_mesh_template",
                "output/external/dynaslam_real/{sequence}/mesh.ply",
                "--neural_pred_mesh_template",
                "output/external/neural_mesh/{sequence}/mesh.ply",
                "--external_require_real",
                "--force",
            ]
            timed = _run_timed(cmd, cwd=project_root, stdout_path=stdout_log, stderr_path=stderr_log)
        else:
            txt = stderr_log.read_text(encoding="utf-8") if stderr_log.exists() else ""
            mt = TIME_PAT.search(txt)
            mr = RSS_PAT.search(txt)
            timed = {
                "returncode": 0.0,
                "elapsed_sec": float(mt.group(1)) if mt else float("nan"),
                "max_rss_kb": float(mr.group(1)) if mr else float("nan"),
            }

        rows_r = _read_csv(recon_csv)
        rows_d = _read_csv(dyn_csv)
        for r in rows_r:
            rr = dict(r)
            rr["runner_method"] = method
            rr["is_native_external_eval"] = 1 if method in {"dynaslam", "neural_implicit", "midfusion"} else 0
            rr["source_root"] = str(method_out)
            recon_rows_all.append(rr)
        for r in rows_d:
            rr = dict(r)
            rr["runner_method"] = method
            rr["is_native_external_eval"] = 1 if method in {"dynaslam", "neural_implicit", "midfusion"} else 0
            rr["source_root"] = str(method_out)
            dyn_rows_all.append(rr)

        n_frames_total = int(args.frames) * len(seqs)
        elapsed = _to_float(timed.get("elapsed_sec"))
        runtime_rows.append(
            {
                "method": method,
                "protocol": str(args.protocol),
                "n_sequences": len(seqs),
                "frames_per_sequence": int(args.frames),
                "total_frames": n_frames_total,
                "elapsed_sec": elapsed,
                "fps_total": (float(n_frames_total) / elapsed) if elapsed > 0 else float("nan"),
                "max_rss_kb": _to_float(timed.get("max_rss_kb")),
                "returncode": int(_to_float(timed.get("returncode"))),
                "out_root": str(method_out),
            }
        )

    # Write outputs
    if recon_rows_all:
        _write_csv(Path(args.out_recon_csv), list(recon_rows_all[0].keys()), recon_rows_all)
    if dyn_rows_all:
        _write_csv(Path(args.out_dynamic_csv), list(dyn_rows_all[0].keys()), dyn_rows_all)
    _write_csv(
        Path(args.out_runtime_csv),
        [
            "method",
            "protocol",
            "n_sequences",
            "frames_per_sequence",
            "total_frames",
            "elapsed_sec",
            "fps_total",
            "max_rss_kb",
            "returncode",
            "out_root",
        ],
        runtime_rows,
    )

    meta = {
        "methods": methods,
        "dynamic_sequences": seqs,
        "protocol": str(args.protocol),
        "frames": int(args.frames),
        "stride": int(args.stride),
        "max_points_per_frame": int(args.max_points_per_frame),
        "voxel_size": float(args.voxel_size),
        "eval_thresh": float(args.eval_thresh),
        "seed": int(args.seed),
        "outputs": {
            "reconstruction": str(args.out_recon_csv),
            "dynamic": str(args.out_dynamic_csv),
            "runtime": str(args.out_runtime_csv),
        },
    }
    (out_root / "p8_native_external_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] reconstruction: {args.out_recon_csv}")
    print(f"[done] dynamic: {args.out_dynamic_csv}")
    print(f"[done] runtime: {args.out_runtime_csv}")


if __name__ == "__main__":
    main()

