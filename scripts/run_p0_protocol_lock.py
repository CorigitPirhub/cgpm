#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CsvData:
    headers: List[str]
    rows: List[Dict[str, str]]


def run_cmd(cmd: Sequence[str], dry_run: bool = False) -> None:
    print("[cmd]", " ".join(str(c) for c in cmd))
    if dry_run:
        return
    subprocess.run(list(cmd), check=True, cwd=str(PROJECT_ROOT))


def read_csv(path: Path) -> CsvData:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return CsvData(headers=headers, rows=rows)


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for row in rows:
            w.writerow({h: row.get(h, "") for h in headers})


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def compare_csv(a: Path, b: Path, key_fields: Sequence[str]) -> Dict[str, object]:
    da = read_csv(a)
    db = read_csv(b)

    if da.headers != db.headers:
        return {
            "ok": False,
            "reason": "header_mismatch",
            "headers_a": da.headers,
            "headers_b": db.headers,
        }

    hset = list(da.headers)
    amap: Dict[Tuple[str, ...], Dict[str, str]] = {}
    bmap: Dict[Tuple[str, ...], Dict[str, str]] = {}

    for r in da.rows:
        k = tuple(str(r.get(kf, "")) for kf in key_fields)
        amap[k] = r
    for r in db.rows:
        k = tuple(str(r.get(kf, "")) for kf in key_fields)
        bmap[k] = r

    if set(amap.keys()) != set(bmap.keys()):
        return {
            "ok": False,
            "reason": "row_key_mismatch",
            "only_a": sorted([list(k) for k in (set(amap.keys()) - set(bmap.keys()))])[:20],
            "only_b": sorted([list(k) for k in (set(bmap.keys()) - set(amap.keys()))])[:20],
            "count_a": len(amap),
            "count_b": len(bmap),
        }

    max_abs_diff = 0.0
    max_abs_field = ""
    max_abs_key: Tuple[str, ...] | None = None
    mismatched_str_fields = 0
    numeric_fields = 0

    for key in sorted(amap.keys()):
        ra = amap[key]
        rb = bmap[key]
        for h in hset:
            if h in key_fields:
                continue
            va = str(ra.get(h, ""))
            vb = str(rb.get(h, ""))
            fa = _to_float(va)
            fb = _to_float(vb)
            if (not (fa != fa)) and (not (fb != fb)):  # both not nan
                numeric_fields += 1
                d = abs(fa - fb)
                if d > max_abs_diff:
                    max_abs_diff = d
                    max_abs_field = h
                    max_abs_key = key
            else:
                if va != vb:
                    mismatched_str_fields += 1

    ok = (max_abs_diff <= 1e-12) and (mismatched_str_fields == 0)
    return {
        "ok": bool(ok),
        "max_abs_diff": float(max_abs_diff),
        "max_abs_field": max_abs_field,
        "max_abs_key": list(max_abs_key) if max_abs_key is not None else [],
        "mismatched_string_fields": int(mismatched_str_fields),
        "numeric_field_compares": int(numeric_fields),
        "rows": int(len(amap)),
    }


def copy_table(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing table: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 protocol lock runner (oracle/slam + reproducibility check).")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--dataset_root", type=str, default="data/tum")
    parser.add_argument("--out_root", type=str, default="output/post_cleanup/p0_protocol_lock")
    parser.add_argument("--summary_root", type=str, default="output/summary_tables")
    parser.add_argument("--static_sequences", type=str, default="rgbd_dataset_freiburg1_xyz")
    parser.add_argument(
        "--dynamic_sequences",
        type=str,
        default="rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere",
    )
    parser.add_argument("--methods", type=str, default="egf,tsdf,simple_removal")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max_points_per_frame", type=int, default=1500)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--ghost_thresh", type=float, default=0.08)
    parser.add_argument("--bg_thresh", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    summary_root = Path(args.summary_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    common = [
        args.python_bin,
        "scripts/run_benchmark.py",
        "--dataset_kind",
        "tum",
        "--dataset_root",
        str(args.dataset_root),
        "--static_sequences",
        str(args.static_sequences),
        "--dynamic_sequences",
        str(args.dynamic_sequences),
        "--methods",
        str(args.methods),
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
    ]
    if args.download:
        common.append("--download")
    if args.force:
        common.append("--force")
    if args.dry_run:
        common.append("--dry_run")

    # 1) Oracle / SLAM split runs
    oracle_out = out_root / "protocol_eval" / "oracle"
    slam_out = out_root / "protocol_eval" / "slam"
    run_cmd([*common, "--protocol", "oracle", "--out_root", str(oracle_out)], dry_run=args.dry_run)
    run_cmd([*common, "--protocol", "slam", "--out_root", str(slam_out)], dry_run=args.dry_run)

    # 2) Repro check: same command twice (SLAM)
    repro1 = out_root / "repro_check" / "run1" / "slam"
    repro2 = out_root / "repro_check" / "run2" / "slam"
    run_cmd([*common, "--protocol", "slam", "--out_root", str(repro1)], dry_run=args.dry_run)
    run_cmd([*common, "--protocol", "slam", "--out_root", str(repro2)], dry_run=args.dry_run)

    if args.dry_run:
        print("[done] dry-run complete")
        return

    # 3) Validate outputs and reproducibility
    oracle_tables = oracle_out / "tables"
    slam_tables = slam_out / "tables"
    repro1_tables = repro1 / "tables"
    repro2_tables = repro2 / "tables"

    recon_cmp = compare_csv(
        repro1_tables / "reconstruction_metrics.csv",
        repro2_tables / "reconstruction_metrics.csv",
        key_fields=["sequence", "scene_type", "protocol", "seed", "method"],
    )
    dyn_cmp = compare_csv(
        repro1_tables / "dynamic_metrics.csv",
        repro2_tables / "dynamic_metrics.csv",
        key_fields=["sequence", "scene_type", "protocol", "seed", "method"],
    )

    # 4) Export locked summary tables to output/summary_tables
    copy_table(oracle_tables / "reconstruction_metrics.csv", summary_root / "tum_reconstruction_metrics_oracle.csv")
    copy_table(oracle_tables / "dynamic_metrics.csv", summary_root / "tum_dynamic_metrics_oracle.csv")
    copy_table(slam_tables / "reconstruction_metrics.csv", summary_root / "tum_reconstruction_metrics_slam.csv")
    copy_table(slam_tables / "dynamic_metrics.csv", summary_root / "tum_dynamic_metrics_slam.csv")
    # Keep main alias to SLAM as default engineering protocol.
    copy_table(slam_tables / "reconstruction_metrics.csv", summary_root / "tum_reconstruction_metrics.csv")
    copy_table(slam_tables / "dynamic_metrics.csv", summary_root / "tum_dynamic_metrics.csv")

    report = {
        "task": "P0 protocol lock",
        "dataset_root": str(args.dataset_root),
        "out_root": str(out_root),
        "summary_root": str(summary_root),
        "protocol_eval": {
            "oracle_tables": str(oracle_tables),
            "slam_tables": str(slam_tables),
        },
        "repro_check": {
            "run1_tables": str(repro1_tables),
            "run2_tables": str(repro2_tables),
            "reconstruction_csv_compare": recon_cmp,
            "dynamic_csv_compare": dyn_cmp,
        },
        "locked_sequences": {
            "static": str(args.static_sequences),
            "dynamic": str(args.dynamic_sequences),
        },
        "seed": int(args.seed),
        "frames": int(args.frames),
        "stride": int(args.stride),
        "methods": str(args.methods),
        "pass": bool(recon_cmp.get("ok", False) and dyn_cmp.get("ok", False)),
    }
    report_path = out_root / "p0_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# P0 Protocol Lock Report",
        "",
        f"- dataset_root: `{args.dataset_root}`",
        f"- out_root: `{out_root}`",
        f"- summary_root: `{summary_root}`",
        f"- static_sequences: `{args.static_sequences}`",
        f"- dynamic_sequences: `{args.dynamic_sequences}`",
        f"- methods: `{args.methods}`",
        f"- seed: `{args.seed}`",
        f"- frames/stride: `{args.frames}/{args.stride}`",
        "",
        "## Reproducibility Check (same command x2, SLAM)",
        "",
        f"- reconstruction csv equal: `{recon_cmp.get('ok', False)}`",
        f"- dynamic csv equal: `{dyn_cmp.get('ok', False)}`",
        f"- recon max abs diff: `{recon_cmp.get('max_abs_diff', 'n/a')}`",
        f"- dynamic max abs diff: `{dyn_cmp.get('max_abs_diff', 'n/a')}`",
        "",
        f"## Final Pass: `{report['pass']}`",
        "",
        f"JSON report: `{report_path}`",
    ]
    (out_root / "P0_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[done] P0 protocol lock finished. report={report_path}")
    if not bool(report["pass"]):
        raise SystemExit("P0 reproducibility check failed")


if __name__ == "__main__":
    main()

