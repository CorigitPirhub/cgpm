#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class CsvData:
    headers: List[str]
    rows: List[Dict[str, str]]


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


def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def row_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        str(row.get("sequence", "")),
        str(row.get("scene_type", "")),
        str(row.get("method", "")),
    )


def sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    scene_order = {"static": 0, "dynamic": 1}
    method_order = {"egf": 0, "tsdf": 1, "simple_removal": 2}

    def k(r: Dict[str, str]) -> Tuple[int, str, int]:
        return (
            scene_order.get(str(r.get("scene_type", "")), 99),
            str(r.get("sequence", "")),
            method_order.get(str(r.get("method", "")), 99),
        )

    return sorted(rows, key=k)


def merge_with_override(
    base_rows: List[Dict[str, str]],
    override_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    merged: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in base_rows:
        merged[row_key(row)] = dict(row)
    for row in override_rows:
        merged[row_key(row)] = dict(row)
    return sort_rows(list(merged.values()))


def copy_if_exists(src: Path, dst: Path, outputs: List[str], missing: List[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    data = read_csv(src)
    write_csv(dst, data.headers, data.rows)
    outputs.append(str(dst))


def build_static_fix_delta(
    base_recon_rows: List[Dict[str, str]],
    static_fix_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    base_map = {row_key(r): r for r in base_recon_rows}
    out: List[Dict[str, object]] = []
    for r in sort_rows(static_fix_rows):
        key = row_key(r)
        b = base_map.get(key)
        if b is None:
            continue
        out.append(
            {
                "sequence": r.get("sequence", ""),
                "scene_type": r.get("scene_type", ""),
                "method": r.get("method", ""),
                "points": to_float(r, "points"),
                "accuracy": to_float(r, "accuracy"),
                "completeness": to_float(r, "completeness"),
                "chamfer_new": to_float(r, "chamfer"),
                "hausdorff": to_float(r, "hausdorff"),
                "precision_new": to_float(r, "precision"),
                "recall": to_float(r, "recall"),
                "fscore_new": to_float(r, "fscore"),
                "ghost_count": to_float(r, "ghost_count"),
                "ghost_ratio": to_float(r, "ghost_ratio"),
                "ghost_tail_count": to_float(r, "ghost_tail_count"),
                "ghost_tail_ratio": to_float(r, "ghost_tail_ratio"),
                "background_recovery": to_float(r, "background_recovery"),
                "fscore_base": to_float(b, "fscore"),
                "precision_base": to_float(b, "precision"),
                "chamfer_base": to_float(b, "chamfer"),
                "fscore_delta": to_float(r, "fscore") - to_float(b, "fscore"),
                "precision_delta": to_float(r, "precision") - to_float(b, "precision"),
                "chamfer_delta": to_float(r, "chamfer") - to_float(b, "chamfer"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Update output/summary_tables from post_cleanup results.")
    parser.add_argument("--post_cleanup_root", type=str, default="output/post_cleanup")
    parser.add_argument("--summary_root", type=str, default="output/summary_tables")
    parser.add_argument("--legacy_ablation_csv", type=str, default="output/ablation_study/summary.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    post_cleanup_root = Path(args.post_cleanup_root)
    summary_root = Path(args.summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    outputs: List[str] = []
    missing: List[str] = []

    # Source CSVs.
    base_recon = post_cleanup_root / "benchmark_tum" / "tables" / "reconstruction_metrics.csv"
    base_dyn = post_cleanup_root / "benchmark_tum" / "tables" / "dynamic_metrics.csv"
    static_recon = post_cleanup_root / "static_fix_fullverify" / "tables" / "reconstruction_metrics.csv"
    static_dyn = post_cleanup_root / "static_fix_fullverify" / "tables" / "dynamic_metrics.csv"
    temporal = post_cleanup_root / "temporal_ablation" / "summary.csv"
    bonn = post_cleanup_root / "benchmark_bonn" / "summary.csv"
    ablation_post = post_cleanup_root / "ablation_summary.csv"
    ablation_legacy = Path(args.legacy_ablation_csv)

    # Required baseline TUM tables.
    if not base_recon.exists() or not base_dyn.exists():
        raise FileNotFoundError(
            f"missing TUM benchmark tables under {post_cleanup_root}/benchmark_tum/tables; "
            "run benchmark first."
        )

    base_recon_data = read_csv(base_recon)
    base_dyn_data = read_csv(base_dyn)

    # TUM merged summary: baseline + static_fix overrides (if available).
    if static_recon.exists():
        static_recon_data = read_csv(static_recon)
        merged_recon_rows = merge_with_override(base_recon_data.rows, static_recon_data.rows)
        write_csv(summary_root / "tum_reconstruction_metrics.csv", base_recon_data.headers, merged_recon_rows)
        outputs.append(str(summary_root / "tum_reconstruction_metrics.csv"))
        write_csv(summary_root / "tum_reconstruction_metrics_static_fix.csv", static_recon_data.headers, sort_rows(static_recon_data.rows))
        outputs.append(str(summary_root / "tum_reconstruction_metrics_static_fix.csv"))
    else:
        write_csv(summary_root / "tum_reconstruction_metrics.csv", base_recon_data.headers, sort_rows(base_recon_data.rows))
        outputs.append(str(summary_root / "tum_reconstruction_metrics.csv"))
        missing.append(str(static_recon))

    if static_dyn.exists():
        static_dyn_data = read_csv(static_dyn)
        merged_dyn_rows = merge_with_override(base_dyn_data.rows, static_dyn_data.rows)
        write_csv(summary_root / "tum_dynamic_metrics.csv", base_dyn_data.headers, merged_dyn_rows)
        outputs.append(str(summary_root / "tum_dynamic_metrics.csv"))
        write_csv(summary_root / "tum_dynamic_metrics_static_fix.csv", static_dyn_data.headers, sort_rows(static_dyn_data.rows))
        outputs.append(str(summary_root / "tum_dynamic_metrics_static_fix.csv"))
    else:
        write_csv(summary_root / "tum_dynamic_metrics.csv", base_dyn_data.headers, sort_rows(base_dyn_data.rows))
        outputs.append(str(summary_root / "tum_dynamic_metrics.csv"))
        missing.append(str(static_dyn))

    # Static-fix delta table (requires static_fix + base).
    if static_recon.exists():
        delta_rows = build_static_fix_delta(base_recon_data.rows, read_csv(static_recon).rows)
        delta_headers = [
            "sequence",
            "scene_type",
            "method",
            "points",
            "accuracy",
            "completeness",
            "chamfer_new",
            "hausdorff",
            "precision_new",
            "recall",
            "fscore_new",
            "ghost_count",
            "ghost_ratio",
            "ghost_tail_count",
            "ghost_tail_ratio",
            "background_recovery",
            "fscore_base",
            "precision_base",
            "chamfer_base",
            "fscore_delta",
            "precision_delta",
            "chamfer_delta",
        ]
        write_csv(summary_root / "static_fix_delta_summary.csv", delta_headers, delta_rows)
        outputs.append(str(summary_root / "static_fix_delta_summary.csv"))

    # Temporal / Bonn / Ablation passthrough.
    copy_if_exists(temporal, summary_root / "temporal_ablation_summary.csv", outputs, missing)
    copy_if_exists(bonn, summary_root / "bonn_summary.csv", outputs, missing)
    if ablation_post.exists():
        copy_if_exists(ablation_post, summary_root / "ablation_summary.csv", outputs, missing)
    else:
        copy_if_exists(ablation_legacy, summary_root / "ablation_summary.csv", outputs, missing)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_cleanup_root": str(post_cleanup_root),
        "summary_root": str(summary_root),
        "outputs": outputs,
        "missing_sources": missing,
    }
    with (summary_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    outputs.append(str(summary_root / "manifest.json"))

    print("[done] updated summary tables")
    if args.verbose:
        for p in outputs:
            print(f"  + {p}")
    if missing:
        print("[warn] some optional sources are missing:")
        for p in missing:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
