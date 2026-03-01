#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_placeholder_source(src: str) -> bool:
    s = src.replace("\\", "/")
    return "../simple_removal/surface_points.ply" in s or "/simple_removal/surface_points.ply" in s


def main() -> None:
    parser = argparse.ArgumentParser(description="Build independent table for external baselines.")
    parser.add_argument("--root", type=str, default="output/post_cleanup/p3_tum_expanded/slam")
    parser.add_argument("--sequences", type=str, default="rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static")
    parser.add_argument("--methods", type=str, default="dynaslam,midfusion,neural_implicit")
    parser.add_argument("--out_csv", type=str, default="output/summary_tables/external_baselines_independent.csv")
    args = parser.parse_args()

    root = Path(args.root)
    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rows: List[Dict[str, object]] = []
    for seq in sequences:
        for method in methods:
            p = root / seq / method / "summary.json"
            if not p.exists():
                rows.append(
                    {
                        "sequence": seq,
                        "method": method,
                        "status": "missing",
                        "source_path": "",
                        "is_real_external": 0,
                        "fscore": "",
                        "chamfer": "",
                        "ghost_ratio": "",
                        "ghost_tail_ratio": "",
                    }
                )
                continue
            d = read_summary(p)
            src = str(d.get("source_path", ""))
            metrics = d.get("metrics", {})
            is_real = 0 if is_placeholder_source(src) else 1
            rows.append(
                {
                    "sequence": seq,
                    "method": method,
                    "status": "ok",
                    "source_path": src,
                    "is_real_external": is_real,
                    "fscore": float(metrics.get("fscore", 0.0)),
                    "chamfer": float(metrics.get("chamfer", 0.0)),
                    "ghost_ratio": float(d.get("ghost_ratio", 0.0)),
                    "ghost_tail_ratio": float(d.get("ghost_tail_ratio", 0.0)),
                }
            )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "sequence",
        "method",
        "status",
        "source_path",
        "is_real_external",
        "fscore",
        "chamfer",
        "ghost_ratio",
        "ghost_tail_ratio",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] external baseline table -> {out}")


if __name__ == "__main__":
    main()

