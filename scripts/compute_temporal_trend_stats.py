#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr


def _read_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items() if v not in {"", None}})
    return rows


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    coef, _ = spearmanr(x, y)
    try:
        return float(coef)
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute temporal trend statistics for local mapping metrics.")
    parser.add_argument("--in_csv", type=str, default="output/post_cleanup/temporal_ablation/summary.csv")
    parser.add_argument("--out_csv", type=str, default="output/summary_tables/temporal_trend_stats.csv")
    parser.add_argument("--out_json", type=str, default="output/summary_tables/temporal_trend_stats.json")
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"missing input csv: {in_csv}")
    rows = _read_csv(in_csv)
    if not rows:
        raise RuntimeError(f"empty csv: {in_csv}")

    frames = np.array([r["frames"] for r in rows], dtype=float)
    fscore = np.array([r.get("fscore", np.nan) for r in rows], dtype=float)
    ghost_main = np.array([r.get("ghost_count_per_frame", np.nan) for r in rows], dtype=float)
    rho_dyn = np.array([r.get("mean_rho_dynamic", np.nan) for r in rows], dtype=float)
    rho_sta = np.array([r.get("mean_rho_static", np.nan) for r in rows], dtype=float)

    stats = {
        "n_points": int(frames.size),
        "frames_min": float(np.min(frames)),
        "frames_max": float(np.max(frames)),
        "fscore_first": float(fscore[0]),
        "fscore_last": float(fscore[-1]),
        "fscore_delta": float(fscore[-1] - fscore[0]),
        "ghost_main_first": float(ghost_main[0]),
        "ghost_main_last": float(ghost_main[-1]),
        "ghost_main_delta": float(ghost_main[-1] - ghost_main[0]),
        "rho_dyn_first": float(rho_dyn[0]),
        "rho_dyn_last": float(rho_dyn[-1]),
        "rho_sta_first": float(rho_sta[0]),
        "rho_sta_last": float(rho_sta[-1]),
        "spearman_frames_fscore": _safe_spearman(frames, fscore),
        "spearman_frames_ghost_main": _safe_spearman(frames, ghost_main),
        "spearman_frames_rho_dynamic": _safe_spearman(frames, rho_dyn),
        "spearman_frames_rho_static": _safe_spearman(frames, rho_sta),
    }
    stats["ghost_main_nonincreasing"] = 1.0 if stats["ghost_main_last"] <= stats["ghost_main_first"] else 0.0
    stats["fscore_nondecreasing"] = 1.0 if stats["fscore_last"] >= stats["fscore_first"] else 0.0
    stats["trend_pass_spearman"] = (
        1.0 if np.isfinite(stats["spearman_frames_ghost_main"]) and stats["spearman_frames_ghost_main"] <= -0.7 else 0.0
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(stats.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(stats)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[done] temporal trend stats -> {out_csv}")
    print(f"[done] temporal trend json -> {out_json}")


if __name__ == "__main__":
    main()

