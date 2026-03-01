#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def to_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def find_row(rows: List[Dict[str, str]], variant: str, sequence: str) -> Dict[str, str]:
    for r in rows:
        if str(r.get("variant", "")) == str(variant) and str(r.get("sequence", "")) == str(sequence):
            return r
    return {}


def monotonic_steps(vals: List[float], increasing: bool) -> int:
    c = 0
    for i in range(1, len(vals)):
        if increasing and vals[i] >= vals[i - 1]:
            c += 1
        if (not increasing) and vals[i] <= vals[i - 1]:
            c += 1
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 mechanism acceptance check (ablation + temporal).")
    parser.add_argument("--ablation_csv", type=str, default="output/summary_tables/ablation_summary.csv")
    parser.add_argument("--temporal_csv", type=str, default="output/summary_tables/temporal_ablation_summary.csv")
    parser.add_argument("--out_root", type=str, default="output/post_cleanup/p2_mechanism_lock")
    parser.add_argument("--summary_root", type=str, default="output/summary_tables")
    parser.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg3_walking_xyz")
    parser.add_argument("--min_gradient_drop", type=float, default=0.05)
    parser.add_argument("--min_temporal_step_agreement", type=int, default=4)
    args = parser.parse_args()

    ablation_csv = Path(args.ablation_csv)
    temporal_csv = Path(args.temporal_csv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_root = Path(args.summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    if not ablation_csv.exists():
        raise FileNotFoundError(f"missing ablation csv: {ablation_csv}")
    if not temporal_csv.exists():
        raise FileNotFoundError(f"missing temporal csv: {temporal_csv}")

    ab_rows = read_csv(ablation_csv)
    tp_rows = read_csv(temporal_csv)

    full = find_row(ab_rows, "EGF-Full-v6 (budget30)", args.sequence)
    no_evidence = find_row(ab_rows, "EGF-No-Evidence", args.sequence)
    no_gradient = find_row(ab_rows, "EGF-No-Gradient", args.sequence)
    classic = find_row(ab_rows, "EGF-Classic-SDF", args.sequence)
    if not (full and no_evidence and no_gradient and classic):
        raise RuntimeError("ablation csv does not contain required variants for the target sequence")

    full_f = to_float(full, "fscore")
    no_evi_f = to_float(no_evidence, "fscore")
    no_gra_f = to_float(no_gradient, "fscore")
    full_ghost_tail = to_float(full, "ghost_tail_ratio")
    no_evi_ghost_tail = to_float(no_evidence, "ghost_tail_ratio")
    full_ghost_count = to_float(full, "ghost_count")
    no_evi_ghost_count = to_float(no_evidence, "ghost_count")

    cond_evidence = (no_evi_ghost_tail > full_ghost_tail) or (no_evi_ghost_count > full_ghost_count)
    cond_gradient = (full_f - no_gra_f) >= float(args.min_gradient_drop)

    tp_rows_sorted = sorted(tp_rows, key=lambda r: float(r.get("frames", "0")))
    frames = [float(r["frames"]) for r in tp_rows_sorted]
    fscores = [to_float(r, "fscore") for r in tp_rows_sorted]
    ghost_main = [to_float(r, "ghost_count_per_frame") for r in tp_rows_sorted]
    rho_dyn = [to_float(r, "mean_rho_dynamic") for r in tp_rows_sorted]
    rho_sta = [to_float(r, "mean_rho_static") for r in tp_rows_sorted]

    # Temporal acceptance: global trend + step consistency.
    cond_fscore_trend = fscores[-1] >= fscores[0]
    cond_ghost_trend = ghost_main[-1] <= ghost_main[0]
    cond_fscore_steps = monotonic_steps(fscores, increasing=True) >= int(args.min_temporal_step_agreement)
    cond_ghost_steps = monotonic_steps(ghost_main, increasing=False) >= int(args.min_temporal_step_agreement)
    cond_temporal = cond_fscore_trend and cond_ghost_trend and cond_fscore_steps and cond_ghost_steps

    pass_all = bool(cond_evidence and cond_gradient and cond_temporal)

    report = {
        "task": "P2 mechanism evidence",
        "inputs": {
            "ablation_csv": str(ablation_csv),
            "temporal_csv": str(temporal_csv),
            "sequence": args.sequence,
        },
        "checks": {
            "evidence_ablation": {
                "full_ghost_tail_ratio": full_ghost_tail,
                "no_evidence_ghost_tail_ratio": no_evi_ghost_tail,
                "full_ghost_count": full_ghost_count,
                "no_evidence_ghost_count": no_evi_ghost_count,
                "pass": bool(cond_evidence),
            },
            "gradient_ablation": {
                "full_fscore": full_f,
                "no_gradient_fscore": no_gra_f,
                "drop": float(full_f - no_gra_f),
                "required_drop": float(args.min_gradient_drop),
                "pass": bool(cond_gradient),
            },
            "temporal_mechanism": {
                "frames": frames,
                "fscore": fscores,
                "ghost_count_per_frame": ghost_main,
                "mean_rho_dynamic": rho_dyn,
                "mean_rho_static": rho_sta,
                "fscore_steps_non_decrease": int(monotonic_steps(fscores, increasing=True)),
                "ghost_steps_non_increase": int(monotonic_steps(ghost_main, increasing=False)),
                "pass": bool(cond_temporal),
            },
        },
        "pass": pass_all,
    }
    report_path = out_root / "p2_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# P2 Mechanism Report",
        "",
        f"- pass: `{pass_all}`",
        f"- ablation_csv: `{ablation_csv}`",
        f"- temporal_csv: `{temporal_csv}`",
        "",
        "## Check-1 Evidence Ablation",
        f"- full ghost_tail_ratio: `{full_ghost_tail:.6f}`",
        f"- no_evidence ghost_tail_ratio: `{no_evi_ghost_tail:.6f}`",
        f"- full ghost_count: `{full_ghost_count:.1f}`",
        f"- no_evidence ghost_count: `{no_evi_ghost_count:.1f}`",
        f"- pass: `{cond_evidence}`",
        "",
        "## Check-2 Gradient Ablation",
        f"- full fscore: `{full_f:.6f}`",
        f"- no_gradient fscore: `{no_gra_f:.6f}`",
        f"- fscore drop: `{(full_f - no_gra_f):.6f}` (required >= `{args.min_gradient_drop}`)",
        f"- pass: `{cond_gradient}`",
        "",
        "## Check-3 Temporal Mechanism (main ghost metric = ghost_count_per_frame)",
        f"- fscore first->last: `{fscores[0]:.6f} -> {fscores[-1]:.6f}`",
        f"- ghost_count_per_frame first->last: `{ghost_main[0]:.6f} -> {ghost_main[-1]:.6f}`",
        f"- fscore non-decrease steps: `{monotonic_steps(fscores, increasing=True)}`",
        f"- ghost non-increase steps: `{monotonic_steps(ghost_main, increasing=False)}`",
        f"- pass: `{cond_temporal}`",
        "",
        f"- json report: `{report_path}`",
    ]
    (out_root / "P2_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Publish locked copies for release usage.
    shutil.copy2(ablation_csv, summary_root / "ablation_summary_p2.csv")
    shutil.copy2(temporal_csv, summary_root / "temporal_ablation_summary_p2.csv")

    print(f"[done] P2 report: {report_path}")
    if not pass_all:
        raise SystemExit("P2 acceptance failed")


if __name__ == "__main__":
    main()

