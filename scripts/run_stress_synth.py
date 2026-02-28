from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd: Sequence[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(list(cmd), check=True, cwd=str(PROJECT_ROOT))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


@dataclass
class SweepCase:
    factor: str
    level: float
    dynamic_ratio: float
    speed: float
    occlusion: float

    @property
    def sequence_name(self) -> str:
        return (
            f"synth_{self.factor}_"
            f"d{int(round(self.dynamic_ratio * 100)):02d}_"
            f"v{int(round(self.speed * 100)):03d}_"
            f"o{int(round(self.occlusion * 100)):02d}"
        )


def make_sweep_cases() -> List[SweepCase]:
    base = {"dynamic_ratio": 0.15, "speed": 1.0, "occlusion": 0.20}
    cases: List[SweepCase] = []
    for r in [0.08, 0.15, 0.28]:
        p = dict(base)
        p["dynamic_ratio"] = r
        cases.append(SweepCase("dynamic_ratio", r, p["dynamic_ratio"], p["speed"], p["occlusion"]))
    for v in [0.60, 1.00, 1.50]:
        p = dict(base)
        p["speed"] = v
        cases.append(SweepCase("speed", v, p["dynamic_ratio"], p["speed"], p["occlusion"]))
    for o in [0.00, 0.20, 0.40]:
        p = dict(base)
        p["occlusion"] = o
        cases.append(SweepCase("occlusion", o, p["dynamic_ratio"], p["speed"], p["occlusion"]))
    return cases


def _render_depth_frame(
    h: int,
    w: int,
    frame_idx: int,
    total_frames: int,
    dynamic_ratio: float,
    speed: float,
    occlusion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Base scene: static far wall at ~2.8m.
    z_bg = 2.8
    z_obj = 1.6
    depth = np.full((h, w), z_bg, dtype=np.float32)

    # Dynamic object area controls dynamic_ratio; square-like footprint.
    area = max(64.0, float(dynamic_ratio) * float(h * w))
    side = int(np.sqrt(area))
    obj_h = int(np.clip(side, 8, h // 2))
    obj_w = int(np.clip(side, 8, w // 2))

    # Motion speed controls travel distance per sequence.
    t = 0.0 if total_frames <= 1 else float(frame_idx) / float(total_frames - 1)
    travel = float(speed) * float(w + obj_w)
    cx = int(-obj_w // 2 + t * travel)
    cy = h // 2 + int(0.10 * h * np.sin(2.0 * np.pi * t))

    x0 = max(0, cx - obj_w // 2)
    x1 = min(w, cx + obj_w // 2)
    y0 = max(0, cy - obj_h // 2)
    y1 = min(h, cy + obj_h // 2)
    if x1 > x0 and y1 > y0:
        depth[y0:y1, x0:x1] = z_obj

        # Occlusion: object region dropouts + edge jitter.
        p_obj_drop = float(np.clip(0.10 + 0.70 * occlusion, 0.0, 0.95))
        mask_obj_drop = rng.random((y1 - y0, x1 - x0)) < p_obj_drop
        depth[y0:y1, x0:x1][mask_obj_drop] = 0.0

        # Shadow/occlusion band behind moving object.
        band = int(max(1, round(6 + 14 * occlusion)))
        bx0 = max(0, x1)
        bx1 = min(w, x1 + band)
        if bx1 > bx0:
            p_band = float(np.clip(0.02 + 0.35 * occlusion, 0.0, 0.8))
            mask_band = rng.random((y1 - y0, bx1 - bx0)) < p_band
            depth[y0:y1, bx0:bx1][mask_band] = 0.0

    # Global missing depth under stronger occlusion.
    p_bg_drop = float(np.clip(0.005 + 0.06 * occlusion, 0.0, 0.2))
    bg_drop = rng.random((h, w)) < p_bg_drop
    depth[bg_drop] = 0.0

    # Depth noise.
    sigma = float(0.003 + 0.008 * occlusion)
    noise = rng.normal(0.0, sigma, size=(h, w)).astype(np.float32)
    valid = depth > 0.0
    depth[valid] += noise[valid]
    depth = np.clip(depth, 0.0, 6.0)
    return depth


def generate_synth_tum_sequence(
    dataset_root: Path,
    case: SweepCase,
    frames: int,
    width: int,
    height: int,
    seed: int,
    force: bool,
) -> Path:
    seq_dir = dataset_root / case.sequence_name
    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"
    rgb_txt = seq_dir / "rgb.txt"
    depth_txt = seq_dir / "depth.txt"
    gt_txt = seq_dir / "groundtruth.txt"

    if seq_dir.exists() and not force:
        return seq_dir

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rgb_lines: List[str] = ["# timestamp filename"]
    depth_lines: List[str] = ["# timestamp filename"]
    gt_lines: List[str] = ["# timestamp tx ty tz qx qy qz qw"]

    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
    dt = 1.0 / 30.0
    for i in range(int(frames)):
        ts = i * dt
        depth = _render_depth_frame(
            h=height,
            w=width,
            frame_idx=i,
            total_frames=int(frames),
            dynamic_ratio=case.dynamic_ratio,
            speed=case.speed,
            occlusion=case.occlusion,
            rng=rng,
        )
        depth_u16 = np.round(depth * 5000.0).astype(np.uint16)

        rgb_name = f"{i:06d}.png"
        depth_name = f"{i:06d}.png"
        imageio.imwrite(rgb_dir / rgb_name, rgb_img)
        imageio.imwrite(depth_dir / depth_name, depth_u16)

        rgb_lines.append(f"{ts:.6f} rgb/{rgb_name}")
        depth_lines.append(f"{ts:.6f} depth/{depth_name}")
        gt_lines.append(f"{ts:.6f} 0 0 0 0 0 0 1")

    rgb_txt.write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    depth_txt.write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    gt_txt.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    return seq_dir


def parse_table(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def as_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def build_curves(summary_rows: Sequence[Dict[str, object]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    factors = ["dynamic_ratio", "speed", "occlusion"]
    methods = ["egf", "tsdf", "simple_removal"]
    method_colors = {"egf": "#1f77b4", "tsdf": "#d62728", "simple_removal": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True)
    for ax, factor in zip(axes, factors):
        subset = [r for r in summary_rows if str(r["factor"]) == factor]
        levels = sorted({float(r["level"]) for r in subset})
        ax2 = ax.twinx()
        for m in methods:
            xs: List[float] = []
            ghosts: List[float] = []
            fscores: List[float] = []
            for lv in levels:
                rows = [r for r in subset if str(r["method"]) == m and abs(float(r["level"]) - lv) < 1e-9]
                if not rows:
                    continue
                xs.append(lv)
                ghosts.append(float(np.mean([float(r["ghost_ratio"]) for r in rows])))
                fscores.append(float(np.mean([float(r["fscore"]) for r in rows])))
            if xs:
                ax.plot(xs, ghosts, marker="o", linewidth=1.8, color=method_colors[m], alpha=0.85, label=f"{m}: ghost")
                ax2.plot(xs, fscores, marker="s", linewidth=1.4, linestyle="--", color=method_colors[m], alpha=0.75, label=f"{m}: fscore")
        ax.set_title(factor)
        ax.set_xlabel("level")
        ax.set_ylabel("ghost_ratio (lower)")
        ax2.set_ylabel("fscore (higher)")
        ax.grid(alpha=0.25)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Synthetic Stress Scan: dynamic_ratio / speed / occlusion", y=1.03)
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/synth_stress")
    parser.add_argument("--out_root", type=str, default="output/post_cleanup/p3_stress_synth")
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_points_per_frame", type=int, default=2500)
    parser.add_argument("--voxel_size", type=float, default=0.03)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--methods", type=str, default="egf,tsdf,simple_removal")
    parser.add_argument("--skip_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    raw_root = out_root / "raw"
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    cases = make_sweep_cases()
    print(f"[info] stress cases={len(cases)}")

    for i, case in enumerate(cases):
        generate_synth_tum_sequence(
            dataset_root=dataset_root,
            case=case,
            frames=args.frames,
            width=args.width,
            height=args.height,
            seed=args.seed + i * 13,
            force=args.force,
        )

    if not args.skip_run:
        for case in cases:
            case_out = raw_root / case.sequence_name
            cmd = [
                sys.executable,
                "scripts/run_benchmark.py",
                "--dataset_kind",
                "tum",
                "--dataset_root",
                str(dataset_root),
                "--static_sequences",
                "",
                "--dynamic_sequences",
                case.sequence_name,
                "--methods",
                args.methods,
                "--frames",
                str(args.frames),
                "--stride",
                str(args.stride),
                "--max_points_per_frame",
                str(args.max_points_per_frame),
                "--voxel_size",
                str(args.voxel_size),
                "--eval_thresh",
                str(args.eval_thresh),
                "--protocol",
                "slam",
                "--seed",
                str(args.seed),
                "--out_root",
                str(case_out),
                "--force",
            ]
            run_cmd(cmd)

    summary_rows: List[Dict[str, object]] = []
    for case in cases:
        case_out = raw_root / case.sequence_name / "slam" / "tables"
        dyn_rows = parse_table(case_out / "dynamic_metrics.csv")
        rec_rows = parse_table(case_out / "reconstruction_metrics.csv")
        rec_by_method = {str(r.get("method", "")): r for r in rec_rows}
        for d in dyn_rows:
            method = str(d.get("method", "")).strip()
            if not method:
                continue
            rr = rec_by_method.get(method, {})
            summary_rows.append(
                {
                    "sequence": case.sequence_name,
                    "factor": case.factor,
                    "level": float(case.level),
                    "dynamic_ratio": float(case.dynamic_ratio),
                    "speed": float(case.speed),
                    "occlusion": float(case.occlusion),
                    "method": method,
                    "fscore": as_float(d, "fscore"),
                    "normal_consistency": as_float(d, "normal_consistency"),
                    "ghost_ratio": as_float(d, "ghost_ratio"),
                    "ghost_tail_ratio": as_float(d, "ghost_tail_ratio"),
                    "background_recovery": as_float(d, "background_recovery"),
                    "chamfer": as_float(rr, "chamfer"),
                    "precision": as_float(rr, "precision"),
                    "recall": as_float(rr, "recall"),
                }
            )

    headers = [
        "sequence",
        "factor",
        "level",
        "dynamic_ratio",
        "speed",
        "occlusion",
        "method",
        "fscore",
        "normal_consistency",
        "ghost_ratio",
        "ghost_tail_ratio",
        "background_recovery",
        "chamfer",
        "precision",
        "recall",
    ]
    write_csv(out_root / "stress_summary.csv", summary_rows, headers)

    # Factor-level aggregate for compact reporting.
    agg_map: Dict[tuple[str, float, str], List[Dict[str, object]]] = {}
    for row in summary_rows:
        k = (str(row["factor"]), float(row["level"]), str(row["method"]))
        agg_map.setdefault(k, []).append(row)
    agg_rows: List[Dict[str, object]] = []
    for (factor, level, method), vals in sorted(agg_map.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        agg_rows.append(
            {
                "factor": factor,
                "level": level,
                "method": method,
                "n_cases": int(len(vals)),
                "fscore_mean": float(np.mean([float(v["fscore"]) for v in vals])),
                "ghost_ratio_mean": float(np.mean([float(v["ghost_ratio"]) for v in vals])),
                "background_recovery_mean": float(np.mean([float(v["background_recovery"]) for v in vals])),
                "chamfer_mean": float(np.mean([float(v["chamfer"]) for v in vals])),
            }
        )
    write_csv(
        out_root / "stress_summary_agg.csv",
        agg_rows,
        ["factor", "level", "method", "n_cases", "fscore_mean", "ghost_ratio_mean", "background_recovery_mean", "chamfer_mean"],
    )

    build_curves(summary_rows, out_root / "stress_curves.png")
    with (out_root / "stress_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": summary_rows, "agg_rows": agg_rows}, f, indent=2)
    print(f"[done] stress summary: {out_root / 'stress_summary.csv'}")
    print(f"[done] stress aggregate: {out_root / 'stress_summary_agg.csv'}")
    print(f"[done] stress figure: {out_root / 'stress_curves.png'}")


if __name__ == "__main__":
    main()

