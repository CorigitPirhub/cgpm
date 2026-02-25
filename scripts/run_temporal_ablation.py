from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import requests
from scipy.spatial import cKDTree

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.data.tum_rgbd import TUMRGBDStream


def infer_tum_group(sequence: str) -> str:
    if "freiburg1" in sequence:
        return "freiburg1"
    if "freiburg2" in sequence:
        return "freiburg2"
    if "freiburg3" in sequence:
        return "freiburg3"
    raise ValueError(f"Cannot infer TUM freiburg group from sequence name: {sequence}")


def download_tum_sequence(dataset_root: Path, sequence: str) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    seq_dir = dataset_root / sequence
    if seq_dir.exists():
        return seq_dir
    group = infer_tum_group(sequence)
    url = f"https://cvg.cit.tum.de/rgbd/dataset/{group}/{sequence}.tgz"
    archive_path = dataset_root / f"{sequence}.tgz"
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with archive_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    print(f"[extract] {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dataset_root)
    return seq_dir


def run_cmd(cmd: Sequence[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(list(cmd), check=True, cwd=str(PROJECT_ROOT))


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=float)
    if pts.shape[0] > 0:
        return pts
    mesh = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(mesh.vertices, dtype=float)


def downsample_points(points: np.ndarray, voxel: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ds = pcd.voxel_down_sample(max(1e-3, float(voxel)))
    return np.asarray(ds.points, dtype=float)


def build_regions(
    sequence_dir: Path,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    voxel_size: float,
    static_ratio: float = 0.65,
    dynamic_ratio: float = 0.35,
    tail_frames: int = 12,
) -> Tuple[np.ndarray, np.ndarray, set[Tuple[int, int, int]], set[Tuple[int, int, int]], float]:
    cfg = EGF3DConfig()
    stream = TUMRGBDStream(
        sequence_dir=sequence_dir,
        cfg=cfg,
        max_frames=frames,
        stride=stride,
        max_points=max_points_per_frame,
        assoc_max_diff=0.02,
        normal_radius=0.08,
    )
    all_frames = [f for f in stream]
    if not all_frames:
        empty = np.zeros((0, 3), dtype=float)
        return empty, empty, set(), set(), max(0.05, 2.0 * float(voxel_size))

    dyn_voxel = max(0.05, 2.0 * float(voxel_size))
    hits: defaultdict[Tuple[int, int, int], int] = defaultdict(int)
    for frame in all_frames:
        idx = np.floor(frame.points_world / dyn_voxel).astype(np.int32)
        uniq = np.unique(idx, axis=0)
        for v in uniq:
            hits[(int(v[0]), int(v[1]), int(v[2]))] += 1

    n = len(all_frames)
    dynamic_max_hits = max(2, int(np.ceil(dynamic_ratio * n)))
    static_min_hits = max(3, int(np.ceil(static_ratio * n)))
    dynamic_region = {k for k, c in hits.items() if c >= 2 and c <= dynamic_max_hits}
    static_region = {k for k, c in hits.items() if c >= static_min_hits}

    if static_region:
        stable_bg = (np.asarray(list(static_region), dtype=float) + 0.5) * dyn_voxel
    else:
        stable_bg = np.zeros((0, 3), dtype=float)
    stable_bg = downsample_points(stable_bg, voxel=max(0.01, 0.5 * dyn_voxel))

    tail = [f.points_world for f in all_frames[max(0, n - int(tail_frames)) :]]
    tail_points = np.vstack(tail) if tail else np.zeros((0, 3), dtype=float)
    tail_points = downsample_points(tail_points, voxel=max(0.01, 0.5 * dyn_voxel))
    return stable_bg, tail_points, dynamic_region, static_region, dyn_voxel


def compute_dynamic_metrics(
    pred_points: np.ndarray,
    stable_bg_points: np.ndarray,
    tail_points: np.ndarray,
    dynamic_region: set[Tuple[int, int, int]],
    dynamic_voxel: float,
    ghost_thresh: float,
    bg_thresh: float,
) -> Dict[str, float]:
    pred = np.asarray(pred_points, dtype=float)
    if pred.shape[0] == 0:
        return {
            "ghost_count": 0.0,
            "ghost_ratio": 0.0,
            "ghost_tail_count": 0.0,
            "ghost_tail_ratio": 0.0,
            "background_recovery": 0.0,
        }

    tail = np.asarray(tail_points, dtype=float)
    if tail.shape[0] == 0:
        ghost_tail_count = float(pred.shape[0])
        ghost_tail_ratio = 1.0
    else:
        tail_tree = cKDTree(tail)
        d_tail, _ = tail_tree.query(pred, k=1)
        ghost_tail_count = float(np.count_nonzero(d_tail > float(ghost_thresh)))
        ghost_tail_ratio = ghost_tail_count / max(1.0, float(pred.shape[0]))

    idx = np.floor(pred / float(dynamic_voxel)).astype(np.int32)
    dyn_hits = 0
    for v in idx:
        if (int(v[0]), int(v[1]), int(v[2])) in dynamic_region:
            dyn_hits += 1
    ghost_count = float(dyn_hits)
    ghost_ratio = ghost_count / max(1.0, float(pred.shape[0]))

    stable_bg = np.asarray(stable_bg_points, dtype=float)
    if stable_bg.shape[0] == 0:
        bg_recovery = 0.0
    else:
        pred_tree = cKDTree(pred)
        d_bg, _ = pred_tree.query(stable_bg, k=1)
        bg_recovery = float(np.mean(d_bg < float(bg_thresh)))

    return {
        "ghost_count": ghost_count,
        "ghost_ratio": ghost_ratio,
        "ghost_tail_count": ghost_tail_count,
        "ghost_tail_ratio": ghost_tail_ratio,
        "background_recovery": bg_recovery,
    }


def compute_fscore(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return 0.0
    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)
    d_pred, _ = gt_tree.query(pred, k=1)
    d_gt, _ = pred_tree.query(gt, k=1)
    precision = float(np.mean(d_pred < threshold))
    recall = float(np.mean(d_gt < threshold))
    if (precision + recall) <= 1e-9:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_rho_stats(
    npz_path: Path,
    dynamic_region: set[Tuple[int, int, int]],
    static_region: set[Tuple[int, int, int]],
    dynamic_voxel: float,
) -> Tuple[float, float]:
    if not npz_path.exists():
        return float("nan"), float("nan")
    data = np.load(npz_path)
    centers = np.asarray(data["centers"], dtype=float)
    rho = np.asarray(data["rho"], dtype=float)
    if centers.shape[0] == 0:
        return float("nan"), float("nan")
    idx = np.floor(centers / float(dynamic_voxel)).astype(np.int32)
    idx_tuples = [(int(v[0]), int(v[1]), int(v[2])) for v in idx]
    dyn_mask = np.fromiter((t in dynamic_region for t in idx_tuples), dtype=bool, count=len(idx_tuples))
    sta_mask = np.fromiter((t in static_region for t in idx_tuples), dtype=bool, count=len(idx_tuples))

    rho_dyn = float(np.mean(rho[dyn_mask])) if np.any(dyn_mask) else float("nan")
    rho_sta = float(np.mean(rho[sta_mask])) if np.any(sta_mask) else float("nan")
    return rho_dyn, rho_sta


def plot_temporal_curve(rows: Sequence[Dict[str, float]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    frames = np.array([int(r["frames"]) for r in rows], dtype=int)
    ghost = np.array([float(r["ghost_count_per_frame"]) for r in rows], dtype=float)
    fscore = np.array([float(r["fscore"]) for r in rows], dtype=float)
    rho_dyn = np.array([float(r["mean_rho_dynamic"]) for r in rows], dtype=float)
    rho_sta = np.array([float(r["mean_rho_static"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.5), constrained_layout=True)

    axes[0].plot(frames, ghost, marker="o", color="#d62728", linewidth=2.0, label="Standardized Ghost Score")
    axes[0].plot(frames, fscore, marker="s", color="#1f77b4", linewidth=2.0, label="F-score")
    axes[0].set_title("Temporal Convergence (Corrected Ghost Metric)")
    axes[0].set_xlabel("Frame Count")
    axes[0].set_ylabel("Metric Value")
    axes[0].grid(alpha=0.25)
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(loc="best")

    axes[1].plot(frames, rho_dyn, marker="o", color="#ff7f0e", linewidth=2.0, label="Mean rho (Dynamic)")
    axes[1].plot(frames, rho_sta, marker="s", color="#2ca02c", linewidth=2.0, label="Mean rho (Static)")
    axes[1].set_title("Evidence Separation Over Time")
    axes[1].set_xlabel("Frame Count")
    axes[1].set_ylabel("Mean rho")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("EGF-DHMap v6 Temporal Ablation on walking_xyz")
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def plot_rho_evolution(
    frame_dirs: Sequence[Tuple[int, Path]],
    out_png: Path,
    max_points: int = 120000,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = len(frame_dirs)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.4), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    rng = np.random.default_rng(9)
    for i, (frame_count, run_dir) in enumerate(frame_dirs):
        ax = axes[i]
        npz_path = run_dir / "dynamic_score_voxels.npz"
        if not npz_path.exists():
            ax.set_title(f"{frame_count} frames (missing)")
            ax.axis("off")
            continue
        d = np.load(npz_path)
        centers = np.asarray(d["centers"], dtype=float)
        rho = np.asarray(d["rho"], dtype=float)
        if centers.shape[0] == 0:
            ax.set_title(f"{frame_count} frames (empty)")
            ax.axis("off")
            continue
        if centers.shape[0] > max_points:
            keep = rng.choice(centers.shape[0], size=max_points, replace=False)
            centers = centers[keep]
            rho = rho[keep]
        vmax = max(1e-6, float(np.quantile(rho, 0.95)))
        sc = ax.scatter(centers[:, 0], centers[:, 1], c=np.clip(rho, 0.0, vmax), s=0.35, cmap="viridis", alpha=0.8)
        fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        ax.set_title(f"{frame_count} frames")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Rho Field Evolution (top view): dynamic traces fade with time")
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/tum")
    parser.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg3_walking_xyz")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--frames_list", type=str, default="15,30,45,60,90,120")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_points_per_frame", type=int, default=3000)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--ghost_thresh", type=float, default=0.08)
    parser.add_argument("--bg_thresh", type=float, default=0.10)
    parser.add_argument("--out_root", type=str, default="output/temporal_ablation")
    parser.add_argument("--curve_png", type=str, default="assets/temporal_convergence_curve.png")
    parser.add_argument("--rho_png", type=str, default="assets/temporal_rho_evolution.png")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.download:
        seq_dir = download_tum_sequence(dataset_root, args.sequence)
    else:
        seq_dir = dataset_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(f"sequence not found: {seq_dir}")

    frame_counts = [int(x.strip()) for x in args.frames_list.split(",") if x.strip()]
    frame_counts = sorted(set(frame_counts))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    rho_panels: List[Tuple[int, Path]] = []
    max_frames = max(frame_counts)
    _, _, fixed_dynamic_region, _, fixed_dynamic_voxel = build_regions(
        sequence_dir=seq_dir,
        frames=max_frames,
        stride=args.stride,
        max_points_per_frame=args.max_points_per_frame,
        voxel_size=args.voxel_size,
    )
    for frames in frame_counts:
        run_dir = out_root / f"frames_{frames:03d}" / "egf"
        summary_path = run_dir / "summary.json"
        if not summary_path.exists() or args.force:
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/run_egf_3d_tum.py",
                "--dataset_root",
                str(dataset_root),
                "--sequence",
                args.sequence,
                "--frames",
                str(frames),
                "--stride",
                str(args.stride),
                "--max_points_per_frame",
                str(args.max_points_per_frame),
                "--voxel_size",
                str(args.voxel_size),
                "--surface_eval_thresh",
                str(args.eval_thresh),
                "--out",
                str(run_dir),
                "--sigma_n0",
                "0.26",
                "--rho_decay",
                "0.998",
                "--phi_w_decay",
                "0.998",
                "--dynamic_forgetting",
                "--forget_mode",
                "local",
                "--dyn_forget_gain",
                "0.0",
                "--dyn_score_alpha",
                "0.08",
                "--dyn_d2_ref",
                "7.0",
                "--dscore_ema",
                "0.12",
                "--residual_score_weight",
                "0.25",
                "--raycast_clear_gain",
                "0.0",
                "--raycast_step_scale",
                "0.75",
                "--raycast_end_margin",
                "0.16",
                "--raycast_max_rays",
                "1500",
                "--raycast_rho_max",
                "20.0",
                "--raycast_phiw_max",
                "220.0",
                "--raycast_dyn_boost",
                "0.6",
                "--surface_phi_thresh",
                "0.8",
                "--surface_rho_thresh",
                "0.0",
                "--surface_min_weight",
                "0.0",
                "--surface_max_dscore",
                "1.0",
                "--surface_max_free_ratio",
                "1000000000.0",
                "--mesh_min_points",
                "100000000",
                "--save_dscore_map",
                "--dscore_min_weight",
                "0.2",
            ]
            run_cmd(cmd)

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        pred_points = load_points(run_dir / "surface_points.ply")
        stable_bg, tail_points, dyn_region, static_region, dyn_voxel = build_regions(
            sequence_dir=seq_dir,
            frames=frames,
            stride=args.stride,
            max_points_per_frame=args.max_points_per_frame,
            voxel_size=args.voxel_size,
        )
        dyn_metrics = compute_dynamic_metrics(
            pred_points=pred_points,
            stable_bg_points=stable_bg,
            tail_points=tail_points,
            dynamic_region=dyn_region,
            dynamic_voxel=dyn_voxel,
            ghost_thresh=args.ghost_thresh,
            bg_thresh=args.bg_thresh,
        )
        # Fixed-region ghost metrics to remove denominator/coverage drift across frame counts.
        fixed_idx = np.floor(pred_points / float(fixed_dynamic_voxel)).astype(np.int32) if pred_points.size > 0 else np.zeros((0, 3), dtype=np.int32)
        fixed_hits = 0
        for v in fixed_idx:
            if (int(v[0]), int(v[1]), int(v[2])) in fixed_dynamic_region:
                fixed_hits += 1
        ghost_count_fixed = float(fixed_hits)
        ghost_ratio_fixed = ghost_count_fixed / max(1.0, float(pred_points.shape[0]))
        ghost_count_per_frame = ghost_count_fixed / max(1.0, float(frames))
        bg_fscore = compute_fscore(pred_points, stable_bg, threshold=args.bg_thresh)
        rho_dyn, rho_sta = compute_rho_stats(
            npz_path=run_dir / "dynamic_score_voxels.npz",
            dynamic_region=dyn_region,
            static_region=static_region,
            dynamic_voxel=dyn_voxel,
        )
        row = {
            "frames": float(frames),
            "surface_points": float(pred_points.shape[0]),
            "fscore": float(summary["metrics"]["fscore"]),
            "ghost_count": float(dyn_metrics["ghost_count"]),
            "ghost_ratio": float(dyn_metrics["ghost_ratio"]),
            "ghost_tail_count": float(dyn_metrics["ghost_tail_count"]),
            "ghost_tail_ratio": float(dyn_metrics["ghost_tail_ratio"]),
            "ghost_count_fixed_region": ghost_count_fixed,
            "ghost_ratio_fixed_region": ghost_ratio_fixed,
            "ghost_count_per_frame": ghost_count_per_frame,
            "background_recovery": float(dyn_metrics["background_recovery"]),
            "background_fscore": float(bg_fscore),
            "mean_rho_dynamic": float(rho_dyn),
            "mean_rho_static": float(rho_sta),
            "assoc_ratio_mean": float(summary.get("assoc_ratio_mean", 0.0)),
        }
        rows.append(row)
        rho_panels.append((frames, run_dir))
        with (out_root / f"frames_{frames:03d}" / "temporal_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

    rows = sorted(rows, key=lambda x: int(x["frames"]))
    headers = [
        "frames",
        "surface_points",
        "fscore",
        "ghost_count",
        "ghost_ratio",
        "ghost_tail_count",
        "ghost_tail_ratio",
        "ghost_count_fixed_region",
        "ghost_ratio_fixed_region",
        "ghost_count_per_frame",
        "background_recovery",
        "background_fscore",
        "mean_rho_dynamic",
        "mean_rho_static",
        "assoc_ratio_mean",
    ]
    write_csv(out_root / "summary.csv", rows, headers=headers)
    plot_temporal_curve(rows, Path(args.curve_png))
    plot_rho_evolution(sorted(rho_panels, key=lambda x: x[0]), Path(args.rho_png))
    print(f"[done] temporal summary: {out_root / 'summary.csv'}")
    print(f"[done] curve: {args.curve_png}")
    print(f"[done] rho evolution: {args.rho_png}")


if __name__ == "__main__":
    main()
