from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from egf_dhmap3d.core.config import EGF3DConfig
from data.bonn_rgbd import BonnRGBDStream, download_bonn_sequence


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


def compute_recon_metrics(pred_points: np.ndarray, gt_points: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = np.asarray(pred_points, dtype=float)
    gt = np.asarray(gt_points, dtype=float)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return {
            "accuracy": float("inf"),
            "completeness": float("inf"),
            "chamfer": float("inf"),
            "hausdorff": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
        }
    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)
    d_pred_to_gt, _ = gt_tree.query(pred, k=1)
    d_gt_to_pred, _ = pred_tree.query(gt, k=1)
    precision = float(np.mean(d_pred_to_gt < threshold))
    recall = float(np.mean(d_gt_to_pred < threshold))
    fscore = 0.0 if (precision + recall) <= 1e-9 else float(2.0 * precision * recall / (precision + recall))
    return {
        "accuracy": float(np.mean(d_pred_to_gt)),
        "completeness": float(np.mean(d_gt_to_pred)),
        "chamfer": float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)),
        "hausdorff": float(max(np.max(d_pred_to_gt), np.max(d_gt_to_pred))),
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def build_regions(
    sequence_dir: Path,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    voxel_size: float,
    static_ratio: float = 0.65,
    dynamic_ratio: float = 0.35,
    tail_frames: int = 12,
) -> Tuple[np.ndarray, np.ndarray, set[Tuple[int, int, int]], float]:
    cfg = EGF3DConfig()
    stream = BonnRGBDStream(
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
        return empty, empty, set(), max(0.05, 2.0 * float(voxel_size))

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
    return stable_bg, tail_points, dynamic_region, dyn_voxel


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


def plot_bonn_compare(
    method_points: Dict[str, np.ndarray],
    stable_bg: np.ndarray,
    table_rows: Dict[str, Dict[str, float]],
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    methods = ["egf", "tsdf"]
    titles = {"egf": "EGF-DHMap v6", "tsdf": "TSDF"}

    stable_bg = np.asarray(stable_bg, dtype=float)
    tree = cKDTree(stable_bg) if stable_bg.shape[0] > 0 else None

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    for ax, m in zip(axes, methods):
        pts = np.asarray(method_points.get(m, np.zeros((0, 3))), dtype=float)
        if pts.shape[0] > 120000:
            rng = np.random.default_rng(5)
            keep = rng.choice(pts.shape[0], size=120000, replace=False)
            pts = pts[keep]
        if pts.shape[0] > 0:
            if tree is not None:
                d, _ = tree.query(pts, k=1)
            else:
                d = np.zeros((pts.shape[0],), dtype=float)
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.clip(d, 0.0, 0.15), s=0.35, cmap="turbo", alpha=0.78)
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        if stable_bg.shape[0] > 0:
            bg = stable_bg
            if bg.shape[0] > 80000:
                rng = np.random.default_rng(9)
                keep = rng.choice(bg.shape[0], size=80000, replace=False)
                bg = bg[keep]
            ax.scatter(bg[:, 0], bg[:, 1], s=0.08, c="lightgray", alpha=0.18)
        row = table_rows.get(m, {})
        ax.set_title(
            f"{titles[m]}\n"
            f"F={row.get('fscore', 0.0):.3f}, "
            f"Ghost={row.get('ghost_ratio', 0.0):.3f}, "
            f"BgRec={row.get('background_recovery', 0.0):.3f}"
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Bonn Generalization: balloon2 (error-colored to stable background)")
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/bonn")
    parser.add_argument("--sequence", type=str, default="rgbd_bonn_balloon2")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_points_per_frame", type=int, default=3000)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--ghost_thresh", type=float, default=0.08)
    parser.add_argument("--bg_thresh", type=float, default=0.10)
    parser.add_argument("--out_root", type=str, default="output/benchmark_bonn")
    parser.add_argument("--compare_png", type=str, default="assets/bonn_comparison.png")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.download:
        seq_dir = download_bonn_sequence(dataset_root, args.sequence)
    else:
        seq_dir = dataset_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(f"Bonn sequence not found: {seq_dir}")

    out_root = Path(args.out_root)
    seq_out = out_root / args.sequence
    egf_out = seq_out / "egf"
    tsdf_out = seq_out / "tsdf"
    seq_out.mkdir(parents=True, exist_ok=True)

    if args.force or not (egf_out / "summary.json").exists():
        run_cmd(
            [
                sys.executable,
                "scripts/run_egf_3d_tum.py",
                "--dataset_root",
                str(dataset_root),
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
                str(egf_out),
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
            ]
        )

    if args.force or not (tsdf_out / "summary.json").exists():
        run_cmd(
            [
                sys.executable,
                "scripts/run_tsdf_baseline.py",
                "--dataset_root",
                str(dataset_root),
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
                str(tsdf_out),
            ]
        )

    stable_bg, tail_points, dynamic_region, dynamic_voxel = build_regions(
        sequence_dir=seq_dir,
        frames=args.frames,
        stride=args.stride,
        max_points_per_frame=args.max_points_per_frame,
        voxel_size=args.voxel_size,
    )
    if stable_bg.shape[0] > 0:
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(stable_bg)
        o3d.io.write_point_cloud(str(seq_out / "stable_background_reference.ply"), bg_pcd)

    gt_points = load_points(egf_out / "reference_points.ply")
    rows: List[Dict[str, object]] = []
    table_rows: Dict[str, Dict[str, float]] = {}
    method_points: Dict[str, np.ndarray] = {}
    for method, out_dir in [("egf", egf_out), ("tsdf", tsdf_out)]:
        pred_points = load_points(out_dir / "surface_points.ply")
        method_points[method] = pred_points
        recon = compute_recon_metrics(pred_points, gt_points, threshold=args.eval_thresh)
        dyn = compute_dynamic_metrics(
            pred_points=pred_points,
            stable_bg_points=stable_bg,
            tail_points=tail_points,
            dynamic_region=dynamic_region,
            dynamic_voxel=dynamic_voxel,
            ghost_thresh=args.ghost_thresh,
            bg_thresh=args.bg_thresh,
        )
        row = {
            "sequence": args.sequence,
            "method": method,
            "points": float(pred_points.shape[0]),
            "accuracy": recon["accuracy"],
            "completeness": recon["completeness"],
            "chamfer": recon["chamfer"],
            "hausdorff": recon["hausdorff"],
            "precision": recon["precision"],
            "recall": recon["recall"],
            "fscore": recon["fscore"],
            "ghost_count": dyn["ghost_count"],
            "ghost_ratio": dyn["ghost_ratio"],
            "ghost_tail_count": dyn["ghost_tail_count"],
            "ghost_tail_ratio": dyn["ghost_tail_ratio"],
            "background_recovery": dyn["background_recovery"],
        }
        rows.append(row)
        table_rows[method] = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        with (out_dir / "benchmark_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

    headers = [
        "sequence",
        "method",
        "points",
        "accuracy",
        "completeness",
        "chamfer",
        "hausdorff",
        "precision",
        "recall",
        "fscore",
        "ghost_count",
        "ghost_ratio",
        "ghost_tail_count",
        "ghost_tail_ratio",
        "background_recovery",
    ]
    write_csv(out_root / "summary.csv", rows, headers=headers)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    plot_bonn_compare(method_points, stable_bg, table_rows, Path(args.compare_png))
    print(f"[done] Bonn summary: {out_root / 'summary.csv'}")
    print(f"[done] Bonn figure: {args.compare_png}")


if __name__ == "__main__":
    main()

