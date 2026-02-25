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
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.data.tum_rgbd import TUMRGBDStream
from data.bonn_rgbd import download_bonn_sequence


def infer_tum_group(sequence: str) -> str:
    if "freiburg1" in sequence:
        return "freiburg1"
    if "freiburg2" in sequence:
        return "freiburg2"
    if "freiburg3" in sequence:
        return "freiburg3"
    raise ValueError(f"Cannot infer TUM freiburg group from sequence name: {sequence}")


def infer_dataset_kind(sequence: str, dataset_kind: str) -> str:
    kind = dataset_kind.strip().lower()
    if kind in {"tum", "bonn"}:
        return kind
    # auto mode
    if sequence.startswith("rgbd_bonn_"):
        return "bonn"
    return "tum"


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


def build_dynamic_references(
    sequence_dir: Path,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    stable_voxel: float = 0.03,
    stable_ratio: float = 0.25,
    tail_frames: int = 12,
    dynamic_voxel: float = 0.05,
    min_dynamic_hits: int = 2,
    max_dynamic_ratio: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, set[Tuple[int, int, int]], float]:
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
    all_frames = [frame for frame in stream]
    if not all_frames:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    voxel_hits: defaultdict[Tuple[int, int, int], int] = defaultdict(int)
    dynamic_hits: defaultdict[Tuple[int, int, int], int] = defaultdict(int)
    for frame in all_frames:
        pts = frame.points_world
        idx = np.floor(pts / float(stable_voxel)).astype(np.int32)
        uniq = np.unique(idx, axis=0)
        for v in uniq:
            voxel_hits[(int(v[0]), int(v[1]), int(v[2]))] += 1
        idx_dyn = np.floor(pts / float(dynamic_voxel)).astype(np.int32)
        uniq_dyn = np.unique(idx_dyn, axis=0)
        for v in uniq_dyn:
            dynamic_hits[(int(v[0]), int(v[1]), int(v[2]))] += 1

    hit_thresh = max(3, int(np.ceil(float(stable_ratio) * float(len(all_frames)))))
    stable_keys = [k for k, c in voxel_hits.items() if c >= hit_thresh]
    if stable_keys:
        stable_bg = (np.asarray(stable_keys, dtype=float) + 0.5) * float(stable_voxel)
    else:
        stable_bg = np.zeros((0, 3), dtype=float)

    tail = [f.points_world for f in all_frames[max(0, len(all_frames) - int(tail_frames)) :]]
    tail_points = np.vstack(tail) if tail else np.zeros((0, 3), dtype=float)
    stable_bg = downsample_points(stable_bg, voxel=max(0.5 * stable_voxel, 0.01))
    tail_points = downsample_points(tail_points, voxel=max(0.5 * stable_voxel, 0.01))
    max_dynamic_hits = max(int(min_dynamic_hits), int(np.ceil(float(max_dynamic_ratio) * float(len(all_frames)))))
    dynamic_region = {k for k, c in dynamic_hits.items() if int(c) >= int(min_dynamic_hits) and int(c) <= int(max_dynamic_hits)}
    return stable_bg, tail_points, dynamic_region, float(dynamic_voxel)


def compute_dynamic_metrics(
    pred_points: np.ndarray,
    stable_bg_points: np.ndarray,
    tail_points: np.ndarray,
    dynamic_region: set[Tuple[int, int, int]] | None,
    dynamic_voxel: float,
    ghost_thresh: float,
    bg_thresh: float,
) -> Dict[str, float]:
    pred = np.asarray(pred_points, dtype=float)
    stable_bg = np.asarray(stable_bg_points, dtype=float)
    tail = np.asarray(tail_points, dtype=float)
    if pred.shape[0] == 0:
        return {
            "ghost_count": 0.0,
            "ghost_ratio": 0.0,
            "ghost_tail_count": 0.0,
            "ghost_tail_ratio": 0.0,
            "background_recovery": 0.0,
        }
    if tail.shape[0] == 0:
        ghost_count = float(pred.shape[0])
        ghost_ratio = 1.0
    else:
        tail_tree = cKDTree(tail)
        d_tail, _ = tail_tree.query(pred, k=1)
        ghost_count = float(np.count_nonzero(d_tail > float(ghost_thresh)))
        ghost_ratio = ghost_count / max(1.0, float(pred.shape[0]))

    if dynamic_region:
        idx = np.floor(pred / float(dynamic_voxel)).astype(np.int32)
        dyn_hits = 0
        for v in idx:
            if (int(v[0]), int(v[1]), int(v[2])) in dynamic_region:
                dyn_hits += 1
        ghost_count_dyn = float(dyn_hits)
        ghost_ratio_dyn = ghost_count_dyn / max(1.0, float(pred.shape[0]))
    else:
        ghost_count_dyn = ghost_count
        ghost_ratio_dyn = ghost_ratio

    if stable_bg.shape[0] == 0:
        bg_recovery = 0.0
    else:
        pred_tree = cKDTree(pred)
        d_bg, _ = pred_tree.query(stable_bg, k=1)
        bg_recovery = float(np.mean(d_bg < float(bg_thresh)))

    return {
        "ghost_count": ghost_count_dyn,
        "ghost_ratio": ghost_ratio_dyn,
        "ghost_tail_count": ghost_count,
        "ghost_tail_ratio": ghost_ratio,
        "background_recovery": bg_recovery,
    }


def plot_error_triptych(
    seq_name: str,
    method_points: Dict[str, np.ndarray],
    stable_bg_points: np.ndarray,
    table_rows: Dict[str, Dict[str, float]],
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stable_bg = np.asarray(stable_bg_points, dtype=float)
    tree = cKDTree(stable_bg) if stable_bg.shape[0] > 0 else None

    methods = ["egf", "tsdf", "simple_removal"]
    titles = {"egf": "EGF-DHMap", "tsdf": "TSDF", "simple_removal": "Simple Removal"}
    colors_max = 0.15

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for ax, m in zip(axes, methods):
        pts = np.asarray(method_points.get(m, np.zeros((0, 3))), dtype=float)
        if pts.shape[0] > 120000:
            rng = np.random.default_rng(7)
            keep = rng.choice(pts.shape[0], size=120000, replace=False)
            pts = pts[keep]
        if pts.shape[0] > 0:
            if tree is not None:
                d, _ = tree.query(pts, k=1)
            else:
                d = np.zeros((pts.shape[0],), dtype=float)
            sc = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=np.clip(d, 0.0, colors_max),
                s=0.35,
                cmap="turbo",
                vmin=0.0,
                vmax=colors_max,
                alpha=0.75,
            )
            fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        if stable_bg.shape[0] > 0:
            bg = stable_bg
            if bg.shape[0] > 90000:
                rng = np.random.default_rng(17)
                keep = rng.choice(bg.shape[0], size=90000, replace=False)
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
    fig.suptitle(f"{seq_name}: Error-Colored Prediction vs Stable Background (gray)")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def write_markdown_table(path: Path, title: str, rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"## {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            vals = []
            for h in headers:
                v = row.get(h, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    print("[cmd]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def run_method(
    method: str,
    sequence: str,
    out_dir: Path,
    dataset_root: Path,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    voxel_size: float,
    eval_thresh: float,
    download: bool,
    is_dynamic: bool,
    force: bool,
    dry_run: bool,
    egf_sigma_n0: float,
    egf_rho_decay: float,
    egf_phi_w_decay: float,
    egf_forget_mode: str,
    egf_dyn_forget_gain: float,
    egf_dyn_score_alpha: float,
    egf_dyn_d2_ref: float,
    egf_dscore_ema: float,
    egf_residual_score_weight: float,
    egf_raycast_clear_gain: float,
    egf_raycast_step_scale: float,
    egf_raycast_end_margin: float,
    egf_raycast_max_rays: int,
    egf_raycast_rho_max: float,
    egf_raycast_phiw_max: float,
    egf_raycast_dyn_boost: float,
    egf_surface_phi_thresh: float,
    egf_surface_rho_thresh: float,
    egf_surface_min_weight: float,
    egf_surface_max_dscore: float,
    egf_surface_max_free_ratio: float,
    egf_mesh_min_points: int,
    egf_ablation_no_evidence: bool,
    egf_ablation_no_gradient: bool,
) -> None:
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"[skip] {method} {sequence}: existing summary")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    base = [
        "--dataset_root",
        str(dataset_root),
        "--sequence",
        sequence,
        "--frames",
        str(frames),
        "--stride",
        str(stride),
        "--max_points_per_frame",
        str(max_points_per_frame),
        "--voxel_size",
        str(voxel_size),
        "--surface_eval_thresh",
        str(eval_thresh),
        "--out",
        str(out_dir),
    ]
    if download:
        base.append("--download")

    if method == "egf":
        cmd = [sys.executable, "scripts/run_egf_3d_tum.py", *base]
        if is_dynamic:
            cmd += [
                "--sigma_n0",
                str(float(egf_sigma_n0)),
                "--rho_decay",
                str(float(egf_rho_decay)),
                "--phi_w_decay",
                str(float(egf_phi_w_decay)),
                "--dynamic_forgetting",
                "--forget_mode",
                str(egf_forget_mode),
                "--dyn_forget_gain",
                str(float(egf_dyn_forget_gain)),
                "--dyn_score_alpha",
                str(float(egf_dyn_score_alpha)),
                "--dyn_d2_ref",
                str(float(egf_dyn_d2_ref)),
                "--dscore_ema",
                str(float(egf_dscore_ema)),
                "--residual_score_weight",
                str(float(egf_residual_score_weight)),
                "--raycast_clear_gain",
                str(float(egf_raycast_clear_gain)),
                "--raycast_step_scale",
                str(float(egf_raycast_step_scale)),
                "--raycast_end_margin",
                str(float(egf_raycast_end_margin)),
                "--raycast_max_rays",
                str(int(egf_raycast_max_rays)),
                "--raycast_rho_max",
                str(float(egf_raycast_rho_max)),
                "--raycast_phiw_max",
                str(float(egf_raycast_phiw_max)),
                "--raycast_dyn_boost",
                str(float(egf_raycast_dyn_boost)),
                "--surface_phi_thresh",
                str(float(egf_surface_phi_thresh)),
                "--surface_rho_thresh",
                str(float(egf_surface_rho_thresh)),
                "--surface_min_weight",
                str(float(egf_surface_min_weight)),
                "--surface_max_dscore",
                str(float(egf_surface_max_dscore)),
                "--surface_max_free_ratio",
                str(float(egf_surface_max_free_ratio)),
                "--mesh_min_points",
                str(int(egf_mesh_min_points)),
            ]
            if egf_ablation_no_evidence:
                cmd.append("--ablation_no_evidence")
            if egf_ablation_no_gradient:
                cmd.append("--ablation_no_gradient")
        else:
            cmd += [
                "--sigma_n0",
                str(float(egf_sigma_n0)),
                "--rho_decay",
                str(float(egf_rho_decay)),
                "--phi_w_decay",
                str(float(egf_phi_w_decay)),
                "--no_dynamic_forgetting",
                "--forget_mode",
                "off",
                "--surface_phi_thresh",
                str(float(max(egf_surface_phi_thresh, 0.70))),
                "--surface_rho_thresh",
                str(float(egf_surface_rho_thresh)),
                "--surface_min_weight",
                str(float(egf_surface_min_weight)),
                "--mesh_min_points",
                str(int(egf_mesh_min_points)),
            ]
            if egf_ablation_no_evidence:
                cmd.append("--ablation_no_evidence")
            if egf_ablation_no_gradient:
                cmd.append("--ablation_no_gradient")
    elif method == "tsdf":
        cmd = [sys.executable, "scripts/run_tsdf_baseline.py", *base]
    elif method == "simple_removal":
        cmd = [
            sys.executable,
            "scripts/run_simple_removal_baseline.py",
            *base,
            "--temporal_window",
            "6",
            "--neighbor_cells",
            "1",
            "--min_temporal_support",
            "2",
            "--min_lifetime_hits",
            "4",
            "--warmup_frames",
            "4",
        ]
    else:
        raise ValueError(f"Unknown method: {method}")
    run_cmd(cmd, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_kind", type=str, default="auto", choices=["auto", "tum", "bonn"])
    parser.add_argument("--dataset_root", type=str, default="data/tum")
    parser.add_argument("--out_root", type=str, default="output/benchmark_results")
    parser.add_argument("--static_sequences", type=str, default="rgbd_dataset_freiburg1_xyz")
    parser.add_argument(
        "--dynamic_sequences",
        type=str,
        default="rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere",
    )
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_points_per_frame", type=int, default=3000)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--ghost_thresh", type=float, default=0.08)
    parser.add_argument("--bg_thresh", type=float, default=0.05)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--methods", type=str, default="egf,tsdf,simple_removal")
    parser.add_argument("--egf_sigma_n0", type=float, default=0.26)
    parser.add_argument("--egf_rho_decay", type=float, default=0.998)
    parser.add_argument("--egf_phi_w_decay", type=float, default=0.998)
    parser.add_argument("--egf_forget_mode", type=str, default="local", choices=["local", "global", "off"])
    parser.add_argument("--egf_dyn_forget_gain", type=float, default=0.0)
    parser.add_argument("--egf_dyn_score_alpha", type=float, default=0.08)
    parser.add_argument("--egf_dyn_d2_ref", type=float, default=7.0)
    parser.add_argument("--egf_dscore_ema", type=float, default=0.12)
    parser.add_argument("--egf_residual_score_weight", type=float, default=0.25)
    parser.add_argument("--egf_raycast_clear_gain", type=float, default=0.0)
    parser.add_argument("--egf_raycast_step_scale", type=float, default=0.75)
    parser.add_argument("--egf_raycast_end_margin", type=float, default=0.16)
    parser.add_argument("--egf_raycast_max_rays", type=int, default=1500)
    parser.add_argument("--egf_raycast_rho_max", type=float, default=20.0)
    parser.add_argument("--egf_raycast_phiw_max", type=float, default=220.0)
    parser.add_argument("--egf_raycast_dyn_boost", type=float, default=0.6)
    parser.add_argument("--egf_surface_phi_thresh", type=float, default=0.80)
    parser.add_argument("--egf_surface_rho_thresh", type=float, default=0.0)
    parser.add_argument("--egf_surface_min_weight", type=float, default=0.0)
    parser.add_argument("--egf_surface_max_dscore", type=float, default=1.0)
    parser.add_argument("--egf_surface_max_free_ratio", type=float, default=1e9)
    parser.add_argument("--egf_mesh_min_points", type=int, default=100000000)
    parser.add_argument("--egf_ablation_no_evidence", action="store_true")
    parser.add_argument("--egf_ablation_no_gradient", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    static_sequences = [s.strip() for s in args.static_sequences.split(",") if s.strip()]
    dynamic_sequences = [s.strip() for s in args.dynamic_sequences.split(",") if s.strip()]
    all_sequences: List[Tuple[str, bool]] = [(s, False) for s in static_sequences] + [(s, True) for s in dynamic_sequences]

    if args.download:
        for seq, _ in all_sequences:
            seq_kind = infer_dataset_kind(seq, args.dataset_kind)
            if seq_kind == "tum":
                download_tum_sequence(dataset_root, seq)
            elif seq_kind == "bonn":
                download_bonn_sequence(dataset_root, seq)
            else:
                raise ValueError(f"Unsupported dataset kind: {seq_kind}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    valid_methods = {"egf", "tsdf", "simple_removal"}
    invalid = [m for m in methods if m not in valid_methods]
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}. valid={sorted(valid_methods)}")
    recon_rows: List[Dict[str, object]] = []
    dynamic_rows: List[Dict[str, object]] = []
    seq_summary_all: Dict[str, Dict] = {}

    for seq, is_dynamic in all_sequences:
        print(f"\n[sequence] {seq} dynamic={is_dynamic}")
        seq_dir = dataset_root / seq
        if not seq_dir.exists():
            raise FileNotFoundError(f"Missing sequence directory: {seq_dir}")
        seq_kind = infer_dataset_kind(seq, args.dataset_kind)
        seq_out = out_root / seq
        seq_out.mkdir(parents=True, exist_ok=True)

        for method in methods:
            run_method(
                method=method,
                sequence=seq,
                out_dir=seq_out / method,
                dataset_root=dataset_root,
                frames=args.frames,
                stride=args.stride,
                max_points_per_frame=args.max_points_per_frame,
                voxel_size=args.voxel_size,
                eval_thresh=args.eval_thresh,
                download=bool(args.download and seq_kind == "tum"),
                is_dynamic=is_dynamic,
                force=args.force,
                dry_run=args.dry_run,
                egf_sigma_n0=args.egf_sigma_n0,
                egf_rho_decay=args.egf_rho_decay,
                egf_phi_w_decay=args.egf_phi_w_decay,
                egf_forget_mode=args.egf_forget_mode,
                egf_dyn_forget_gain=args.egf_dyn_forget_gain,
                egf_dyn_score_alpha=args.egf_dyn_score_alpha,
                egf_dyn_d2_ref=args.egf_dyn_d2_ref,
                egf_dscore_ema=args.egf_dscore_ema,
                egf_residual_score_weight=args.egf_residual_score_weight,
                egf_raycast_clear_gain=args.egf_raycast_clear_gain,
                egf_raycast_step_scale=args.egf_raycast_step_scale,
                egf_raycast_end_margin=args.egf_raycast_end_margin,
                egf_raycast_max_rays=args.egf_raycast_max_rays,
                egf_raycast_rho_max=args.egf_raycast_rho_max,
                egf_raycast_phiw_max=args.egf_raycast_phiw_max,
                egf_raycast_dyn_boost=args.egf_raycast_dyn_boost,
                egf_surface_phi_thresh=args.egf_surface_phi_thresh,
                egf_surface_rho_thresh=args.egf_surface_rho_thresh,
                egf_surface_min_weight=args.egf_surface_min_weight,
                egf_surface_max_dscore=args.egf_surface_max_dscore,
                egf_surface_max_free_ratio=args.egf_surface_max_free_ratio,
                egf_mesh_min_points=args.egf_mesh_min_points,
                egf_ablation_no_evidence=args.egf_ablation_no_evidence,
                egf_ablation_no_gradient=args.egf_ablation_no_gradient,
            )

        if args.dry_run:
            continue

        stable_bg, tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
            sequence_dir=seq_dir,
            frames=args.frames,
            stride=args.stride,
            max_points_per_frame=args.max_points_per_frame,
            stable_voxel=max(0.03, args.voxel_size),
            stable_ratio=0.25 if is_dynamic else 0.35,
            tail_frames=12,
            dynamic_voxel=max(0.05, 2.0 * args.voxel_size),
            min_dynamic_hits=2,
            max_dynamic_ratio=0.35,
        )
        if stable_bg.shape[0] > 0:
            pcd_bg = o3d.geometry.PointCloud()
            pcd_bg.points = o3d.utility.Vector3dVector(stable_bg)
            o3d.io.write_point_cloud(str(seq_out / "stable_background_reference.ply"), pcd_bg)
        if tail_points.shape[0] > 0:
            pcd_tail = o3d.geometry.PointCloud()
            pcd_tail.points = o3d.utility.Vector3dVector(tail_points)
            o3d.io.write_point_cloud(str(seq_out / "tail_reference.ply"), pcd_tail)

        method_points: Dict[str, np.ndarray] = {}
        per_method: Dict[str, Dict[str, float]] = {}

        gt_points = load_points(seq_out / "egf" / "reference_points.ply")
        for method in methods:
            m_out = seq_out / method
            pred_points = load_points(m_out / "surface_points.ply")
            method_points[method] = pred_points

            recon = compute_recon_metrics(pred_points, gt_points, threshold=args.eval_thresh)
            dyn = compute_dynamic_metrics(
                pred_points=pred_points,
                stable_bg_points=stable_bg,
                tail_points=tail_points,
                dynamic_region=dynamic_region if is_dynamic else None,
                dynamic_voxel=dynamic_voxel,
                ghost_thresh=args.ghost_thresh,
                bg_thresh=args.bg_thresh,
            )
            row = {
                "sequence": seq,
                "scene_type": "dynamic" if is_dynamic else "static",
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
            recon_rows.append(row)
            dynamic_rows.append(
                {
                    "sequence": seq,
                    "scene_type": "dynamic" if is_dynamic else "static",
                    "method": method,
                    "ghost_count": dyn["ghost_count"],
                    "ghost_ratio": dyn["ghost_ratio"],
                    "ghost_tail_count": dyn["ghost_tail_count"],
                    "ghost_tail_ratio": dyn["ghost_tail_ratio"],
                    "background_recovery": dyn["background_recovery"],
                    "fscore": recon["fscore"],
                }
            )
            per_method[method] = row
            with (m_out / "benchmark_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

        if {"egf", "tsdf", "simple_removal"}.issubset(set(methods)):
            plot_error_triptych(
                seq_name=seq,
                method_points=method_points,
                stable_bg_points=stable_bg,
                table_rows=per_method,
                out_png=seq_out / "qualitative_triptych.png",
            )

        seq_summary = {
            "sequence": seq,
            "scene_type": "dynamic" if is_dynamic else "static",
            "frames": int(args.frames),
            "stride": int(args.stride),
            "voxel_size": float(args.voxel_size),
            "metrics": per_method,
        }
        with (seq_out / "sequence_summary.json").open("w", encoding="utf-8") as f:
            json.dump(seq_summary, f, indent=2)
        seq_summary_all[seq] = seq_summary

    if not args.dry_run:
        tables_dir = out_root / "tables"
        recon_headers = [
            "sequence",
            "scene_type",
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
        dyn_headers = [
            "sequence",
            "scene_type",
            "method",
            "ghost_count",
            "ghost_ratio",
            "ghost_tail_count",
            "ghost_tail_ratio",
            "background_recovery",
            "fscore",
        ]
        write_csv(tables_dir / "reconstruction_metrics.csv", recon_rows, recon_headers)
        write_csv(tables_dir / "dynamic_metrics.csv", dynamic_rows, dyn_headers)
        write_markdown_table(tables_dir / "reconstruction_metrics.md", "Reconstruction Metrics", recon_rows, recon_headers)
        write_markdown_table(tables_dir / "dynamic_metrics.md", "Dynamic Metrics", dynamic_rows, dyn_headers)
        with (tables_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
            json.dump({"sequences": seq_summary_all, "reconstruction_rows": recon_rows, "dynamic_rows": dynamic_rows}, f, indent=2)
        print(f"[done] results written to {out_root}")


if __name__ == "__main__":
    main()
