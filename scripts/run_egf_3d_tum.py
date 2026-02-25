from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
import requests
import sys
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.data.tum_rgbd import TUMRGBDStream
from egf_dhmap3d.eval.metrics import compute_reconstruction_metrics
from egf_dhmap3d.modules.pipeline import EGFDHMap3D


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


def save_point_cloud(path: Path, points: np.ndarray, normals: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals is not None and normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/tum")
    parser.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg1_xyz")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max_points_per_frame", type=int, default=5000)
    parser.add_argument("--surface_eval_thresh", type=float, default=0.05)
    parser.add_argument("--out", type=str, default="output/egf3d/freiburg1_xyz")
    parser.add_argument("--use_gt_pose", action="store_true", default=True)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--rho_decay", type=float, default=None)
    parser.add_argument("--phi_w_decay", type=float, default=None)
    parser.add_argument("--surface_phi_thresh", type=float, default=0.04)
    parser.add_argument("--surface_rho_thresh", type=float, default=0.20)
    parser.add_argument("--surface_min_weight", type=float, default=1.5)
    parser.add_argument("--surface_max_dscore", type=float, default=1.0)
    parser.add_argument("--surface_max_free_ratio", type=float, default=1e9)
    parser.add_argument("--surface_prune_free_min", type=float, default=1e9)
    parser.add_argument("--surface_prune_residual_min", type=float, default=1e9)
    parser.add_argument("--surface_max_clear_hits", type=float, default=1e9)
    parser.add_argument("--poisson_depth", type=int, default=8)
    parser.add_argument("--mesh_min_points", type=int, default=800)
    parser.add_argument("--sigma_n0", type=float, default=0.18)
    parser.add_argument("--huber_delta_n", type=float, default=0.20)
    parser.add_argument("--dynamic_forgetting", dest="dynamic_forgetting", action="store_true")
    parser.add_argument("--no_dynamic_forgetting", dest="dynamic_forgetting", action="store_false")
    parser.set_defaults(dynamic_forgetting=True)
    parser.add_argument("--forget_mode", type=str, default="local", choices=["local", "global", "off"])
    parser.add_argument("--dyn_forget_gain", type=float, default=0.12)
    parser.add_argument("--dyn_score_alpha", type=float, default=0.10)
    parser.add_argument("--dyn_d2_ref", type=float, default=8.0)
    parser.add_argument("--dscore_ema", type=float, default=0.12)
    parser.add_argument("--residual_score_weight", type=float, default=0.25)
    parser.add_argument("--raycast_clear_gain", type=float, default=0.0)
    parser.add_argument("--raycast_step_scale", type=float, default=1.0)
    parser.add_argument("--raycast_end_margin", type=float, default=0.12)
    parser.add_argument("--raycast_max_rays", type=int, default=600)
    parser.add_argument("--raycast_rho_max", type=float, default=5.0)
    parser.add_argument("--raycast_phiw_max", type=float, default=40.0)
    parser.add_argument("--raycast_dyn_boost", type=float, default=0.25)
    parser.add_argument("--ablation_no_evidence", action="store_true")
    parser.add_argument("--ablation_no_gradient", action="store_true")
    parser.add_argument("--save_dscore_map", action="store_true")
    parser.add_argument("--dscore_min_weight", type=float, default=0.2)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.download:
        seq_dir = download_tum_sequence(dataset_root, args.sequence)
    else:
        seq_dir = dataset_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(
            f"sequence not found: {seq_dir}. You can pass --download to fetch it automatically."
        )

    cfg = EGF3DConfig()
    cfg.map3d.voxel_size = float(args.voxel_size)
    cfg.map3d.truncation = max(0.08, 3.0 * cfg.map3d.voxel_size)
    if args.rho_decay is not None:
        cfg.map3d.rho_decay = float(args.rho_decay)
    if args.phi_w_decay is not None:
        cfg.map3d.phi_w_decay = float(args.phi_w_decay)
    cfg.surface.phi_thresh = float(args.surface_phi_thresh)
    cfg.surface.rho_thresh = float(args.surface_rho_thresh)
    cfg.surface.min_weight = float(args.surface_min_weight)
    cfg.surface.max_d_score = float(args.surface_max_dscore)
    cfg.surface.max_free_ratio = float(args.surface_max_free_ratio)
    cfg.surface.prune_free_min = float(args.surface_prune_free_min)
    cfg.surface.prune_residual_min = float(args.surface_prune_residual_min)
    cfg.surface.max_clear_hits = float(args.surface_max_clear_hits)
    cfg.surface.poisson_depth = int(args.poisson_depth)
    cfg.update.poisson_iters = 1
    cfg.update.eikonal_lambda = 0.02
    cfg.assoc.gate_threshold = 14.0
    cfg.assoc.sigma_n0 = float(args.sigma_n0)
    cfg.assoc.huber_delta_n = float(args.huber_delta_n)
    cfg.update.forget_mode = str(args.forget_mode)
    cfg.update.dyn_forget_gain = float(args.dyn_forget_gain if args.dynamic_forgetting else 0.0)
    if not args.dynamic_forgetting:
        cfg.update.forget_mode = "off"
    cfg.update.dyn_score_alpha = float(args.dyn_score_alpha)
    cfg.update.dyn_d2_ref = float(args.dyn_d2_ref)
    cfg.update.dscore_ema = float(args.dscore_ema)
    cfg.update.residual_score_weight = float(args.residual_score_weight)
    cfg.update.raycast_clear_gain = float(args.raycast_clear_gain)
    cfg.update.raycast_step_scale = float(args.raycast_step_scale)
    cfg.update.raycast_end_margin = float(args.raycast_end_margin)
    cfg.update.raycast_max_rays = int(args.raycast_max_rays)
    cfg.update.raycast_rho_max = float(args.raycast_rho_max)
    cfg.update.raycast_phiw_max = float(args.raycast_phiw_max)
    cfg.update.raycast_dyn_boost = float(args.raycast_dyn_boost)
    cfg.update.enable_evidence = bool(not args.ablation_no_evidence)
    cfg.assoc.use_evidence_in_noise = bool(not args.ablation_no_evidence)
    cfg.update.enable_gradient_fusion = bool(not args.ablation_no_gradient)
    cfg.assoc.use_normal_residual = bool(not args.ablation_no_gradient)

    stream = TUMRGBDStream(
        sequence_dir=seq_dir,
        cfg=cfg,
        max_frames=args.frames,
        stride=args.stride,
        max_points=args.max_points_per_frame,
        assoc_max_diff=0.02,
        normal_radius=0.08,
    )
    model = EGFDHMap3D(cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_refs: List[np.ndarray] = []
    assoc_ratios: List[float] = []
    touched_voxels: List[float] = []
    dyn_scores: List[float] = []
    rng = np.random.default_rng(7)

    total = len(stream)
    print(f"[run] sequence={seq_dir.name} frames={total}")
    for i, frame in enumerate(stream):
        stat = model.step(frame, use_gt_pose=args.use_gt_pose)
        assoc_ratios.append(float(stat["assoc_ratio"]))
        touched_voxels.append(float(stat["touched_voxels"]))
        dyn_scores.append(float(stat.get("dynamic_score", 0.0)))

        ref = frame.points_world
        if ref.shape[0] > 2500:
            keep = rng.choice(ref.shape[0], size=2500, replace=False)
            ref = ref[keep]
        gt_refs.append(ref)

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == total:
            print(
                f"  frame={i + 1:04d}/{total:04d} "
                f"assoc={stat['assoc_ratio']:.3f} "
                f"vox={int(stat['active_voxels'])} "
                f"dyn={stat.get('dynamic_score', 0.0):.3f}"
            )

    pred_points, pred_normals = model.extract_surface_points()
    gt_points = np.vstack(gt_refs) if gt_refs else np.zeros((0, 3), dtype=float)
    metrics = compute_reconstruction_metrics(pred_points, gt_points, threshold=args.surface_eval_thresh)

    save_point_cloud(out_dir / "surface_points.ply", pred_points, pred_normals)
    save_point_cloud(out_dir / "reference_points.ply", gt_points, None)
    mesh_info = model.save_poisson_mesh(out_dir / "surface_mesh.ply", min_points=int(args.mesh_min_points))

    if args.save_dscore_map:
        (
            centers,
            d_score,
            rho_vals,
            phi_w_vals,
            surf_vals,
            free_vals,
            residual_vals,
            clear_vals,
        ) = model.export_dynamic_voxel_map(min_phi_w=float(args.dscore_min_weight))
        np.savez(
            out_dir / "dynamic_score_voxels.npz",
            centers=centers,
            d_score=d_score,
            rho=rho_vals,
            phi_w=phi_w_vals,
            surf_evidence=surf_vals,
            free_evidence=free_vals,
            residual_evidence=residual_vals,
            clear_hits=clear_vals,
        )
        fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
        if centers.shape[0] > 0:
            sc = ax.scatter(centers[:, 0], centers[:, 1], c=d_score, s=0.7, cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.85)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("d_score")
        ax.set_title("Dynamic Score Voxel Map (top view)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
        fig.savefig(out_dir / "dynamic_score_map.png", dpi=220)
        plt.close(fig)

    trajectory = model.get_trajectory()
    np.save(out_dir / "trajectory.npy", trajectory)

    summary = {
        "sequence": seq_dir.name,
        "frames_used": int(total),
        "stride": int(args.stride),
        "use_gt_pose": bool(args.use_gt_pose),
        "voxel_size": float(cfg.map3d.voxel_size),
        "rho_decay": float(cfg.map3d.rho_decay),
        "phi_w_decay": float(cfg.map3d.phi_w_decay),
        "sigma_n0": float(cfg.assoc.sigma_n0),
        "surface_max_dscore": float(cfg.surface.max_d_score),
        "surface_max_free_ratio": float(cfg.surface.max_free_ratio),
        "surface_prune_free_min": float(cfg.surface.prune_free_min),
        "surface_prune_residual_min": float(cfg.surface.prune_residual_min),
        "surface_max_clear_hits": float(cfg.surface.max_clear_hits),
        "huber_delta_n": float(cfg.assoc.huber_delta_n),
        "dynamic_forgetting": bool(args.dynamic_forgetting),
        "forget_mode": str(cfg.update.forget_mode),
        "dyn_forget_gain": float(cfg.update.dyn_forget_gain),
        "dyn_score_alpha": float(cfg.update.dyn_score_alpha),
        "dyn_d2_ref": float(cfg.update.dyn_d2_ref),
        "dscore_ema": float(cfg.update.dscore_ema),
        "residual_score_weight": float(cfg.update.residual_score_weight),
        "raycast_clear_gain": float(cfg.update.raycast_clear_gain),
        "raycast_step_scale": float(cfg.update.raycast_step_scale),
        "raycast_end_margin": float(cfg.update.raycast_end_margin),
        "raycast_max_rays": int(cfg.update.raycast_max_rays),
        "raycast_rho_max": float(cfg.update.raycast_rho_max),
        "raycast_phiw_max": float(cfg.update.raycast_phiw_max),
        "raycast_dyn_boost": float(cfg.update.raycast_dyn_boost),
        "ablation_no_evidence": bool(args.ablation_no_evidence),
        "ablation_no_gradient": bool(args.ablation_no_gradient),
        "active_voxels": int(len(model.voxel_map)),
        "surface_points": int(pred_points.shape[0]),
        "reference_points": int(gt_points.shape[0]),
        "assoc_ratio_mean": float(np.mean(assoc_ratios)) if assoc_ratios else 0.0,
        "touched_voxels_mean": float(np.mean(touched_voxels)) if touched_voxels else 0.0,
        "dynamic_score_mean": float(np.mean(dyn_scores)) if dyn_scores else 0.0,
        "metrics": {
            "chamfer": float(metrics.chamfer),
            "hausdorff": float(metrics.hausdorff),
            "precision": float(metrics.precision),
            "recall": float(metrics.recall),
            "fscore": float(metrics.fscore),
            "threshold": float(args.surface_eval_thresh),
        },
        "mesh": mesh_info,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
