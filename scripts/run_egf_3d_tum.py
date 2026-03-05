from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path
from typing import Dict, List

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
from egf_dhmap3d.eval.metrics import compute_reconstruction_metrics, compute_trajectory_metrics
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


def save_poisson_mesh_from_surface(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    voxel_size: float,
    depth: int,
) -> Dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    pcd = pcd.voxel_down_sample(max(0.5 * float(voxel_size), 0.01))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(depth),
    )
    densities = np.asarray(densities, dtype=float)
    if densities.size > 0:
        remove_mask = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(remove_mask)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    mesh = mesh.crop(bbox)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh)
    return {
        "mode": "mesh",
        "surface_points": float(points.shape[0]),
        "vertices": float(np.asarray(mesh.vertices).shape[0]),
        "triangles": float(np.asarray(mesh.triangles).shape[0]),
    }


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
    gt_pose_group = parser.add_mutually_exclusive_group()
    gt_pose_group.add_argument("--use_gt_pose", dest="use_gt_pose", action="store_true")
    gt_pose_group.add_argument("--no_gt_pose", dest="use_gt_pose", action="store_false")
    parser.set_defaults(use_gt_pose=False)
    slam_delta_group = parser.add_mutually_exclusive_group()
    slam_delta_group.add_argument("--slam_use_gt_delta_odom", dest="slam_use_gt_delta_odom", action="store_true")
    slam_delta_group.add_argument("--slam_no_gt_delta_odom", dest="slam_use_gt_delta_odom", action="store_false")
    # Fairness default: do not use GT delta in SLAM mode unless explicitly requested.
    parser.set_defaults(slam_use_gt_delta_odom=False)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--truncation", type=float, default=None)
    parser.add_argument("--rho_decay", type=float, default=None)
    parser.add_argument("--phi_w_decay", type=float, default=None)
    parser.add_argument("--surface_phi_thresh", type=float, default=0.04)
    parser.add_argument("--surface_rho_thresh", type=float, default=0.20)
    parser.add_argument("--surface_min_weight", type=float, default=1.5)
    parser.add_argument("--surface_max_age_frames", type=int, default=1000000000)
    parser.add_argument("--surface_max_dscore", type=float, default=1.0)
    parser.add_argument("--surface_max_free_ratio", type=float, default=1e9)
    parser.add_argument("--surface_prune_free_min", type=float, default=1e9)
    parser.add_argument("--surface_prune_residual_min", type=float, default=1e9)
    parser.add_argument("--surface_max_clear_hits", type=float, default=1e9)
    parser.add_argument("--surface_use_zero_crossing", dest="surface_use_zero_crossing", action="store_true")
    parser.add_argument("--surface_no_zero_crossing", dest="surface_use_zero_crossing", action="store_false")
    parser.set_defaults(surface_use_zero_crossing=True)
    parser.add_argument("--surface_use_phi_geo_channel", dest="surface_use_phi_geo_channel", action="store_true")
    parser.add_argument("--surface_no_phi_geo_channel", dest="surface_use_phi_geo_channel", action="store_false")
    parser.set_defaults(surface_use_phi_geo_channel=False)
    parser.add_argument("--surface_zero_crossing_max_offset", type=float, default=0.06)
    parser.add_argument("--surface_zero_crossing_phi_gate", type=float, default=0.05)
    parser.add_argument("--surface_consistency_enable", action="store_true")
    parser.add_argument("--surface_consistency_radius", type=int, default=1)
    parser.add_argument("--surface_consistency_min_neighbors", type=int, default=4)
    parser.add_argument("--surface_consistency_normal_cos", type=float, default=0.55)
    parser.add_argument("--surface_consistency_phi_diff", type=float, default=0.04)
    parser.add_argument("--surface_snef_local_enable", action="store_true")
    parser.add_argument("--surface_snef_block_size_cells", type=int, default=8)
    parser.add_argument("--surface_snef_dscore_quantile", type=float, default=0.80)
    parser.add_argument("--surface_snef_dscore_margin", type=float, default=0.05)
    parser.add_argument("--surface_snef_free_ratio_quantile", type=float, default=0.85)
    parser.add_argument("--surface_snef_free_ratio_margin", type=float, default=0.10)
    parser.add_argument("--surface_snef_abs_phi_quantile", type=float, default=1.0)
    parser.add_argument("--surface_snef_abs_phi_margin", type=float, default=0.0)
    parser.add_argument("--surface_snef_min_keep_per_block", type=int, default=16)
    parser.add_argument("--surface_snef_min_keep_ratio_per_block", type=float, default=0.0)
    parser.add_argument("--surface_snef_min_candidates_per_block", type=int, default=10)
    parser.add_argument("--surface_snef_anchor_rho_quantile", type=float, default=0.90)
    parser.add_argument("--surface_snef_anchor_dscore_quantile", type=float, default=0.25)
    parser.add_argument("--surface_snef_anchor_min_per_block", type=int, default=2)
    parser.add_argument("--surface_two_stage_enable", action="store_true")
    parser.add_argument("--surface_two_stage_geom_margin", type=float, default=0.02)
    parser.add_argument("--surface_two_stage_dynamic_dscore_quantile", type=float, default=0.70)
    parser.add_argument("--surface_two_stage_dynamic_free_quantile", type=float, default=0.70)
    parser.add_argument("--surface_two_stage_dynamic_rho_quantile", type=float, default=0.40)
    parser.add_argument("--surface_two_stage_dynamic_rho_margin", type=float, default=0.0)
    parser.add_argument(
        "--surface_two_stage_dynamic_require_low_rho",
        dest="surface_two_stage_dynamic_require_low_rho",
        action="store_true",
    )
    parser.add_argument(
        "--surface_two_stage_dynamic_no_require_low_rho",
        dest="surface_two_stage_dynamic_require_low_rho",
        action="store_false",
    )
    parser.set_defaults(surface_two_stage_dynamic_require_low_rho=True)
    parser.add_argument("--surface_adaptive_enable", action="store_true")
    parser.add_argument("--surface_adaptive_rho_ref", type=float, default=2.0)
    parser.add_argument("--surface_adaptive_phi_min_scale", type=float, default=0.55)
    parser.add_argument("--surface_adaptive_phi_max_scale", type=float, default=1.15)
    parser.add_argument("--surface_adaptive_min_weight_gain", type=float, default=0.8)
    parser.add_argument("--surface_adaptive_free_ratio_gain", type=float, default=0.5)
    parser.add_argument("--surface_lzcd_apply_in_extraction", action="store_true")
    parser.add_argument("--surface_lzcd_bias_scale", type=float, default=1.0)
    parser.add_argument("--surface_stcg_enable", action="store_true")
    parser.add_argument("--surface_stcg_min_score", type=float, default=0.35)
    parser.add_argument("--surface_stcg_rho_ref", type=float, default=1.8)
    parser.add_argument("--surface_stcg_free_shrink", type=float, default=0.45)
    parser.add_argument("--surface_stcg_phi_shrink", type=float, default=0.25)
    parser.add_argument("--surface_stcg_dscore_shrink", type=float, default=0.30)
    parser.add_argument("--surface_stcg_weight_gain", type=float, default=0.50)
    parser.add_argument("--surface_stcg_static_protect", type=float, default=0.70)
    parser.add_argument("--poisson_depth", type=int, default=8)
    parser.add_argument("--poisson_iters", type=int, default=1)
    parser.add_argument("--poisson_lr", type=float, default=0.08)
    parser.add_argument("--eikonal_lambda", type=float, default=0.02)
    parser.add_argument("--mesh_min_points", type=int, default=800)
    parser.add_argument("--skip_mesh_export", action="store_true")
    parser.add_argument("--sigma_n0", type=float, default=0.18)
    parser.add_argument("--assoc_hetero_enable", dest="assoc_hetero_enable", action="store_true")
    parser.add_argument("--assoc_hetero_disable", dest="assoc_hetero_enable", action="store_false")
    parser.set_defaults(assoc_hetero_enable=False)
    parser.add_argument("--assoc_hetero_inc_ref_cos", type=float, default=0.65)
    parser.add_argument("--assoc_hetero_depth_ref_m", type=float, default=2.5)
    parser.add_argument("--assoc_hetero_normal_ref", type=float, default=0.20)
    parser.add_argument("--assoc_hetero_k_inc", type=float, default=0.45)
    parser.add_argument("--assoc_hetero_k_depth", type=float, default=0.12)
    parser.add_argument("--assoc_hetero_k_normal", type=float, default=0.55)
    parser.add_argument("--assoc_hetero_good_cos", type=float, default=0.90)
    parser.add_argument("--assoc_hetero_good_bonus", type=float, default=0.20)
    parser.add_argument("--assoc_hetero_sigma_d_min_scale", type=float, default=0.75)
    parser.add_argument("--assoc_hetero_sigma_d_max_scale", type=float, default=1.75)
    parser.add_argument("--assoc_hetero_sigma_n_min_scale", type=float, default=0.70)
    parser.add_argument("--assoc_hetero_sigma_n_max_scale", type=float, default=2.20)
    parser.add_argument("--assoc_gate_threshold", type=float, default=14.0)
    parser.add_argument("--assoc_search_radius_cells", type=int, default=2)
    parser.add_argument("--assoc_strict_surface_weight", type=float, default=0.8)
    parser.add_argument("--huber_delta_n", type=float, default=0.20)
    parser.add_argument("--frontier_boost", type=float, default=0.45)
    parser.add_argument("--assoc_seed_fallback_enable", dest="assoc_seed_fallback_enable", action="store_true")
    parser.add_argument("--assoc_seed_fallback_disable", dest="assoc_seed_fallback_enable", action="store_false")
    parser.set_defaults(assoc_seed_fallback_enable=True)
    parser.add_argument("--assoc_seed_fallback_low_support_scale", type=float, default=0.7)
    parser.add_argument("--assoc_seed_fallback_frontier_scale", type=float, default=0.7)
    parser.add_argument("--dynamic_forgetting", dest="dynamic_forgetting", action="store_true")
    parser.add_argument("--no_dynamic_forgetting", dest="dynamic_forgetting", action="store_false")
    parser.set_defaults(dynamic_forgetting=True)
    parser.add_argument("--forget_mode", type=str, default="local", choices=["local", "global", "off"])
    parser.add_argument("--dyn_forget_gain", type=float, default=0.12)
    parser.add_argument("--dyn_score_alpha", type=float, default=0.10)
    parser.add_argument("--dyn_d2_ref", type=float, default=8.0)
    parser.add_argument("--dscore_ema", type=float, default=0.12)
    parser.add_argument("--residual_score_weight", type=float, default=0.25)
    parser.add_argument("--integration_radius_scale", type=float, default=1.0)
    parser.add_argument("--integration_min_radius_vox", type=float, default=1.2)
    parser.add_argument("--decay_interval_frames", type=int, default=1)
    parser.add_argument("--lzcd_enable", action="store_true")
    parser.add_argument("--lzcd_interval", type=int, default=2)
    parser.add_argument("--lzcd_radius_cells", type=int, default=1)
    parser.add_argument("--lzcd_min_neighbors", type=int, default=6)
    parser.add_argument("--lzcd_min_phi_w", type=float, default=0.40)
    parser.add_argument("--lzcd_min_rho", type=float, default=0.05)
    parser.add_argument("--lzcd_max_dscore", type=float, default=0.85)
    parser.add_argument("--lzcd_neighbor_phi_gate", type=float, default=0.25)
    parser.add_argument("--lzcd_normal_cos_min", type=float, default=0.45)
    parser.add_argument("--lzcd_bias_alpha", type=float, default=0.18)
    parser.add_argument("--lzcd_correction_gain", type=float, default=0.35)
    parser.add_argument("--lzcd_max_bias", type=float, default=0.06)
    parser.add_argument("--lzcd_max_step", type=float, default=0.02)
    parser.add_argument("--lzcd_trim_quantile", type=float, default=0.75)
    parser.add_argument("--lzcd_use_geo_channel", dest="lzcd_use_geo_channel", action="store_true")
    parser.add_argument("--lzcd_no_geo_channel", dest="lzcd_use_geo_channel", action="store_false")
    parser.set_defaults(lzcd_use_geo_channel=True)
    parser.add_argument("--stcg_enable", action="store_true")
    parser.add_argument("--stcg_alpha", type=float, default=0.12)
    parser.add_argument("--stcg_conflict_weight", type=float, default=0.60)
    parser.add_argument("--stcg_residual_weight", type=float, default=0.25)
    parser.add_argument("--stcg_osc_weight", type=float, default=0.15)
    parser.add_argument("--stcg_free_ratio_ref", type=float, default=0.90)
    parser.add_argument("--stcg_on_thresh", type=float, default=0.58)
    parser.add_argument("--stcg_off_thresh", type=float, default=0.42)
    parser.add_argument("--raycast_clear_gain", type=float, default=0.0)
    parser.add_argument("--raycast_step_scale", type=float, default=1.0)
    parser.add_argument("--raycast_end_margin", type=float, default=0.12)
    parser.add_argument("--raycast_max_rays", type=int, default=600)
    parser.add_argument("--raycast_rho_max", type=float, default=5.0)
    parser.add_argument("--raycast_phiw_max", type=float, default=40.0)
    parser.add_argument("--raycast_dyn_boost", type=float, default=0.25)
    parser.add_argument("--icp_voxel_size", type=float, default=0.04)
    parser.add_argument("--icp_max_corr", type=float, default=0.12)
    parser.add_argument("--icp_max_iters", type=int, default=30)
    parser.add_argument("--icp_min_fitness", type=float, default=0.10)
    parser.add_argument("--icp_max_rmse", type=float, default=0.10)
    parser.add_argument("--icp_max_trans_step", type=float, default=0.35)
    parser.add_argument("--icp_max_rot_deg_step", type=float, default=30.0)
    parser.add_argument("--ablation_no_evidence", action="store_true")
    parser.add_argument("--ablation_no_gradient", action="store_true")
    parser.add_argument("--save_dscore_map", action="store_true")
    parser.add_argument("--dscore_min_weight", type=float, default=0.2)
    parser.add_argument("--stress_occlusion_ratio", type=float, default=0.0)
    parser.add_argument(
        "--stress_occlusion_mode",
        type=str,
        default="moving_band",
        choices=["moving_band", "fixed_center"],
    )
    parser.add_argument(
        "--stress_occlusion_axis",
        type=str,
        default="x",
        choices=["x", "y"],
    )
    parser.add_argument("--seed", type=int, default=42)
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
    if args.truncation is not None:
        cfg.map3d.truncation = float(max(1.5 * cfg.map3d.voxel_size, args.truncation))
    if args.rho_decay is not None:
        cfg.map3d.rho_decay = float(args.rho_decay)
    if args.phi_w_decay is not None:
        cfg.map3d.phi_w_decay = float(args.phi_w_decay)
    cfg.surface.phi_thresh = float(args.surface_phi_thresh)
    cfg.surface.rho_thresh = float(args.surface_rho_thresh)
    cfg.surface.min_weight = float(args.surface_min_weight)
    cfg.surface.max_age_frames = int(args.surface_max_age_frames)
    cfg.surface.max_d_score = float(args.surface_max_dscore)
    cfg.surface.max_free_ratio = float(args.surface_max_free_ratio)
    cfg.surface.prune_free_min = float(args.surface_prune_free_min)
    cfg.surface.prune_residual_min = float(args.surface_prune_residual_min)
    cfg.surface.max_clear_hits = float(args.surface_max_clear_hits)
    cfg.surface.use_zero_crossing = bool(args.surface_use_zero_crossing)
    cfg.surface.use_phi_geo_channel = bool(args.surface_use_phi_geo_channel)
    cfg.surface.zero_crossing_max_offset = float(max(0.0, args.surface_zero_crossing_max_offset))
    cfg.surface.zero_crossing_phi_gate = float(max(1e-4, args.surface_zero_crossing_phi_gate))
    cfg.surface.consistency_enable = bool(args.surface_consistency_enable)
    cfg.surface.consistency_radius = int(max(1, args.surface_consistency_radius))
    cfg.surface.consistency_min_neighbors = int(max(0, args.surface_consistency_min_neighbors))
    cfg.surface.consistency_normal_cos = float(np.clip(args.surface_consistency_normal_cos, 0.0, 1.0))
    cfg.surface.consistency_phi_diff = float(max(1e-4, args.surface_consistency_phi_diff))
    cfg.surface.snef_local_enable = bool(args.surface_snef_local_enable)
    cfg.surface.snef_block_size_cells = int(max(1, args.surface_snef_block_size_cells))
    cfg.surface.snef_dscore_quantile = float(np.clip(args.surface_snef_dscore_quantile, 0.0, 1.0))
    cfg.surface.snef_dscore_margin = float(max(0.0, args.surface_snef_dscore_margin))
    cfg.surface.snef_free_ratio_quantile = float(np.clip(args.surface_snef_free_ratio_quantile, 0.0, 1.0))
    cfg.surface.snef_free_ratio_margin = float(max(0.0, args.surface_snef_free_ratio_margin))
    cfg.surface.snef_abs_phi_quantile = float(np.clip(args.surface_snef_abs_phi_quantile, 0.0, 1.0))
    cfg.surface.snef_abs_phi_margin = float(max(0.0, args.surface_snef_abs_phi_margin))
    cfg.surface.snef_min_keep_per_block = int(max(1, args.surface_snef_min_keep_per_block))
    cfg.surface.snef_min_keep_ratio_per_block = float(np.clip(args.surface_snef_min_keep_ratio_per_block, 0.0, 1.0))
    cfg.surface.snef_min_candidates_per_block = int(max(1, args.surface_snef_min_candidates_per_block))
    cfg.surface.snef_anchor_rho_quantile = float(np.clip(args.surface_snef_anchor_rho_quantile, 0.0, 1.0))
    cfg.surface.snef_anchor_dscore_quantile = float(np.clip(args.surface_snef_anchor_dscore_quantile, 0.0, 1.0))
    cfg.surface.snef_anchor_min_per_block = int(max(0, args.surface_snef_anchor_min_per_block))
    cfg.surface.two_stage_enable = bool(args.surface_two_stage_enable)
    cfg.surface.two_stage_geom_margin = float(max(0.0, args.surface_two_stage_geom_margin))
    cfg.surface.two_stage_dynamic_dscore_quantile = float(np.clip(args.surface_two_stage_dynamic_dscore_quantile, 0.0, 1.0))
    cfg.surface.two_stage_dynamic_free_quantile = float(np.clip(args.surface_two_stage_dynamic_free_quantile, 0.0, 1.0))
    cfg.surface.two_stage_dynamic_rho_quantile = float(np.clip(args.surface_two_stage_dynamic_rho_quantile, 0.0, 1.0))
    cfg.surface.two_stage_dynamic_rho_margin = float(max(0.0, args.surface_two_stage_dynamic_rho_margin))
    cfg.surface.two_stage_dynamic_require_low_rho = bool(args.surface_two_stage_dynamic_require_low_rho)
    cfg.surface.adaptive_enable = bool(args.surface_adaptive_enable)
    cfg.surface.adaptive_rho_ref = float(max(1e-6, args.surface_adaptive_rho_ref))
    cfg.surface.adaptive_phi_min_scale = float(np.clip(args.surface_adaptive_phi_min_scale, 0.25, 2.0))
    cfg.surface.adaptive_phi_max_scale = float(np.clip(args.surface_adaptive_phi_max_scale, 0.25, 2.0))
    if cfg.surface.adaptive_phi_max_scale < cfg.surface.adaptive_phi_min_scale:
        cfg.surface.adaptive_phi_max_scale = cfg.surface.adaptive_phi_min_scale
    cfg.surface.adaptive_min_weight_gain = float(max(0.0, args.surface_adaptive_min_weight_gain))
    cfg.surface.adaptive_free_ratio_gain = float(np.clip(args.surface_adaptive_free_ratio_gain, 0.0, 0.99))
    cfg.surface.lzcd_apply_in_extraction = bool(args.surface_lzcd_apply_in_extraction)
    cfg.surface.lzcd_bias_scale = float(max(0.0, args.surface_lzcd_bias_scale))
    cfg.surface.stcg_enable = bool(args.surface_stcg_enable)
    cfg.surface.stcg_min_score = float(np.clip(args.surface_stcg_min_score, 0.0, 1.0))
    cfg.surface.stcg_rho_ref = float(max(1e-6, args.surface_stcg_rho_ref))
    cfg.surface.stcg_free_shrink = float(np.clip(args.surface_stcg_free_shrink, 0.0, 1.0))
    cfg.surface.stcg_phi_shrink = float(np.clip(args.surface_stcg_phi_shrink, 0.0, 1.0))
    cfg.surface.stcg_dscore_shrink = float(np.clip(args.surface_stcg_dscore_shrink, 0.0, 1.0))
    cfg.surface.stcg_weight_gain = float(max(0.0, args.surface_stcg_weight_gain))
    cfg.surface.stcg_static_protect = float(np.clip(args.surface_stcg_static_protect, 0.0, 1.0))
    cfg.surface.poisson_depth = int(args.poisson_depth)
    cfg.update.poisson_iters = int(max(0, args.poisson_iters))
    cfg.update.poisson_lr = float(max(1e-4, args.poisson_lr))
    cfg.update.eikonal_lambda = float(max(0.0, args.eikonal_lambda))
    cfg.assoc.gate_threshold = float(max(1.0, args.assoc_gate_threshold))
    cfg.assoc.search_radius_cells = int(max(0, args.assoc_search_radius_cells))
    cfg.assoc.strict_surface_weight = float(max(0.1, args.assoc_strict_surface_weight))
    cfg.assoc.sigma_n0 = float(args.sigma_n0)
    cfg.assoc.hetero_enable = bool(args.assoc_hetero_enable)
    cfg.assoc.hetero_inc_ref_cos = float(np.clip(args.assoc_hetero_inc_ref_cos, 1e-3, 1.0))
    cfg.assoc.hetero_depth_ref_m = float(max(1e-3, args.assoc_hetero_depth_ref_m))
    cfg.assoc.hetero_normal_ref = float(max(1e-4, args.assoc_hetero_normal_ref))
    cfg.assoc.hetero_k_inc = float(max(0.0, args.assoc_hetero_k_inc))
    cfg.assoc.hetero_k_depth = float(max(0.0, args.assoc_hetero_k_depth))
    cfg.assoc.hetero_k_normal = float(max(0.0, args.assoc_hetero_k_normal))
    cfg.assoc.hetero_good_cos = float(np.clip(args.assoc_hetero_good_cos, 0.0, 1.0))
    cfg.assoc.hetero_good_bonus = float(np.clip(args.assoc_hetero_good_bonus, 0.0, 0.9))
    cfg.assoc.hetero_sigma_d_min_scale = float(max(1e-3, args.assoc_hetero_sigma_d_min_scale))
    cfg.assoc.hetero_sigma_d_max_scale = float(max(cfg.assoc.hetero_sigma_d_min_scale, args.assoc_hetero_sigma_d_max_scale))
    cfg.assoc.hetero_sigma_n_min_scale = float(max(1e-3, args.assoc_hetero_sigma_n_min_scale))
    cfg.assoc.hetero_sigma_n_max_scale = float(max(cfg.assoc.hetero_sigma_n_min_scale, args.assoc_hetero_sigma_n_max_scale))
    cfg.assoc.huber_delta_n = float(args.huber_delta_n)
    cfg.assoc.seed_fallback_enable = bool(args.assoc_seed_fallback_enable)
    cfg.assoc.seed_fallback_low_support_scale = float(max(0.0, args.assoc_seed_fallback_low_support_scale))
    cfg.assoc.seed_fallback_frontier_scale = float(max(0.0, args.assoc_seed_fallback_frontier_scale))
    cfg.update.frontier_boost = float(max(0.0, args.frontier_boost))
    cfg.update.forget_mode = str(args.forget_mode)
    cfg.update.dyn_forget_gain = float(args.dyn_forget_gain if args.dynamic_forgetting else 0.0)
    if not args.dynamic_forgetting:
        cfg.update.forget_mode = "off"
    cfg.update.dyn_score_alpha = float(args.dyn_score_alpha)
    cfg.update.dyn_d2_ref = float(args.dyn_d2_ref)
    cfg.update.dscore_ema = float(args.dscore_ema)
    cfg.update.residual_score_weight = float(args.residual_score_weight)
    cfg.update.integration_radius_scale = float(np.clip(args.integration_radius_scale, 0.20, 1.0))
    cfg.update.integration_min_radius_vox = float(max(0.6, args.integration_min_radius_vox))
    cfg.update.decay_interval_frames = int(max(1, args.decay_interval_frames))
    cfg.update.lzcd_enable = bool(args.lzcd_enable)
    cfg.update.lzcd_interval = int(max(1, args.lzcd_interval))
    cfg.update.lzcd_radius_cells = int(max(1, args.lzcd_radius_cells))
    cfg.update.lzcd_min_neighbors = int(max(3, args.lzcd_min_neighbors))
    cfg.update.lzcd_min_phi_w = float(max(0.0, args.lzcd_min_phi_w))
    cfg.update.lzcd_min_rho = float(max(0.0, args.lzcd_min_rho))
    cfg.update.lzcd_max_dscore = float(np.clip(args.lzcd_max_dscore, 0.0, 1.0))
    cfg.update.lzcd_neighbor_phi_gate = float(max(1e-4, args.lzcd_neighbor_phi_gate))
    cfg.update.lzcd_normal_cos_min = float(np.clip(args.lzcd_normal_cos_min, 0.0, 1.0))
    cfg.update.lzcd_bias_alpha = float(np.clip(args.lzcd_bias_alpha, 0.01, 0.95))
    cfg.update.lzcd_correction_gain = float(np.clip(args.lzcd_correction_gain, 0.0, 1.0))
    cfg.update.lzcd_max_bias = float(max(1e-4, args.lzcd_max_bias))
    cfg.update.lzcd_max_step = float(max(1e-4, args.lzcd_max_step))
    cfg.update.lzcd_trim_quantile = float(np.clip(args.lzcd_trim_quantile, 0.55, 1.0))
    cfg.update.lzcd_use_geo_channel = bool(args.lzcd_use_geo_channel)
    cfg.update.stcg_enable = bool(args.stcg_enable)
    cfg.update.stcg_alpha = float(np.clip(args.stcg_alpha, 0.01, 0.95))
    cfg.update.stcg_conflict_weight = float(max(0.0, args.stcg_conflict_weight))
    cfg.update.stcg_residual_weight = float(max(0.0, args.stcg_residual_weight))
    cfg.update.stcg_osc_weight = float(max(0.0, args.stcg_osc_weight))
    cfg.update.stcg_free_ratio_ref = float(max(1e-6, args.stcg_free_ratio_ref))
    cfg.update.stcg_on_thresh = float(np.clip(args.stcg_on_thresh, 0.05, 0.95))
    cfg.update.stcg_off_thresh = float(np.clip(args.stcg_off_thresh, 0.01, 0.90))
    cfg.update.raycast_clear_gain = float(args.raycast_clear_gain)
    cfg.update.raycast_step_scale = float(args.raycast_step_scale)
    cfg.update.raycast_end_margin = float(args.raycast_end_margin)
    cfg.update.raycast_max_rays = int(args.raycast_max_rays)
    cfg.update.raycast_rho_max = float(args.raycast_rho_max)
    cfg.update.raycast_phiw_max = float(args.raycast_phiw_max)
    cfg.update.raycast_dyn_boost = float(args.raycast_dyn_boost)
    cfg.predict.icp_voxel_size = float(args.icp_voxel_size)
    cfg.predict.icp_max_corr = float(args.icp_max_corr)
    cfg.predict.icp_max_iters = int(args.icp_max_iters)
    cfg.predict.icp_min_fitness = float(args.icp_min_fitness)
    cfg.predict.icp_max_rmse = float(args.icp_max_rmse)
    cfg.predict.icp_max_trans_step = float(args.icp_max_trans_step)
    cfg.predict.icp_max_rot_deg_step = float(args.icp_max_rot_deg_step)
    cfg.predict.slam_use_gt_delta_odom = bool(args.slam_use_gt_delta_odom)
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
        stress_occlusion_ratio=float(max(0.0, args.stress_occlusion_ratio)),
        stress_occlusion_mode=str(args.stress_occlusion_mode),
        stress_occlusion_axis=str(args.stress_occlusion_axis),
        seed=int(args.seed),
    )
    model = EGFDHMap3D(cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_refs: List[np.ndarray] = []
    gt_norm_refs: List[np.ndarray] = []
    gt_traj: List[np.ndarray] = []
    assoc_ratios: List[float] = []
    touched_voxels: List[float] = []
    dyn_scores: List[float] = []
    odom_fitness: List[float] = []
    odom_rmse: List[float] = []
    odom_valid: List[float] = []
    rng = np.random.default_rng(int(args.seed))
    run_wall_t0 = time.perf_counter()
    step_total_sec = 0.0
    step_predict_sec = 0.0
    step_assoc_sec = 0.0
    step_update_sec = 0.0
    step_map_refine_sec = 0.0

    total = len(stream)
    print(f"[run] sequence={seq_dir.name} frames={total}")
    for i, frame in enumerate(stream):
        t_step0 = time.perf_counter()
        stat = model.step(frame, use_gt_pose=args.use_gt_pose)
        t_step1 = time.perf_counter()
        assoc_ratios.append(float(stat["assoc_ratio"]))
        touched_voxels.append(float(stat["touched_voxels"]))
        dyn_scores.append(float(stat.get("dynamic_score", 0.0)))
        odom_fitness.append(float(stat.get("odom_fitness", 0.0)))
        odom_rmse.append(float(stat.get("odom_rmse", 0.0)))
        odom_valid.append(float(stat.get("odom_valid", 0.0)))
        step_total_sec += float(stat.get("step_total_sec", t_step1 - t_step0))
        step_predict_sec += float(stat.get("predict_sec", 0.0))
        step_assoc_sec += float(stat.get("associate_sec", 0.0))
        step_update_sec += float(stat.get("update_sec", 0.0))
        step_map_refine_sec += float(stat.get("map_refine_sec", 0.0))

        ref = frame.points_world
        ref_n = frame.normals_world
        if ref.shape[0] > 2500:
            keep = rng.choice(ref.shape[0], size=2500, replace=False)
            ref = ref[keep]
            ref_n = ref_n[keep]
        gt_refs.append(ref)
        gt_norm_refs.append(ref_n)
        gt_traj.append(np.asarray(frame.pose_w_c, dtype=float))

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == total:
            print(
                f"  frame={i + 1:04d}/{total:04d} "
                f"assoc={stat['assoc_ratio']:.3f} "
                f"vox={int(stat['active_voxels'])} "
                f"dyn={stat.get('dynamic_score', 0.0):.3f} "
                f"odom={stat.get('odom_valid', 0.0):.0f}/{stat.get('odom_fitness', 0.0):.2f}"
            )

    t_extract0 = time.perf_counter()
    pred_points, pred_normals = model.extract_surface_points()
    extract_total_sec = float(time.perf_counter() - t_extract0)
    gt_points = np.vstack(gt_refs) if gt_refs else np.zeros((0, 3), dtype=float)
    gt_normals = np.vstack(gt_norm_refs) if gt_norm_refs else np.zeros((0, 3), dtype=float)
    t_eval0 = time.perf_counter()
    metrics = compute_reconstruction_metrics(
        pred_points,
        gt_points,
        threshold=args.surface_eval_thresh,
        pred_normals=pred_normals,
        gt_normals=gt_normals,
    )
    eval_recon_sec = float(time.perf_counter() - t_eval0)

    t_io0 = time.perf_counter()
    save_point_cloud(out_dir / "surface_points.ply", pred_points, pred_normals)
    save_point_cloud(out_dir / "reference_points.ply", gt_points, gt_normals)
    mesh_min_points = int(max(0, args.mesh_min_points))
    if args.skip_mesh_export:
        mesh_info = {
            "mode": "skipped",
            "surface_points": float(pred_points.shape[0]),
            "vertices": 0.0,
            "triangles": 0.0,
        }
    elif pred_points.shape[0] < mesh_min_points:
        mesh_info = {
            "mode": "pointcloud",
            "surface_points": float(pred_points.shape[0]),
            "vertices": 0.0,
            "triangles": 0.0,
        }
    else:
        mesh_info = save_poisson_mesh_from_surface(
            out_dir / "surface_mesh.ply",
            pred_points,
            pred_normals,
            voxel_size=float(cfg.map3d.voxel_size),
            depth=int(cfg.surface.poisson_depth),
        )
    io_mesh_sec = float(time.perf_counter() - t_io0)

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
    gt_trajectory = np.asarray(gt_traj, dtype=float) if gt_traj else np.zeros((0, 4, 4), dtype=float)
    t_eval_traj0 = time.perf_counter()
    traj_metrics = compute_trajectory_metrics(
        pred_poses=trajectory,
        gt_poses=gt_trajectory,
        step=1,
        align_first=True,
    )
    eval_traj_sec = float(time.perf_counter() - t_eval_traj0)
    t_io_traj0 = time.perf_counter()
    np.save(out_dir / "trajectory.npy", trajectory)
    np.save(out_dir / "gt_trajectory.npy", gt_trajectory)
    io_traj_sec = float(time.perf_counter() - t_io_traj0)
    runtime_model = model.get_runtime_stats()
    wall_total_sec = float(time.perf_counter() - run_wall_t0)

    summary = {
        "sequence": seq_dir.name,
        "frames_used": int(total),
        "stride": int(args.stride),
        "use_gt_pose": bool(args.use_gt_pose),
        "seed": int(args.seed),
        "voxel_size": float(cfg.map3d.voxel_size),
        "truncation": float(cfg.map3d.truncation),
        "rho_decay": float(cfg.map3d.rho_decay),
        "phi_w_decay": float(cfg.map3d.phi_w_decay),
        "sigma_n0": float(cfg.assoc.sigma_n0),
        "assoc_hetero_enable": bool(cfg.assoc.hetero_enable),
        "assoc_hetero_inc_ref_cos": float(cfg.assoc.hetero_inc_ref_cos),
        "assoc_hetero_depth_ref_m": float(cfg.assoc.hetero_depth_ref_m),
        "assoc_hetero_normal_ref": float(cfg.assoc.hetero_normal_ref),
        "assoc_hetero_k_inc": float(cfg.assoc.hetero_k_inc),
        "assoc_hetero_k_depth": float(cfg.assoc.hetero_k_depth),
        "assoc_hetero_k_normal": float(cfg.assoc.hetero_k_normal),
        "assoc_hetero_good_cos": float(cfg.assoc.hetero_good_cos),
        "assoc_hetero_good_bonus": float(cfg.assoc.hetero_good_bonus),
        "assoc_hetero_sigma_d_min_scale": float(cfg.assoc.hetero_sigma_d_min_scale),
        "assoc_hetero_sigma_d_max_scale": float(cfg.assoc.hetero_sigma_d_max_scale),
        "assoc_hetero_sigma_n_min_scale": float(cfg.assoc.hetero_sigma_n_min_scale),
        "assoc_hetero_sigma_n_max_scale": float(cfg.assoc.hetero_sigma_n_max_scale),
        "assoc_gate_threshold": float(cfg.assoc.gate_threshold),
        "assoc_search_radius_cells": int(cfg.assoc.search_radius_cells),
        "assoc_strict_surface_weight": float(cfg.assoc.strict_surface_weight),
        "surface_max_dscore": float(cfg.surface.max_d_score),
        "surface_max_age_frames": int(cfg.surface.max_age_frames),
        "surface_max_free_ratio": float(cfg.surface.max_free_ratio),
        "surface_prune_free_min": float(cfg.surface.prune_free_min),
        "surface_prune_residual_min": float(cfg.surface.prune_residual_min),
        "surface_max_clear_hits": float(cfg.surface.max_clear_hits),
        "surface_use_zero_crossing": bool(cfg.surface.use_zero_crossing),
        "surface_use_phi_geo_channel": bool(cfg.surface.use_phi_geo_channel),
        "surface_zero_crossing_max_offset": float(cfg.surface.zero_crossing_max_offset),
        "surface_zero_crossing_phi_gate": float(cfg.surface.zero_crossing_phi_gate),
        "surface_consistency_enable": bool(cfg.surface.consistency_enable),
        "surface_consistency_radius": int(cfg.surface.consistency_radius),
        "surface_consistency_min_neighbors": int(cfg.surface.consistency_min_neighbors),
        "surface_consistency_normal_cos": float(cfg.surface.consistency_normal_cos),
        "surface_consistency_phi_diff": float(cfg.surface.consistency_phi_diff),
        "surface_snef_local_enable": bool(cfg.surface.snef_local_enable),
        "surface_snef_block_size_cells": int(cfg.surface.snef_block_size_cells),
        "surface_snef_dscore_quantile": float(cfg.surface.snef_dscore_quantile),
        "surface_snef_dscore_margin": float(cfg.surface.snef_dscore_margin),
        "surface_snef_free_ratio_quantile": float(cfg.surface.snef_free_ratio_quantile),
        "surface_snef_free_ratio_margin": float(cfg.surface.snef_free_ratio_margin),
        "surface_snef_abs_phi_quantile": float(cfg.surface.snef_abs_phi_quantile),
        "surface_snef_abs_phi_margin": float(cfg.surface.snef_abs_phi_margin),
        "surface_snef_min_keep_per_block": int(cfg.surface.snef_min_keep_per_block),
        "surface_snef_min_keep_ratio_per_block": float(cfg.surface.snef_min_keep_ratio_per_block),
        "surface_snef_min_candidates_per_block": int(cfg.surface.snef_min_candidates_per_block),
        "surface_snef_anchor_rho_quantile": float(cfg.surface.snef_anchor_rho_quantile),
        "surface_snef_anchor_dscore_quantile": float(cfg.surface.snef_anchor_dscore_quantile),
        "surface_snef_anchor_min_per_block": int(cfg.surface.snef_anchor_min_per_block),
        "surface_two_stage_enable": bool(cfg.surface.two_stage_enable),
        "surface_two_stage_geom_margin": float(cfg.surface.two_stage_geom_margin),
        "surface_two_stage_dynamic_dscore_quantile": float(cfg.surface.two_stage_dynamic_dscore_quantile),
        "surface_two_stage_dynamic_free_quantile": float(cfg.surface.two_stage_dynamic_free_quantile),
        "surface_two_stage_dynamic_rho_quantile": float(cfg.surface.two_stage_dynamic_rho_quantile),
        "surface_two_stage_dynamic_rho_margin": float(cfg.surface.two_stage_dynamic_rho_margin),
        "surface_two_stage_dynamic_require_low_rho": bool(cfg.surface.two_stage_dynamic_require_low_rho),
        "surface_adaptive_enable": bool(cfg.surface.adaptive_enable),
        "surface_adaptive_rho_ref": float(cfg.surface.adaptive_rho_ref),
        "surface_adaptive_phi_min_scale": float(cfg.surface.adaptive_phi_min_scale),
        "surface_adaptive_phi_max_scale": float(cfg.surface.adaptive_phi_max_scale),
        "surface_adaptive_min_weight_gain": float(cfg.surface.adaptive_min_weight_gain),
        "surface_adaptive_free_ratio_gain": float(cfg.surface.adaptive_free_ratio_gain),
        "surface_lzcd_apply_in_extraction": bool(cfg.surface.lzcd_apply_in_extraction),
        "surface_lzcd_bias_scale": float(cfg.surface.lzcd_bias_scale),
        "surface_stcg_enable": bool(cfg.surface.stcg_enable),
        "surface_stcg_min_score": float(cfg.surface.stcg_min_score),
        "surface_stcg_rho_ref": float(cfg.surface.stcg_rho_ref),
        "surface_stcg_free_shrink": float(cfg.surface.stcg_free_shrink),
        "surface_stcg_phi_shrink": float(cfg.surface.stcg_phi_shrink),
        "surface_stcg_dscore_shrink": float(cfg.surface.stcg_dscore_shrink),
        "surface_stcg_weight_gain": float(cfg.surface.stcg_weight_gain),
        "surface_stcg_static_protect": float(cfg.surface.stcg_static_protect),
        "huber_delta_n": float(cfg.assoc.huber_delta_n),
        "poisson_iters": int(cfg.update.poisson_iters),
        "poisson_lr": float(cfg.update.poisson_lr),
        "eikonal_lambda": float(cfg.update.eikonal_lambda),
        "frontier_boost": float(cfg.update.frontier_boost),
        "assoc_seed_fallback_enable": bool(cfg.assoc.seed_fallback_enable),
        "assoc_seed_fallback_low_support_scale": float(cfg.assoc.seed_fallback_low_support_scale),
        "assoc_seed_fallback_frontier_scale": float(cfg.assoc.seed_fallback_frontier_scale),
        "dynamic_forgetting": bool(args.dynamic_forgetting),
        "forget_mode": str(cfg.update.forget_mode),
        "dyn_forget_gain": float(cfg.update.dyn_forget_gain),
        "dyn_score_alpha": float(cfg.update.dyn_score_alpha),
        "dyn_d2_ref": float(cfg.update.dyn_d2_ref),
        "dscore_ema": float(cfg.update.dscore_ema),
        "residual_score_weight": float(cfg.update.residual_score_weight),
        "integration_radius_scale": float(cfg.update.integration_radius_scale),
        "integration_min_radius_vox": float(cfg.update.integration_min_radius_vox),
        "decay_interval_frames": int(cfg.update.decay_interval_frames),
        "lzcd_enable": bool(cfg.update.lzcd_enable),
        "lzcd_interval": int(cfg.update.lzcd_interval),
        "lzcd_radius_cells": int(cfg.update.lzcd_radius_cells),
        "lzcd_min_neighbors": int(cfg.update.lzcd_min_neighbors),
        "lzcd_min_phi_w": float(cfg.update.lzcd_min_phi_w),
        "lzcd_min_rho": float(cfg.update.lzcd_min_rho),
        "lzcd_max_dscore": float(cfg.update.lzcd_max_dscore),
        "lzcd_neighbor_phi_gate": float(cfg.update.lzcd_neighbor_phi_gate),
        "lzcd_normal_cos_min": float(cfg.update.lzcd_normal_cos_min),
        "lzcd_bias_alpha": float(cfg.update.lzcd_bias_alpha),
        "lzcd_correction_gain": float(cfg.update.lzcd_correction_gain),
        "lzcd_max_bias": float(cfg.update.lzcd_max_bias),
        "lzcd_max_step": float(cfg.update.lzcd_max_step),
        "lzcd_trim_quantile": float(cfg.update.lzcd_trim_quantile),
        "lzcd_use_geo_channel": bool(cfg.update.lzcd_use_geo_channel),
        "stcg_enable": bool(cfg.update.stcg_enable),
        "stcg_alpha": float(cfg.update.stcg_alpha),
        "stcg_conflict_weight": float(cfg.update.stcg_conflict_weight),
        "stcg_residual_weight": float(cfg.update.stcg_residual_weight),
        "stcg_osc_weight": float(cfg.update.stcg_osc_weight),
        "stcg_free_ratio_ref": float(cfg.update.stcg_free_ratio_ref),
        "stcg_on_thresh": float(cfg.update.stcg_on_thresh),
        "stcg_off_thresh": float(cfg.update.stcg_off_thresh),
        "raycast_clear_gain": float(cfg.update.raycast_clear_gain),
        "raycast_step_scale": float(cfg.update.raycast_step_scale),
        "raycast_end_margin": float(cfg.update.raycast_end_margin),
        "raycast_max_rays": int(cfg.update.raycast_max_rays),
        "raycast_rho_max": float(cfg.update.raycast_rho_max),
        "raycast_phiw_max": float(cfg.update.raycast_phiw_max),
        "raycast_dyn_boost": float(cfg.update.raycast_dyn_boost),
        "icp_voxel_size": float(cfg.predict.icp_voxel_size),
        "icp_max_corr": float(cfg.predict.icp_max_corr),
        "icp_max_iters": int(cfg.predict.icp_max_iters),
        "icp_min_fitness": float(cfg.predict.icp_min_fitness),
        "icp_max_rmse": float(cfg.predict.icp_max_rmse),
        "icp_max_trans_step": float(cfg.predict.icp_max_trans_step),
        "icp_max_rot_deg_step": float(cfg.predict.icp_max_rot_deg_step),
        "slam_use_gt_delta_odom": bool(cfg.predict.slam_use_gt_delta_odom),
        "ablation_no_evidence": bool(args.ablation_no_evidence),
        "ablation_no_gradient": bool(args.ablation_no_gradient),
        "stress_occlusion_ratio": float(max(0.0, args.stress_occlusion_ratio)),
        "stress_occlusion_mode": str(args.stress_occlusion_mode),
        "stress_occlusion_axis": str(args.stress_occlusion_axis),
        "active_voxels": int(len(model.voxel_map)),
        "surface_points": int(pred_points.shape[0]),
        "reference_points": int(gt_points.shape[0]),
        "assoc_ratio_mean": float(np.mean(assoc_ratios)) if assoc_ratios else 0.0,
        "touched_voxels_mean": float(np.mean(touched_voxels)) if touched_voxels else 0.0,
        "dynamic_score_mean": float(np.mean(dyn_scores)) if dyn_scores else 0.0,
        "odom_fitness_mean": float(np.mean(odom_fitness)) if odom_fitness else 0.0,
        "odom_rmse_mean": float(np.mean(odom_rmse)) if odom_rmse else 0.0,
        "odom_valid_ratio": float(np.mean(odom_valid)) if odom_valid else 0.0,
        "runtime": {
            "wall_total_sec": float(wall_total_sec),
            "mapping_step_total_sec": float(step_total_sec),
            "mapping_step_mean_sec": float(step_total_sec / max(1, int(total))),
            "mapping_fps": float(total / step_total_sec) if step_total_sec > 1e-9 else 0.0,
            "extract_total_sec": float(extract_total_sec),
            "eval_recon_sec": float(eval_recon_sec),
            "eval_traj_sec": float(eval_traj_sec),
            "io_mesh_sec": float(io_mesh_sec),
            "io_traj_sec": float(io_traj_sec),
            "stage_predict_sec": float(step_predict_sec),
            "stage_associate_sec": float(step_assoc_sec),
            "stage_update_sec": float(step_update_sec),
            "stage_map_refine_sec": float(step_map_refine_sec),
            "model_runtime_stats": runtime_model,
        },
        "trajectory_metrics": {
            "ate_rmse": float(traj_metrics.ate_rmse),
            "ate_mean": float(traj_metrics.ate_mean),
            "ate_median": float(traj_metrics.ate_median),
            "ate_max": float(traj_metrics.ate_max),
            "rpe_trans_rmse": float(traj_metrics.rpe_trans_rmse),
            "rpe_trans_mean": float(traj_metrics.rpe_trans_mean),
            "rpe_rot_deg_rmse": float(traj_metrics.rpe_rot_deg_rmse),
            "rpe_rot_deg_mean": float(traj_metrics.rpe_rot_deg_mean),
            "frame_count": int(traj_metrics.frame_count),
            "valid_pair_count": int(traj_metrics.valid_pair_count),
            "finite_ratio": float(traj_metrics.finite_ratio),
        },
        "metrics": {
            "chamfer": float(metrics.chamfer),
            "hausdorff": float(metrics.hausdorff),
            "precision": float(metrics.precision),
            "recall": float(metrics.recall),
            "fscore": float(metrics.fscore),
            "normal_consistency": float(metrics.normal_consistency),
            "precision_2cm": float(metrics.precision_2cm),
            "recall_2cm": float(metrics.recall_2cm),
            "fscore_2cm": float(metrics.fscore_2cm),
            "precision_5cm": float(metrics.precision_5cm),
            "recall_5cm": float(metrics.recall_5cm),
            "fscore_5cm": float(metrics.fscore_5cm),
            "precision_10cm": float(metrics.precision_10cm),
            "recall_10cm": float(metrics.recall_10cm),
            "fscore_10cm": float(metrics.fscore_10cm),
            "threshold": float(args.surface_eval_thresh),
        },
        "mesh": mesh_info,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
