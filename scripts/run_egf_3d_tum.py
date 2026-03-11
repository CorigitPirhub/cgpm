from __future__ import annotations

import argparse
import csv
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
from egf_dhmap3d.P10_method.geometry_chain import apply_geometry_chain_coupling
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


def save_feature_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--depth_bias_offset_m", type=float, default=0.0)
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
    parser.add_argument("--surface_point_bias_along_normal_m", type=float, default=0.0)
    parser.add_argument("--surface_geometry_chain_coupling_enable", action="store_true")
    parser.add_argument("--surface_geometry_chain_coupling_mode", type=str, default="direct", choices=["direct", "projected"])
    parser.add_argument("--surface_geometry_chain_coupling_donor_root", type=str, default="")
    parser.add_argument("--surface_geometry_chain_coupling_project_dist", type=float, default=0.05)
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
    parser.add_argument("--surface_ptdsf_persistent_only_enable", action="store_true")
    parser.add_argument("--surface_ptdsf_persistent_min_rho", type=float, default=0.15)
    parser.add_argument("--surface_ptdsf_static_rho_weight", type=float, default=0.35)
    parser.add_argument("--surface_zcbf_apply_in_extraction", action="store_true")
    parser.add_argument("--surface_zcbf_bias_scale", type=float, default=1.0)
    parser.add_argument("--surface_stcg_enable", action="store_true")
    parser.add_argument("--surface_dccm_enable", action="store_true")
    parser.add_argument("--surface_dccm_commit_weight", type=float, default=0.30)
    parser.add_argument("--surface_dccm_static_guard", type=float, default=0.65)
    parser.add_argument("--surface_dccm_drop_gain", type=float, default=0.22)
    parser.add_argument("--surface_stcg_min_score", type=float, default=0.35)
    parser.add_argument("--surface_stcg_rho_ref", type=float, default=1.8)
    parser.add_argument("--surface_stcg_free_shrink", type=float, default=0.45)
    parser.add_argument("--surface_stcg_phi_shrink", type=float, default=0.25)
    parser.add_argument("--surface_stcg_dscore_shrink", type=float, default=0.30)
    parser.add_argument("--surface_stcg_weight_gain", type=float, default=0.50)
    parser.add_argument("--surface_stcg_static_protect", type=float, default=0.70)
    parser.add_argument("--surface_use_dual_static_channel", dest="surface_use_dual_static_channel", action="store_true")
    parser.add_argument("--surface_no_dual_static_channel", dest="surface_use_dual_static_channel", action="store_false")
    parser.set_defaults(surface_use_dual_static_channel=False)
    parser.add_argument("--surface_dual_p_static_min", type=float, default=0.0)
    parser.add_argument("--surface_structural_decouple_enable", dest="surface_structural_decouple_enable", action="store_true")
    parser.add_argument("--surface_structural_decouple_disable", dest="surface_structural_decouple_enable", action="store_false")
    parser.set_defaults(surface_structural_decouple_enable=True)
    parser.add_argument("--surface_decouple_min_geo_weight_ratio", type=float, default=0.35)
    parser.add_argument("--surface_decouple_dyn_drop_thresh", type=float, default=0.78)
    parser.add_argument("--surface_decouple_dyn_rho_guard", type=float, default=1.2)
    parser.add_argument("--surface_decouple_dyn_free_ratio_thresh", type=float, default=1.10)
    parser.add_argument("--surface_decouple_channel_div_enable", action="store_true")
    parser.add_argument("--surface_decouple_channel_div_thresh", type=float, default=0.04)
    parser.add_argument("--surface_decouple_channel_div_weight", type=float, default=0.35)
    parser.add_argument("--surface_decouple_channel_div_static_guard", type=float, default=0.70)
    parser.add_argument("--surface_dual_layer_extract_enable", action="store_true")
    parser.add_argument("--surface_dual_layer_geo_min_weight_ratio", type=float, default=0.30)
    parser.add_argument(
        "--surface_dual_layer_dyn_use_zdyn",
        dest="surface_dual_layer_dyn_use_zdyn",
        action="store_true",
    )
    parser.add_argument(
        "--surface_dual_layer_dyn_no_zdyn",
        dest="surface_dual_layer_dyn_use_zdyn",
        action="store_false",
    )
    parser.set_defaults(surface_dual_layer_dyn_use_zdyn=True)
    parser.add_argument("--surface_dual_layer_dyn_prob_weight", type=float, default=0.38)
    parser.add_argument("--surface_dual_layer_dyn_stmem_weight", type=float, default=0.22)
    parser.add_argument("--surface_dual_layer_dyn_contra_weight", type=float, default=0.20)
    parser.add_argument("--surface_dual_layer_dyn_transient_weight", type=float, default=0.20)
    parser.add_argument("--surface_dual_layer_dyn_phi_div_weight", type=float, default=0.16)
    parser.add_argument("--surface_dual_layer_dyn_phi_ratio_weight", type=float, default=0.10)
    parser.add_argument("--surface_dual_layer_dyn_phi_div_ref", type=float, default=0.04)
    parser.add_argument(
        "--surface_dual_layer_dyn_use_phi_dyn",
        dest="surface_dual_layer_dyn_use_phi_dyn",
        action="store_true",
    )
    parser.add_argument(
        "--surface_dual_layer_dyn_no_phi_dyn",
        dest="surface_dual_layer_dyn_use_phi_dyn",
        action="store_false",
    )
    parser.set_defaults(surface_dual_layer_dyn_use_phi_dyn=True)
    parser.add_argument(
        "--surface_dual_layer_compete_enable",
        dest="surface_dual_layer_compete_enable",
        action="store_true",
    )
    parser.add_argument(
        "--surface_dual_layer_compete_disable",
        dest="surface_dual_layer_compete_enable",
        action="store_false",
    )
    parser.set_defaults(surface_dual_layer_compete_enable=False)
    parser.add_argument("--surface_dual_layer_compete_margin", type=float, default=0.08)
    parser.add_argument("--surface_dual_layer_compete_geo_weight", type=float, default=0.62)
    parser.add_argument("--surface_dual_layer_compete_dyn_mix_weight", type=float, default=0.55)
    parser.add_argument("--surface_dual_layer_compete_dyn_conf_weight", type=float, default=0.25)
    parser.add_argument("--surface_dual_layer_dyn_drop_thresh", type=float, default=0.72)
    parser.add_argument("--surface_dual_layer_dyn_free_ratio_min", type=float, default=0.90)
    parser.add_argument("--surface_dual_layer_static_anchor_rho", type=float, default=0.90)
    parser.add_argument("--surface_dual_layer_static_anchor_p", type=float, default=0.70)
    parser.add_argument("--surface_dual_layer_static_anchor_ratio", type=float, default=1.70)
    parser.add_argument("--surface_csr_enable", action="store_true")
    parser.add_argument("--surface_csr_min_score", type=float, default=0.38)
    parser.add_argument("--surface_csr_geo_blend", type=float, default=0.18)
    parser.add_argument("--surface_csr_geo_agree_min", type=float, default=0.70)
    parser.add_argument("--surface_xmap_enable", action="store_true")
    parser.add_argument("--surface_xmap_dyn_min_score", type=float, default=0.52)
    parser.add_argument("--surface_xmap_static_min_score", type=float, default=0.42)
    parser.add_argument("--surface_xmap_sep_ref_vox", type=float, default=0.90)
    parser.add_argument("--surface_omhs_enable", action="store_true")
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
    parser.add_argument("--assoc_contra_gate_enable", dest="assoc_contra_gate_enable", action="store_true")
    parser.add_argument("--assoc_contra_gate_disable", dest="assoc_contra_gate_enable", action="store_false")
    parser.set_defaults(assoc_contra_gate_enable=True)
    parser.add_argument("--assoc_contra_stmem_weight", type=float, default=0.65)
    parser.add_argument("--assoc_contra_visibility_weight", type=float, default=0.20)
    parser.add_argument("--assoc_contra_residual_weight", type=float, default=0.15)
    parser.add_argument("--assoc_contra_free_ratio_ref", type=float, default=1.0)
    parser.add_argument("--assoc_contra_rho_ref", type=float, default=1.6)
    parser.add_argument("--assoc_contra_static_guard", type=float, default=0.70)
    parser.add_argument("--assoc_contra_rho_guard", type=float, default=0.55)
    parser.add_argument("--assoc_contra_d2_boost_max", type=float, default=2.2)
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
    parser.add_argument("--dual_state_enable", action="store_true")
    parser.add_argument("--dual_state_assoc_weight", type=float, default=0.45)
    parser.add_argument("--dual_state_free_weight", type=float, default=0.25)
    parser.add_argument("--dual_state_residual_weight", type=float, default=0.15)
    parser.add_argument("--dual_state_osc_weight", type=float, default=0.10)
    parser.add_argument("--dual_state_pose_weight", type=float, default=0.05)
    parser.add_argument("--dual_state_bias", type=float, default=0.45)
    parser.add_argument("--dual_state_temp", type=float, default=0.25)
    parser.add_argument("--dual_pose_var_ref", type=float, default=0.05)
    parser.add_argument("--dual_state_static_ema", type=float, default=0.12)
    parser.add_argument("--dual_state_min_static_ratio", type=float, default=0.06)
    parser.add_argument("--dual_state_commit_thresh", type=float, default=0.70)
    parser.add_argument("--dual_state_rollback_thresh", type=float, default=0.32)
    parser.add_argument("--dual_state_commit_gain", type=float, default=0.25)
    parser.add_argument("--dual_state_rollback_gain", type=float, default=0.10)
    parser.add_argument("--dual_state_static_protect_rho", type=float, default=0.90)
    parser.add_argument("--dual_state_static_protect_ratio", type=float, default=1.60)
    parser.add_argument("--dual_state_static_decay_mult", type=float, default=1.0)
    parser.add_argument("--dual_state_transient_decay_mult", type=float, default=2.2)
    parser.add_argument("--ptdsf_enable", action="store_true")
    parser.add_argument("--ptdsf_rho_alpha", type=float, default=0.18)
    parser.add_argument("--ptdsf_static_blend", type=float, default=0.55)
    parser.add_argument("--ptdsf_commit_age_ref", type=float, default=3.0)
    parser.add_argument("--ptdsf_commit_bonus", type=float, default=0.08)
    parser.add_argument("--ptdsf_rollback_bonus", type=float, default=0.06)
    parser.add_argument("--wod_enable", action="store_true")
    parser.add_argument("--rps_enable", action="store_true")
    parser.add_argument("--rps_hard_commit_enable", action="store_true")
    parser.add_argument("--rps_surface_bank_enable", action="store_true")
    parser.add_argument("--rps_bank_margin", type=float, default=0.08)
    parser.add_argument("--rps_bank_separation_ref", type=float, default=0.04)
    parser.add_argument("--rps_bank_rear_min_score", type=float, default=0.52)
    parser.add_argument("--rps_bank_sep_gate", type=float, default=0.22)
    parser.add_argument("--rps_bank_bg_support_gain", type=float, default=0.0)
    parser.add_argument("--rps_bank_front_dyn_penalty_gain", type=float, default=0.0)
    parser.add_argument("--rps_bank_rear_score_bias", type=float, default=0.0)
    parser.add_argument("--rps_bank_soft_competition_enable", action="store_true")
    parser.add_argument("--rps_bank_soft_competition_gap", type=float, default=0.0)
    parser.add_argument("--rps_bank_soft_sep_relax", type=float, default=0.0)
    parser.add_argument("--rps_bank_soft_rear_min_relax", type=float, default=0.0)
    parser.add_argument("--rps_bank_soft_support_min", type=float, default=0.45)
    parser.add_argument("--rps_bank_soft_local_support_gain", type=float, default=1.0)
    parser.add_argument("--rps_soft_bank_export_enable", action="store_true")
    parser.add_argument("--rps_soft_bank_min_score", type=float, default=0.18)
    parser.add_argument("--rps_soft_bank_gain", type=float, default=0.65)
    parser.add_argument("--rps_soft_bank_commit_relax", type=float, default=0.70)
    parser.add_argument("--rps_candidate_rescue_enable", action="store_true")
    parser.add_argument("--rps_candidate_support_gain", type=float, default=0.28)
    parser.add_argument("--rps_candidate_bg_gain", type=float, default=0.22)
    parser.add_argument("--rps_candidate_rho_gain", type=float, default=0.18)
    parser.add_argument("--rps_candidate_front_relax", type=float, default=0.20)
    parser.add_argument("--rps_commit_activation_enable", action="store_true")
    parser.add_argument("--rps_commit_threshold", type=float, default=0.62)
    parser.add_argument("--rps_commit_release", type=float, default=0.40)
    parser.add_argument("--rps_commit_age_threshold", type=float, default=2.0)
    parser.add_argument("--rps_commit_rho_ref", type=float, default=0.08)
    parser.add_argument("--rps_commit_weight_ref", type=float, default=0.80)
    parser.add_argument("--rps_commit_min_cand_rho", type=float, default=0.02)
    parser.add_argument("--rps_commit_min_cand_w", type=float, default=0.08)
    parser.add_argument("--rps_commit_evidence_weight", type=float, default=0.34)
    parser.add_argument("--rps_commit_geometry_weight", type=float, default=0.28)
    parser.add_argument("--rps_commit_bg_weight", type=float, default=0.20)
    parser.add_argument("--rps_commit_static_weight", type=float, default=0.18)
    parser.add_argument("--rps_commit_front_penalty", type=float, default=0.22)
    parser.add_argument("--rps_admission_support_enable", action="store_true")
    parser.add_argument("--rps_admission_support_on", type=float, default=0.42)
    parser.add_argument("--rps_admission_support_gain", type=float, default=0.55)
    parser.add_argument("--rps_admission_score_relax", type=float, default=0.10)
    parser.add_argument("--rps_admission_active_floor", type=float, default=0.32)
    parser.add_argument("--rps_admission_rho_ref", type=float, default=0.08)
    parser.add_argument("--rps_admission_weight_ref", type=float, default=0.35)
    parser.add_argument("--rps_admission_geometry_enable", action="store_true")
    parser.add_argument("--rps_admission_geometry_weight", type=float, default=0.25)
    parser.add_argument("--rps_admission_geometry_floor", type=float, default=0.40)
    parser.add_argument("--rps_admission_occlusion_enable", action="store_true")
    parser.add_argument("--rps_admission_occlusion_weight", type=float, default=0.12)
    parser.add_argument("--rps_space_redirect_history_enable", action="store_true")
    parser.add_argument("--rps_space_redirect_history_weight", type=float, default=0.32)
    parser.add_argument("--rps_space_redirect_history_bg_weight", type=float, default=0.60)
    parser.add_argument("--rps_space_redirect_history_static_weight", type=float, default=0.40)
    parser.add_argument("--rps_space_redirect_history_floor", type=float, default=0.30)
    parser.add_argument("--rps_space_redirect_ghost_suppress_enable", action="store_true")
    parser.add_argument("--rps_space_redirect_ghost_suppress_weight", type=float, default=0.22)
    parser.add_argument("--rps_space_redirect_visual_anchor_enable", action="store_true")
    parser.add_argument("--rps_space_redirect_visual_anchor_weight", type=float, default=0.16)
    parser.add_argument("--rps_space_redirect_visual_anchor_min", type=float, default=0.36)
    parser.add_argument("--rps_history_obstructed_gate_enable", action="store_true")
    parser.add_argument("--rps_history_visible_min", type=float, default=0.45)
    parser.add_argument("--rps_obstruction_min", type=float, default=0.28)
    parser.add_argument("--rps_non_hole_min", type=float, default=0.30)
    parser.add_argument("--rps_history_manifold_enable", action="store_true")
    parser.add_argument("--rps_history_manifold_visible_min", type=float, default=0.45)
    parser.add_argument("--rps_history_manifold_obstruction_min", type=float, default=0.28)
    parser.add_argument("--rps_history_manifold_bg_weight", type=float, default=0.50)
    parser.add_argument("--rps_history_manifold_geo_weight", type=float, default=0.30)
    parser.add_argument("--rps_history_manifold_static_weight", type=float, default=0.20)
    parser.add_argument("--rps_history_manifold_blend", type=float, default=0.75)
    parser.add_argument("--rps_history_manifold_max_offset", type=float, default=0.04)
    parser.add_argument("--rps_bg_manifold_state_enable", action="store_true")
    parser.add_argument("--rps_bg_manifold_alpha_up", type=float, default=0.08)
    parser.add_argument("--rps_bg_manifold_alpha_down", type=float, default=0.02)
    parser.add_argument("--rps_bg_manifold_rho_alpha", type=float, default=0.10)
    parser.add_argument("--rps_bg_manifold_weight_gain", type=float, default=0.55)
    parser.add_argument("--rps_bg_manifold_rho_ref", type=float, default=0.08)
    parser.add_argument("--rps_bg_manifold_weight_ref", type=float, default=0.35)
    parser.add_argument("--rps_bg_manifold_history_weight", type=float, default=0.30)
    parser.add_argument("--rps_bg_manifold_obstruction_weight", type=float, default=0.20)
    parser.add_argument("--rps_bg_manifold_visible_lo", type=float, default=0.25)
    parser.add_argument("--rps_bg_manifold_visible_hi", type=float, default=0.50)
    parser.add_argument("--rps_bg_dense_state_enable", action="store_true")
    parser.add_argument("--rps_bg_dense_neighbor_radius", type=int, default=1)
    parser.add_argument("--rps_bg_dense_neighbor_weight", type=float, default=0.55)
    parser.add_argument("--rps_bg_dense_geometry_weight", type=float, default=0.30)
    parser.add_argument("--rps_bg_dense_max_weight", type=float, default=1.0)
    parser.add_argument("--rps_bg_dense_support_floor", type=float, default=0.18)
    parser.add_argument("--rps_bg_dense_decay", type=float, default=0.996)
    parser.add_argument("--rps_bg_surface_constrained_enable", action="store_true")
    parser.add_argument("--rps_bg_surface_min_conf", type=float, default=0.12)
    parser.add_argument("--rps_bg_surface_agree_weight", type=float, default=0.40)
    parser.add_argument("--rps_bg_surface_tangent_enable", action="store_true")
    parser.add_argument("--rps_bg_surface_tangent_weight", type=float, default=0.65)
    parser.add_argument("--rps_bg_surface_tangent_floor", type=float, default=0.15)
    parser.add_argument("--rps_bg_bridge_enable", action="store_true")
    parser.add_argument("--rps_bg_bridge_min_visible", type=float, default=0.35)
    parser.add_argument("--rps_bg_bridge_min_obstruction", type=float, default=0.30)
    parser.add_argument("--rps_bg_bridge_min_step", type=int, default=1)
    parser.add_argument("--rps_bg_bridge_max_step", type=int, default=3)
    parser.add_argument("--rps_bg_bridge_gain", type=float, default=0.65)
    parser.add_argument("--rps_bg_bridge_phi_blend", type=float, default=0.85)
    parser.add_argument("--rps_bg_bridge_target_dyn_max", type=float, default=0.35)
    parser.add_argument("--rps_bg_bridge_target_surface_max", type=float, default=0.35)
    parser.add_argument("--rps_bg_bridge_ghost_suppress_enable", action="store_true")
    parser.add_argument("--rps_bg_bridge_ghost_suppress_weight", type=float, default=0.22)
    parser.add_argument("--rps_bg_bridge_relaxed_dyn_max", type=float, default=0.45)
    parser.add_argument("--rps_bg_bridge_keep_multi_hits", action="store_true")
    parser.add_argument("--rps_bg_bridge_max_hits_per_source", type=int, default=3)
    parser.add_argument("--rps_bg_bridge_cone_enable", action="store_true")
    parser.add_argument("--rps_bg_bridge_cone_radius_cells", type=int, default=1)
    parser.add_argument("--rps_bg_bridge_cone_gain_scale", type=float, default=0.65)
    parser.add_argument("--rps_bg_bridge_patch_radius_cells", type=int, default=0)
    parser.add_argument("--rps_bg_bridge_patch_gain_scale", type=float, default=0.55)
    parser.add_argument("--rps_bg_bridge_depth_hypothesis_count", type=int, default=0)
    parser.add_argument("--rps_bg_bridge_depth_step_scale", type=float, default=0.50)
    parser.add_argument("--rps_bg_bridge_rear_synth_enable", action="store_true")
    parser.add_argument("--rps_bg_bridge_rear_support_gain", type=float, default=0.28)
    parser.add_argument("--rps_bg_bridge_rear_rho_gain", type=float, default=0.10)
    parser.add_argument("--rps_bg_bridge_rear_phi_blend", type=float, default=0.80)
    parser.add_argument("--rps_bg_bridge_rear_score_floor", type=float, default=0.22)
    parser.add_argument("--rps_bg_bridge_rear_active_floor", type=float, default=0.52)
    parser.add_argument("--rps_bg_bridge_rear_age_floor", type=float, default=1.0)
    parser.add_argument("--rps_rear_hybrid_filter_enable", action="store_true")
    parser.add_argument("--rps_rear_hybrid_bridge_support_min", type=float, default=0.20)
    parser.add_argument("--rps_rear_hybrid_dyn_max", type=float, default=0.22)
    parser.add_argument("--rps_rear_hybrid_manifold_min", type=float, default=0.25)
    parser.add_argument("--rps_rear_density_gate_enable", action="store_true")
    parser.add_argument("--rps_rear_density_radius_cells", type=int, default=1)
    parser.add_argument("--rps_rear_density_min_neighbors", type=int, default=2)
    parser.add_argument("--rps_rear_density_support_min", type=float, default=0.45)
    parser.add_argument("--rps_rear_selectivity_enable", action="store_true")
    parser.add_argument("--rps_rear_selectivity_support_weight", type=float, default=0.18)
    parser.add_argument("--rps_rear_selectivity_history_weight", type=float, default=0.24)
    parser.add_argument("--rps_rear_selectivity_static_weight", type=float, default=0.16)
    parser.add_argument("--rps_rear_selectivity_geom_weight", type=float, default=0.22)
    parser.add_argument("--rps_rear_selectivity_bridge_weight", type=float, default=0.10)
    parser.add_argument("--rps_rear_selectivity_density_weight", type=float, default=0.10)
    parser.add_argument("--rps_rear_selectivity_rear_score_weight", type=float, default=0.28)
    parser.add_argument("--rps_rear_selectivity_front_score_weight", type=float, default=0.28)
    parser.add_argument("--rps_rear_selectivity_competition_weight", type=float, default=0.34)
    parser.add_argument("--rps_rear_selectivity_competition_alpha", type=float, default=0.80)
    parser.add_argument("--rps_rear_selectivity_gap_weight", type=float, default=0.18)
    parser.add_argument("--rps_rear_selectivity_sep_weight", type=float, default=0.08)
    parser.add_argument("--rps_rear_selectivity_dyn_weight", type=float, default=0.22)
    parser.add_argument("--rps_rear_selectivity_ghost_weight", type=float, default=0.18)
    parser.add_argument("--rps_rear_selectivity_front_weight", type=float, default=0.16)
    parser.add_argument("--rps_rear_selectivity_geom_risk_weight", type=float, default=0.22)
    parser.add_argument("--rps_rear_selectivity_history_risk_weight", type=float, default=0.16)
    parser.add_argument("--rps_rear_selectivity_density_risk_weight", type=float, default=0.10)
    parser.add_argument("--rps_rear_selectivity_bridge_relief_weight", type=float, default=0.10)
    parser.add_argument("--rps_rear_selectivity_static_relief_weight", type=float, default=0.08)
    parser.add_argument("--rps_rear_selectivity_gap_risk_weight", type=float, default=0.18)
    parser.add_argument("--rps_rear_selectivity_score_min", type=float, default=0.46)
    parser.add_argument("--rps_rear_selectivity_risk_max", type=float, default=0.45)
    parser.add_argument("--rps_rear_selectivity_geom_floor", type=float, default=0.48)
    parser.add_argument("--rps_rear_selectivity_history_floor", type=float, default=0.36)
    parser.add_argument("--rps_rear_selectivity_bridge_floor", type=float, default=0.12)
    parser.add_argument("--rps_rear_selectivity_competition_floor", type=float, default=-0.02)
    parser.add_argument("--rps_rear_selectivity_front_score_max", type=float, default=0.92)
    parser.add_argument("--rps_rear_selectivity_gap_min", type=float, default=0.018)
    parser.add_argument("--rps_rear_selectivity_gap_max", type=float, default=0.090)
    parser.add_argument("--rps_rear_selectivity_gap_valid_min", type=float, default=0.28)
    parser.add_argument("--rps_rear_selectivity_occlusion_order_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_occlusion_order_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_occlusion_order_risk_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_local_conflict_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_local_conflict_max", type=float, default=1.5)
    parser.add_argument("--rps_rear_selectivity_front_residual_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_front_residual_max", type=float, default=1.5)
    parser.add_argument("--rps_rear_selectivity_occluder_protect_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_occluder_protect_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_occluder_relief_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_dynamic_trail_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_dynamic_trail_max", type=float, default=1.5)
    parser.add_argument("--rps_rear_selectivity_dynamic_trail_relief_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_history_anchor_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_history_anchor_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_history_anchor_relief_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_surface_anchor_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_surface_anchor_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_surface_anchor_risk_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_surface_distance_ref", type=float, default=0.05)
    parser.add_argument("--rps_rear_selectivity_dynamic_shell_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_dynamic_shell_max", type=float, default=1.5)
    parser.add_argument("--rps_rear_selectivity_dynamic_shell_gap_ref", type=float, default=0.05)
    parser.add_argument("--rps_rear_selectivity_conflict_radius_cells", type=int, default=1)
    parser.add_argument("--rps_rear_selectivity_conflict_front_score_min", type=float, default=0.20)
    parser.add_argument("--rps_rear_selectivity_conflict_static_score_min", type=float, default=0.35)
    parser.add_argument("--rps_rear_selectivity_conflict_dist_scale", type=float, default=1.2)
    parser.add_argument("--rps_rear_selectivity_conflict_gap_ref", type=float, default=0.06)
    parser.add_argument("--rps_rear_selectivity_conflict_ref", type=float, default=1.8)
    parser.add_argument("--rps_rear_selectivity_trail_radius_cells", type=int, default=1)
    parser.add_argument("--rps_rear_selectivity_trail_ref", type=float, default=2.0)
    parser.add_argument("--rps_rear_selectivity_density_radius_cells", type=int, default=1)
    parser.add_argument("--rps_rear_selectivity_density_ref", type=int, default=8)
    parser.add_argument("--rps_rear_selectivity_topk", type=int, default=0)
    parser.add_argument("--rps_rear_selectivity_rank_risk_weight", type=float, default=0.55)
    parser.add_argument("--rps_rear_selectivity_penetration_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_penetration_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_penetration_risk_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_penetration_free_ref", type=float, default=0.05)
    parser.add_argument("--rps_rear_selectivity_penetration_max_steps", type=int, default=10)
    parser.add_argument("--rps_rear_selectivity_observation_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_observation_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_observation_risk_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_observation_count_ref", type=float, default=6.0)
    parser.add_argument("--rps_rear_selectivity_observation_min_count", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_unobserved_veto_enable", action="store_true")
    parser.add_argument("--rps_rear_selectivity_static_coherence_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_static_coherence_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_static_coherence_relief_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_static_coherence_radius_cells", type=int, default=1)
    parser.add_argument("--rps_rear_selectivity_static_coherence_ref", type=float, default=0.35)
    parser.add_argument("--rps_rear_selectivity_static_neighbor_min_weight", type=float, default=0.20)
    parser.add_argument("--rps_rear_selectivity_static_neighbor_dyn_max", type=float, default=0.35)
    parser.add_argument("--rps_rear_selectivity_thickness_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_thickness_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_thickness_risk_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_thickness_ref", type=float, default=0.08)
    parser.add_argument("--rps_rear_selectivity_normal_consistency_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_normal_consistency_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_normal_consistency_relief_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_normal_consistency_radius_cells", type=int, default=1)
    parser.add_argument("--rps_rear_selectivity_normal_consistency_dyn_max", type=float, default=0.35)
    parser.add_argument("--rps_rear_selectivity_ray_convergence_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_ray_convergence_floor", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_ray_convergence_relief_weight", type=float, default=0.0)
    parser.add_argument("--rps_rear_selectivity_ray_convergence_radius_cells", type=int, default=1)
    parser.add_argument("--rps_rear_selectivity_ray_convergence_gap_ref", type=float, default=0.06)
    parser.add_argument("--rps_rear_selectivity_ray_convergence_normal_cos", type=float, default=0.75)
    parser.add_argument("--rps_rear_selectivity_ray_convergence_thickness_ref", type=float, default=0.08)
    parser.add_argument("--rps_rear_selectivity_ray_convergence_ref", type=float, default=2.0)
    parser.add_argument("--rps_rear_state_protect_enable", action="store_true")
    parser.add_argument("--rps_rear_state_decay_relax", type=float, default=0.45)
    parser.add_argument("--rps_rear_state_active_floor", type=float, default=0.28)
    parser.add_argument("--rps_commit_quality_enable", action="store_true")
    parser.add_argument("--rps_commit_quality_transfer_gain", type=float, default=0.18)
    parser.add_argument("--rps_commit_quality_rho_gain", type=float, default=0.55)
    parser.add_argument("--rps_commit_quality_geom_blend", type=float, default=0.35)
    parser.add_argument("--rps_commit_quality_sep_scale", type=float, default=0.65)
    parser.add_argument("--joint_bg_state_enable", action="store_true")
    parser.add_argument("--joint_bg_state_on", type=float, default=0.20)
    parser.add_argument("--joint_bg_state_gain", type=float, default=0.55)
    parser.add_argument("--joint_bg_state_rho_gain", type=float, default=0.20)
    parser.add_argument("--joint_bg_state_front_penalty", type=float, default=0.22)
    parser.add_argument("--wdsg_enable", action="store_true")
    parser.add_argument("--wdsg_front_shift_vox", type=float, default=0.90)
    parser.add_argument("--wdsg_rear_shift_vox", type=float, default=1.10)
    parser.add_argument("--wdsg_shell_shift_vox", type=float, default=0.40)
    parser.add_argument("--wdsg_front_mix_gain", type=float, default=0.95)
    parser.add_argument("--wdsg_rear_mix_gain", type=float, default=1.00)
    parser.add_argument("--wdsg_synth_mode", type=str, default="legacy", choices=["legacy", "anchor", "counterfactual", "energy"])
    parser.add_argument("--wdsg_synth_anchor_gain", type=float, default=0.55)
    parser.add_argument("--wdsg_synth_geo_gain", type=float, default=0.35)
    parser.add_argument("--wdsg_synth_bg_gain", type=float, default=0.20)
    parser.add_argument("--wdsg_synth_counterfactual_gain", type=float, default=0.45)
    parser.add_argument("--wdsg_synth_front_repel_gain", type=float, default=0.35)
    parser.add_argument("--wdsg_synth_energy_temp", type=float, default=0.18)
    parser.add_argument("--wdsg_synth_clip_vox", type=float, default=2.40)
    parser.add_argument("--wdsg_conservative_enable", action="store_true")
    parser.add_argument("--wdsg_conservative_ref_vox", type=float, default=0.60)
    parser.add_argument("--wdsg_conservative_min_clip_scale", type=float, default=0.20)
    parser.add_argument("--wdsg_conservative_static_gain", type=float, default=0.45)
    parser.add_argument("--wdsg_conservative_rear_gain", type=float, default=0.25)
    parser.add_argument("--wdsg_conservative_geo_gain", type=float, default=0.20)
    parser.add_argument("--wdsg_conservative_front_penalty", type=float, default=0.35)
    parser.add_argument("--wdsg_local_clip_enable", action="store_true")
    parser.add_argument("--wdsg_local_clip_min_scale", type=float, default=0.70)
    parser.add_argument("--wdsg_local_clip_max_scale", type=float, default=1.18)
    parser.add_argument("--wdsg_local_clip_risk_gain", type=float, default=0.52)
    parser.add_argument("--wdsg_local_clip_expand_gain", type=float, default=0.22)
    parser.add_argument("--wdsg_local_clip_front_gate", type=float, default=0.48)
    parser.add_argument("--wdsg_local_clip_support_gate", type=float, default=0.52)
    parser.add_argument("--wdsg_local_clip_ambiguity_gate", type=float, default=0.12)
    parser.add_argument("--wdsg_local_clip_pfv_gain", type=float, default=0.20)
    parser.add_argument("--wdsg_local_clip_bg_gain", type=float, default=0.18)
    parser.add_argument("--wdsg_route_enable", action="store_true")
    parser.add_argument("--spg_enable", action="store_true")
    parser.add_argument("--otv_enable", action="store_true")
    parser.add_argument("--xmem_enable", action="store_true")
    parser.add_argument("--obl_enable", action="store_true")
    parser.add_argument("--dual_map_enable", action="store_true")
    parser.add_argument("--cmct_enable", action="store_true")
    parser.add_argument("--cgcc_enable", action="store_true")
    parser.add_argument("--pfv_enable", action="store_true")
    parser.add_argument("--pfv_exclusive_enable", action="store_true")
    parser.add_argument("--pfv_commit_delay_enable", action="store_true")
    parser.add_argument("--pfv_bg_candidate_enable", action="store_true")
    parser.add_argument("--tri_map_enable", action="store_true")
    parser.add_argument("--tri_map_promotion_rescue_enable", action="store_true")
    parser.add_argument("--tri_map_hole_rescue_enable", action="store_true")
    parser.add_argument("--pfvp_enable", action="store_true")
    parser.add_argument("--xmem_sep_ref_vox", type=float, default=0.90)
    parser.add_argument("--xmem_occ_alpha", type=float, default=0.18)
    parser.add_argument("--xmem_free_alpha", type=float, default=0.14)
    parser.add_argument("--xmem_score_alpha", type=float, default=0.20)
    parser.add_argument("--xmem_support_ref", type=float, default=0.24)
    parser.add_argument("--xmem_commit_on", type=float, default=0.60)
    parser.add_argument("--xmem_commit_off", type=float, default=0.42)
    parser.add_argument("--xmem_age_ref", type=float, default=1.0)
    parser.add_argument("--xmem_static_guard", type=float, default=0.78)
    parser.add_argument("--xmem_free_gain", type=float, default=0.85)
    parser.add_argument("--xmem_static_veto", type=float, default=0.96)
    parser.add_argument("--xmem_geo_veto", type=float, default=0.92)
    parser.add_argument("--xmem_transient_boost", type=float, default=0.90)
    parser.add_argument("--xmem_dyn_boost", type=float, default=0.82)
    parser.add_argument("--xmem_decay", type=float, default=0.97)
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
    parser.add_argument("--lzcd_solver_iters", type=int, default=3)
    parser.add_argument("--lzcd_solver_lambda_smooth", type=float, default=0.35)
    parser.add_argument("--lzcd_solver_step", type=float, default=0.85)
    parser.add_argument("--lzcd_solver_tol", type=float, default=5e-4)
    parser.add_argument("--lzcd_residual_anchor_weight", type=float, default=0.25)
    parser.add_argument("--lzcd_residual_alpha", type=float, default=0.12)
    parser.add_argument("--lzcd_residual_hit_ref", type=float, default=10.0)
    parser.add_argument("--lzcd_residual_max_abs", type=float, default=0.10)
    parser.add_argument("--lzcd_max_candidates", type=int, default=6000)
    parser.add_argument("--zcbf_enable", action="store_true")
    parser.add_argument("--zcbf_block_size_cells", type=int, default=6)
    parser.add_argument("--zcbf_min_rho", type=float, default=0.25)
    parser.add_argument("--zcbf_min_phi_w", type=float, default=0.6)
    parser.add_argument("--zcbf_max_dscore", type=float, default=0.55)
    parser.add_argument("--zcbf_alpha", type=float, default=0.18)
    parser.add_argument("--zcbf_trim_quantile", type=float, default=0.70)
    parser.add_argument("--zcbf_apply_gain", type=float, default=0.30)
    parser.add_argument("--zcbf_max_bias", type=float, default=0.04)
    parser.add_argument("--zcbf_static_rho_ref", type=float, default=1.0)
    parser.add_argument("--lzcd_use_geo_channel", dest="lzcd_use_geo_channel", action="store_true")
    parser.add_argument("--lzcd_no_geo_channel", dest="lzcd_use_geo_channel", action="store_false")
    parser.set_defaults(lzcd_use_geo_channel=True)
    parser.add_argument("--stcg_enable", action="store_true")
    parser.add_argument("--dccm_enable", action="store_true")
    parser.add_argument("--dccm_alpha", type=float, default=0.16)
    parser.add_argument("--dccm_age_gain", type=float, default=0.10)
    parser.add_argument("--dccm_age_decay", type=float, default=0.94)
    parser.add_argument("--dccm_commit_thresh", type=float, default=0.62)
    parser.add_argument("--dccm_free_weight", type=float, default=0.40)
    parser.add_argument("--dccm_rear_weight", type=float, default=0.22)
    parser.add_argument("--dccm_age_weight", type=float, default=0.18)
    parser.add_argument("--dccm_surface_weight", type=float, default=0.12)
    parser.add_argument("--dccm_rho_weight", type=float, default=0.08)
    parser.add_argument("--stcg_alpha", type=float, default=0.12)
    parser.add_argument("--stcg_conflict_weight", type=float, default=0.60)
    parser.add_argument("--stcg_residual_weight", type=float, default=0.25)
    parser.add_argument("--stcg_osc_weight", type=float, default=0.15)
    parser.add_argument("--stcg_free_ratio_ref", type=float, default=0.90)
    parser.add_argument("--stcg_on_thresh", type=float, default=0.58)
    parser.add_argument("--stcg_off_thresh", type=float, default=0.42)
    parser.add_argument("--zdyn_enable", action="store_true")
    parser.add_argument("--zdyn_alpha_up", type=float, default=0.26)
    parser.add_argument("--zdyn_alpha_down", type=float, default=0.10)
    parser.add_argument("--zdyn_decay", type=float, default=0.985)
    parser.add_argument("--zdyn_conflict_weight", type=float, default=0.40)
    parser.add_argument("--zdyn_visibility_weight", type=float, default=0.25)
    parser.add_argument("--zdyn_residual_weight", type=float, default=0.20)
    parser.add_argument("--zdyn_osc_weight", type=float, default=0.10)
    parser.add_argument("--zdyn_free_ratio_weight", type=float, default=0.05)
    parser.add_argument("--zdyn_free_ratio_ref", type=float, default=1.0)
    parser.add_argument("--sse_em_enable", action="store_true")
    parser.add_argument("--sse_em_prior_temp", type=float, default=0.9)
    parser.add_argument("--sse_em_mstep_alpha", type=float, default=0.20)
    parser.add_argument("--sse_em_static_floor", type=float, default=0.05)
    parser.add_argument("--sse_em_dynamic_ceil", type=float, default=0.95)
    parser.add_argument("--lbr_enable", action="store_true")
    parser.add_argument("--lbr_alpha", type=float, default=0.14)
    parser.add_argument("--lbr_max_bias", type=float, default=0.05)
    parser.add_argument("--lbr_depth_ref", type=float, default=2.5)
    parser.add_argument("--lbr_apply_gain", type=float, default=0.35)
    parser.add_argument("--vcr_enable", action="store_true")
    parser.add_argument("--vcr_alpha", type=float, default=0.16)
    parser.add_argument("--vcr_on_thresh", type=float, default=0.55)
    parser.add_argument("--vcr_off_thresh", type=float, default=0.40)
    parser.add_argument("--rbi_enable", action="store_true")
    parser.add_argument("--rbi_decay", type=float, default=0.92)
    parser.add_argument("--rbi_commit_static_p", type=float, default=0.72)
    parser.add_argument("--rbi_dyn_gate", type=float, default=0.35)
    parser.add_argument("--rbi_min_weight", type=float, default=1.5)
    parser.add_argument("--rbi_recover_gain", type=float, default=0.30)
    parser.add_argument("--rbi_max_step", type=float, default=0.02)
    parser.add_argument("--surface_ebcut_enable", action="store_true")
    parser.add_argument("--surface_ebcut_energy_thresh", type=float, default=0.58)
    parser.add_argument("--surface_ebcut_w_phi", type=float, default=0.30)
    parser.add_argument("--surface_ebcut_w_dyn", type=float, default=0.35)
    parser.add_argument("--surface_ebcut_w_free", type=float, default=0.20)
    parser.add_argument("--surface_ebcut_w_conf", type=float, default=0.15)
    parser.add_argument("--surface_ebcut_w_smooth", type=float, default=0.10)
    parser.add_argument("--surface_ebcut_smooth_radius", type=int, default=1)
    parser.add_argument("--surface_mopc_enable", action="store_true")
    parser.add_argument("--surface_mopc_step", type=float, default=0.02)
    parser.add_argument("--surface_mopc_dyn_target", type=float, default=0.20)
    parser.add_argument("--surface_mopc_rej_target", type=float, default=0.08)
    parser.add_argument("--surface_mopc_drop_min", type=float, default=0.60)
    parser.add_argument("--surface_mopc_drop_max", type=float, default=0.90)
    parser.add_argument("--surface_mopc_maxd_min", type=float, default=0.70)
    parser.add_argument("--surface_mopc_maxd_max", type=float, default=1.00)
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
    cfg.update.depth_bias_offset_m = float(args.depth_bias_offset_m)
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
    cfg.surface.point_bias_along_normal_m = float(args.surface_point_bias_along_normal_m)
    cfg.surface.geometry_chain_coupling_enable = bool(args.surface_geometry_chain_coupling_enable)
    cfg.surface.geometry_chain_coupling_mode = str(args.surface_geometry_chain_coupling_mode)
    cfg.surface.geometry_chain_coupling_donor_root = str(args.surface_geometry_chain_coupling_donor_root)
    cfg.surface.geometry_chain_coupling_project_dist = float(max(0.0, args.surface_geometry_chain_coupling_project_dist))
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
    cfg.surface.ptdsf_persistent_only_enable = bool(args.surface_ptdsf_persistent_only_enable)
    cfg.surface.ptdsf_persistent_min_rho = float(max(0.0, args.surface_ptdsf_persistent_min_rho))
    cfg.surface.ptdsf_static_rho_weight = float(max(0.0, args.surface_ptdsf_static_rho_weight))
    cfg.surface.zcbf_apply_in_extraction = bool(args.surface_zcbf_apply_in_extraction)
    cfg.surface.zcbf_bias_scale = float(max(0.0, args.surface_zcbf_bias_scale))
    cfg.surface.stcg_enable = bool(args.surface_stcg_enable)
    cfg.surface.dccm_enable = bool(args.surface_dccm_enable)
    cfg.surface.dccm_commit_weight = float(max(0.0, args.surface_dccm_commit_weight))
    cfg.surface.dccm_static_guard = float(np.clip(args.surface_dccm_static_guard, 0.0, 1.0))
    cfg.surface.dccm_drop_gain = float(np.clip(args.surface_dccm_drop_gain, 0.0, 1.0))
    cfg.surface.stcg_min_score = float(np.clip(args.surface_stcg_min_score, 0.0, 1.0))
    cfg.surface.stcg_rho_ref = float(max(1e-6, args.surface_stcg_rho_ref))
    cfg.surface.stcg_free_shrink = float(np.clip(args.surface_stcg_free_shrink, 0.0, 1.0))
    cfg.surface.stcg_phi_shrink = float(np.clip(args.surface_stcg_phi_shrink, 0.0, 1.0))
    cfg.surface.stcg_dscore_shrink = float(np.clip(args.surface_stcg_dscore_shrink, 0.0, 1.0))
    cfg.surface.stcg_weight_gain = float(max(0.0, args.surface_stcg_weight_gain))
    cfg.surface.stcg_static_protect = float(np.clip(args.surface_stcg_static_protect, 0.0, 1.0))
    cfg.surface.use_dual_static_channel = bool(args.surface_use_dual_static_channel)
    cfg.surface.dual_p_static_min = float(max(0.0, args.surface_dual_p_static_min))
    cfg.surface.structural_decouple_enable = bool(args.surface_structural_decouple_enable)
    cfg.surface.decouple_min_geo_weight_ratio = float(max(1e-6, args.surface_decouple_min_geo_weight_ratio))
    cfg.surface.decouple_dyn_drop_thresh = float(np.clip(args.surface_decouple_dyn_drop_thresh, 0.0, 1.5))
    cfg.surface.decouple_dyn_rho_guard = float(max(0.0, args.surface_decouple_dyn_rho_guard))
    cfg.surface.decouple_dyn_free_ratio_thresh = float(max(0.0, args.surface_decouple_dyn_free_ratio_thresh))
    cfg.surface.decouple_channel_div_enable = bool(args.surface_decouple_channel_div_enable)
    cfg.surface.decouple_channel_div_thresh = float(max(1e-6, args.surface_decouple_channel_div_thresh))
    cfg.surface.decouple_channel_div_weight = float(max(0.0, args.surface_decouple_channel_div_weight))
    cfg.surface.decouple_channel_div_static_guard = float(np.clip(args.surface_decouple_channel_div_static_guard, 0.0, 1.0))
    cfg.surface.dual_layer_extract_enable = bool(args.surface_dual_layer_extract_enable)
    cfg.surface.dual_layer_geo_min_weight_ratio = float(np.clip(args.surface_dual_layer_geo_min_weight_ratio, 0.0, 1.0))
    cfg.surface.dual_layer_dyn_use_zdyn = bool(args.surface_dual_layer_dyn_use_zdyn)
    cfg.surface.dual_layer_dyn_prob_weight = float(max(0.0, args.surface_dual_layer_dyn_prob_weight))
    cfg.surface.dual_layer_dyn_stmem_weight = float(max(0.0, args.surface_dual_layer_dyn_stmem_weight))
    cfg.surface.dual_layer_dyn_contra_weight = float(max(0.0, args.surface_dual_layer_dyn_contra_weight))
    cfg.surface.dual_layer_dyn_transient_weight = float(max(0.0, args.surface_dual_layer_dyn_transient_weight))
    cfg.surface.dual_layer_dyn_phi_div_weight = float(max(0.0, args.surface_dual_layer_dyn_phi_div_weight))
    cfg.surface.dual_layer_dyn_phi_ratio_weight = float(max(0.0, args.surface_dual_layer_dyn_phi_ratio_weight))
    cfg.surface.dual_layer_dyn_phi_div_ref = float(max(1e-6, args.surface_dual_layer_dyn_phi_div_ref))
    cfg.surface.dual_layer_dyn_use_phi_dyn = bool(args.surface_dual_layer_dyn_use_phi_dyn)
    cfg.surface.dual_layer_compete_enable = bool(args.surface_dual_layer_compete_enable)
    cfg.surface.dual_layer_compete_margin = float(max(0.0, args.surface_dual_layer_compete_margin))
    cfg.surface.dual_layer_compete_geo_weight = float(np.clip(args.surface_dual_layer_compete_geo_weight, 0.0, 1.0))
    cfg.surface.dual_layer_compete_dyn_mix_weight = float(np.clip(args.surface_dual_layer_compete_dyn_mix_weight, 0.0, 1.0))
    cfg.surface.dual_layer_compete_dyn_conf_weight = float(np.clip(args.surface_dual_layer_compete_dyn_conf_weight, 0.0, 1.0))
    cfg.surface.dual_layer_dyn_drop_thresh = float(np.clip(args.surface_dual_layer_dyn_drop_thresh, 0.0, 1.2))
    cfg.surface.dual_layer_dyn_free_ratio_min = float(max(0.0, args.surface_dual_layer_dyn_free_ratio_min))
    cfg.surface.dual_layer_static_anchor_rho = float(max(0.0, args.surface_dual_layer_static_anchor_rho))
    cfg.surface.dual_layer_static_anchor_p = float(np.clip(args.surface_dual_layer_static_anchor_p, 0.0, 1.0))
    cfg.surface.dual_layer_static_anchor_ratio = float(max(1e-6, args.surface_dual_layer_static_anchor_ratio))
    cfg.surface.csr_enable = bool(args.surface_csr_enable)
    cfg.surface.csr_min_score = float(np.clip(args.surface_csr_min_score, 0.0, 1.0))
    cfg.surface.csr_geo_blend = float(np.clip(args.surface_csr_geo_blend, 0.0, 1.0))
    cfg.surface.csr_geo_agree_min = float(np.clip(args.surface_csr_geo_agree_min, 0.0, 1.0))
    cfg.surface.xmap_enable = bool(args.surface_xmap_enable)
    cfg.surface.xmap_dyn_min_score = float(np.clip(args.surface_xmap_dyn_min_score, 0.0, 1.0))
    cfg.surface.xmap_static_min_score = float(np.clip(args.surface_xmap_static_min_score, 0.0, 1.0))
    cfg.surface.xmap_sep_ref_vox = float(max(0.1, args.surface_xmap_sep_ref_vox))
    cfg.surface.omhs_enable = bool(args.surface_omhs_enable)
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
    cfg.assoc.contra_gate_enable = bool(args.assoc_contra_gate_enable)
    cfg.assoc.contra_stmem_weight = float(max(0.0, args.assoc_contra_stmem_weight))
    cfg.assoc.contra_visibility_weight = float(max(0.0, args.assoc_contra_visibility_weight))
    cfg.assoc.contra_residual_weight = float(max(0.0, args.assoc_contra_residual_weight))
    cfg.assoc.contra_free_ratio_ref = float(max(1e-6, args.assoc_contra_free_ratio_ref))
    cfg.assoc.contra_rho_ref = float(max(1e-6, args.assoc_contra_rho_ref))
    cfg.assoc.contra_static_guard = float(np.clip(args.assoc_contra_static_guard, 0.0, 1.0))
    cfg.assoc.contra_rho_guard = float(np.clip(args.assoc_contra_rho_guard, 0.0, 1.0))
    cfg.assoc.contra_d2_boost_max = float(max(1.0, args.assoc_contra_d2_boost_max))
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
    cfg.update.dual_state_enable = bool(args.dual_state_enable)
    cfg.update.dual_state_assoc_weight = float(max(0.0, args.dual_state_assoc_weight))
    cfg.update.dual_state_free_weight = float(max(0.0, args.dual_state_free_weight))
    cfg.update.dual_state_residual_weight = float(max(0.0, args.dual_state_residual_weight))
    cfg.update.dual_state_osc_weight = float(max(0.0, args.dual_state_osc_weight))
    cfg.update.dual_state_pose_weight = float(max(0.0, args.dual_state_pose_weight))
    cfg.update.dual_state_bias = float(np.clip(args.dual_state_bias, 0.0, 1.0))
    cfg.update.dual_state_temp = float(max(1e-3, args.dual_state_temp))
    cfg.update.dual_pose_var_ref = float(max(1e-6, args.dual_pose_var_ref))
    cfg.update.dual_state_static_ema = float(np.clip(args.dual_state_static_ema, 0.01, 0.95))
    cfg.update.dual_state_min_static_ratio = float(np.clip(args.dual_state_min_static_ratio, 0.0, 0.5))
    cfg.update.dual_state_commit_thresh = float(np.clip(args.dual_state_commit_thresh, 0.0, 1.0))
    cfg.update.dual_state_rollback_thresh = float(np.clip(args.dual_state_rollback_thresh, 0.0, 1.0))
    cfg.update.dual_state_commit_gain = float(np.clip(args.dual_state_commit_gain, 0.0, 1.0))
    cfg.update.dual_state_rollback_gain = float(np.clip(args.dual_state_rollback_gain, 0.0, 1.0))
    cfg.update.dual_state_static_protect_rho = float(max(0.0, args.dual_state_static_protect_rho))
    cfg.update.dual_state_static_protect_ratio = float(max(1e-6, args.dual_state_static_protect_ratio))
    cfg.update.dual_state_static_decay_mult = float(max(0.2, args.dual_state_static_decay_mult))
    cfg.update.dual_state_transient_decay_mult = float(max(0.2, args.dual_state_transient_decay_mult))
    cfg.update.ptdsf_enable = bool(args.ptdsf_enable)
    cfg.update.ptdsf_rho_alpha = float(np.clip(args.ptdsf_rho_alpha, 0.01, 1.0))
    cfg.update.ptdsf_static_blend = float(np.clip(args.ptdsf_static_blend, 0.0, 1.0))
    cfg.update.ptdsf_commit_age_ref = float(max(1.0, args.ptdsf_commit_age_ref))
    cfg.update.ptdsf_commit_bonus = float(max(0.0, args.ptdsf_commit_bonus))
    cfg.update.ptdsf_rollback_bonus = float(max(0.0, args.ptdsf_rollback_bonus))
    cfg.update.wod_enable = bool(args.wod_enable)
    cfg.update.rps_enable = bool(args.rps_enable)
    cfg.update.rps_hard_commit_enable = bool(args.rps_hard_commit_enable)
    cfg.update.rps_surface_bank_enable = bool(args.rps_surface_bank_enable)
    cfg.update.rps_bank_margin = float(max(0.0, args.rps_bank_margin))
    cfg.update.rps_bank_separation_ref = float(max(1e-6, args.rps_bank_separation_ref))
    cfg.update.rps_bank_rear_min_score = float(min(1.0, max(0.0, args.rps_bank_rear_min_score)))
    cfg.update.rps_bank_sep_gate = float(min(1.0, max(0.0, args.rps_bank_sep_gate)))
    cfg.update.rps_bank_bg_support_gain = float(max(0.0, args.rps_bank_bg_support_gain))
    cfg.update.rps_bank_front_dyn_penalty_gain = float(max(0.0, args.rps_bank_front_dyn_penalty_gain))
    cfg.update.rps_bank_rear_score_bias = float(max(0.0, args.rps_bank_rear_score_bias))
    cfg.update.rps_bank_soft_competition_enable = bool(args.rps_bank_soft_competition_enable)
    cfg.update.rps_bank_soft_competition_gap = float(max(0.0, args.rps_bank_soft_competition_gap))
    cfg.update.rps_bank_soft_sep_relax = float(max(0.0, args.rps_bank_soft_sep_relax))
    cfg.update.rps_bank_soft_rear_min_relax = float(max(0.0, args.rps_bank_soft_rear_min_relax))
    cfg.update.rps_bank_soft_support_min = float(min(1.0, max(0.0, args.rps_bank_soft_support_min)))
    cfg.update.rps_bank_soft_local_support_gain = float(max(0.0, args.rps_bank_soft_local_support_gain))
    cfg.update.rps_soft_bank_export_enable = bool(args.rps_soft_bank_export_enable)
    cfg.update.rps_soft_bank_min_score = float(min(1.0, max(0.0, args.rps_soft_bank_min_score)))
    cfg.update.rps_soft_bank_gain = float(max(0.0, args.rps_soft_bank_gain))
    cfg.update.rps_soft_bank_commit_relax = float(min(1.0, max(0.0, args.rps_soft_bank_commit_relax)))
    cfg.update.rps_candidate_rescue_enable = bool(args.rps_candidate_rescue_enable)
    cfg.update.rps_candidate_support_gain = float(max(0.0, args.rps_candidate_support_gain))
    cfg.update.rps_candidate_bg_gain = float(max(0.0, args.rps_candidate_bg_gain))
    cfg.update.rps_candidate_rho_gain = float(max(0.0, args.rps_candidate_rho_gain))
    cfg.update.rps_candidate_front_relax = float(min(1.0, max(0.0, args.rps_candidate_front_relax)))
    cfg.update.rps_commit_activation_enable = bool(args.rps_commit_activation_enable)
    cfg.update.rps_commit_threshold = float(min(1.0, max(0.0, args.rps_commit_threshold)))
    cfg.update.rps_commit_release = float(min(1.0, max(0.0, args.rps_commit_release)))
    cfg.update.rps_commit_age_threshold = float(max(1.0, args.rps_commit_age_threshold))
    cfg.update.rps_commit_rho_ref = float(max(1e-6, args.rps_commit_rho_ref))
    cfg.update.rps_commit_weight_ref = float(max(1e-6, args.rps_commit_weight_ref))
    cfg.update.rps_commit_min_cand_rho = float(max(0.0, args.rps_commit_min_cand_rho))
    cfg.update.rps_commit_min_cand_w = float(max(0.0, args.rps_commit_min_cand_w))
    cfg.update.rps_commit_evidence_weight = float(max(0.0, args.rps_commit_evidence_weight))
    cfg.update.rps_commit_geometry_weight = float(max(0.0, args.rps_commit_geometry_weight))
    cfg.update.rps_commit_bg_weight = float(max(0.0, args.rps_commit_bg_weight))
    cfg.update.rps_commit_static_weight = float(max(0.0, args.rps_commit_static_weight))
    cfg.update.rps_commit_front_penalty = float(max(0.0, args.rps_commit_front_penalty))
    cfg.update.rps_admission_support_enable = bool(args.rps_admission_support_enable)
    cfg.update.rps_admission_support_on = float(min(1.0, max(0.0, args.rps_admission_support_on)))
    cfg.update.rps_admission_support_gain = float(max(0.0, args.rps_admission_support_gain))
    cfg.update.rps_admission_score_relax = float(max(0.0, args.rps_admission_score_relax))
    cfg.update.rps_admission_active_floor = float(min(1.0, max(0.0, args.rps_admission_active_floor)))
    cfg.update.rps_admission_rho_ref = float(max(1e-6, args.rps_admission_rho_ref))
    cfg.update.rps_admission_weight_ref = float(max(1e-6, args.rps_admission_weight_ref))
    cfg.update.rps_admission_geometry_enable = bool(args.rps_admission_geometry_enable)
    cfg.update.rps_admission_geometry_weight = float(max(0.0, args.rps_admission_geometry_weight))
    cfg.update.rps_admission_geometry_floor = float(min(1.0, max(0.0, args.rps_admission_geometry_floor)))
    cfg.update.rps_admission_occlusion_enable = bool(args.rps_admission_occlusion_enable)
    cfg.update.rps_admission_occlusion_weight = float(max(0.0, args.rps_admission_occlusion_weight))
    cfg.update.rps_space_redirect_history_enable = bool(args.rps_space_redirect_history_enable)
    cfg.update.rps_space_redirect_history_weight = float(max(0.0, args.rps_space_redirect_history_weight))
    cfg.update.rps_space_redirect_history_bg_weight = float(max(0.0, args.rps_space_redirect_history_bg_weight))
    cfg.update.rps_space_redirect_history_static_weight = float(max(0.0, args.rps_space_redirect_history_static_weight))
    cfg.update.rps_space_redirect_history_floor = float(min(1.0, max(0.0, args.rps_space_redirect_history_floor)))
    cfg.update.rps_space_redirect_ghost_suppress_enable = bool(args.rps_space_redirect_ghost_suppress_enable)
    cfg.update.rps_space_redirect_ghost_suppress_weight = float(max(0.0, args.rps_space_redirect_ghost_suppress_weight))
    cfg.update.rps_space_redirect_visual_anchor_enable = bool(args.rps_space_redirect_visual_anchor_enable)
    cfg.update.rps_space_redirect_visual_anchor_weight = float(max(0.0, args.rps_space_redirect_visual_anchor_weight))
    cfg.update.rps_space_redirect_visual_anchor_min = float(min(1.0, max(0.0, args.rps_space_redirect_visual_anchor_min)))
    cfg.update.rps_history_obstructed_gate_enable = bool(args.rps_history_obstructed_gate_enable)
    cfg.update.rps_history_visible_min = float(min(1.0, max(0.0, args.rps_history_visible_min)))
    cfg.update.rps_obstruction_min = float(min(1.0, max(0.0, args.rps_obstruction_min)))
    cfg.update.rps_non_hole_min = float(min(1.0, max(0.0, args.rps_non_hole_min)))
    cfg.update.rps_history_manifold_enable = bool(args.rps_history_manifold_enable)
    cfg.update.rps_history_manifold_visible_min = float(min(1.0, max(0.0, args.rps_history_manifold_visible_min)))
    cfg.update.rps_history_manifold_obstruction_min = float(min(1.0, max(0.0, args.rps_history_manifold_obstruction_min)))
    cfg.update.rps_history_manifold_bg_weight = float(max(0.0, args.rps_history_manifold_bg_weight))
    cfg.update.rps_history_manifold_geo_weight = float(max(0.0, args.rps_history_manifold_geo_weight))
    cfg.update.rps_history_manifold_static_weight = float(max(0.0, args.rps_history_manifold_static_weight))
    cfg.update.rps_history_manifold_blend = float(min(1.0, max(0.0, args.rps_history_manifold_blend)))
    cfg.update.rps_history_manifold_max_offset = float(max(0.0, args.rps_history_manifold_max_offset))
    cfg.update.rps_bg_manifold_state_enable = bool(args.rps_bg_manifold_state_enable)
    cfg.update.rps_bg_manifold_alpha_up = float(max(0.0, args.rps_bg_manifold_alpha_up))
    cfg.update.rps_bg_manifold_alpha_down = float(max(0.0, args.rps_bg_manifold_alpha_down))
    cfg.update.rps_bg_manifold_rho_alpha = float(max(0.0, args.rps_bg_manifold_rho_alpha))
    cfg.update.rps_bg_manifold_weight_gain = float(max(0.0, args.rps_bg_manifold_weight_gain))
    cfg.update.rps_bg_manifold_rho_ref = float(max(1e-6, args.rps_bg_manifold_rho_ref))
    cfg.update.rps_bg_manifold_weight_ref = float(max(1e-6, args.rps_bg_manifold_weight_ref))
    cfg.update.rps_bg_manifold_history_weight = float(max(0.0, args.rps_bg_manifold_history_weight))
    cfg.update.rps_bg_manifold_obstruction_weight = float(max(0.0, args.rps_bg_manifold_obstruction_weight))
    cfg.update.rps_bg_manifold_visible_lo = float(min(1.0, max(0.0, args.rps_bg_manifold_visible_lo)))
    cfg.update.rps_bg_manifold_visible_hi = float(min(1.0, max(0.0, args.rps_bg_manifold_visible_hi)))
    cfg.update.rps_bg_dense_state_enable = bool(args.rps_bg_dense_state_enable)
    cfg.update.rps_bg_dense_neighbor_radius = int(max(1, args.rps_bg_dense_neighbor_radius))
    cfg.update.rps_bg_dense_neighbor_weight = float(max(0.0, args.rps_bg_dense_neighbor_weight))
    cfg.update.rps_bg_dense_geometry_weight = float(max(0.0, args.rps_bg_dense_geometry_weight))
    cfg.update.rps_bg_dense_max_weight = float(max(0.0, args.rps_bg_dense_max_weight))
    cfg.update.rps_bg_dense_support_floor = float(min(1.0, max(0.0, args.rps_bg_dense_support_floor)))
    cfg.update.rps_bg_dense_decay = float(max(0.0, args.rps_bg_dense_decay))
    cfg.update.rps_bg_surface_constrained_enable = bool(args.rps_bg_surface_constrained_enable)
    cfg.update.rps_bg_surface_min_conf = float(min(1.0, max(0.0, args.rps_bg_surface_min_conf)))
    cfg.update.rps_bg_surface_agree_weight = float(max(0.0, args.rps_bg_surface_agree_weight))
    cfg.update.rps_bg_surface_tangent_enable = bool(args.rps_bg_surface_tangent_enable)
    cfg.update.rps_bg_surface_tangent_weight = float(max(0.0, args.rps_bg_surface_tangent_weight))
    cfg.update.rps_bg_surface_tangent_floor = float(min(1.0, max(0.0, args.rps_bg_surface_tangent_floor)))
    cfg.update.rps_bg_bridge_enable = bool(args.rps_bg_bridge_enable)
    cfg.update.rps_bg_bridge_min_visible = float(min(1.0, max(0.0, args.rps_bg_bridge_min_visible)))
    cfg.update.rps_bg_bridge_min_obstruction = float(min(1.0, max(0.0, args.rps_bg_bridge_min_obstruction)))
    cfg.update.rps_bg_bridge_min_step = int(max(1, args.rps_bg_bridge_min_step))
    cfg.update.rps_bg_bridge_max_step = int(max(1, args.rps_bg_bridge_max_step))
    cfg.update.rps_bg_bridge_gain = float(max(0.0, args.rps_bg_bridge_gain))
    cfg.update.rps_bg_bridge_phi_blend = float(min(1.0, max(0.0, args.rps_bg_bridge_phi_blend)))
    cfg.update.rps_bg_bridge_target_dyn_max = float(min(1.0, max(0.0, args.rps_bg_bridge_target_dyn_max)))
    cfg.update.rps_bg_bridge_target_surface_max = float(min(1.0, max(0.0, args.rps_bg_bridge_target_surface_max)))
    cfg.update.rps_bg_bridge_ghost_suppress_enable = bool(args.rps_bg_bridge_ghost_suppress_enable)
    cfg.update.rps_bg_bridge_ghost_suppress_weight = float(max(0.0, args.rps_bg_bridge_ghost_suppress_weight))
    cfg.update.rps_bg_bridge_relaxed_dyn_max = float(min(1.0, max(0.0, args.rps_bg_bridge_relaxed_dyn_max)))
    cfg.update.rps_bg_bridge_keep_multi_hits = bool(args.rps_bg_bridge_keep_multi_hits)
    cfg.update.rps_bg_bridge_max_hits_per_source = int(max(1, args.rps_bg_bridge_max_hits_per_source))
    cfg.update.rps_bg_bridge_cone_enable = bool(args.rps_bg_bridge_cone_enable)
    cfg.update.rps_bg_bridge_cone_radius_cells = int(max(0, args.rps_bg_bridge_cone_radius_cells))
    cfg.update.rps_bg_bridge_cone_gain_scale = float(max(0.0, args.rps_bg_bridge_cone_gain_scale))
    cfg.update.rps_bg_bridge_patch_radius_cells = int(max(0, args.rps_bg_bridge_patch_radius_cells))
    cfg.update.rps_bg_bridge_patch_gain_scale = float(max(0.0, args.rps_bg_bridge_patch_gain_scale))
    cfg.update.rps_bg_bridge_depth_hypothesis_count = int(max(0, args.rps_bg_bridge_depth_hypothesis_count))
    cfg.update.rps_bg_bridge_depth_step_scale = float(max(0.0, args.rps_bg_bridge_depth_step_scale))
    cfg.update.rps_bg_bridge_rear_synth_enable = bool(args.rps_bg_bridge_rear_synth_enable)
    cfg.update.rps_bg_bridge_rear_support_gain = float(max(0.0, args.rps_bg_bridge_rear_support_gain))
    cfg.update.rps_bg_bridge_rear_rho_gain = float(max(0.0, args.rps_bg_bridge_rear_rho_gain))
    cfg.update.rps_bg_bridge_rear_phi_blend = float(min(1.0, max(0.0, args.rps_bg_bridge_rear_phi_blend)))
    cfg.update.rps_bg_bridge_rear_score_floor = float(min(1.0, max(0.0, args.rps_bg_bridge_rear_score_floor)))
    cfg.update.rps_bg_bridge_rear_active_floor = float(min(1.0, max(0.0, args.rps_bg_bridge_rear_active_floor)))
    cfg.update.rps_bg_bridge_rear_age_floor = float(max(0.0, args.rps_bg_bridge_rear_age_floor))
    cfg.update.rps_rear_hybrid_filter_enable = bool(args.rps_rear_hybrid_filter_enable)
    cfg.update.rps_rear_hybrid_bridge_support_min = float(min(1.0, max(0.0, args.rps_rear_hybrid_bridge_support_min)))
    cfg.update.rps_rear_hybrid_dyn_max = float(min(1.0, max(0.0, args.rps_rear_hybrid_dyn_max)))
    cfg.update.rps_rear_hybrid_manifold_min = float(min(1.0, max(0.0, args.rps_rear_hybrid_manifold_min)))
    cfg.update.rps_rear_density_gate_enable = bool(args.rps_rear_density_gate_enable)
    cfg.update.rps_rear_density_radius_cells = int(max(1, args.rps_rear_density_radius_cells))
    cfg.update.rps_rear_density_min_neighbors = int(max(0, args.rps_rear_density_min_neighbors))
    cfg.update.rps_rear_density_support_min = float(min(1.0, max(0.0, args.rps_rear_density_support_min)))
    cfg.update.rps_rear_selectivity_enable = bool(args.rps_rear_selectivity_enable)
    cfg.update.rps_rear_selectivity_support_weight = float(max(0.0, args.rps_rear_selectivity_support_weight))
    cfg.update.rps_rear_selectivity_history_weight = float(max(0.0, args.rps_rear_selectivity_history_weight))
    cfg.update.rps_rear_selectivity_static_weight = float(max(0.0, args.rps_rear_selectivity_static_weight))
    cfg.update.rps_rear_selectivity_geom_weight = float(max(0.0, args.rps_rear_selectivity_geom_weight))
    cfg.update.rps_rear_selectivity_bridge_weight = float(max(0.0, args.rps_rear_selectivity_bridge_weight))
    cfg.update.rps_rear_selectivity_density_weight = float(max(0.0, args.rps_rear_selectivity_density_weight))
    cfg.update.rps_rear_selectivity_rear_score_weight = float(max(0.0, args.rps_rear_selectivity_rear_score_weight))
    cfg.update.rps_rear_selectivity_front_score_weight = float(max(0.0, args.rps_rear_selectivity_front_score_weight))
    cfg.update.rps_rear_selectivity_competition_weight = float(max(0.0, args.rps_rear_selectivity_competition_weight))
    cfg.update.rps_rear_selectivity_competition_alpha = float(max(0.0, args.rps_rear_selectivity_competition_alpha))
    cfg.update.rps_rear_selectivity_gap_weight = float(max(0.0, args.rps_rear_selectivity_gap_weight))
    cfg.update.rps_rear_selectivity_sep_weight = float(max(0.0, args.rps_rear_selectivity_sep_weight))
    cfg.update.rps_rear_selectivity_dyn_weight = float(max(0.0, args.rps_rear_selectivity_dyn_weight))
    cfg.update.rps_rear_selectivity_ghost_weight = float(max(0.0, args.rps_rear_selectivity_ghost_weight))
    cfg.update.rps_rear_selectivity_front_weight = float(max(0.0, args.rps_rear_selectivity_front_weight))
    cfg.update.rps_rear_selectivity_geom_risk_weight = float(max(0.0, args.rps_rear_selectivity_geom_risk_weight))
    cfg.update.rps_rear_selectivity_history_risk_weight = float(max(0.0, args.rps_rear_selectivity_history_risk_weight))
    cfg.update.rps_rear_selectivity_density_risk_weight = float(max(0.0, args.rps_rear_selectivity_density_risk_weight))
    cfg.update.rps_rear_selectivity_bridge_relief_weight = float(max(0.0, args.rps_rear_selectivity_bridge_relief_weight))
    cfg.update.rps_rear_selectivity_static_relief_weight = float(max(0.0, args.rps_rear_selectivity_static_relief_weight))
    cfg.update.rps_rear_selectivity_gap_risk_weight = float(max(0.0, args.rps_rear_selectivity_gap_risk_weight))
    cfg.update.rps_rear_selectivity_score_min = float(min(1.0, max(0.0, args.rps_rear_selectivity_score_min)))
    cfg.update.rps_rear_selectivity_risk_max = float(max(0.0, args.rps_rear_selectivity_risk_max))
    cfg.update.rps_rear_selectivity_geom_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_geom_floor)))
    cfg.update.rps_rear_selectivity_history_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_history_floor)))
    cfg.update.rps_rear_selectivity_bridge_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_bridge_floor)))
    cfg.update.rps_rear_selectivity_competition_floor = float(np.clip(args.rps_rear_selectivity_competition_floor, -1.0, 1.0))
    cfg.update.rps_rear_selectivity_front_score_max = float(min(1.0, max(0.0, args.rps_rear_selectivity_front_score_max)))
    cfg.update.rps_rear_selectivity_gap_min = float(max(0.0, args.rps_rear_selectivity_gap_min))
    cfg.update.rps_rear_selectivity_gap_max = float(max(cfg.update.rps_rear_selectivity_gap_min + 1e-6, args.rps_rear_selectivity_gap_max))
    cfg.update.rps_rear_selectivity_gap_valid_min = float(min(1.0, max(0.0, args.rps_rear_selectivity_gap_valid_min)))
    cfg.update.rps_rear_selectivity_occlusion_order_weight = float(max(0.0, args.rps_rear_selectivity_occlusion_order_weight))
    cfg.update.rps_rear_selectivity_occlusion_order_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_occlusion_order_floor)))
    cfg.update.rps_rear_selectivity_occlusion_order_risk_weight = float(max(0.0, args.rps_rear_selectivity_occlusion_order_risk_weight))
    cfg.update.rps_rear_selectivity_local_conflict_weight = float(max(0.0, args.rps_rear_selectivity_local_conflict_weight))
    cfg.update.rps_rear_selectivity_local_conflict_max = float(max(0.0, args.rps_rear_selectivity_local_conflict_max))
    cfg.update.rps_rear_selectivity_front_residual_weight = float(max(0.0, args.rps_rear_selectivity_front_residual_weight))
    cfg.update.rps_rear_selectivity_front_residual_max = float(max(0.0, args.rps_rear_selectivity_front_residual_max))
    cfg.update.rps_rear_selectivity_occluder_protect_weight = float(max(0.0, args.rps_rear_selectivity_occluder_protect_weight))
    cfg.update.rps_rear_selectivity_occluder_protect_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_occluder_protect_floor)))
    cfg.update.rps_rear_selectivity_occluder_relief_weight = float(max(0.0, args.rps_rear_selectivity_occluder_relief_weight))
    cfg.update.rps_rear_selectivity_dynamic_trail_weight = float(max(0.0, args.rps_rear_selectivity_dynamic_trail_weight))
    cfg.update.rps_rear_selectivity_dynamic_trail_max = float(max(0.0, args.rps_rear_selectivity_dynamic_trail_max))
    cfg.update.rps_rear_selectivity_dynamic_trail_relief_weight = float(max(0.0, args.rps_rear_selectivity_dynamic_trail_relief_weight))
    cfg.update.rps_rear_selectivity_history_anchor_weight = float(max(0.0, args.rps_rear_selectivity_history_anchor_weight))
    cfg.update.rps_rear_selectivity_history_anchor_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_history_anchor_floor)))
    cfg.update.rps_rear_selectivity_history_anchor_relief_weight = float(max(0.0, args.rps_rear_selectivity_history_anchor_relief_weight))
    cfg.update.rps_rear_selectivity_surface_anchor_weight = float(max(0.0, args.rps_rear_selectivity_surface_anchor_weight))
    cfg.update.rps_rear_selectivity_surface_anchor_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_surface_anchor_floor)))
    cfg.update.rps_rear_selectivity_surface_anchor_risk_weight = float(max(0.0, args.rps_rear_selectivity_surface_anchor_risk_weight))
    cfg.update.rps_rear_selectivity_surface_distance_ref = float(max(1e-6, args.rps_rear_selectivity_surface_distance_ref))
    cfg.update.rps_rear_selectivity_dynamic_shell_weight = float(max(0.0, args.rps_rear_selectivity_dynamic_shell_weight))
    cfg.update.rps_rear_selectivity_dynamic_shell_max = float(max(0.0, args.rps_rear_selectivity_dynamic_shell_max))
    cfg.update.rps_rear_selectivity_dynamic_shell_gap_ref = float(max(1e-6, args.rps_rear_selectivity_dynamic_shell_gap_ref))
    cfg.update.rps_rear_selectivity_conflict_radius_cells = int(max(1, args.rps_rear_selectivity_conflict_radius_cells))
    cfg.update.rps_rear_selectivity_conflict_front_score_min = float(min(1.0, max(0.0, args.rps_rear_selectivity_conflict_front_score_min)))
    cfg.update.rps_rear_selectivity_conflict_static_score_min = float(min(1.0, max(0.0, args.rps_rear_selectivity_conflict_static_score_min)))
    cfg.update.rps_rear_selectivity_conflict_dist_scale = float(max(0.1, args.rps_rear_selectivity_conflict_dist_scale))
    cfg.update.rps_rear_selectivity_conflict_gap_ref = float(max(1e-6, args.rps_rear_selectivity_conflict_gap_ref))
    cfg.update.rps_rear_selectivity_conflict_ref = float(max(1e-6, args.rps_rear_selectivity_conflict_ref))
    cfg.update.rps_rear_selectivity_trail_radius_cells = int(max(1, args.rps_rear_selectivity_trail_radius_cells))
    cfg.update.rps_rear_selectivity_trail_ref = float(max(1e-6, args.rps_rear_selectivity_trail_ref))
    cfg.update.rps_rear_selectivity_density_radius_cells = int(max(1, args.rps_rear_selectivity_density_radius_cells))
    cfg.update.rps_rear_selectivity_density_ref = int(max(1, args.rps_rear_selectivity_density_ref))
    cfg.update.rps_rear_selectivity_topk = int(max(0, args.rps_rear_selectivity_topk))
    cfg.update.rps_rear_selectivity_rank_risk_weight = float(max(0.0, args.rps_rear_selectivity_rank_risk_weight))
    cfg.update.rps_rear_selectivity_penetration_weight = float(max(0.0, args.rps_rear_selectivity_penetration_weight))
    cfg.update.rps_rear_selectivity_penetration_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_penetration_floor)))
    cfg.update.rps_rear_selectivity_penetration_risk_weight = float(max(0.0, args.rps_rear_selectivity_penetration_risk_weight))
    cfg.update.rps_rear_selectivity_penetration_free_ref = float(max(1e-6, args.rps_rear_selectivity_penetration_free_ref))
    cfg.update.rps_rear_selectivity_penetration_max_steps = int(max(2, args.rps_rear_selectivity_penetration_max_steps))
    cfg.update.rps_rear_selectivity_observation_weight = float(max(0.0, args.rps_rear_selectivity_observation_weight))
    cfg.update.rps_rear_selectivity_observation_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_observation_floor)))
    cfg.update.rps_rear_selectivity_observation_risk_weight = float(max(0.0, args.rps_rear_selectivity_observation_risk_weight))
    cfg.update.rps_rear_selectivity_observation_count_ref = float(max(1e-6, args.rps_rear_selectivity_observation_count_ref))
    cfg.update.rps_rear_selectivity_observation_min_count = float(max(0.0, args.rps_rear_selectivity_observation_min_count))
    cfg.update.rps_rear_selectivity_unobserved_veto_enable = bool(args.rps_rear_selectivity_unobserved_veto_enable)
    cfg.update.rps_rear_selectivity_static_coherence_weight = float(max(0.0, args.rps_rear_selectivity_static_coherence_weight))
    cfg.update.rps_rear_selectivity_static_coherence_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_static_coherence_floor)))
    cfg.update.rps_rear_selectivity_static_coherence_relief_weight = float(max(0.0, args.rps_rear_selectivity_static_coherence_relief_weight))
    cfg.update.rps_rear_selectivity_static_coherence_radius_cells = int(max(1, args.rps_rear_selectivity_static_coherence_radius_cells))
    cfg.update.rps_rear_selectivity_static_coherence_ref = float(max(1e-6, args.rps_rear_selectivity_static_coherence_ref))
    cfg.update.rps_rear_selectivity_static_neighbor_min_weight = float(max(0.0, args.rps_rear_selectivity_static_neighbor_min_weight))
    cfg.update.rps_rear_selectivity_static_neighbor_dyn_max = float(min(1.0, max(0.0, args.rps_rear_selectivity_static_neighbor_dyn_max)))
    cfg.update.rps_rear_selectivity_thickness_weight = float(max(0.0, args.rps_rear_selectivity_thickness_weight))
    cfg.update.rps_rear_selectivity_thickness_floor = float(max(0.0, args.rps_rear_selectivity_thickness_floor))
    cfg.update.rps_rear_selectivity_thickness_risk_weight = float(max(0.0, args.rps_rear_selectivity_thickness_risk_weight))
    cfg.update.rps_rear_selectivity_thickness_ref = float(max(1e-6, args.rps_rear_selectivity_thickness_ref))
    cfg.update.rps_rear_selectivity_normal_consistency_weight = float(max(0.0, args.rps_rear_selectivity_normal_consistency_weight))
    cfg.update.rps_rear_selectivity_normal_consistency_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_normal_consistency_floor)))
    cfg.update.rps_rear_selectivity_normal_consistency_relief_weight = float(max(0.0, args.rps_rear_selectivity_normal_consistency_relief_weight))
    cfg.update.rps_rear_selectivity_normal_consistency_radius_cells = int(max(1, args.rps_rear_selectivity_normal_consistency_radius_cells))
    cfg.update.rps_rear_selectivity_normal_consistency_dyn_max = float(min(1.0, max(0.0, args.rps_rear_selectivity_normal_consistency_dyn_max)))
    cfg.update.rps_rear_selectivity_ray_convergence_weight = float(max(0.0, args.rps_rear_selectivity_ray_convergence_weight))
    cfg.update.rps_rear_selectivity_ray_convergence_floor = float(min(1.0, max(0.0, args.rps_rear_selectivity_ray_convergence_floor)))
    cfg.update.rps_rear_selectivity_ray_convergence_relief_weight = float(max(0.0, args.rps_rear_selectivity_ray_convergence_relief_weight))
    cfg.update.rps_rear_selectivity_ray_convergence_radius_cells = int(max(1, args.rps_rear_selectivity_ray_convergence_radius_cells))
    cfg.update.rps_rear_selectivity_ray_convergence_gap_ref = float(max(1e-6, args.rps_rear_selectivity_ray_convergence_gap_ref))
    cfg.update.rps_rear_selectivity_ray_convergence_normal_cos = float(min(1.0, max(0.0, args.rps_rear_selectivity_ray_convergence_normal_cos)))
    cfg.update.rps_rear_selectivity_ray_convergence_thickness_ref = float(max(1e-6, args.rps_rear_selectivity_ray_convergence_thickness_ref))
    cfg.update.rps_rear_selectivity_ray_convergence_ref = float(max(1e-6, args.rps_rear_selectivity_ray_convergence_ref))
    cfg.update.rps_rear_state_protect_enable = bool(args.rps_rear_state_protect_enable)
    cfg.update.rps_rear_state_decay_relax = float(max(0.0, args.rps_rear_state_decay_relax))
    cfg.update.rps_rear_state_active_floor = float(min(1.0, max(0.0, args.rps_rear_state_active_floor)))
    cfg.update.rps_commit_quality_enable = bool(args.rps_commit_quality_enable)
    cfg.update.rps_commit_quality_transfer_gain = float(max(0.0, args.rps_commit_quality_transfer_gain))
    cfg.update.rps_commit_quality_rho_gain = float(max(0.0, args.rps_commit_quality_rho_gain))
    cfg.update.rps_commit_quality_geom_blend = float(min(1.0, max(0.0, args.rps_commit_quality_geom_blend)))
    cfg.update.rps_commit_quality_sep_scale = float(max(0.0, args.rps_commit_quality_sep_scale))
    cfg.update.joint_bg_state_enable = bool(args.joint_bg_state_enable)
    cfg.update.joint_bg_state_on = float(min(1.0, max(0.0, args.joint_bg_state_on)))
    cfg.update.joint_bg_state_gain = float(max(0.0, args.joint_bg_state_gain))
    cfg.update.joint_bg_state_rho_gain = float(max(0.0, args.joint_bg_state_rho_gain))
    cfg.update.joint_bg_state_front_penalty = float(min(1.0, max(0.0, args.joint_bg_state_front_penalty)))
    cfg.update.wdsg_enable = bool(args.wdsg_enable)
    cfg.update.wdsg_front_shift_vox = float(max(0.0, args.wdsg_front_shift_vox))
    cfg.update.wdsg_rear_shift_vox = float(max(0.0, args.wdsg_rear_shift_vox))
    cfg.update.wdsg_shell_shift_vox = float(max(0.0, args.wdsg_shell_shift_vox))
    cfg.update.wdsg_front_mix_gain = float(max(0.0, args.wdsg_front_mix_gain))
    cfg.update.wdsg_rear_mix_gain = float(max(0.0, args.wdsg_rear_mix_gain))
    cfg.update.wdsg_synth_mode = str(args.wdsg_synth_mode)
    cfg.update.wdsg_synth_anchor_gain = float(max(0.0, args.wdsg_synth_anchor_gain))
    cfg.update.wdsg_synth_geo_gain = float(max(0.0, args.wdsg_synth_geo_gain))
    cfg.update.wdsg_synth_bg_gain = float(max(0.0, args.wdsg_synth_bg_gain))
    cfg.update.wdsg_synth_counterfactual_gain = float(max(0.0, args.wdsg_synth_counterfactual_gain))
    cfg.update.wdsg_synth_front_repel_gain = float(max(0.0, args.wdsg_synth_front_repel_gain))
    cfg.update.wdsg_synth_energy_temp = float(max(1e-3, args.wdsg_synth_energy_temp))
    cfg.update.wdsg_synth_clip_vox = float(max(0.1, args.wdsg_synth_clip_vox))
    cfg.update.wdsg_conservative_enable = bool(args.wdsg_conservative_enable)
    cfg.update.wdsg_conservative_ref_vox = float(max(0.1, args.wdsg_conservative_ref_vox))
    cfg.update.wdsg_conservative_min_clip_scale = float(max(0.0, min(1.0, args.wdsg_conservative_min_clip_scale)))
    cfg.update.wdsg_conservative_static_gain = float(max(0.0, args.wdsg_conservative_static_gain))
    cfg.update.wdsg_conservative_rear_gain = float(max(0.0, args.wdsg_conservative_rear_gain))
    cfg.update.wdsg_conservative_geo_gain = float(max(0.0, args.wdsg_conservative_geo_gain))
    cfg.update.wdsg_conservative_front_penalty = float(max(0.0, args.wdsg_conservative_front_penalty))
    cfg.update.wdsg_local_clip_enable = bool(args.wdsg_local_clip_enable)
    cfg.update.wdsg_local_clip_min_scale = float(np.clip(args.wdsg_local_clip_min_scale, 0.25, 1.0))
    cfg.update.wdsg_local_clip_max_scale = float(max(cfg.update.wdsg_local_clip_min_scale, args.wdsg_local_clip_max_scale))
    cfg.update.wdsg_local_clip_risk_gain = float(max(0.0, args.wdsg_local_clip_risk_gain))
    cfg.update.wdsg_local_clip_expand_gain = float(max(0.0, args.wdsg_local_clip_expand_gain))
    cfg.update.wdsg_local_clip_front_gate = float(np.clip(args.wdsg_local_clip_front_gate, 0.0, 0.95))
    cfg.update.wdsg_local_clip_support_gate = float(np.clip(args.wdsg_local_clip_support_gate, 1e-3, 1.0))
    cfg.update.wdsg_local_clip_ambiguity_gate = float(np.clip(args.wdsg_local_clip_ambiguity_gate, 0.0, 0.95))
    cfg.update.wdsg_local_clip_pfv_gain = float(max(0.0, args.wdsg_local_clip_pfv_gain))
    cfg.update.wdsg_local_clip_bg_gain = float(max(0.0, args.wdsg_local_clip_bg_gain))
    cfg.update.wdsg_route_enable = bool(args.wdsg_route_enable)
    cfg.update.spg_enable = bool(args.spg_enable)
    cfg.update.otv_enable = bool(args.otv_enable)
    cfg.update.xmem_enable = bool(args.xmem_enable)
    cfg.update.obl_enable = bool(args.obl_enable)
    cfg.update.dual_map_enable = bool(args.dual_map_enable)
    cfg.update.cmct_enable = bool(args.cmct_enable)
    cfg.update.cgcc_enable = bool(args.cgcc_enable)
    cfg.update.pfv_enable = bool(args.pfv_enable)
    cfg.update.pfv_exclusive_enable = bool(args.pfv_exclusive_enable)
    cfg.update.pfv_commit_delay_enable = bool(args.pfv_commit_delay_enable)
    cfg.update.pfv_bg_candidate_enable = bool(args.pfv_bg_candidate_enable)
    cfg.update.tri_map_enable = bool(args.tri_map_enable)
    cfg.update.tri_map_promotion_rescue_enable = bool(args.tri_map_promotion_rescue_enable)
    cfg.update.tri_map_hole_rescue_enable = bool(args.tri_map_hole_rescue_enable)
    cfg.update.pfvp_enable = bool(args.pfvp_enable)
    cfg.update.xmem_sep_ref_vox = float(max(0.1, args.xmem_sep_ref_vox))
    cfg.update.xmem_occ_alpha = float(np.clip(args.xmem_occ_alpha, 0.01, 0.95))
    cfg.update.xmem_free_alpha = float(np.clip(args.xmem_free_alpha, 0.01, 0.95))
    cfg.update.xmem_score_alpha = float(np.clip(args.xmem_score_alpha, 0.01, 0.95))
    cfg.update.xmem_support_ref = float(np.clip(args.xmem_support_ref, 0.01, 0.95))
    cfg.update.xmem_commit_on = float(np.clip(args.xmem_commit_on, 0.0, 1.0))
    cfg.update.xmem_commit_off = float(np.clip(args.xmem_commit_off, 0.0, 1.0))
    cfg.update.xmem_age_ref = float(max(1.0, args.xmem_age_ref))
    cfg.update.xmem_static_guard = float(np.clip(args.xmem_static_guard, 0.0, 1.0))
    cfg.update.xmem_free_gain = float(max(0.0, args.xmem_free_gain))
    cfg.update.xmem_static_veto = float(np.clip(args.xmem_static_veto, 0.0, 1.2))
    cfg.update.xmem_geo_veto = float(np.clip(args.xmem_geo_veto, 0.0, 1.2))
    cfg.update.xmem_transient_boost = float(max(0.0, args.xmem_transient_boost))
    cfg.update.xmem_dyn_boost = float(np.clip(args.xmem_dyn_boost, 0.0, 1.5))
    cfg.update.xmem_decay = float(np.clip(args.xmem_decay, 0.80, 1.0))
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
    cfg.update.lzcd_solver_iters = int(max(1, args.lzcd_solver_iters))
    cfg.update.lzcd_solver_lambda_smooth = float(max(0.0, args.lzcd_solver_lambda_smooth))
    cfg.update.lzcd_solver_step = float(np.clip(args.lzcd_solver_step, 0.10, 1.0))
    cfg.update.lzcd_solver_tol = float(max(1e-6, args.lzcd_solver_tol))
    cfg.update.lzcd_residual_anchor_weight = float(np.clip(args.lzcd_residual_anchor_weight, 0.0, 0.95))
    cfg.update.lzcd_residual_alpha = float(np.clip(args.lzcd_residual_alpha, 0.01, 0.95))
    cfg.update.lzcd_residual_hit_ref = float(max(1.0, args.lzcd_residual_hit_ref))
    cfg.update.lzcd_residual_max_abs = float(max(1e-4, args.lzcd_residual_max_abs))
    cfg.update.lzcd_max_candidates = int(max(0, args.lzcd_max_candidates))
    cfg.update.zcbf_enable = bool(args.zcbf_enable)
    cfg.update.zcbf_block_size_cells = int(max(1, args.zcbf_block_size_cells))
    cfg.update.zcbf_min_rho = float(max(0.0, args.zcbf_min_rho))
    cfg.update.zcbf_min_phi_w = float(max(0.0, args.zcbf_min_phi_w))
    cfg.update.zcbf_max_dscore = float(np.clip(args.zcbf_max_dscore, 0.0, 1.0))
    cfg.update.zcbf_alpha = float(np.clip(args.zcbf_alpha, 0.01, 0.95))
    cfg.update.zcbf_trim_quantile = float(np.clip(args.zcbf_trim_quantile, 0.55, 1.0))
    cfg.update.zcbf_apply_gain = float(np.clip(args.zcbf_apply_gain, 0.0, 1.0))
    cfg.update.zcbf_max_bias = float(max(1e-5, args.zcbf_max_bias))
    cfg.update.zcbf_static_rho_ref = float(max(1e-6, args.zcbf_static_rho_ref))
    cfg.update.lzcd_use_geo_channel = bool(args.lzcd_use_geo_channel)
    cfg.update.stcg_enable = bool(args.stcg_enable)
    cfg.update.dccm_enable = bool(args.dccm_enable)
    cfg.update.dccm_alpha = float(np.clip(args.dccm_alpha, 0.01, 0.95))
    cfg.update.dccm_age_gain = float(max(0.0, args.dccm_age_gain))
    cfg.update.dccm_age_decay = float(np.clip(args.dccm_age_decay, 0.70, 1.0))
    cfg.update.dccm_commit_thresh = float(np.clip(args.dccm_commit_thresh, 0.0, 1.0))
    cfg.update.dccm_free_weight = float(max(0.0, args.dccm_free_weight))
    cfg.update.dccm_rear_weight = float(max(0.0, args.dccm_rear_weight))
    cfg.update.dccm_age_weight = float(max(0.0, args.dccm_age_weight))
    cfg.update.dccm_surface_weight = float(max(0.0, args.dccm_surface_weight))
    cfg.update.dccm_rho_weight = float(max(0.0, args.dccm_rho_weight))
    cfg.update.stcg_alpha = float(np.clip(args.stcg_alpha, 0.01, 0.95))
    cfg.update.stcg_conflict_weight = float(max(0.0, args.stcg_conflict_weight))
    cfg.update.stcg_residual_weight = float(max(0.0, args.stcg_residual_weight))
    cfg.update.stcg_osc_weight = float(max(0.0, args.stcg_osc_weight))
    cfg.update.stcg_free_ratio_ref = float(max(1e-6, args.stcg_free_ratio_ref))
    cfg.update.stcg_on_thresh = float(np.clip(args.stcg_on_thresh, 0.05, 0.95))
    cfg.update.stcg_off_thresh = float(np.clip(args.stcg_off_thresh, 0.01, 0.90))
    cfg.update.zdyn_enable = bool(args.zdyn_enable)
    cfg.update.zdyn_alpha_up = float(np.clip(args.zdyn_alpha_up, 0.01, 0.95))
    cfg.update.zdyn_alpha_down = float(np.clip(args.zdyn_alpha_down, 0.01, 0.95))
    cfg.update.zdyn_decay = float(np.clip(args.zdyn_decay, 0.80, 1.0))
    cfg.update.zdyn_conflict_weight = float(max(0.0, args.zdyn_conflict_weight))
    cfg.update.zdyn_visibility_weight = float(max(0.0, args.zdyn_visibility_weight))
    cfg.update.zdyn_residual_weight = float(max(0.0, args.zdyn_residual_weight))
    cfg.update.zdyn_osc_weight = float(max(0.0, args.zdyn_osc_weight))
    cfg.update.zdyn_free_ratio_weight = float(max(0.0, args.zdyn_free_ratio_weight))
    cfg.update.zdyn_free_ratio_ref = float(max(1e-6, args.zdyn_free_ratio_ref))
    cfg.update.sse_em_enable = bool(args.sse_em_enable)
    cfg.update.sse_em_prior_temp = float(max(1e-3, args.sse_em_prior_temp))
    cfg.update.sse_em_mstep_alpha = float(np.clip(args.sse_em_mstep_alpha, 0.01, 0.95))
    cfg.update.sse_em_static_floor = float(np.clip(args.sse_em_static_floor, 0.0, 0.49))
    cfg.update.sse_em_dynamic_ceil = float(np.clip(args.sse_em_dynamic_ceil, 0.51, 1.0))
    cfg.update.lbr_enable = bool(args.lbr_enable)
    cfg.update.lbr_alpha = float(np.clip(args.lbr_alpha, 0.01, 0.95))
    cfg.update.lbr_max_bias = float(max(1e-5, args.lbr_max_bias))
    cfg.update.lbr_depth_ref = float(max(1e-5, args.lbr_depth_ref))
    cfg.update.lbr_apply_gain = float(np.clip(args.lbr_apply_gain, 0.0, 1.0))
    cfg.update.vcr_enable = bool(args.vcr_enable)
    cfg.update.vcr_alpha = float(np.clip(args.vcr_alpha, 0.01, 0.95))
    cfg.update.vcr_on_thresh = float(np.clip(args.vcr_on_thresh, 0.05, 0.95))
    cfg.update.vcr_off_thresh = float(np.clip(args.vcr_off_thresh, 0.01, 0.90))
    cfg.update.rbi_enable = bool(args.rbi_enable)
    cfg.update.rbi_decay = float(np.clip(args.rbi_decay, 0.60, 0.999))
    cfg.update.rbi_commit_static_p = float(np.clip(args.rbi_commit_static_p, 0.0, 1.0))
    cfg.update.rbi_dyn_gate = float(np.clip(args.rbi_dyn_gate, 0.0, 1.0))
    cfg.update.rbi_min_weight = float(max(1e-6, args.rbi_min_weight))
    cfg.update.rbi_recover_gain = float(np.clip(args.rbi_recover_gain, 0.0, 1.0))
    cfg.update.rbi_max_step = float(max(1e-5, args.rbi_max_step))
    cfg.surface.ebcut_enable = bool(args.surface_ebcut_enable)
    cfg.surface.ebcut_energy_thresh = float(max(0.0, args.surface_ebcut_energy_thresh))
    cfg.surface.ebcut_w_phi = float(max(0.0, args.surface_ebcut_w_phi))
    cfg.surface.ebcut_w_dyn = float(max(0.0, args.surface_ebcut_w_dyn))
    cfg.surface.ebcut_w_free = float(max(0.0, args.surface_ebcut_w_free))
    cfg.surface.ebcut_w_conf = float(max(0.0, args.surface_ebcut_w_conf))
    cfg.surface.ebcut_w_smooth = float(max(0.0, args.surface_ebcut_w_smooth))
    cfg.surface.ebcut_smooth_radius = int(max(0, args.surface_ebcut_smooth_radius))
    cfg.surface.mopc_enable = bool(args.surface_mopc_enable)
    cfg.surface.mopc_step = float(max(1e-6, args.surface_mopc_step))
    cfg.surface.mopc_dyn_target = float(np.clip(args.surface_mopc_dyn_target, 0.0, 1.0))
    cfg.surface.mopc_rej_target = float(np.clip(args.surface_mopc_rej_target, 0.0, 1.0))
    cfg.surface.mopc_drop_min = float(np.clip(args.surface_mopc_drop_min, 0.0, 1.5))
    cfg.surface.mopc_drop_max = float(np.clip(args.surface_mopc_drop_max, 0.0, 1.5))
    cfg.surface.mopc_maxd_min = float(np.clip(args.surface_mopc_maxd_min, 0.0, 1.5))
    cfg.surface.mopc_maxd_max = float(np.clip(args.surface_mopc_maxd_max, 0.0, 1.5))
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
    trimap_delayed: List[float] = []
    trimap_dual: List[float] = []
    trimap_promoted: List[float] = []
    trimap_hybrid: List[float] = []
    trimap_support_gap: List[float] = []
    trimap_gap_score: List[float] = []
    trimap_bg_support: List[float] = []
    trimap_centered_gap: List[float] = []
    trimap_norm_gap: List[float] = []
    trimap_gap_bias: List[float] = []
    trimap_front_occ: List[float] = []
    trimap_delay_score: List[float] = []
    trimap_q_strong_thresh: List[float] = []
    trimap_q_soft_thresh: List[float] = []
    trimap_q_strong_budget: List[float] = []
    trimap_q_soft_budget: List[float] = []
    trimap_hold_blocked: List[float] = []
    trimap_hold_mean: List[float] = []
    trimap_hysteresis_mean: List[float] = []
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
        trimap_delayed.append(float(stat.get("trimap_delayed_only", 0.0)))
        trimap_dual.append(float(stat.get("trimap_dual", 0.0)))
        trimap_promoted.append(float(stat.get("trimap_promoted", 0.0)))
        trimap_hybrid.append(float(stat.get("trimap_hybrid_mean", 0.0)))
        trimap_support_gap.append(float(stat.get("trimap_support_gap_mean", 0.0)))
        trimap_gap_score.append(float(stat.get("trimap_gap_score_mean", 0.0)))
        trimap_bg_support.append(float(stat.get("trimap_bg_support_mean", 0.0)))
        trimap_centered_gap.append(float(stat.get("trimap_centered_gap_mean", 0.0)))
        trimap_norm_gap.append(float(stat.get("trimap_norm_gap_mean", 0.0)))
        trimap_gap_bias.append(float(stat.get("trimap_gap_bias_mean", 0.0)))
        trimap_front_occ.append(float(stat.get("trimap_front_occ_mean", 0.0)))
        trimap_delay_score.append(float(stat.get("trimap_delay_score_mean", 0.0)))
        trimap_q_strong_thresh.append(float(stat.get("trimap_quantile_strong_thresh", 0.0)))
        trimap_q_soft_thresh.append(float(stat.get("trimap_quantile_soft_thresh", 0.0)))
        trimap_q_strong_budget.append(float(stat.get("trimap_quantile_strong_budget", 0.0)))
        trimap_q_soft_budget.append(float(stat.get("trimap_quantile_soft_budget", 0.0)))
        trimap_hold_blocked.append(float(stat.get("trimap_hold_blocked", 0.0)))
        trimap_hold_mean.append(float(stat.get("trimap_hold_mean", 0.0)))
        trimap_hysteresis_mean.append(float(stat.get("trimap_hysteresis_mean", 0.0)))
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
    extract_stats = dict(getattr(model.voxel_map, 'last_extract_stats', {}))
    extract_bank_points = dict(getattr(model.voxel_map, 'last_extract_bank_points', {})) if hasattr(model, 'voxel_map') else {}
    if bool(getattr(cfg.surface, "geometry_chain_coupling_enable", False)):
        donor_root = str(getattr(cfg.surface, "geometry_chain_coupling_donor_root", "")).strip()
        if donor_root:
            front_points = np.asarray(extract_bank_points.get('front_points', np.zeros((0, 3), dtype=float)), dtype=float)
            front_normals = np.asarray(extract_bank_points.get('front_normals', np.zeros((0, 3), dtype=float)), dtype=float)
            coupled_points, coupled_normals, coupled_rear, coupled_rear_normals, coupled_rows, coupling_stats = apply_geometry_chain_coupling(
                sequence=str(args.sequence),
                front_points=front_points,
                front_normals=front_normals if front_normals.shape == front_points.shape else np.zeros_like(front_points),
                surface_points=pred_points,
                surface_normals=pred_normals,
                donor_root=donor_root,
                mode=str(getattr(cfg.surface, "geometry_chain_coupling_mode", "direct")),
                project_dist=float(getattr(cfg.surface, "geometry_chain_coupling_project_dist", 0.05)),
            )
            if coupling_stats.get("geometry_chain_applied", 0.0) > 0.5:
                pred_points = coupled_points
                pred_normals = coupled_normals
                extract_bank_points['rear_points'] = coupled_rear
                extract_bank_points['rear_normals'] = coupled_rear_normals
                extract_bank_points['rear_feature_rows'] = coupled_rows
            extract_stats.update(coupling_stats)
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
    if extract_bank_points:
        save_point_cloud(out_dir / "rear_surface_points.ply", extract_bank_points.get('rear_points', np.zeros((0, 3), dtype=float)), extract_bank_points.get('rear_normals', np.zeros((0, 3), dtype=float)))
        save_point_cloud(out_dir / "front_surface_points.ply", extract_bank_points.get('front_points', np.zeros((0, 3), dtype=float)), extract_bank_points.get('front_normals', np.zeros((0, 3), dtype=float)))
        save_point_cloud(out_dir / "background_surface_points.ply", extract_bank_points.get('background_points', np.zeros((0, 3), dtype=float)), extract_bank_points.get('background_normals', np.zeros((0, 3), dtype=float)))
        save_feature_rows(out_dir / "rear_surface_features.csv", list(extract_bank_points.get('rear_feature_rows', [])))
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

    ptdsf_diag = dict(getattr(model.updater, '_ptdsf_diag', {})) if hasattr(model, 'updater') else {}
    ptdsf_export_diag = dict(getattr(model.voxel_map, '_ptdsf_export_diag', {})) if hasattr(model, 'voxel_map') else {}
    rps_commit_diag = dict(getattr(model.updater, '_rps_commit_diag', {})) if hasattr(model, 'updater') else {}
    rps_competition_diag = dict(getattr(model.voxel_map, '_rps_competition_diag', {})) if hasattr(model, 'voxel_map') else {}
    rps_admission_diag = dict(getattr(model.voxel_map, '_rps_admission_diag', {})) if hasattr(model, 'voxel_map') else {}
    bg_manifold_diag = dict(getattr(model.updater, '_bg_manifold_diag', {})) if hasattr(model, 'updater') else {}
    state_diag = {}
    if hasattr(model, 'voxel_map'):
        cells = list(model.voxel_map.cells.values())
        state_diag = {
            'rear_w_nonzero': float(sum(1 for c in cells if float(getattr(c, 'phi_rear_w', 0.0)) > 1e-9)),
            'rear_cand_w_nonzero': float(sum(1 for c in cells if float(getattr(c, 'phi_rear_cand_w', 0.0)) > 1e-9)),
            'bg_w_nonzero': float(sum(1 for c in cells if float(getattr(c, 'phi_bg_w', 0.0)) > 1e-9)),
            'bg_cand_w_nonzero': float(sum(1 for c in cells if float(getattr(c, 'phi_bg_cand_w', 0.0)) > 1e-9)),
            'rps_active_nonzero': float(sum(1 for c in cells if float(getattr(c, 'rps_active', 0.0)) > 1e-9)),
            'rps_commit_score_nonzero': float(sum(1 for c in cells if float(getattr(c, 'rps_commit_score', 0.0)) > 1e-9)),
            'rps_commit_score_ge_release': float(sum(1 for c in cells if float(getattr(c, 'rps_commit_score', 0.0)) >= float(cfg.update.rps_commit_release))),
            'rps_commit_score_ge_on': float(sum(1 for c in cells if float(getattr(c, 'rps_commit_score', 0.0)) >= float(cfg.update.rps_commit_threshold))),
            'rps_commit_age_ge_thr': float(sum(1 for c in cells if float(getattr(c, 'rps_commit_age', 0.0)) >= float(cfg.update.rps_commit_age_threshold))),
            'rps_commit_score_sum': float(sum(float(getattr(c, 'rps_commit_score', 0.0)) for c in cells)),
            'rho_rear_sum': float(sum(float(getattr(c, 'rho_rear', 0.0)) for c in cells)),
            'rho_rear_cand_sum': float(sum(float(getattr(c, 'rho_rear_cand', 0.0)) for c in cells)),
            'rho_bg_sum': float(sum(float(getattr(c, 'rho_bg', 0.0)) for c in cells)),
            'rho_bg_cand_sum': float(sum(float(getattr(c, 'rho_bg_cand', 0.0)) for c in cells)),
        }
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
        "assoc_contra_gate_enable": bool(cfg.assoc.contra_gate_enable),
        "assoc_contra_stmem_weight": float(cfg.assoc.contra_stmem_weight),
        "assoc_contra_visibility_weight": float(cfg.assoc.contra_visibility_weight),
        "assoc_contra_residual_weight": float(cfg.assoc.contra_residual_weight),
        "assoc_contra_free_ratio_ref": float(cfg.assoc.contra_free_ratio_ref),
        "assoc_contra_rho_ref": float(cfg.assoc.contra_rho_ref),
        "assoc_contra_static_guard": float(cfg.assoc.contra_static_guard),
        "assoc_contra_rho_guard": float(cfg.assoc.contra_rho_guard),
        "assoc_contra_d2_boost_max": float(cfg.assoc.contra_d2_boost_max),
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
        "surface_use_dual_static_channel": bool(cfg.surface.use_dual_static_channel),
        "surface_structural_decouple_enable": bool(cfg.surface.structural_decouple_enable),
        "surface_decouple_min_geo_weight_ratio": float(cfg.surface.decouple_min_geo_weight_ratio),
        "surface_decouple_dyn_drop_thresh": float(cfg.surface.decouple_dyn_drop_thresh),
        "surface_decouple_dyn_rho_guard": float(cfg.surface.decouple_dyn_rho_guard),
        "surface_decouple_dyn_free_ratio_thresh": float(cfg.surface.decouple_dyn_free_ratio_thresh),
        "surface_decouple_channel_div_enable": bool(cfg.surface.decouple_channel_div_enable),
        "surface_decouple_channel_div_thresh": float(cfg.surface.decouple_channel_div_thresh),
        "surface_decouple_channel_div_weight": float(cfg.surface.decouple_channel_div_weight),
        "surface_decouple_channel_div_static_guard": float(cfg.surface.decouple_channel_div_static_guard),
        "surface_dual_layer_extract_enable": bool(cfg.surface.dual_layer_extract_enable),
        "surface_dual_layer_geo_min_weight_ratio": float(cfg.surface.dual_layer_geo_min_weight_ratio),
        "surface_dual_layer_dyn_use_zdyn": bool(cfg.surface.dual_layer_dyn_use_zdyn),
        "surface_dual_layer_dyn_prob_weight": float(cfg.surface.dual_layer_dyn_prob_weight),
        "surface_dual_layer_dyn_stmem_weight": float(cfg.surface.dual_layer_dyn_stmem_weight),
        "surface_dual_layer_dyn_contra_weight": float(cfg.surface.dual_layer_dyn_contra_weight),
        "surface_dual_layer_dyn_transient_weight": float(cfg.surface.dual_layer_dyn_transient_weight),
        "surface_dual_layer_dyn_phi_div_weight": float(cfg.surface.dual_layer_dyn_phi_div_weight),
        "surface_dual_layer_dyn_phi_ratio_weight": float(cfg.surface.dual_layer_dyn_phi_ratio_weight),
        "surface_dual_layer_dyn_phi_div_ref": float(cfg.surface.dual_layer_dyn_phi_div_ref),
        "surface_dual_layer_dyn_use_phi_dyn": bool(cfg.surface.dual_layer_dyn_use_phi_dyn),
        "surface_dual_layer_compete_enable": bool(cfg.surface.dual_layer_compete_enable),
        "surface_dual_layer_compete_margin": float(cfg.surface.dual_layer_compete_margin),
        "surface_dual_layer_compete_geo_weight": float(cfg.surface.dual_layer_compete_geo_weight),
        "surface_dual_layer_compete_dyn_mix_weight": float(cfg.surface.dual_layer_compete_dyn_mix_weight),
        "surface_dual_layer_compete_dyn_conf_weight": float(cfg.surface.dual_layer_compete_dyn_conf_weight),
        "surface_dual_layer_dyn_drop_thresh": float(cfg.surface.dual_layer_dyn_drop_thresh),
        "surface_dual_layer_dyn_free_ratio_min": float(cfg.surface.dual_layer_dyn_free_ratio_min),
        "surface_ebcut_enable": bool(cfg.surface.ebcut_enable),
        "surface_ebcut_energy_thresh": float(cfg.surface.ebcut_energy_thresh),
        "surface_ebcut_w_phi": float(cfg.surface.ebcut_w_phi),
        "surface_ebcut_w_dyn": float(cfg.surface.ebcut_w_dyn),
        "surface_ebcut_w_free": float(cfg.surface.ebcut_w_free),
        "surface_ebcut_w_conf": float(cfg.surface.ebcut_w_conf),
        "surface_ebcut_w_smooth": float(cfg.surface.ebcut_w_smooth),
        "surface_ebcut_smooth_radius": int(cfg.surface.ebcut_smooth_radius),
        "surface_mopc_enable": bool(cfg.surface.mopc_enable),
        "surface_mopc_step": float(cfg.surface.mopc_step),
        "surface_mopc_dyn_target": float(cfg.surface.mopc_dyn_target),
        "surface_mopc_rej_target": float(cfg.surface.mopc_rej_target),
        "surface_mopc_drop_min": float(cfg.surface.mopc_drop_min),
        "surface_mopc_drop_max": float(cfg.surface.mopc_drop_max),
        "surface_mopc_maxd_min": float(cfg.surface.mopc_maxd_min),
        "surface_mopc_maxd_max": float(cfg.surface.mopc_maxd_max),
        "sse_em_enable": bool(cfg.update.sse_em_enable),
        "sse_em_prior_temp": float(cfg.update.sse_em_prior_temp),
        "sse_em_mstep_alpha": float(cfg.update.sse_em_mstep_alpha),
        "sse_em_static_floor": float(cfg.update.sse_em_static_floor),
        "sse_em_dynamic_ceil": float(cfg.update.sse_em_dynamic_ceil),
        "lbr_enable": bool(cfg.update.lbr_enable),
        "lbr_alpha": float(cfg.update.lbr_alpha),
        "lbr_max_bias": float(cfg.update.lbr_max_bias),
        "lbr_depth_ref": float(cfg.update.lbr_depth_ref),
        "lbr_apply_gain": float(cfg.update.lbr_apply_gain),
        "vcr_enable": bool(cfg.update.vcr_enable),
        "vcr_alpha": float(cfg.update.vcr_alpha),
        "vcr_on_thresh": float(cfg.update.vcr_on_thresh),
        "vcr_off_thresh": float(cfg.update.vcr_off_thresh),
        "rbi_enable": bool(cfg.update.rbi_enable),
        "rbi_decay": float(cfg.update.rbi_decay),
        "rbi_commit_static_p": float(cfg.update.rbi_commit_static_p),
        "rbi_dyn_gate": float(cfg.update.rbi_dyn_gate),
        "rbi_min_weight": float(cfg.update.rbi_min_weight),
        "rbi_recover_gain": float(cfg.update.rbi_recover_gain),
        "rbi_max_step": float(cfg.update.rbi_max_step),
        "surface_dual_layer_static_anchor_rho": float(cfg.surface.dual_layer_static_anchor_rho),
        "surface_dual_layer_static_anchor_p": float(cfg.surface.dual_layer_static_anchor_p),
        "surface_dual_layer_static_anchor_ratio": float(cfg.surface.dual_layer_static_anchor_ratio),
        "surface_csr_enable": bool(cfg.surface.csr_enable),
        "surface_csr_min_score": float(cfg.surface.csr_min_score),
        "surface_csr_geo_blend": float(cfg.surface.csr_geo_blend),
        "surface_csr_geo_agree_min": float(cfg.surface.csr_geo_agree_min),
        "surface_xmap_enable": bool(cfg.surface.xmap_enable),
        "surface_xmap_dyn_min_score": float(cfg.surface.xmap_dyn_min_score),
        "surface_xmap_static_min_score": float(cfg.surface.xmap_static_min_score),
        "surface_xmap_sep_ref_vox": float(cfg.surface.xmap_sep_ref_vox),
        "surface_omhs_enable": bool(cfg.surface.omhs_enable),
        "wod_enable": bool(cfg.update.wod_enable),
        "rps_enable": bool(cfg.update.rps_enable),
        "rps_hard_commit_enable": bool(cfg.update.rps_hard_commit_enable),
        "rps_surface_bank_enable": bool(cfg.update.rps_surface_bank_enable),
        "rps_bank_margin": float(cfg.update.rps_bank_margin),
        "rps_bank_separation_ref": float(cfg.update.rps_bank_separation_ref),
        "rps_bank_rear_min_score": float(cfg.update.rps_bank_rear_min_score),
        "rps_bank_sep_gate": float(cfg.update.rps_bank_sep_gate),
        "rps_bank_bg_support_gain": float(cfg.update.rps_bank_bg_support_gain),
        "rps_bank_front_dyn_penalty_gain": float(cfg.update.rps_bank_front_dyn_penalty_gain),
        "rps_bank_rear_score_bias": float(cfg.update.rps_bank_rear_score_bias),
        "rps_bank_soft_competition_enable": bool(cfg.update.rps_bank_soft_competition_enable),
        "rps_bank_soft_competition_gap": float(cfg.update.rps_bank_soft_competition_gap),
        "rps_bank_soft_sep_relax": float(cfg.update.rps_bank_soft_sep_relax),
        "rps_bank_soft_rear_min_relax": float(cfg.update.rps_bank_soft_rear_min_relax),
        "rps_bank_soft_support_min": float(cfg.update.rps_bank_soft_support_min),
        "rps_bank_soft_local_support_gain": float(cfg.update.rps_bank_soft_local_support_gain),
        "rps_soft_bank_export_enable": bool(cfg.update.rps_soft_bank_export_enable),
        "rps_soft_bank_min_score": float(cfg.update.rps_soft_bank_min_score),
        "rps_soft_bank_gain": float(cfg.update.rps_soft_bank_gain),
        "rps_soft_bank_commit_relax": float(cfg.update.rps_soft_bank_commit_relax),
        "rps_candidate_rescue_enable": bool(cfg.update.rps_candidate_rescue_enable),
        "rps_candidate_support_gain": float(cfg.update.rps_candidate_support_gain),
        "rps_candidate_bg_gain": float(cfg.update.rps_candidate_bg_gain),
        "rps_candidate_rho_gain": float(cfg.update.rps_candidate_rho_gain),
        "rps_candidate_front_relax": float(cfg.update.rps_candidate_front_relax),
        "rps_commit_activation_enable": bool(cfg.update.rps_commit_activation_enable),
        "rps_commit_threshold": float(cfg.update.rps_commit_threshold),
        "rps_commit_release": float(cfg.update.rps_commit_release),
        "rps_commit_age_threshold": float(cfg.update.rps_commit_age_threshold),
        "rps_commit_rho_ref": float(cfg.update.rps_commit_rho_ref),
        "rps_commit_weight_ref": float(cfg.update.rps_commit_weight_ref),
        "rps_commit_min_cand_rho": float(cfg.update.rps_commit_min_cand_rho),
        "rps_commit_min_cand_w": float(cfg.update.rps_commit_min_cand_w),
        "rps_commit_evidence_weight": float(cfg.update.rps_commit_evidence_weight),
        "rps_commit_geometry_weight": float(cfg.update.rps_commit_geometry_weight),
        "rps_commit_bg_weight": float(cfg.update.rps_commit_bg_weight),
        "rps_commit_static_weight": float(cfg.update.rps_commit_static_weight),
        "rps_commit_front_penalty": float(cfg.update.rps_commit_front_penalty),
        "rps_admission_support_enable": bool(cfg.update.rps_admission_support_enable),
        "rps_admission_support_on": float(cfg.update.rps_admission_support_on),
        "rps_admission_support_gain": float(cfg.update.rps_admission_support_gain),
        "rps_admission_score_relax": float(cfg.update.rps_admission_score_relax),
        "rps_admission_active_floor": float(cfg.update.rps_admission_active_floor),
        "rps_admission_rho_ref": float(cfg.update.rps_admission_rho_ref),
        "rps_admission_weight_ref": float(cfg.update.rps_admission_weight_ref),
        "rps_admission_geometry_enable": bool(cfg.update.rps_admission_geometry_enable),
        "rps_admission_geometry_weight": float(cfg.update.rps_admission_geometry_weight),
        "rps_admission_geometry_floor": float(cfg.update.rps_admission_geometry_floor),
        "rps_admission_occlusion_enable": bool(cfg.update.rps_admission_occlusion_enable),
        "rps_admission_occlusion_weight": float(cfg.update.rps_admission_occlusion_weight),
        "rps_space_redirect_history_enable": bool(cfg.update.rps_space_redirect_history_enable),
        "rps_space_redirect_history_weight": float(cfg.update.rps_space_redirect_history_weight),
        "rps_space_redirect_history_bg_weight": float(cfg.update.rps_space_redirect_history_bg_weight),
        "rps_space_redirect_history_static_weight": float(cfg.update.rps_space_redirect_history_static_weight),
        "rps_space_redirect_history_floor": float(cfg.update.rps_space_redirect_history_floor),
        "rps_space_redirect_ghost_suppress_enable": bool(cfg.update.rps_space_redirect_ghost_suppress_enable),
        "rps_space_redirect_ghost_suppress_weight": float(cfg.update.rps_space_redirect_ghost_suppress_weight),
        "rps_space_redirect_visual_anchor_enable": bool(cfg.update.rps_space_redirect_visual_anchor_enable),
        "rps_space_redirect_visual_anchor_weight": float(cfg.update.rps_space_redirect_visual_anchor_weight),
        "rps_space_redirect_visual_anchor_min": float(cfg.update.rps_space_redirect_visual_anchor_min),
        "rps_history_obstructed_gate_enable": bool(cfg.update.rps_history_obstructed_gate_enable),
        "rps_history_visible_min": float(cfg.update.rps_history_visible_min),
        "rps_obstruction_min": float(cfg.update.rps_obstruction_min),
        "rps_non_hole_min": float(cfg.update.rps_non_hole_min),
        "rps_history_manifold_enable": bool(cfg.update.rps_history_manifold_enable),
        "rps_history_manifold_visible_min": float(cfg.update.rps_history_manifold_visible_min),
        "rps_history_manifold_obstruction_min": float(cfg.update.rps_history_manifold_obstruction_min),
        "rps_history_manifold_bg_weight": float(cfg.update.rps_history_manifold_bg_weight),
        "rps_history_manifold_geo_weight": float(cfg.update.rps_history_manifold_geo_weight),
        "rps_history_manifold_static_weight": float(cfg.update.rps_history_manifold_static_weight),
        "rps_history_manifold_blend": float(cfg.update.rps_history_manifold_blend),
        "rps_history_manifold_max_offset": float(cfg.update.rps_history_manifold_max_offset),
        "rps_bg_manifold_state_enable": bool(cfg.update.rps_bg_manifold_state_enable),
        "rps_bg_manifold_alpha_up": float(cfg.update.rps_bg_manifold_alpha_up),
        "rps_bg_manifold_alpha_down": float(cfg.update.rps_bg_manifold_alpha_down),
        "rps_bg_manifold_rho_alpha": float(cfg.update.rps_bg_manifold_rho_alpha),
        "rps_bg_manifold_weight_gain": float(cfg.update.rps_bg_manifold_weight_gain),
        "rps_bg_manifold_rho_ref": float(cfg.update.rps_bg_manifold_rho_ref),
        "rps_bg_manifold_weight_ref": float(cfg.update.rps_bg_manifold_weight_ref),
        "rps_bg_manifold_history_weight": float(cfg.update.rps_bg_manifold_history_weight),
        "rps_bg_manifold_obstruction_weight": float(cfg.update.rps_bg_manifold_obstruction_weight),
        "rps_bg_manifold_visible_lo": float(cfg.update.rps_bg_manifold_visible_lo),
        "rps_bg_manifold_visible_hi": float(cfg.update.rps_bg_manifold_visible_hi),
        "rps_bg_dense_state_enable": bool(cfg.update.rps_bg_dense_state_enable),
        "rps_bg_dense_neighbor_radius": int(cfg.update.rps_bg_dense_neighbor_radius),
        "rps_bg_dense_neighbor_weight": float(cfg.update.rps_bg_dense_neighbor_weight),
        "rps_bg_dense_geometry_weight": float(cfg.update.rps_bg_dense_geometry_weight),
        "rps_bg_dense_max_weight": float(cfg.update.rps_bg_dense_max_weight),
        "rps_bg_dense_support_floor": float(cfg.update.rps_bg_dense_support_floor),
        "rps_bg_dense_decay": float(cfg.update.rps_bg_dense_decay),
        "rps_bg_surface_constrained_enable": bool(cfg.update.rps_bg_surface_constrained_enable),
        "rps_bg_surface_min_conf": float(cfg.update.rps_bg_surface_min_conf),
        "rps_bg_surface_agree_weight": float(cfg.update.rps_bg_surface_agree_weight),
        "rps_bg_surface_tangent_enable": bool(cfg.update.rps_bg_surface_tangent_enable),
        "rps_bg_surface_tangent_weight": float(cfg.update.rps_bg_surface_tangent_weight),
        "rps_bg_surface_tangent_floor": float(cfg.update.rps_bg_surface_tangent_floor),
        "rps_bg_bridge_enable": bool(cfg.update.rps_bg_bridge_enable),
        "rps_bg_bridge_min_visible": float(cfg.update.rps_bg_bridge_min_visible),
        "rps_bg_bridge_min_obstruction": float(cfg.update.rps_bg_bridge_min_obstruction),
        "rps_bg_bridge_min_step": int(cfg.update.rps_bg_bridge_min_step),
        "rps_bg_bridge_max_step": int(cfg.update.rps_bg_bridge_max_step),
        "rps_bg_bridge_gain": float(cfg.update.rps_bg_bridge_gain),
        "rps_bg_bridge_phi_blend": float(cfg.update.rps_bg_bridge_phi_blend),
        "rps_bg_bridge_target_dyn_max": float(cfg.update.rps_bg_bridge_target_dyn_max),
        "rps_bg_bridge_target_surface_max": float(cfg.update.rps_bg_bridge_target_surface_max),
        "rps_bg_bridge_ghost_suppress_enable": bool(cfg.update.rps_bg_bridge_ghost_suppress_enable),
        "rps_bg_bridge_ghost_suppress_weight": float(cfg.update.rps_bg_bridge_ghost_suppress_weight),
        "rps_bg_bridge_relaxed_dyn_max": float(cfg.update.rps_bg_bridge_relaxed_dyn_max),
        "rps_bg_bridge_keep_multi_hits": bool(cfg.update.rps_bg_bridge_keep_multi_hits),
        "rps_bg_bridge_max_hits_per_source": int(cfg.update.rps_bg_bridge_max_hits_per_source),
        "rps_bg_bridge_cone_enable": bool(cfg.update.rps_bg_bridge_cone_enable),
        "rps_bg_bridge_cone_radius_cells": int(cfg.update.rps_bg_bridge_cone_radius_cells),
        "rps_bg_bridge_cone_gain_scale": float(cfg.update.rps_bg_bridge_cone_gain_scale),
        "rps_bg_bridge_patch_radius_cells": int(cfg.update.rps_bg_bridge_patch_radius_cells),
        "rps_bg_bridge_patch_gain_scale": float(cfg.update.rps_bg_bridge_patch_gain_scale),
        "rps_bg_bridge_depth_hypothesis_count": int(cfg.update.rps_bg_bridge_depth_hypothesis_count),
        "rps_bg_bridge_depth_step_scale": float(cfg.update.rps_bg_bridge_depth_step_scale),
        "rps_bg_bridge_rear_synth_enable": bool(cfg.update.rps_bg_bridge_rear_synth_enable),
        "rps_bg_bridge_rear_support_gain": float(cfg.update.rps_bg_bridge_rear_support_gain),
        "rps_bg_bridge_rear_rho_gain": float(cfg.update.rps_bg_bridge_rear_rho_gain),
        "rps_bg_bridge_rear_phi_blend": float(cfg.update.rps_bg_bridge_rear_phi_blend),
        "rps_bg_bridge_rear_score_floor": float(cfg.update.rps_bg_bridge_rear_score_floor),
        "rps_bg_bridge_rear_active_floor": float(cfg.update.rps_bg_bridge_rear_active_floor),
        "rps_bg_bridge_rear_age_floor": float(cfg.update.rps_bg_bridge_rear_age_floor),
        "rps_rear_hybrid_filter_enable": bool(cfg.update.rps_rear_hybrid_filter_enable),
        "rps_rear_hybrid_bridge_support_min": float(cfg.update.rps_rear_hybrid_bridge_support_min),
        "rps_rear_hybrid_dyn_max": float(cfg.update.rps_rear_hybrid_dyn_max),
        "rps_rear_hybrid_manifold_min": float(cfg.update.rps_rear_hybrid_manifold_min),
        "rps_rear_density_gate_enable": bool(cfg.update.rps_rear_density_gate_enable),
        "rps_rear_density_radius_cells": int(cfg.update.rps_rear_density_radius_cells),
        "rps_rear_density_min_neighbors": int(cfg.update.rps_rear_density_min_neighbors),
        "rps_rear_density_support_min": float(cfg.update.rps_rear_density_support_min),
        "rps_rear_selectivity_enable": bool(cfg.update.rps_rear_selectivity_enable),
        "rps_rear_selectivity_support_weight": float(cfg.update.rps_rear_selectivity_support_weight),
        "rps_rear_selectivity_history_weight": float(cfg.update.rps_rear_selectivity_history_weight),
        "rps_rear_selectivity_static_weight": float(cfg.update.rps_rear_selectivity_static_weight),
        "rps_rear_selectivity_geom_weight": float(cfg.update.rps_rear_selectivity_geom_weight),
        "rps_rear_selectivity_bridge_weight": float(cfg.update.rps_rear_selectivity_bridge_weight),
        "rps_rear_selectivity_density_weight": float(cfg.update.rps_rear_selectivity_density_weight),
        "rps_rear_selectivity_rear_score_weight": float(cfg.update.rps_rear_selectivity_rear_score_weight),
        "rps_rear_selectivity_front_score_weight": float(cfg.update.rps_rear_selectivity_front_score_weight),
        "rps_rear_selectivity_competition_weight": float(cfg.update.rps_rear_selectivity_competition_weight),
        "rps_rear_selectivity_competition_alpha": float(cfg.update.rps_rear_selectivity_competition_alpha),
        "rps_rear_selectivity_gap_weight": float(cfg.update.rps_rear_selectivity_gap_weight),
        "rps_rear_selectivity_sep_weight": float(cfg.update.rps_rear_selectivity_sep_weight),
        "rps_rear_selectivity_dyn_weight": float(cfg.update.rps_rear_selectivity_dyn_weight),
        "rps_rear_selectivity_ghost_weight": float(cfg.update.rps_rear_selectivity_ghost_weight),
        "rps_rear_selectivity_front_weight": float(cfg.update.rps_rear_selectivity_front_weight),
        "rps_rear_selectivity_geom_risk_weight": float(cfg.update.rps_rear_selectivity_geom_risk_weight),
        "rps_rear_selectivity_history_risk_weight": float(cfg.update.rps_rear_selectivity_history_risk_weight),
        "rps_rear_selectivity_density_risk_weight": float(cfg.update.rps_rear_selectivity_density_risk_weight),
        "rps_rear_selectivity_bridge_relief_weight": float(cfg.update.rps_rear_selectivity_bridge_relief_weight),
        "rps_rear_selectivity_static_relief_weight": float(cfg.update.rps_rear_selectivity_static_relief_weight),
        "rps_rear_selectivity_gap_risk_weight": float(cfg.update.rps_rear_selectivity_gap_risk_weight),
        "rps_rear_selectivity_score_min": float(cfg.update.rps_rear_selectivity_score_min),
        "rps_rear_selectivity_risk_max": float(cfg.update.rps_rear_selectivity_risk_max),
        "rps_rear_selectivity_geom_floor": float(cfg.update.rps_rear_selectivity_geom_floor),
        "rps_rear_selectivity_history_floor": float(cfg.update.rps_rear_selectivity_history_floor),
        "rps_rear_selectivity_bridge_floor": float(cfg.update.rps_rear_selectivity_bridge_floor),
        "rps_rear_selectivity_competition_floor": float(cfg.update.rps_rear_selectivity_competition_floor),
        "rps_rear_selectivity_front_score_max": float(cfg.update.rps_rear_selectivity_front_score_max),
        "rps_rear_selectivity_gap_min": float(cfg.update.rps_rear_selectivity_gap_min),
        "rps_rear_selectivity_gap_max": float(cfg.update.rps_rear_selectivity_gap_max),
        "rps_rear_selectivity_gap_valid_min": float(cfg.update.rps_rear_selectivity_gap_valid_min),
        "rps_rear_selectivity_occlusion_order_weight": float(cfg.update.rps_rear_selectivity_occlusion_order_weight),
        "rps_rear_selectivity_occlusion_order_floor": float(cfg.update.rps_rear_selectivity_occlusion_order_floor),
        "rps_rear_selectivity_occlusion_order_risk_weight": float(cfg.update.rps_rear_selectivity_occlusion_order_risk_weight),
        "rps_rear_selectivity_local_conflict_weight": float(cfg.update.rps_rear_selectivity_local_conflict_weight),
        "rps_rear_selectivity_local_conflict_max": float(cfg.update.rps_rear_selectivity_local_conflict_max),
        "rps_rear_selectivity_front_residual_weight": float(cfg.update.rps_rear_selectivity_front_residual_weight),
        "rps_rear_selectivity_front_residual_max": float(cfg.update.rps_rear_selectivity_front_residual_max),
        "rps_rear_selectivity_occluder_protect_weight": float(cfg.update.rps_rear_selectivity_occluder_protect_weight),
        "rps_rear_selectivity_occluder_protect_floor": float(cfg.update.rps_rear_selectivity_occluder_protect_floor),
        "rps_rear_selectivity_occluder_relief_weight": float(cfg.update.rps_rear_selectivity_occluder_relief_weight),
        "rps_rear_selectivity_dynamic_trail_weight": float(cfg.update.rps_rear_selectivity_dynamic_trail_weight),
        "rps_rear_selectivity_dynamic_trail_max": float(cfg.update.rps_rear_selectivity_dynamic_trail_max),
        "rps_rear_selectivity_dynamic_trail_relief_weight": float(cfg.update.rps_rear_selectivity_dynamic_trail_relief_weight),
        "rps_rear_selectivity_history_anchor_weight": float(cfg.update.rps_rear_selectivity_history_anchor_weight),
        "rps_rear_selectivity_history_anchor_floor": float(cfg.update.rps_rear_selectivity_history_anchor_floor),
        "rps_rear_selectivity_history_anchor_relief_weight": float(cfg.update.rps_rear_selectivity_history_anchor_relief_weight),
        "rps_rear_selectivity_surface_anchor_weight": float(cfg.update.rps_rear_selectivity_surface_anchor_weight),
        "rps_rear_selectivity_surface_anchor_floor": float(cfg.update.rps_rear_selectivity_surface_anchor_floor),
        "rps_rear_selectivity_surface_anchor_risk_weight": float(cfg.update.rps_rear_selectivity_surface_anchor_risk_weight),
        "rps_rear_selectivity_surface_distance_ref": float(cfg.update.rps_rear_selectivity_surface_distance_ref),
        "rps_rear_selectivity_dynamic_shell_weight": float(cfg.update.rps_rear_selectivity_dynamic_shell_weight),
        "rps_rear_selectivity_dynamic_shell_max": float(cfg.update.rps_rear_selectivity_dynamic_shell_max),
        "rps_rear_selectivity_dynamic_shell_gap_ref": float(cfg.update.rps_rear_selectivity_dynamic_shell_gap_ref),
        "rps_rear_selectivity_conflict_radius_cells": int(cfg.update.rps_rear_selectivity_conflict_radius_cells),
        "rps_rear_selectivity_conflict_front_score_min": float(cfg.update.rps_rear_selectivity_conflict_front_score_min),
        "rps_rear_selectivity_conflict_static_score_min": float(cfg.update.rps_rear_selectivity_conflict_static_score_min),
        "rps_rear_selectivity_conflict_dist_scale": float(cfg.update.rps_rear_selectivity_conflict_dist_scale),
        "rps_rear_selectivity_conflict_gap_ref": float(cfg.update.rps_rear_selectivity_conflict_gap_ref),
        "rps_rear_selectivity_conflict_ref": float(cfg.update.rps_rear_selectivity_conflict_ref),
        "rps_rear_selectivity_trail_radius_cells": int(cfg.update.rps_rear_selectivity_trail_radius_cells),
        "rps_rear_selectivity_trail_ref": float(cfg.update.rps_rear_selectivity_trail_ref),
        "rps_rear_selectivity_density_radius_cells": int(cfg.update.rps_rear_selectivity_density_radius_cells),
        "rps_rear_selectivity_density_ref": int(cfg.update.rps_rear_selectivity_density_ref),
        "rps_rear_selectivity_topk": int(cfg.update.rps_rear_selectivity_topk),
        "rps_rear_selectivity_rank_risk_weight": float(cfg.update.rps_rear_selectivity_rank_risk_weight),
        "rps_rear_selectivity_penetration_weight": float(cfg.update.rps_rear_selectivity_penetration_weight),
        "rps_rear_selectivity_penetration_floor": float(cfg.update.rps_rear_selectivity_penetration_floor),
        "rps_rear_selectivity_penetration_risk_weight": float(cfg.update.rps_rear_selectivity_penetration_risk_weight),
        "rps_rear_selectivity_penetration_free_ref": float(cfg.update.rps_rear_selectivity_penetration_free_ref),
        "rps_rear_selectivity_penetration_max_steps": int(cfg.update.rps_rear_selectivity_penetration_max_steps),
        "rps_rear_selectivity_observation_weight": float(cfg.update.rps_rear_selectivity_observation_weight),
        "rps_rear_selectivity_observation_floor": float(cfg.update.rps_rear_selectivity_observation_floor),
        "rps_rear_selectivity_observation_risk_weight": float(cfg.update.rps_rear_selectivity_observation_risk_weight),
        "rps_rear_selectivity_observation_count_ref": float(cfg.update.rps_rear_selectivity_observation_count_ref),
        "rps_rear_selectivity_observation_min_count": float(cfg.update.rps_rear_selectivity_observation_min_count),
        "rps_rear_selectivity_unobserved_veto_enable": bool(cfg.update.rps_rear_selectivity_unobserved_veto_enable),
        "rps_rear_selectivity_static_coherence_weight": float(cfg.update.rps_rear_selectivity_static_coherence_weight),
        "rps_rear_selectivity_static_coherence_floor": float(cfg.update.rps_rear_selectivity_static_coherence_floor),
        "rps_rear_selectivity_static_coherence_relief_weight": float(cfg.update.rps_rear_selectivity_static_coherence_relief_weight),
        "rps_rear_selectivity_static_coherence_radius_cells": int(cfg.update.rps_rear_selectivity_static_coherence_radius_cells),
        "rps_rear_selectivity_static_coherence_ref": float(cfg.update.rps_rear_selectivity_static_coherence_ref),
        "rps_rear_selectivity_static_neighbor_min_weight": float(cfg.update.rps_rear_selectivity_static_neighbor_min_weight),
        "rps_rear_selectivity_static_neighbor_dyn_max": float(cfg.update.rps_rear_selectivity_static_neighbor_dyn_max),
        "rps_rear_selectivity_thickness_weight": float(cfg.update.rps_rear_selectivity_thickness_weight),
        "rps_rear_selectivity_thickness_floor": float(cfg.update.rps_rear_selectivity_thickness_floor),
        "rps_rear_selectivity_thickness_risk_weight": float(cfg.update.rps_rear_selectivity_thickness_risk_weight),
        "rps_rear_selectivity_thickness_ref": float(cfg.update.rps_rear_selectivity_thickness_ref),
        "rps_rear_selectivity_normal_consistency_weight": float(cfg.update.rps_rear_selectivity_normal_consistency_weight),
        "rps_rear_selectivity_normal_consistency_floor": float(cfg.update.rps_rear_selectivity_normal_consistency_floor),
        "rps_rear_selectivity_normal_consistency_relief_weight": float(cfg.update.rps_rear_selectivity_normal_consistency_relief_weight),
        "rps_rear_selectivity_normal_consistency_radius_cells": int(cfg.update.rps_rear_selectivity_normal_consistency_radius_cells),
        "rps_rear_selectivity_normal_consistency_dyn_max": float(cfg.update.rps_rear_selectivity_normal_consistency_dyn_max),
        "rps_rear_selectivity_ray_convergence_weight": float(cfg.update.rps_rear_selectivity_ray_convergence_weight),
        "rps_rear_selectivity_ray_convergence_floor": float(cfg.update.rps_rear_selectivity_ray_convergence_floor),
        "rps_rear_selectivity_ray_convergence_relief_weight": float(cfg.update.rps_rear_selectivity_ray_convergence_relief_weight),
        "rps_rear_selectivity_ray_convergence_radius_cells": int(cfg.update.rps_rear_selectivity_ray_convergence_radius_cells),
        "rps_rear_selectivity_ray_convergence_gap_ref": float(cfg.update.rps_rear_selectivity_ray_convergence_gap_ref),
        "rps_rear_selectivity_ray_convergence_normal_cos": float(cfg.update.rps_rear_selectivity_ray_convergence_normal_cos),
        "rps_rear_selectivity_ray_convergence_thickness_ref": float(cfg.update.rps_rear_selectivity_ray_convergence_thickness_ref),
        "rps_rear_selectivity_ray_convergence_ref": float(cfg.update.rps_rear_selectivity_ray_convergence_ref),
        "rps_rear_state_protect_enable": bool(cfg.update.rps_rear_state_protect_enable),
        "rps_rear_state_decay_relax": float(cfg.update.rps_rear_state_decay_relax),
        "rps_rear_state_active_floor": float(cfg.update.rps_rear_state_active_floor),
        "rps_commit_front_penalty": float(cfg.update.rps_commit_front_penalty),
        "rps_commit_quality_enable": bool(cfg.update.rps_commit_quality_enable),
        "rps_commit_quality_transfer_gain": float(cfg.update.rps_commit_quality_transfer_gain),
        "rps_commit_quality_rho_gain": float(cfg.update.rps_commit_quality_rho_gain),
        "rps_commit_quality_geom_blend": float(cfg.update.rps_commit_quality_geom_blend),
        "rps_commit_quality_sep_scale": float(cfg.update.rps_commit_quality_sep_scale),
        "joint_bg_state_enable": bool(cfg.update.joint_bg_state_enable),
        "joint_bg_state_on": float(cfg.update.joint_bg_state_on),
        "joint_bg_state_gain": float(cfg.update.joint_bg_state_gain),
        "joint_bg_state_rho_gain": float(cfg.update.joint_bg_state_rho_gain),
        "joint_bg_state_front_penalty": float(cfg.update.joint_bg_state_front_penalty),
        "wdsg_enable": bool(cfg.update.wdsg_enable),
        "wdsg_synth_mode": str(cfg.update.wdsg_synth_mode),
        "wdsg_synth_anchor_gain": float(cfg.update.wdsg_synth_anchor_gain),
        "wdsg_synth_geo_gain": float(cfg.update.wdsg_synth_geo_gain),
        "wdsg_synth_bg_gain": float(cfg.update.wdsg_synth_bg_gain),
        "wdsg_synth_counterfactual_gain": float(cfg.update.wdsg_synth_counterfactual_gain),
        "wdsg_synth_front_repel_gain": float(cfg.update.wdsg_synth_front_repel_gain),
        "wdsg_synth_energy_temp": float(cfg.update.wdsg_synth_energy_temp),
        "wdsg_synth_clip_vox": float(cfg.update.wdsg_synth_clip_vox),
        "wdsg_conservative_enable": bool(cfg.update.wdsg_conservative_enable),
        "wdsg_conservative_ref_vox": float(cfg.update.wdsg_conservative_ref_vox),
        "wdsg_conservative_min_clip_scale": float(cfg.update.wdsg_conservative_min_clip_scale),
        "wdsg_conservative_static_gain": float(cfg.update.wdsg_conservative_static_gain),
        "wdsg_conservative_rear_gain": float(cfg.update.wdsg_conservative_rear_gain),
        "wdsg_conservative_geo_gain": float(cfg.update.wdsg_conservative_geo_gain),
        "wdsg_conservative_front_penalty": float(cfg.update.wdsg_conservative_front_penalty),
        "wdsg_local_clip_enable": bool(cfg.update.wdsg_local_clip_enable),
        "wdsg_local_clip_min_scale": float(cfg.update.wdsg_local_clip_min_scale),
        "wdsg_local_clip_max_scale": float(cfg.update.wdsg_local_clip_max_scale),
        "wdsg_local_clip_risk_gain": float(cfg.update.wdsg_local_clip_risk_gain),
        "wdsg_local_clip_expand_gain": float(cfg.update.wdsg_local_clip_expand_gain),
        "wdsg_local_clip_front_gate": float(cfg.update.wdsg_local_clip_front_gate),
        "wdsg_local_clip_support_gate": float(cfg.update.wdsg_local_clip_support_gate),
        "wdsg_local_clip_ambiguity_gate": float(cfg.update.wdsg_local_clip_ambiguity_gate),
        "wdsg_local_clip_pfv_gain": float(cfg.update.wdsg_local_clip_pfv_gain),
        "wdsg_local_clip_bg_gain": float(cfg.update.wdsg_local_clip_bg_gain),
        "wdsg_route_enable": bool(cfg.update.wdsg_route_enable),
        "spg_enable": bool(cfg.update.spg_enable),
        "otv_enable": bool(cfg.update.otv_enable),
        "xmem_enable": bool(cfg.update.xmem_enable),
        "obl_enable": bool(cfg.update.obl_enable),
        "obl_sep_ref_vox": float(cfg.update.obl_sep_ref_vox),
        "obl_rear_gain": float(cfg.update.obl_rear_gain),
        "obl_static_gain": float(cfg.update.obl_static_gain),
        "obl_commit_on": float(cfg.update.obl_commit_on),
        "obl_commit_off": float(cfg.update.obl_commit_off),
        "obl_static_veto": float(cfg.update.obl_static_veto),
        "obl_geo_veto": float(cfg.update.obl_geo_veto),
        "obl_extract_gain": float(cfg.update.obl_extract_gain),
        "obl_dyn_static_guard": float(cfg.update.obl_dyn_static_guard),
        "dual_map_enable": bool(cfg.update.dual_map_enable),
        "dual_map_bg_front_veto": float(cfg.update.dual_map_bg_front_veto),
        "dual_map_bg_rear_gain": float(cfg.update.dual_map_bg_rear_gain),
        "dual_map_bg_static_floor": float(cfg.update.dual_map_bg_static_floor),
        "dual_map_fg_front_boost": float(cfg.update.dual_map_fg_front_boost),
        "dual_map_fg_static_leak": float(cfg.update.dual_map_fg_static_leak),
        "dual_map_fg_dynamic_score_bias": float(cfg.update.dual_map_fg_dynamic_score_bias),
        "cmct_enable": bool(cfg.update.cmct_enable),
        "cmct_alpha": float(cfg.update.cmct_alpha),
        "cmct_commit_on": float(cfg.update.cmct_commit_on),
        "cmct_commit_off": float(cfg.update.cmct_commit_off),
        "cmct_bg_decay": float(cfg.update.cmct_bg_decay),
        "cmct_geo_decay": float(cfg.update.cmct_geo_decay),
        "cmct_rho_decay": float(cfg.update.cmct_rho_decay),
        "cmct_static_guard": float(cfg.update.cmct_static_guard),
        "cmct_bg_rho_protect": float(cfg.update.cmct_bg_rho_protect),
        "cmct_radius_cells": int(cfg.update.cmct_radius_cells),
        "cgcc_enable": bool(cfg.update.cgcc_enable),
        "cgcc_conf_on": float(cfg.update.cgcc_conf_on),
        "cgcc_conf_off": float(cfg.update.cgcc_conf_off),
        "cgcc_front_margin_vox": float(cfg.update.cgcc_front_margin_vox),
        "cgcc_rear_margin_vox": float(cfg.update.cgcc_rear_margin_vox),
        "cgcc_step_scale": float(cfg.update.cgcc_step_scale),
        "cgcc_lateral_radius_cells": int(cfg.update.cgcc_lateral_radius_cells),
        "cgcc_bg_decay": float(cfg.update.cgcc_bg_decay),
        "cgcc_geo_decay": float(cfg.update.cgcc_geo_decay),
        "cgcc_rho_decay": float(cfg.update.cgcc_rho_decay),
        "cgcc_bg_layer_decay": float(cfg.update.cgcc_bg_layer_decay),
        "cgcc_static_guard": float(cfg.update.cgcc_static_guard),
        "cgcc_fg_weight_floor": float(cfg.update.cgcc_fg_weight_floor),
        "pfv_enable": bool(cfg.update.pfv_enable),
        "pfv_exclusive_enable": bool(cfg.update.pfv_exclusive_enable),
        "pfv_commit_delay_enable": bool(cfg.update.pfv_commit_delay_enable),
        "pfv_bg_candidate_enable": bool(cfg.update.pfv_bg_candidate_enable),
        "tri_map_enable": bool(cfg.update.tri_map_enable),
        "tri_map_promotion_rescue_enable": bool(cfg.update.tri_map_promotion_rescue_enable),
        "tri_map_hole_rescue_enable": bool(cfg.update.tri_map_hole_rescue_enable),
        "pfv_alpha": float(cfg.update.pfv_alpha),
        "pfv_commit_on": float(cfg.update.pfv_commit_on),
        "pfv_commit_off": float(cfg.update.pfv_commit_off),
        "pfv_step_scale": float(cfg.update.pfv_step_scale),
        "pfv_end_margin": float(cfg.update.pfv_end_margin),
        "pfv_bg_support_ref": float(cfg.update.pfv_bg_support_ref),
        "pfv_fg_guard": float(cfg.update.pfv_fg_guard),
        "pfv_static_guard": float(cfg.update.pfv_static_guard),
        "pfv_extract_thresh": float(cfg.update.pfv_extract_thresh),
        "pfv_bg_decay": float(cfg.update.pfv_bg_decay),
        "pfv_geo_decay": float(cfg.update.pfv_geo_decay),
        "pfv_rho_decay": float(cfg.update.pfv_rho_decay),
        "pfvp_enable": bool(cfg.update.pfvp_enable),
        "pfvp_margin": float(cfg.update.pfvp_margin),
        "pfvp_fg_on": float(cfg.update.pfvp_fg_on),
        "pfvp_bg_on": float(cfg.update.pfvp_bg_on),
        "pfvp_bg_keep_floor": float(cfg.update.pfvp_bg_keep_floor),
        "pfvp_pfv_weight": float(cfg.update.pfvp_pfv_weight),
        "pfvp_fg_hist_weight": float(cfg.update.pfvp_fg_hist_weight),
        "pfvp_assoc_weight": float(cfg.update.pfvp_assoc_weight),
        "pfvp_static_weight": float(cfg.update.pfvp_static_weight),
        "pfvp_bg_rho_weight": float(cfg.update.pfvp_bg_rho_weight),
        "pfvp_bg_obl_weight": float(cfg.update.pfvp_bg_obl_weight),
        "xmem_sep_ref_vox": float(cfg.update.xmem_sep_ref_vox),
        "xmem_occ_alpha": float(cfg.update.xmem_occ_alpha),
        "xmem_free_alpha": float(cfg.update.xmem_free_alpha),
        "xmem_score_alpha": float(cfg.update.xmem_score_alpha),
        "xmem_support_ref": float(cfg.update.xmem_support_ref),
        "xmem_commit_on": float(cfg.update.xmem_commit_on),
        "xmem_commit_off": float(cfg.update.xmem_commit_off),
        "xmem_age_ref": float(cfg.update.xmem_age_ref),
        "xmem_static_guard": float(cfg.update.xmem_static_guard),
        "xmem_free_gain": float(cfg.update.xmem_free_gain),
        "xmem_static_veto": float(cfg.update.xmem_static_veto),
        "xmem_geo_veto": float(cfg.update.xmem_geo_veto),
        "xmem_transient_boost": float(cfg.update.xmem_transient_boost),
        "xmem_dyn_boost": float(cfg.update.xmem_dyn_boost),
        "xmem_decay": float(cfg.update.xmem_decay),
        "xmem_clear_alpha": float(cfg.update.xmem_clear_alpha),
        "xmem_clear_on": float(cfg.update.xmem_clear_on),
        "xmem_clear_off": float(cfg.update.xmem_clear_off),
        "xmem_clear_static_release": float(cfg.update.xmem_clear_static_release),
        "xmem_clear_weight_decay": float(cfg.update.xmem_clear_weight_decay),
        "xmem_raycast_gain": float(cfg.update.xmem_raycast_gain),
        "xmem_raycast_gate": float(cfg.update.xmem_raycast_gate),
        "xmem_raycast_static_decay": float(cfg.update.xmem_raycast_static_decay),
        "surface_dual_p_static_min": float(cfg.surface.dual_p_static_min),
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
        "dual_state_enable": bool(cfg.update.dual_state_enable),
        "dual_state_assoc_weight": float(cfg.update.dual_state_assoc_weight),
        "dual_state_free_weight": float(cfg.update.dual_state_free_weight),
        "dual_state_residual_weight": float(cfg.update.dual_state_residual_weight),
        "dual_state_osc_weight": float(cfg.update.dual_state_osc_weight),
        "dual_state_pose_weight": float(cfg.update.dual_state_pose_weight),
        "dual_state_bias": float(cfg.update.dual_state_bias),
        "dual_state_temp": float(cfg.update.dual_state_temp),
        "dual_pose_var_ref": float(cfg.update.dual_pose_var_ref),
        "dual_state_static_ema": float(cfg.update.dual_state_static_ema),
        "dual_state_min_static_ratio": float(cfg.update.dual_state_min_static_ratio),
        "dual_state_commit_thresh": float(cfg.update.dual_state_commit_thresh),
        "dual_state_rollback_thresh": float(cfg.update.dual_state_rollback_thresh),
        "dual_state_commit_gain": float(cfg.update.dual_state_commit_gain),
        "dual_state_rollback_gain": float(cfg.update.dual_state_rollback_gain),
        "dual_state_static_protect_rho": float(cfg.update.dual_state_static_protect_rho),
        "dual_state_static_protect_ratio": float(cfg.update.dual_state_static_protect_ratio),
        "dual_state_static_decay_mult": float(cfg.update.dual_state_static_decay_mult),
        "dual_state_transient_decay_mult": float(cfg.update.dual_state_transient_decay_mult),
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
        "lzcd_solver_iters": int(cfg.update.lzcd_solver_iters),
        "lzcd_solver_lambda_smooth": float(cfg.update.lzcd_solver_lambda_smooth),
        "lzcd_solver_step": float(cfg.update.lzcd_solver_step),
        "lzcd_solver_tol": float(cfg.update.lzcd_solver_tol),
        "lzcd_residual_anchor_weight": float(cfg.update.lzcd_residual_anchor_weight),
        "lzcd_residual_alpha": float(cfg.update.lzcd_residual_alpha),
        "lzcd_residual_hit_ref": float(cfg.update.lzcd_residual_hit_ref),
        "lzcd_residual_max_abs": float(cfg.update.lzcd_residual_max_abs),
        "lzcd_max_candidates": int(cfg.update.lzcd_max_candidates),
        "lzcd_use_geo_channel": bool(cfg.update.lzcd_use_geo_channel),
        "stcg_enable": bool(cfg.update.stcg_enable),
        "stcg_alpha": float(cfg.update.stcg_alpha),
        "stcg_conflict_weight": float(cfg.update.stcg_conflict_weight),
        "stcg_residual_weight": float(cfg.update.stcg_residual_weight),
        "stcg_osc_weight": float(cfg.update.stcg_osc_weight),
        "stcg_free_ratio_ref": float(cfg.update.stcg_free_ratio_ref),
        "stcg_on_thresh": float(cfg.update.stcg_on_thresh),
        "stcg_off_thresh": float(cfg.update.stcg_off_thresh),
        "zdyn_enable": bool(cfg.update.zdyn_enable),
        "zdyn_alpha_up": float(cfg.update.zdyn_alpha_up),
        "zdyn_alpha_down": float(cfg.update.zdyn_alpha_down),
        "zdyn_decay": float(cfg.update.zdyn_decay),
        "zdyn_conflict_weight": float(cfg.update.zdyn_conflict_weight),
        "zdyn_visibility_weight": float(cfg.update.zdyn_visibility_weight),
        "zdyn_residual_weight": float(cfg.update.zdyn_residual_weight),
        "zdyn_osc_weight": float(cfg.update.zdyn_osc_weight),
        "zdyn_free_ratio_weight": float(cfg.update.zdyn_free_ratio_weight),
        "zdyn_free_ratio_ref": float(cfg.update.zdyn_free_ratio_ref),
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
        "rear_surface_points": int(np.asarray(extract_bank_points.get('rear_points', np.zeros((0, 3), dtype=float))).shape[0]) if extract_bank_points else 0,
        "front_surface_points": int(np.asarray(extract_bank_points.get('front_points', np.zeros((0, 3), dtype=float))).shape[0]) if extract_bank_points else 0,
        "background_surface_points": int(np.asarray(extract_bank_points.get('background_points', np.zeros((0, 3), dtype=float))).shape[0]) if extract_bank_points else 0,
        "reference_points": int(gt_points.shape[0]),
        "ptdsf_diag": ptdsf_diag,
        "ptdsf_export_diag": ptdsf_export_diag,
        "rps_commit_diag": rps_commit_diag,
        "rps_competition_diag": rps_competition_diag,
        "rps_admission_diag": rps_admission_diag,
        "bg_manifold_diag": bg_manifold_diag,
        "rear_bg_state_diag": state_diag,
        "assoc_ratio_mean": float(np.mean(assoc_ratios)) if assoc_ratios else 0.0,
        "touched_voxels_mean": float(np.mean(touched_voxels)) if touched_voxels else 0.0,
        "dynamic_score_mean": float(np.mean(dyn_scores)) if dyn_scores else 0.0,
        "odom_fitness_mean": float(np.mean(odom_fitness)) if odom_fitness else 0.0,
        "odom_rmse_mean": float(np.mean(odom_rmse)) if odom_rmse else 0.0,
        "odom_valid_ratio": float(np.mean(odom_valid)) if odom_valid else 0.0,
        "trimap_delayed_mean": float(np.mean(trimap_delayed)) if trimap_delayed else 0.0,
        "trimap_dual_mean": float(np.mean(trimap_dual)) if trimap_dual else 0.0,
        "trimap_promoted_mean": float(np.mean(trimap_promoted)) if trimap_promoted else 0.0,
        "trimap_hybrid_mean": float(np.mean(trimap_hybrid)) if trimap_hybrid else 0.0,
        "trimap_support_gap_mean": float(np.mean(trimap_support_gap)) if trimap_support_gap else 0.0,
        "trimap_gap_score_mean": float(np.mean(trimap_gap_score)) if trimap_gap_score else 0.0,
        "trimap_bg_support_mean": float(np.mean(trimap_bg_support)) if trimap_bg_support else 0.0,
        "trimap_centered_gap_mean": float(np.mean(trimap_centered_gap)) if trimap_centered_gap else 0.0,
        "trimap_norm_gap_mean": float(np.mean(trimap_norm_gap)) if trimap_norm_gap else 0.0,
        "trimap_gap_bias_mean": float(np.mean(trimap_gap_bias)) if trimap_gap_bias else 0.0,
        "trimap_front_occ_mean": float(np.mean(trimap_front_occ)) if trimap_front_occ else 0.0,
        "trimap_delay_score_mean": float(np.mean(trimap_delay_score)) if trimap_delay_score else 0.0,
        "trimap_quantile_strong_thresh_mean": float(np.mean(trimap_q_strong_thresh)) if trimap_q_strong_thresh else 0.0,
        "trimap_quantile_soft_thresh_mean": float(np.mean(trimap_q_soft_thresh)) if trimap_q_soft_thresh else 0.0,
        "trimap_quantile_strong_budget_mean": float(np.mean(trimap_q_strong_budget)) if trimap_q_strong_budget else 0.0,
        "trimap_quantile_soft_budget_mean": float(np.mean(trimap_q_soft_budget)) if trimap_q_soft_budget else 0.0,
        "trimap_hold_blocked_mean": float(np.mean(trimap_hold_blocked)) if trimap_hold_blocked else 0.0,
        "trimap_hold_mean": float(np.mean(trimap_hold_mean)) if trimap_hold_mean else 0.0,
        "trimap_hysteresis_mean": float(np.mean(trimap_hysteresis_mean)) if trimap_hysteresis_mean else 0.0,
        "trimap_export_added": float(extract_stats.get('trimap_export_added', 0.0)),
        "trimap_export_replaced": float(extract_stats.get('trimap_export_replaced', 0.0)),
        "trimap_export_candidates": float(extract_stats.get('trimap_export_candidates', 0.0)),
        "trimap_export_residency": float(extract_stats.get('trimap_export_residency', 0.0)),
        "trimap_export_route_mean": float(extract_stats.get('trimap_export_route_mean', 0.0)),
        "trimap_export_compete_mean": float(extract_stats.get('trimap_export_compete_mean', 0.0)),
        "trimap_export_normal_cos_mean": float(extract_stats.get('trimap_export_normal_cos_mean', 0.0)),
        "trimap_delayed_refine_offset_mean": float(extract_stats.get('trimap_delayed_refine_offset_mean', 0.0)),
        "trimap_delayed_refine_normal_cos_mean": float(extract_stats.get('trimap_delayed_refine_normal_cos_mean', 1.0)),
        "trimap_delayed_bank_points": float(extract_stats.get('trimap_delayed_bank_points', 0.0)),
        "trimap_delayed_bank_conf_mean": float(extract_stats.get('trimap_delayed_bank_conf_mean', 0.0)),
        "trimap_delayed_bank_residency_mean": float(extract_stats.get('trimap_delayed_bank_residency_mean', 0.0)),
        "rear_selectivity_pre_count": float(extract_stats.get('rear_selectivity_pre_count', 0.0)),
        "rear_selectivity_kept_count": float(extract_stats.get('rear_selectivity_kept_count', 0.0)),
        "rear_selectivity_drop_count": float(extract_stats.get('rear_selectivity_drop_count', 0.0)),
        "rear_selectivity_topk_drop_count": float(extract_stats.get('rear_selectivity_topk_drop_count', 0.0)),
        "rear_selectivity_score_sum": float(extract_stats.get('rear_selectivity_score_sum', 0.0)),
        "rear_selectivity_risk_sum": float(extract_stats.get('rear_selectivity_risk_sum', 0.0)),
        "rear_selectivity_pre_front_score_sum": float(extract_stats.get('rear_selectivity_pre_front_score_sum', 0.0)),
        "rear_selectivity_pre_front_residual_sum": float(extract_stats.get('rear_selectivity_pre_front_residual_sum', 0.0)),
        "rear_selectivity_pre_occlusion_order_sum": float(extract_stats.get('rear_selectivity_pre_occlusion_order_sum', 0.0)),
        "rear_selectivity_pre_local_conflict_sum": float(extract_stats.get('rear_selectivity_pre_local_conflict_sum', 0.0)),
        "rear_selectivity_pre_dynamic_trail_sum": float(extract_stats.get('rear_selectivity_pre_dynamic_trail_sum', 0.0)),
        "rear_selectivity_pre_dyn_risk_sum": float(extract_stats.get('rear_selectivity_pre_dyn_risk_sum', 0.0)),
        "rear_selectivity_pre_history_anchor_sum": float(extract_stats.get('rear_selectivity_pre_history_anchor_sum', 0.0)),
        "rear_selectivity_pre_surface_anchor_sum": float(extract_stats.get('rear_selectivity_pre_surface_anchor_sum', 0.0)),
        "rear_selectivity_pre_surface_distance_sum": float(extract_stats.get('rear_selectivity_pre_surface_distance_sum', 0.0)),
        "rear_selectivity_pre_dynamic_shell_sum": float(extract_stats.get('rear_selectivity_pre_dynamic_shell_sum', 0.0)),
        "rear_selectivity_front_score_sum": float(extract_stats.get('rear_selectivity_front_score_sum', 0.0)),
        "rear_selectivity_rear_score_sum": float(extract_stats.get('rear_selectivity_rear_score_sum', 0.0)),
        "rear_selectivity_gap_sum": float(extract_stats.get('rear_selectivity_gap_sum', 0.0)),
        "rear_selectivity_competition_sum": float(extract_stats.get('rear_selectivity_competition_sum', 0.0)),
        "rear_selectivity_occlusion_order_sum": float(extract_stats.get('rear_selectivity_occlusion_order_sum', 0.0)),
        "rear_selectivity_occluder_protect_sum": float(extract_stats.get('rear_selectivity_occluder_protect_sum', 0.0)),
        "rear_selectivity_local_conflict_sum": float(extract_stats.get('rear_selectivity_local_conflict_sum', 0.0)),
        "rear_selectivity_front_residual_sum": float(extract_stats.get('rear_selectivity_front_residual_sum', 0.0)),
        "rear_selectivity_dynamic_trail_sum": float(extract_stats.get('rear_selectivity_dynamic_trail_sum', 0.0)),
        "rear_selectivity_dyn_risk_sum": float(extract_stats.get('rear_selectivity_dyn_risk_sum', 0.0)),
        "rear_selectivity_history_anchor_sum": float(extract_stats.get('rear_selectivity_history_anchor_sum', 0.0)),
        "rear_selectivity_surface_anchor_sum": float(extract_stats.get('rear_selectivity_surface_anchor_sum', 0.0)),
        "rear_selectivity_surface_distance_sum": float(extract_stats.get('rear_selectivity_surface_distance_sum', 0.0)),
        "rear_selectivity_dynamic_shell_sum": float(extract_stats.get('rear_selectivity_dynamic_shell_sum', 0.0)),
        "rear_selectivity_pre_penetration_sum": float(extract_stats.get('rear_selectivity_pre_penetration_sum', 0.0)),
        "rear_selectivity_pre_penetration_free_span_sum": float(extract_stats.get('rear_selectivity_pre_penetration_free_span_sum', 0.0)),
        "rear_selectivity_pre_observation_count_sum": float(extract_stats.get('rear_selectivity_pre_observation_count_sum', 0.0)),
        "rear_selectivity_pre_observation_support_sum": float(extract_stats.get('rear_selectivity_pre_observation_support_sum', 0.0)),
        "rear_selectivity_pre_static_coherence_sum": float(extract_stats.get('rear_selectivity_pre_static_coherence_sum', 0.0)),
        "rear_selectivity_pre_topology_thickness_sum": float(extract_stats.get('rear_selectivity_pre_topology_thickness_sum', 0.0)),
        "rear_selectivity_pre_normal_consistency_sum": float(extract_stats.get('rear_selectivity_pre_normal_consistency_sum', 0.0)),
        "rear_selectivity_pre_ray_convergence_sum": float(extract_stats.get('rear_selectivity_pre_ray_convergence_sum', 0.0)),
        "rear_selectivity_penetration_sum": float(extract_stats.get('rear_selectivity_penetration_sum', 0.0)),
        "rear_selectivity_penetration_free_span_sum": float(extract_stats.get('rear_selectivity_penetration_free_span_sum', 0.0)),
        "rear_selectivity_observation_count_sum": float(extract_stats.get('rear_selectivity_observation_count_sum', 0.0)),
        "rear_selectivity_observation_support_sum": float(extract_stats.get('rear_selectivity_observation_support_sum', 0.0)),
        "rear_selectivity_static_coherence_sum": float(extract_stats.get('rear_selectivity_static_coherence_sum', 0.0)),
        "rear_selectivity_topology_thickness_sum": float(extract_stats.get('rear_selectivity_topology_thickness_sum', 0.0)),
        "rear_selectivity_normal_consistency_sum": float(extract_stats.get('rear_selectivity_normal_consistency_sum', 0.0)),
        "rear_selectivity_ray_convergence_sum": float(extract_stats.get('rear_selectivity_ray_convergence_sum', 0.0)),
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
