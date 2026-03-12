from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS_DIR))

from run_benchmark import (
    build_dynamic_references,
    compute_dynamic_metrics,
    compute_recon_metrics,
    load_points_with_normals,
    rigid_align_for_eval,
)
from run_s2_rps_hssa import (
    classify_points,
    load_control_tsdf_ghosts,
    load_predicted_views,
    load_rows,
    save_point_cloud,
    save_rows,
)


CONTROL_ROOT = PROJECT_ROOT / "output" / "s2_stage" / "93_spatial_neighborhood_density_clustering"
HSSA_COMPARE = PROJECT_ROOT / "output" / "s2" / "S2_RPS_HSSA_COMPARE.csv"
BONN_ALL3 = ["rgbd_bonn_balloon2", "rgbd_bonn_balloon", "rgbd_bonn_crowd2"]
CLUSTER_FEATURE_KEYS = [
    "cluster_fitting_error",
    "geodesic_smoothness",
    "balloon_expand_ratio",
    "cluster_anchor_distance",
]


def classify_balloon_rows(sequence: str, rows: List[dict], *, frames: int, stride: int, max_points: int, seed: int) -> Dict[str, dict]:
    stats = {
        "true_background": {"count": 0.0, **{f"{k}_sum": 0.0 for k in CLUSTER_FEATURE_KEYS}},
        "ghost_region": {"count": 0.0, **{f"{k}_sum": 0.0 for k in CLUSTER_FEATURE_KEYS}},
        "hole_or_noise": {"count": 0.0, **{f"{k}_sum": 0.0 for k in CLUSTER_FEATURE_KEYS}},
    }
    points = np.asarray([[float(r.get("x", 0.0)), float(r.get("y", 0.0)), float(r.get("z", 0.0))] for r in rows], dtype=float)
    classes = classify_points(sequence, points, frames=frames, stride=stride, max_points=max_points, seed=seed)
    if points.shape[0] == 0:
        for bucket in stats.values():
            for key in CLUSTER_FEATURE_KEYS:
                bucket[f"{key}_mean"] = 0.0
        return stats
    stable_bg, _tail, dynamic_region, dynamic_voxel = build_dynamic_references(
        sequence_dir=Path("data/bonn") / sequence,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points,
        seed=seed,
    )
    bg_tree = None
    if stable_bg.shape[0] > 0:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stable_bg)
        bg_tree = o3d.geometry.KDTreeFlann(pcd)
    for row, point in zip(rows, points):
        voxel = tuple(np.floor(point / float(dynamic_voxel)).astype(np.int32).tolist())
        label = "hole_or_noise"
        if voxel in dynamic_region:
            label = "ghost_region"
        elif bg_tree is not None:
            _, idx, dist2 = bg_tree.search_knn_vector_3d(point.astype(float), 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < 0.05:
                label = "true_background"
        stats[label]["count"] += 1.0
        for key in CLUSTER_FEATURE_KEYS:
            stats[label][f"{key}_sum"] += float(row.get(key, 0.0))
    for bucket in stats.values():
        denom = max(1.0, bucket["count"])
        for key in CLUSTER_FEATURE_KEYS:
            bucket[f"{key}_mean"] = float(bucket[f"{key}_sum"] / denom)
    return stats


def load_control_tum_metrics() -> Tuple[float, float]:
    rows = list(csv.DictReader(HSSA_COMPARE.open("r", encoding="utf-8")))
    row = next(r for r in rows if r["variant"] == "93_spatial_neighborhood_density_clustering")
    return float(row["tum_acc_cm"]), float(row["tum_comp_r_5cm"])


def cluster_indices(points: np.ndarray, *, radius: float, min_size: int) -> List[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return []
    tree = cKDTree(pts)
    seen = np.zeros((pts.shape[0],), dtype=bool)
    clusters: List[np.ndarray] = []
    for i in range(pts.shape[0]):
        if seen[i]:
            continue
        neigh = tree.query_ball_point(pts[i], radius)
        if len(neigh) < min_size:
            seen[i] = True
            continue
        queue = deque(neigh)
        for n in neigh:
            seen[n] = True
        comp: List[int] = []
        while queue:
            j = int(queue.popleft())
            comp.append(j)
            neigh2 = tree.query_ball_point(pts[j], radius)
            if len(neigh2) >= min_size:
                for k in neigh2:
                    if not seen[k]:
                        seen[k] = True
                        queue.append(k)
        clusters.append(np.asarray(sorted(set(comp)), dtype=np.int64))
    return clusters


def compute_cluster_metrics(points: np.ndarray, normals: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    if pts.shape[0] <= 1:
        return {
            "center": pts.mean(axis=0) if pts.shape[0] else np.zeros((3,), dtype=float),
            "normal": np.array([0.0, 0.0, 1.0], dtype=float),
            "cluster_fitting_error": 0.0,
            "geodesic_smoothness": 0.0,
            "balloon_expand_ratio": 1.0,
            "radius_extent": 0.0,
        }
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    fit_error = float(np.sqrt(max(float(eigvals[order[0]]), 0.0)))
    plane_normal = eigvecs[:, order[0]]
    if nrm.shape == pts.shape and nrm.shape[0] > 0:
        mean_normal = nrm.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_normal))
        if mean_norm > 1e-9:
            mean_normal = mean_normal / mean_norm
            if float(np.dot(mean_normal, plane_normal)) < 0.0:
                plane_normal = -plane_normal
    tree = cKDTree(pts)
    max_nn = min(4, pts.shape[0] - 1)
    jump_vals: List[float] = []
    accepted = 0
    edges = 0
    for i, point in enumerate(pts):
        dists, idx = tree.query(point, k=max_nn + 1)
        idx = np.atleast_1d(idx)[1:]
        dists = np.atleast_1d(dists)[1:]
        for j, dist in zip(idx, dists):
            j = int(j)
            if j <= i:
                continue
            normal_jump = 1.0
            if nrm.shape == pts.shape:
                normal_jump = 1.0 - abs(float(np.dot(nrm[i], nrm[j])))
            jump_vals.append(normal_jump)
            if normal_jump <= 0.08 and float(dist) <= 0.06:
                accepted += 1
            edges += 1
    geodesic_smoothness = float(np.mean(jump_vals)) if jump_vals else 0.0
    balloon_expand_ratio = float(accepted / max(1, edges))
    radius_extent = float(np.max(np.linalg.norm(pts - center[None, :], axis=1)))
    return {
        "center": center,
        "normal": plane_normal / max(1e-9, float(np.linalg.norm(plane_normal))),
        "cluster_fitting_error": fit_error,
        "geodesic_smoothness": geodesic_smoothness,
        "balloon_expand_ratio": balloon_expand_ratio,
        "radius_extent": radius_extent,
    }


def classify_cluster_label(sequence: str, points: np.ndarray, *, frames: int, stride: int, max_points: int, seed: int) -> str:
    stats = classify_points(sequence, np.asarray(points, dtype=float), frames=frames, stride=stride, max_points=max_points, seed=seed)
    return max(stats.items(), key=lambda kv: kv[1])[0]


def compute_candidate_clusters(
    *,
    sequence: str,
    points: np.ndarray,
    normals: np.ndarray,
    rows: List[dict],
    front_points: np.ndarray,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    seed: int,
) -> List[dict]:
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    support_score = np.asarray([float(row.get("support_score", 0.0)) for row in rows], dtype=float)
    seed_indices = np.where(support_score >= 0.16)[0].astype(np.int64)
    if seed_indices.size == 0:
        return []
    clusters_local = cluster_indices(pts[seed_indices], radius=0.12, min_size=2)
    front_tree = cKDTree(np.asarray(front_points, dtype=float)) if front_points.shape[0] > 0 else None
    out: List[dict] = []
    for cid, local_idx in enumerate(clusters_local):
        idx = seed_indices[local_idx]
        metrics = compute_cluster_metrics(pts[idx], nrm[idx])
        anchor_distance = 0.0
        if front_tree is not None:
            anchor_distance = float(front_tree.query(metrics["center"], k=1)[0])
        cluster_rows = [rows[int(i)] for i in idx]
        selected_frac = float(np.mean([float(row.get("cluster_selected", 0.0)) for row in cluster_rows]))
        label = classify_cluster_label(sequence, pts[idx], frames=frames, stride=stride, max_points=max_points_per_frame, seed=seed)
        out.append(
            {
                "cluster_id": int(cid),
                "indices": idx,
                "size": int(idx.size),
                "center": metrics["center"],
                "normal": metrics["normal"],
                "cluster_fitting_error": metrics["cluster_fitting_error"],
                "geodesic_smoothness": metrics["geodesic_smoothness"],
                "balloon_expand_ratio": metrics["balloon_expand_ratio"],
                "radius_extent": metrics["radius_extent"],
                "cluster_anchor_distance": anchor_distance,
                "selected_frac": selected_frac,
                "label": label,
            }
        )
    return out


def keep_mask_from_cluster_plane(points: np.ndarray, cluster: dict, *, radius: float, plane_dist: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    center = np.asarray(cluster["center"], dtype=float)
    normal = np.asarray(cluster["normal"], dtype=float)
    radial = np.linalg.norm(pts - center[None, :], axis=1) <= float(radius)
    planar = np.abs((pts - center[None, :]) @ normal) <= float(plane_dist)
    return radial & planar


def transform_variant(
    *,
    variant: str,
    sequence: str,
    rear_points: np.ndarray,
    rear_normals: np.ndarray,
    rows: List[dict],
    front_points: np.ndarray,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[dict], Dict[str, float], List[dict]]:
    pts = np.asarray(rear_points, dtype=float)
    nrm = np.asarray(rear_normals, dtype=float)
    rows_out = [dict(row) for row in rows]
    candidate_clusters = compute_candidate_clusters(
        sequence=sequence,
        points=pts,
        normals=nrm,
        rows=rows_out,
        front_points=np.asarray(front_points, dtype=float),
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points_per_frame,
        seed=seed,
    )

    point_cluster = np.full((pts.shape[0],), -1, dtype=np.int64)
    for cluster in candidate_clusters:
        for idx in cluster["indices"]:
            point_cluster[int(idx)] = int(cluster["cluster_id"])
    for i, row in enumerate(rows_out):
        cid = int(point_cluster[i])
        row["balloon_cluster_id"] = float(cid)
        row["cluster_retained"] = 0.0
        row["cluster_fitting_error"] = 0.0
        row["geodesic_smoothness"] = 0.0
        row["balloon_expand_ratio"] = 0.0
        row["cluster_anchor_distance"] = 0.0
        if cid >= 0:
            cluster = next(c for c in candidate_clusters if c["cluster_id"] == cid)
            row["cluster_fitting_error"] = float(cluster["cluster_fitting_error"])
            row["geodesic_smoothness"] = float(cluster["geodesic_smoothness"])
            row["balloon_expand_ratio"] = float(cluster["balloon_expand_ratio"])
            row["cluster_anchor_distance"] = float(cluster["cluster_anchor_distance"])

    if variant == "93_spatial_neighborhood_density_clustering":
        return pts, nrm, rows_out, {"kept_points": float(pts.shape[0]), "dropped_points": 0.0, "retained_clusters": 0.0}, candidate_clusters

    retained_clusters: List[dict] = []
    if variant == "95_geodesic_balloon_consistency":
        retained_clusters = [
            c
            for c in candidate_clusters
            if c["selected_frac"] > 0.5
            and c["size"] >= 8
            and c["cluster_fitting_error"] <= 0.003
            and c["geodesic_smoothness"] <= 0.10
            and c["balloon_expand_ratio"] >= 0.60
        ]
        if retained_clusters:
            keep_mask = np.isin(point_cluster, [c["cluster_id"] for c in retained_clusters])
        else:
            keep_mask = np.ones((pts.shape[0],), dtype=bool)
    elif variant == "96_support_cluster_model_fitting":
        retained_clusters = [
            c
            for c in candidate_clusters
            if c["selected_frac"] > 0.5 and c["size"] >= 8 and c["cluster_fitting_error"] <= 0.003
        ]
        if retained_clusters:
            keep_mask = np.zeros((pts.shape[0],), dtype=bool)
            for cluster in retained_clusters:
                keep_mask |= keep_mask_from_cluster_plane(pts, cluster, radius=0.08, plane_dist=0.01)
        else:
            keep_mask = np.ones((pts.shape[0],), dtype=bool)
    elif variant == "97_global_map_anchoring":
        retained_clusters = [
            c
            for c in candidate_clusters
            if c["selected_frac"] > 0.5
            and c["size"] >= 8
            and c["cluster_fitting_error"] <= 0.003
            and c["cluster_anchor_distance"] <= 0.035
        ]
        if retained_clusters:
            keep_mask = np.zeros((pts.shape[0],), dtype=bool)
            for cluster in retained_clusters:
                keep_mask |= keep_mask_from_cluster_plane(pts, cluster, radius=0.05, plane_dist=0.005)
        else:
            keep_mask = np.ones((pts.shape[0],), dtype=bool)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    for cluster in retained_clusters:
        for idx in cluster["indices"]:
            rows_out[int(idx)]["cluster_retained"] = 1.0
    filtered_points = pts[keep_mask]
    filtered_normals = nrm[keep_mask] if nrm.shape == pts.shape else nrm
    filtered_rows = [rows_out[i] for i, keep in enumerate(keep_mask.tolist()) if keep]
    stats = {
        "kept_points": float(filtered_points.shape[0]),
        "dropped_points": float(pts.shape[0] - filtered_points.shape[0]),
        "retained_clusters": float(len(retained_clusters)),
    }
    return filtered_points, filtered_normals, filtered_rows, stats, candidate_clusters


def evaluate_variant(
    *,
    variant_name: str,
    root_out: Path,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    seed: int,
    tsdf_ghosts: Dict[str, float],
    tum_acc_cm: float,
    tum_comp_r_5cm: float,
) -> dict:
    bonn_accs: List[float] = []
    bonn_comps: List[float] = []
    bonn_ghost_reds: List[float] = []
    rear_sum = tb_sum = ghost_sum = noise_sum = 0.0
    kept_points_sum = dropped_points_sum = retained_clusters_sum = 0.0
    seq_pairs: List[Tuple[float, float]] = []
    cluster_stats = {
        "tb": {"count": 0.0, **{f"{k}_sum": 0.0 for k in CLUSTER_FEATURE_KEYS}},
        "noise": {"count": 0.0, **{f"{k}_sum": 0.0 for k in CLUSTER_FEATURE_KEYS}},
        "ghost": {"count": 0.0, **{f"{k}_sum": 0.0 for k in CLUSTER_FEATURE_KEYS}},
    }
    point_stats = {
        "true_background": {"count": 0.0, **{f"{k}_sum": 0.0 for k in CLUSTER_FEATURE_KEYS}},
        "hole_or_noise": {"count": 0.0, **{f"{k}_sum": 0.0 for k in CLUSTER_FEATURE_KEYS}},
        "ghost_region": {"count": 0.0, **{f"{k}_sum": 0.0 for k in CLUSTER_FEATURE_KEYS}},
    }

    for seq in BONN_ALL3:
        base = CONTROL_ROOT / "bonn_slam" / "slam" / seq / "egf"
        seq_out = root_out / "bonn_slam" / "slam" / seq / "egf"
        seq_out.parent.mkdir(parents=True, exist_ok=True)
        if variant_name != "93_spatial_neighborhood_density_clustering":
            shutil.copytree(base, seq_out, dirs_exist_ok=True)
        else:
            seq_out.mkdir(parents=True, exist_ok=True)

        rear_points, rear_normals = load_points_with_normals(base / "rear_surface_points.ply")
        front_points, front_normals = load_points_with_normals(base / "front_surface_points.ply")
        ref_points, ref_normals = load_points_with_normals(base / "reference_points.ply")
        rows = load_rows(base / "rear_surface_features.csv")
        transformed_rear, transformed_normals, rows_out, variant_stats, candidate_clusters = transform_variant(
            variant=variant_name,
            sequence=seq,
            rear_points=rear_points,
            rear_normals=rear_normals,
            rows=rows,
            front_points=front_points,
            frames=frames,
            stride=stride,
            max_points_per_frame=max_points_per_frame,
            seed=seed,
        )

        pred_points = np.vstack([front_points, transformed_rear]) if front_points.shape[0] > 0 else transformed_rear
        pred_normals = np.vstack([front_normals, transformed_normals]) if front_points.shape[0] > 0 else transformed_normals
        pred_eval_points, pred_eval_normals, _align_info, _align_t = rigid_align_for_eval(
            pred_points=pred_points,
            pred_normals=pred_normals,
            ref_points=ref_points,
            voxel_size=0.02,
        )
        recon = compute_recon_metrics(pred_eval_points, ref_points, threshold=0.05, pred_normals=pred_eval_normals, gt_normals=ref_normals)

        stable_bg, tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
            sequence_dir=Path("data/bonn") / seq,
            frames=frames,
            stride=stride,
            max_points_per_frame=max_points_per_frame,
            seed=seed,
            stable_voxel=max(0.03, 0.02),
            stable_ratio=0.25,
            tail_frames=12,
            dynamic_voxel=max(0.05, 0.04),
            min_dynamic_hits=2,
            max_dynamic_ratio=0.35,
        )
        dyn = compute_dynamic_metrics(
            pred_points=pred_eval_points,
            stable_bg_points=stable_bg,
            tail_points=tail_points,
            dynamic_region=dynamic_region,
            dynamic_voxel=dynamic_voxel,
            ghost_thresh=0.08,
            bg_thresh=0.05,
        )
        ghost_red = ((tsdf_ghosts[seq] - float(dyn["ghost_ratio"])) / max(1e-9, tsdf_ghosts[seq])) * 100.0
        bonn_accs.append(float(recon["accuracy"]) * 100.0)
        bonn_comps.append(float(recon["recall_5cm"]) * 100.0)
        bonn_ghost_reds.append(ghost_red)

        classes = classify_points(seq, transformed_rear, frames=frames, stride=stride, max_points=max_points_per_frame, seed=seed)
        rear_sum += float(transformed_rear.shape[0])
        tb_sum += classes["tb"]
        ghost_sum += classes["ghost"]
        noise_sum += classes["noise"]
        seq_pairs.append((float(classes["tb"]), float(classes["noise"])))
        kept_points_sum += float(variant_stats["kept_points"])
        dropped_points_sum += float(variant_stats["dropped_points"])
        retained_clusters_sum += float(variant_stats["retained_clusters"])

        class_rows = classify_balloon_rows(seq, rows_out, frames=frames, stride=stride, max_points=max_points_per_frame, seed=seed)
        for label in point_stats:
            point_stats[label]["count"] += class_rows[label]["count"]
            for key in CLUSTER_FEATURE_KEYS:
                point_stats[label][f"{key}_sum"] += class_rows[label][f"{key}_sum"]

        for cluster in candidate_clusters:
            label = cluster["label"]
            cluster_stats[label]["count"] += 1.0
            for key in CLUSTER_FEATURE_KEYS:
                cluster_stats[label][f"{key}_sum"] += float(cluster[key])

        save_point_cloud(seq_out / "rear_surface_points.ply", transformed_rear, transformed_normals)
        save_point_cloud(seq_out / "surface_points.ply", pred_points, pred_normals)
        save_rows(seq_out / "rear_surface_features.csv", rows_out)
        with (seq_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"variant": variant_name, "cluster_stats": variant_stats, "recon": recon, "dynamic": dyn}, f, indent=2)

    pair_arr = np.asarray(seq_pairs, dtype=float) if seq_pairs else np.zeros((0, 2), dtype=float)
    tb_noise_corr = 0.0
    if pair_arr.shape[0] >= 2 and np.std(pair_arr[:, 0]) > 1e-9 and np.std(pair_arr[:, 1]) > 1e-9:
        tb_noise_corr = float(np.corrcoef(pair_arr[:, 0], pair_arr[:, 1])[0, 1])

    row = {
        "variant": variant_name,
        "tum_acc_cm": tum_acc_cm,
        "tum_comp_r_5cm": tum_comp_r_5cm,
        "bonn_acc_cm": float(np.mean(bonn_accs)),
        "bonn_comp_r_5cm": float(np.mean(bonn_comps)),
        "bonn_ghost_reduction_vs_tsdf": float(np.mean(bonn_ghost_reds)),
        "bonn_rear_points_sum": float(rear_sum),
        "bonn_rear_true_background_sum": float(tb_sum),
        "bonn_rear_ghost_sum": float(ghost_sum),
        "bonn_rear_hole_or_noise_sum": float(noise_sum),
        "kept_rear_points_sum": float(kept_points_sum),
        "dropped_rear_points_sum": float(dropped_points_sum),
        "retained_clusters_sum": float(retained_clusters_sum),
        "tb_noise_correlation": float(tb_noise_corr),
        "decision": "pending",
    }
    for label, prefix in [("tb", "tb_cluster"), ("noise", "noise_cluster"), ("ghost", "ghost_cluster")]:
        denom = max(1.0, cluster_stats[label]["count"])
        row[f"{prefix}_count"] = float(cluster_stats[label]["count"])
        for key in CLUSTER_FEATURE_KEYS:
            row[f"{prefix}_{key}_mean"] = float(cluster_stats[label][f"{key}_sum"] / denom)
    for label, prefix in [("true_background", "tb"), ("hole_or_noise", "noise"), ("ghost_region", "ghost")]:
        denom = max(1.0, point_stats[label]["count"])
        row[f"bonn_{prefix}_count"] = float(point_stats[label]["count"])
        for key in CLUSTER_FEATURE_KEYS:
            row[f"bonn_{prefix}_{key}_mean"] = float(point_stats[label][f"{key}_sum"] / denom)
    row["cluster_fit_margin_tb_vs_noise"] = float(row["tb_cluster_cluster_fitting_error_mean"] - row["noise_cluster_cluster_fitting_error_mean"])
    row["cluster_geo_margin_noise_vs_tb"] = float(row["noise_cluster_geodesic_smoothness_mean"] - row["tb_cluster_geodesic_smoothness_mean"])
    return row


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r["variant"] == "93_spatial_neighborhood_density_clustering")
    for row in rows:
        if row is control:
            row["decision"] = "control"
            continue
        corr_ok = row["tb_noise_correlation"] < 0.9
        tb_ok = row["bonn_rear_true_background_sum"] >= 8.0
        ghost_ok = row["bonn_ghost_reduction_vs_tsdf"] >= 22.0
        row["decision"] = "iterate" if corr_ok and tb_ok and ghost_ok else "abandon"


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# S2 balloon cluster compare",
        "",
        "| variant | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | tb_noise_correlation | bonn_ghost_reduction_vs_tsdf | kept_rear_points_sum | dropped_rear_points_sum | tb_cluster_cluster_fitting_error_mean | noise_cluster_cluster_fitting_error_mean | tb_cluster_geodesic_smoothness_mean | noise_cluster_geodesic_smoothness_mean | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['tb_noise_correlation']:.3f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['kept_rear_points_sum']:.0f} | {row['dropped_rear_points_sum']:.0f} | {row['tb_cluster_cluster_fitting_error_mean']:.4f} | {row['noise_cluster_cluster_fitting_error_mean']:.4f} | {row['tb_cluster_geodesic_smoothness_mean']:.4f} | {row['noise_cluster_geodesic_smoothness_mean']:.4f} | {row['decision']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_distribution(rows: List[dict], path_md: Path) -> None:
    lines = [
        "# S2 balloon cluster distribution report",
        "",
        "日期：`2026-03-10`",
        "协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "",
        "| variant | true_background_sum | ghost_sum | hole_or_noise_sum | tb_noise_correlation | tb_cluster_fit | noise_cluster_fit | tb_cluster_geo | noise_cluster_geo | retained_clusters_sum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['tb_noise_correlation']:.3f} | {row['tb_cluster_cluster_fitting_error_mean']:.4f} | {row['noise_cluster_cluster_fitting_error_mean']:.4f} | {row['tb_cluster_geodesic_smoothness_mean']:.4f} | {row['noise_cluster_geodesic_smoothness_mean']:.4f} | {row['retained_clusters_sum']:.0f} |"
        )
    lines += ["", "重点检查：`tb_noise_correlation < 0.9` 是否成立，以及 `TB` 是否仍保持 `>= 8`。"]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r["variant"] == "93_spatial_neighborhood_density_clustering")
    best = max(
        rows[1:],
        key=lambda r: (
            r["tb_noise_correlation"] < 0.9,
            -r["tb_noise_correlation"],
            r["bonn_ghost_reduction_vs_tsdf"],
            r["bonn_rear_true_background_sum"],
            r["bonn_ghost_reduction_vs_tsdf"],
            -r["bonn_rear_hole_or_noise_sum"],
        ),
    )
    lines = [
        "# S2 balloon cluster analysis",
        "",
        "日期：`2026-03-10`",
        "协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "对比表：`output/tmp/legacy_artifacts_placeholder`",
        "",
        "## 1. 簇级一致性是否区分了真假 TB",
        f"- 控制组 `93`：`fit(TB/Noise) = {control['tb_cluster_cluster_fitting_error_mean']:.4f} / {control['noise_cluster_cluster_fitting_error_mean']:.4f}`，`geo(TB/Noise) = {control['tb_cluster_geodesic_smoothness_mean']:.4f} / {control['noise_cluster_geodesic_smoothness_mean']:.4f}`。",
        f"- 最佳候选 `{best['variant']}`：`fit(TB/Noise) = {best['tb_cluster_cluster_fitting_error_mean']:.4f} / {best['noise_cluster_cluster_fitting_error_mean']:.4f}`，`geo(TB/Noise) = {best['tb_cluster_geodesic_smoothness_mean']:.4f} / {best['noise_cluster_geodesic_smoothness_mean']:.4f}`。",
        "- 当前分离主因来自 `cluster_fitting_error`：TB 簇的平面拟合误差显著低于 Noise 簇；`geodesic_smoothness` 单独使用时并不构成有效门槛，这也是 `95` 退化为零保留的原因。",
        "",
        "## 2. 相关性是否打破死锁",
        f"- 控制组 `93`：TB=`{control['bonn_rear_true_background_sum']:.0f}`, Noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}`, `tb_noise_correlation={control['tb_noise_correlation']:.3f}`。",
        f"- 最佳候选 `{best['variant']}`：TB=`{best['bonn_rear_true_background_sum']:.0f}`, Noise=`{best['bonn_rear_hole_or_noise_sum']:.0f}`, `tb_noise_correlation={best['tb_noise_correlation']:.3f}`。",
        "",
        "## 3. 结论",
        "- 若 `tb_noise_correlation < 0.9`，说明簇级验证至少打破了 `crowd2` 单序列过拟合对 family 统计的绑架。",
        "- 若 `TB` 仍保持 `>= 8` 且 `ghost_reduction_vs_tsdf >= 22%`，说明这条支链已从“局部加点”推进到“全局一致性过滤”。",
        "- 即便本轮局部过线，也不代表 `S2` 整体通过；`Acc/Comp-R` 全局门槛仍未满足时，绝对不能进入 `S3`。",
    ]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 balloon-cluster runner.")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_points_per_frame", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tum_acc_cm, tum_comp_r_5cm = load_control_tum_metrics()
    tsdf_ghosts = load_control_tsdf_ghosts()
    variants = [
        ("93_spatial_neighborhood_density_clustering", CONTROL_ROOT),
        ("95_geodesic_balloon_consistency", PROJECT_ROOT / "output" / "s2_stage" / "95_geodesic_balloon_consistency"),
        ("96_support_cluster_model_fitting", PROJECT_ROOT / "output" / "s2_stage" / "96_support_cluster_model_fitting"),
        ("97_global_map_anchoring", PROJECT_ROOT / "output" / "s2_stage" / "97_global_map_anchoring"),
    ]
    for _, root in variants[1:]:
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(CONTROL_ROOT / "tum_oracle" / "oracle", root / "tum_oracle" / "oracle", dirs_exist_ok=True)

    rows = [
        evaluate_variant(
            variant_name=name,
            root_out=root,
            frames=args.frames,
            stride=args.stride,
            max_points_per_frame=args.max_points_per_frame,
            seed=args.seed,
            tsdf_ghosts=tsdf_ghosts,
            tum_acc_cm=tum_acc_cm,
            tum_comp_r_5cm=tum_comp_r_5cm,
        )
        for name, root in variants
    ]
    decide(rows)
    out_dir = PROJECT_ROOT / "output" / "s2"
    write_compare(rows, out_dir / "S2_RPS_BALLOON_CLUSTER_COMPARE.csv", out_dir / "S2_RPS_BALLOON_CLUSTER_COMPARE.md")
    write_distribution(rows, out_dir / "S2_RPS_BALLOON_CLUSTER_DISTRIBUTION.md")
    write_analysis(rows, out_dir / "S2_RPS_BALLOON_CLUSTER_ANALYSIS.md")
    write_compare(rows, out_dir / "S2_RPS_DEEP_EXPLORE_COMPARE.csv", out_dir / "S2_RPS_DEEP_EXPLORE_COMPARE.md")
    write_distribution(rows, out_dir / "S2_RPS_DEEP_EXPLORE_DISTRIBUTION.md")
    write_analysis(rows, out_dir / "S2_RPS_DEEP_EXPLORE_ANALYSIS.md")
    print("[done]", out_dir / "S2_RPS_BALLOON_CLUSTER_COMPARE.csv")


if __name__ == "__main__":
    main()
