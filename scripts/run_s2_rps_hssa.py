from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egf_dhmap3d.core.config import EGF3DConfig
from run_benchmark import (
    build_dynamic_references,
    compute_dynamic_metrics,
    compute_recon_metrics,
    load_points_with_normals,
    rigid_align_for_eval,
)


CONTROL_ROOT = PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "80_ray_penetration_consistency"
RAY_COMPARE = PROJECT_ROOT / "processes" / "s2" / "S2_RPS_RAY_CONSISTENCY_COMPARE.csv"
DATA_BONN = PROJECT_ROOT / "data" / "bonn"
BONN_ALL3 = ["rgbd_bonn_balloon2", "rgbd_bonn_balloon", "rgbd_bonn_crowd2"]
BG_THRESH = 0.05
SUPPORT_FEATURE_KEYS = [
    "support_score",
    "neighbor_count",
    "history_reactivate",
    "cluster_thickness",
]


def load_control_tum_metrics() -> Tuple[float, float]:
    rows = list(csv.DictReader(RAY_COMPARE.open("r", encoding="utf-8")))
    row = next(r for r in rows if r["variant"] == "80_ray_penetration_consistency")
    return float(row["tum_acc_cm"]), float(row["tum_comp_r_5cm"])


def load_control_tsdf_ghosts() -> Dict[str, float]:
    path = CONTROL_ROOT / "bonn_slam" / "slam" / "tables" / "dynamic_metrics.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return {str(r["sequence"]): float(r["ghost_ratio"]) for r in rows if str(r.get("method", "")).lower() == "tsdf"}


def load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    out: List[dict] = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            if value in (None, ""):
                parsed[key] = 0.0
            else:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
        out.append(parsed)
    return out


def save_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["x", "y", "z"])
        return
    fieldnames: List[str] = []
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


def save_point_cloud(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd)


def parse_tum_style_text(path: Path) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items = line.split()
            if len(items) < 2:
                continue
            out.append((float(items[0]), items[1]))
    return out


def associate_depth_paths(
    sequence_dir: Path,
    *,
    frames: int,
    stride: int,
    assoc_max_diff: float = 0.02,
) -> List[Tuple[float, Path]]:
    rgb_entries = parse_tum_style_text(sequence_dir / "rgb.txt")
    depth_entries = parse_tum_style_text(sequence_dir / "depth.txt")
    if not rgb_entries or not depth_entries:
        return []
    depth_times = np.asarray([t for t, _ in depth_entries], dtype=float)
    matched: List[Tuple[float, Path]] = []
    for t_rgb, _ in rgb_entries:
        j = int(np.searchsorted(depth_times, t_rgb))
        candidates: List[int] = []
        if j < depth_times.size:
            candidates.append(j)
        if j - 1 >= 0:
            candidates.append(j - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda k: abs(float(depth_times[k]) - t_rgb))
        if abs(float(depth_times[best]) - t_rgb) > assoc_max_diff:
            continue
        matched.append((float(t_rgb), sequence_dir / depth_entries[best][1]))
    return matched[:: max(1, int(stride))][: int(frames)]


def load_predicted_views(sequence: str, *, frames: int, stride: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    sequence_dir = DATA_BONN / sequence
    matched = associate_depth_paths(sequence_dir, frames=frames, stride=stride)
    traj = np.load(CONTROL_ROOT / "bonn_slam" / "slam" / sequence / "egf" / "trajectory.npy")
    views: List[Tuple[np.ndarray, np.ndarray]] = []
    for pose_w_c, (_timestamp, depth_path) in zip(traj, matched):
        depth = imageio.imread(depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        views.append((np.asarray(pose_w_c, dtype=float), depth.astype(np.float32)))
    return views


def classify_points(sequence: str, points: np.ndarray, *, frames: int, stride: int, max_points: int, seed: int) -> Dict[str, float]:
    stable_bg, _tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
        sequence_dir=DATA_BONN / sequence,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points,
        seed=seed,
    )
    pts = np.asarray(points, dtype=float)
    bg_tree = None
    if stable_bg.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stable_bg)
        bg_tree = o3d.geometry.KDTreeFlann(pcd)
    out = {"tb": 0.0, "ghost": 0.0, "noise": 0.0}
    for point in pts:
        voxel = tuple(np.floor(point / float(dynamic_voxel)).astype(np.int32).tolist())
        if voxel in dynamic_region:
            out["ghost"] += 1.0
            continue
        if bg_tree is not None:
            _, idx, dist2 = bg_tree.search_knn_vector_3d(point.astype(float), 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < BG_THRESH:
                out["tb"] += 1.0
                continue
        out["noise"] += 1.0
    return out


def classify_feature_rows(sequence: str, rows: List[dict], *, frames: int, stride: int, max_points: int, seed: int) -> Dict[str, dict]:
    stable_bg, _tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
        sequence_dir=DATA_BONN / sequence,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points,
        seed=seed,
    )
    bg_tree = None
    if stable_bg.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stable_bg)
        bg_tree = o3d.geometry.KDTreeFlann(pcd)
    stats = {
        "true_background": {"count": 0.0, **{f"{k}_sum": 0.0 for k in SUPPORT_FEATURE_KEYS}},
        "ghost_region": {"count": 0.0, **{f"{k}_sum": 0.0 for k in SUPPORT_FEATURE_KEYS}},
        "hole_or_noise": {"count": 0.0, **{f"{k}_sum": 0.0 for k in SUPPORT_FEATURE_KEYS}},
    }
    for row in rows:
        point = np.array([row.get("x", 0.0), row.get("y", 0.0), row.get("z", 0.0)], dtype=float)
        voxel = tuple(np.floor(point / float(dynamic_voxel)).astype(np.int32).tolist())
        label = "hole_or_noise"
        if voxel in dynamic_region:
            label = "ghost_region"
        elif bg_tree is not None:
            _, idx, dist2 = bg_tree.search_knn_vector_3d(point.astype(float), 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < BG_THRESH:
                label = "true_background"
        stats[label]["count"] += 1.0
        for key in SUPPORT_FEATURE_KEYS:
            stats[label][f"{key}_sum"] += float(row.get(key, 0.0))
    for bucket in stats.values():
        denom = max(1.0, bucket["count"])
        for key in SUPPORT_FEATURE_KEYS:
            bucket[f"{key}_mean"] = float(bucket[f"{key}_sum"] / denom)
    return stats


def make_support_score(
    history_anchor: np.ndarray,
    observation_support: np.ndarray,
    direct_count: np.ndarray,
    occlusion_count: np.ndarray,
    neighbor_count: np.ndarray,
) -> np.ndarray:
    history_reactivate = 0.65 * history_anchor + 0.35 * observation_support
    direct_boost = 1.0 + 0.25 * np.maximum(direct_count - 1.0, 0.0)
    occ_penalty = 1.0 + 0.35 * np.maximum(occlusion_count, 0.0)
    crowd_penalty = 1.0 + 0.08 * np.maximum(neighbor_count - 4.0, 0.0)
    return history_reactivate * direct_boost / np.maximum(1e-6, occ_penalty * crowd_penalty)


def compute_support_features(
    points: np.ndarray,
    rows: List[dict],
    views: Sequence[Tuple[np.ndarray, np.ndarray]],
    cfg: EGF3DConfig,
) -> Dict[str, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    size = int(pts.shape[0])
    history_anchor = np.asarray([float(row.get("history_anchor", 0.0)) for row in rows], dtype=float)
    observation_support = np.asarray([float(row.get("observation_support", 0.0)) for row in rows], dtype=float)
    direct_count = np.zeros((size,), dtype=float)
    occlusion_count = np.zeros((size,), dtype=float)
    anchor_spread = np.zeros((size,), dtype=float)
    anchor_mean = pts.copy()
    cam = cfg.camera
    if size > 0:
        tree = cKDTree(pts)
        neighbor_count = np.asarray([max(0, len(tree.query_ball_point(point, 0.10)) - 1) for point in pts], dtype=float)
    else:
        neighbor_count = np.zeros((0,), dtype=float)

    for i, point in enumerate(pts):
        anchors: List[np.ndarray] = []
        for pose_w_c, depth_raw in views:
            pose_c_w = np.linalg.inv(pose_w_c)
            point_cam = pose_c_w @ np.array([point[0], point[1], point[2], 1.0], dtype=float)
            z = float(point_cam[2])
            if z <= cam.depth_min or z >= cam.depth_max:
                continue
            u = int(round(float(cam.fx) * float(point_cam[0]) / z + float(cam.cx)))
            v = int(round(float(cam.fy) * float(point_cam[1]) / z + float(cam.cy)))
            if v < 0 or u < 0 or v >= depth_raw.shape[0] or u >= depth_raw.shape[1]:
                continue
            depth_obs = float(depth_raw[v, u]) / float(cam.depth_scale)
            if depth_obs <= cam.depth_min or depth_obs >= cam.depth_max:
                continue
            delta = z - depth_obs
            if abs(delta) <= 0.05:
                direct_count[i] += 1.0
                x = (float(u) - float(cam.cx)) * depth_obs / float(cam.fx)
                y = (float(v) - float(cam.cy)) * depth_obs / float(cam.fy)
                anchor_world = pose_w_c[:3, :3] @ np.array([x, y, depth_obs], dtype=float) + pose_w_c[:3, 3]
                anchors.append(anchor_world)
            elif 0.05 < delta <= 0.40:
                occlusion_count[i] += 1.0
        if anchors:
            anc = np.asarray(anchors, dtype=float)
            anchor_mean[i] = anc.mean(axis=0)
            if anc.shape[0] >= 2:
                anchor_spread[i] = float(np.mean(np.linalg.norm(anc - anchor_mean[i][None, :], axis=1)))
    history_reactivate = 0.65 * history_anchor + 0.35 * observation_support
    support_score = make_support_score(history_anchor, observation_support, direct_count, occlusion_count, neighbor_count)
    return {
        "direct_count": direct_count,
        "occlusion_count": occlusion_count,
        "anchor_spread": anchor_spread,
        "anchor_mean": anchor_mean,
        "neighbor_count": neighbor_count,
        "history_reactivate": history_reactivate,
        "support_score": support_score,
    }


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


def compute_cluster_info(
    *,
    seed_indices: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    anchor_mean: np.ndarray,
    support: Dict[str, np.ndarray],
    rows: List[dict],
) -> List[dict]:
    if seed_indices.size == 0:
        return []
    seed_anchor = anchor_mean[seed_indices]
    clusters_local = cluster_indices(seed_anchor, radius=0.12, min_size=3)
    cluster_info: List[dict] = []
    for cid, local_idx in enumerate(clusters_local):
        idx = seed_indices[local_idx]
        cluster_points = np.asarray(points[idx], dtype=float)
        cluster_anchors = np.asarray(anchor_mean[idx], dtype=float)
        cluster_normals = np.asarray(normals[idx], dtype=float)
        center = cluster_anchors.mean(axis=0)
        centroid = cluster_anchors.mean(axis=0)
        if cluster_anchors.shape[0] >= 3:
            cov = np.cov((cluster_anchors - centroid).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)
            pca_normal = eigvecs[:, order[0]]
            tangent_1 = eigvecs[:, order[-1]]
            tangent_2 = eigvecs[:, order[-2]]
            normal = pca_normal
            if cluster_normals.shape == cluster_points.shape and cluster_normals.shape[0] > 0:
                mean_normal = cluster_normals.mean(axis=0)
                mean_norm = float(np.linalg.norm(mean_normal))
                if mean_norm > 1e-9:
                    mean_normal = mean_normal / mean_norm
                    if float(np.dot(mean_normal, normal)) < 0.0:
                        normal = -normal
            d = np.abs((cluster_anchors - centroid) @ normal)
            thickness = float(np.quantile(d, 0.95))
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
            tangent_1 = np.array([1.0, 0.0, 0.0], dtype=float)
            tangent_2 = np.array([0.0, 1.0, 0.0], dtype=float)
            thickness = 1.0
        tangent_1 = tangent_1 / max(1e-9, float(np.linalg.norm(tangent_1)))
        tangent_2 = tangent_2 / max(1e-9, float(np.linalg.norm(tangent_2)))
        row_vals = [rows[int(i)] for i in idx]
        cluster_info.append(
            {
                "cluster_id": int(cid),
                "indices": idx,
                "center": center,
                "centroid": centroid,
                "normal": normal,
                "tangent_1": tangent_1,
                "tangent_2": tangent_2,
                "size": int(idx.size),
                "thickness": thickness,
                "support_mean": float(np.mean(support["support_score"][idx])),
                "history_mean": float(np.mean(support["history_reactivate"][idx])),
                "history_anchor_mean": float(np.mean([float(row.get("history_anchor", 0.0)) for row in row_vals])),
                "obs_mean": float(np.mean([float(row.get("observation_support", 0.0)) for row in row_vals])),
                "occ_mean": float(np.mean(support["occlusion_count"][idx])),
                "direct_mean": float(np.mean(support["direct_count"][idx])),
            }
        )
    return cluster_info


def patch_offsets(grid_size: int, step: float) -> List[Tuple[float, float]]:
    coords = np.linspace(-step * float(grid_size - 1) / 2.0, step * float(grid_size - 1) / 2.0, grid_size)
    return [(float(a), float(b)) for a in coords for b in coords]


def synthesize_row(
    template: dict,
    *,
    point: np.ndarray,
    normal: np.ndarray,
    cluster: dict,
    support_score: float,
    neighbor_count: float,
) -> dict:
    row = dict(template)
    row["x"] = float(point[0])
    row["y"] = float(point[1])
    row["z"] = float(point[2])
    row["nx"] = float(normal[0])
    row["ny"] = float(normal[1])
    row["nz"] = float(normal[2])
    row["support_score"] = float(support_score)
    row["neighbor_count"] = float(neighbor_count)
    row["history_reactivate"] = float(cluster["history_mean"])
    row["cluster_thickness"] = float(cluster["thickness"])
    row["cluster_size"] = float(cluster["size"])
    row["anchor_direct_count"] = float(cluster["direct_mean"])
    row["anchor_occ_count"] = float(cluster["occ_mean"])
    row["cluster_selected"] = 1.0
    row["is_support_seed"] = 1.0
    return row


def transform_variant(
    *,
    variant: str,
    rear_points: np.ndarray,
    rear_normals: np.ndarray,
    rows: List[dict],
    views: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, List[dict], Dict[str, float]]:
    points = np.asarray(rear_points, dtype=float)
    normals = np.asarray(rear_normals, dtype=float)
    out_rows = [dict(row) for row in rows]
    if points.shape[0] == 0:
        return points, normals, out_rows, {"added_points": 0.0, "seed_points": 0.0, "selected_clusters": 0.0}

    cfg = EGF3DConfig()
    support = compute_support_features(points, out_rows, views, cfg)
    for i, row in enumerate(out_rows):
        row["support_score"] = float(support["support_score"][i])
        row["neighbor_count"] = float(support["neighbor_count"][i])
        row["history_reactivate"] = float(support["history_reactivate"][i])
        row["anchor_direct_count"] = float(support["direct_count"][i])
        row["anchor_occ_count"] = float(support["occlusion_count"][i])
        row["anchor_spread"] = float(support["anchor_spread"][i])
        row["cluster_thickness"] = 0.0
        row["cluster_size"] = 0.0
        row["cluster_selected"] = 0.0
        row["is_support_seed"] = 0.0

    if variant == "80_ray_penetration_consistency":
        return points, normals, out_rows, {"added_points": 0.0, "seed_points": 0.0, "selected_clusters": 0.0}

    seed_threshold = 0.18 if variant == "92_multi_view_ray_support_aggregation" else 0.19
    seed_indices = np.where(support["support_score"] >= seed_threshold)[0].astype(np.int64)
    cluster_info = compute_cluster_info(
        seed_indices=seed_indices,
        points=points,
        normals=normals,
        anchor_mean=support["anchor_mean"],
        support=support,
        rows=out_rows,
    )

    for cluster in cluster_info:
        idx = cluster["indices"]
        for i in idx:
            out_rows[int(i)]["cluster_thickness"] = float(cluster["thickness"])
            out_rows[int(i)]["cluster_size"] = float(cluster["size"])
            out_rows[int(i)]["is_support_seed"] = 1.0

    added_points: List[np.ndarray] = []
    added_normals: List[np.ndarray] = []
    added_rows: List[dict] = []
    selected_clusters = 0
    template = dict(out_rows[0]) if out_rows else {}

    for cluster in cluster_info:
        use_cluster = False
        grid = 1
        step = 0.0
        if variant == "92_multi_view_ray_support_aggregation":
            use_cluster = cluster["size"] >= 3 and cluster["support_mean"] >= 0.18 and cluster["occ_mean"] <= 0.25
            grid = 1
            step = 0.0
        elif variant == "93_spatial_neighborhood_density_clustering":
            use_cluster = cluster["size"] >= 4 and cluster["thickness"] <= 0.005 and cluster["support_mean"] >= 0.23 and cluster["occ_mean"] <= 0.10
            grid = 3
            step = 0.02
        elif variant == "94_historical_tsdf_consistency_reactivation":
            use_cluster = (
                cluster["size"] >= 4
                and cluster["thickness"] <= 0.005
                and cluster["support_mean"] >= 0.23
                and cluster["history_anchor_mean"] >= 0.24
                and cluster["obs_mean"] >= 0.18
                and cluster["occ_mean"] <= 0.10
            )
            grid = 2
            step = 0.03
        if not use_cluster:
            continue
        selected_clusters += 1
        for i in cluster["indices"]:
            out_rows[int(i)]["cluster_selected"] = 1.0
        if grid == 1:
            candidates = [(0.0, 0.0)]
        else:
            candidates = patch_offsets(grid, step)
        for du, dv in candidates:
            point = cluster["center"] + du * cluster["tangent_1"] + dv * cluster["tangent_2"]
            normal = cluster["normal"]
            added_points.append(point.astype(float))
            added_normals.append(normal.astype(float))
            added_rows.append(
                synthesize_row(
                    template,
                    point=point,
                    normal=normal,
                    cluster=cluster,
                    support_score=float(cluster["support_mean"]),
                    neighbor_count=float(cluster["size"] - 1),
                )
            )

    if added_points:
        out_points = np.vstack([points, np.asarray(added_points, dtype=float)])
        out_normals = np.vstack([normals, np.asarray(added_normals, dtype=float)]) if normals.shape == points.shape else np.asarray(added_normals, dtype=float)
        out_rows.extend(added_rows)
    else:
        out_points = points
        out_normals = normals
    return out_points, out_normals, out_rows, {
        "added_points": float(len(added_points)),
        "seed_points": float(seed_indices.size),
        "selected_clusters": float(selected_clusters),
    }


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
    added_support_points_sum = 0.0
    seed_points_sum = 0.0
    seq_pairs: List[Tuple[float, float]] = []
    feature_class = {
        "true_background": {"count": 0.0, **{f"{k}_sum": 0.0 for k in SUPPORT_FEATURE_KEYS}},
        "ghost_region": {"count": 0.0, **{f"{k}_sum": 0.0 for k in SUPPORT_FEATURE_KEYS}},
        "hole_or_noise": {"count": 0.0, **{f"{k}_sum": 0.0 for k in SUPPORT_FEATURE_KEYS}},
    }

    for seq in BONN_ALL3:
        base = CONTROL_ROOT / "bonn_slam" / "slam" / seq / "egf"
        seq_out = root_out / "bonn_slam" / "slam" / seq / "egf"
        seq_out.parent.mkdir(parents=True, exist_ok=True)
        if variant_name != "80_ray_penetration_consistency":
            shutil.copytree(base, seq_out, dirs_exist_ok=True)
        else:
            seq_out.mkdir(parents=True, exist_ok=True)

        rear_points, rear_normals = load_points_with_normals(base / "rear_surface_points.ply")
        front_points, front_normals = load_points_with_normals(base / "front_surface_points.ply")
        ref_points, ref_normals = load_points_with_normals(base / "reference_points.ply")
        rows = load_rows(base / "rear_surface_features.csv")
        views = load_predicted_views(seq, frames=frames, stride=stride)

        transformed_rear, transformed_normals, rows_out, hssa_stats = transform_variant(
            variant=variant_name,
            rear_points=rear_points,
            rear_normals=rear_normals,
            rows=rows,
            views=views,
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
            sequence_dir=DATA_BONN / seq,
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

        class_stats = classify_feature_rows(seq, rows_out, frames=frames, stride=stride, max_points=max_points_per_frame, seed=seed)
        for label in feature_class:
            feature_class[label]["count"] += class_stats[label]["count"]
            for key in SUPPORT_FEATURE_KEYS:
                feature_class[label][f"{key}_sum"] += class_stats[label][f"{key}_sum"]

        added_support_points_sum += float(hssa_stats.get("added_points", 0.0))
        seed_points_sum += float(hssa_stats.get("seed_points", 0.0))

        save_point_cloud(seq_out / "rear_surface_points.ply", transformed_rear, transformed_normals)
        save_point_cloud(seq_out / "surface_points.ply", pred_points, pred_normals)
        save_rows(seq_out / "rear_surface_features.csv", rows_out)
        with (seq_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"variant": variant_name, "hssa_stats": hssa_stats, "recon": recon, "dynamic": dyn}, f, indent=2)

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
        "added_support_points_sum": float(added_support_points_sum),
        "support_seed_points_sum": float(seed_points_sum),
        "tb_noise_correlation": float(tb_noise_corr),
        "decision": "pending",
    }
    for label, prefix in [("true_background", "tb"), ("hole_or_noise", "noise"), ("ghost_region", "ghost")]:
        denom = max(1.0, feature_class[label]["count"])
        row[f"bonn_{prefix}_count"] = float(feature_class[label]["count"])
        for key in SUPPORT_FEATURE_KEYS:
            row[f"bonn_{prefix}_{key}_mean"] = float(feature_class[label][f"{key}_sum"] / denom)
    row["support_margin_tb_vs_noise"] = float(row["bonn_tb_support_score_mean"] - row["bonn_noise_support_score_mean"])
    return row


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r["variant"] == "80_ray_penetration_consistency")
    for row in rows:
        if row is control:
            row["decision"] = "control"
            continue
        tb_ok = row["bonn_rear_true_background_sum"] > 6.0
        corr_ok = row["tb_noise_correlation"] < 0.9
        support_ok = row["support_margin_tb_vs_noise"] > 0.0
        ghost_ok = row["bonn_ghost_reduction_vs_tsdf"] >= 22.0 and row["bonn_rear_ghost_sum"] <= 25.0
        row["decision"] = "iterate" if tb_ok and corr_ok and support_ok and ghost_ok else "abandon"


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# S2 HSSA compare",
        "",
        "| variant | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | added_support_points_sum | bonn_tb_support_score_mean | bonn_noise_support_score_mean | tb_noise_correlation | bonn_comp_r_5cm | bonn_acc_cm | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['added_support_points_sum']:.0f} | {row['bonn_tb_support_score_mean']:.3f} | {row['bonn_noise_support_score_mean']:.3f} | {row['tb_noise_correlation']:.3f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.3f} | {row['decision']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_distribution(rows: List[dict], path_md: Path) -> None:
    lines = [
        "# S2 HSSA distribution report",
        "",
        "日期：`2026-03-10`",
        "协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "",
        "| variant | true_background_sum | ghost_sum | hole_or_noise_sum | added_support_points_sum | tb_support_score_mean | noise_support_score_mean | tb_neighbor_count_mean | noise_neighbor_count_mean | tb_history_reactivate_mean | noise_history_reactivate_mean | tb_noise_correlation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['added_support_points_sum']:.0f} | {row['bonn_tb_support_score_mean']:.3f} | {row['bonn_noise_support_score_mean']:.3f} | {row['bonn_tb_neighbor_count_mean']:.3f} | {row['bonn_noise_neighbor_count_mean']:.3f} | {row['bonn_tb_history_reactivate_mean']:.3f} | {row['bonn_noise_history_reactivate_mean']:.3f} | {row['tb_noise_correlation']:.3f} |"
        )
    lines += ["", "重点检查：`support_score(TB) > support_score(Noise)` 是否成立，以及 `TB` 是否突破 `6`。"]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r["variant"] == "80_ray_penetration_consistency")
    best = max(
        rows[1:],
        key=lambda r: (
            r["bonn_rear_true_background_sum"],
            r["support_margin_tb_vs_noise"],
            r["bonn_ghost_reduction_vs_tsdf"],
            -r["bonn_rear_ghost_sum"],
            -r["bonn_rear_hole_or_noise_sum"],
        ),
    )
    lines = [
        "# S2 HSSA analysis",
        "",
        "日期：`2026-03-10`",
        "协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "对比表：`processes/s2/S2_RPS_HSSA_COMPARE.csv`",
        "",
        "## 1. 支持度特征是否区分了 TB 与 Noise",
        f"- 控制组 `80`：`support_score(TB/Noise) = {control['bonn_tb_support_score_mean']:.3f} / {control['bonn_noise_support_score_mean']:.3f}`，`history_reactivate(TB/Noise) = {control['bonn_tb_history_reactivate_mean']:.3f} / {control['bonn_noise_history_reactivate_mean']:.3f}`。",
        f"- 最佳候选 `{best['variant']}`：`support_score(TB/Noise) = {best['bonn_tb_support_score_mean']:.3f} / {best['bonn_noise_support_score_mean']:.3f}`，差值=`{best['support_margin_tb_vs_noise']:.3f}`。",
        "",
        "## 2. HSSA 是否打破了 TB=4 的停滞",
        f"- 控制组 `80`：TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, Noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}`。",
        f"- 最佳候选 `{best['variant']}`：TB=`{best['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best['bonn_rear_ghost_sum']:.0f}`, Noise=`{best['bonn_rear_hole_or_noise_sum']:.0f}`, 新增支持点=`{best['added_support_points_sum']:.0f}`。",
        "",
        "## 3. 相关性是否解耦",
        f"- 控制组相关性：`{control['tb_noise_correlation']:.3f}`。",
        f"- 候选相关性：`{best['tb_noise_correlation']:.3f}`。",
        "- 若相关性仍接近 `1.0`，说明支持度聚合只在单个序列形成局部增益，尚未在 family 层打破 `TB-Noise` 耦合。",
        "",
        "## 4. 阶段判断",
        "- 若未同时满足 `TB > 6`、`tb_noise_correlation < 0.9`、`support_score(TB) > support_score(Noise)`、`ghost_reduction_vs_tsdf >= 22%`，则 `S2` 仍未通过，绝对不能进入 `S3`。",
    ]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 HSSA runner.")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_points_per_frame", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tum_acc_cm, tum_comp_r_5cm = load_control_tum_metrics()
    tsdf_ghosts = load_control_tsdf_ghosts()
    variants = [
        ("80_ray_penetration_consistency", CONTROL_ROOT),
        ("92_multi_view_ray_support_aggregation", PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "92_multi_view_ray_support_aggregation"),
        ("93_spatial_neighborhood_density_clustering", PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "93_spatial_neighborhood_density_clustering"),
        ("94_historical_tsdf_consistency_reactivation", PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "94_historical_tsdf_consistency_reactivation"),
    ]

    for variant_name, root in variants[1:]:
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
    out_dir = PROJECT_ROOT / "processes" / "s2"
    write_compare(rows, out_dir / "S2_RPS_HSSA_COMPARE.csv", out_dir / "S2_RPS_HSSA_COMPARE.md")
    write_distribution(rows, out_dir / "S2_RPS_HSSA_DISTRIBUTION.md")
    write_analysis(rows, out_dir / "S2_RPS_HSSA_ANALYSIS.md")
    print("[done]", out_dir / "S2_RPS_HSSA_COMPARE.csv")


if __name__ == "__main__":
    main()
