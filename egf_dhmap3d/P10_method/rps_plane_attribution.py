from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import open3d as o3d


@dataclass
class PlaneModel:
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    inlier_count: int
    extent_xy: float
    extent_z: float


def extract_dominant_planes(
    points: np.ndarray,
    *,
    distance_threshold: float = 0.04,
    min_plane_points: int = 24,
    max_planes: int = 4,
    min_extent_xy: float = 0.25,
) -> List[PlaneModel]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < max(8, min_plane_points):
        return []
    remain = pts.copy()
    planes: List[PlaneModel] = []
    for _ in range(max_planes):
        if remain.shape[0] < min_plane_points:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remain)
        try:
            model, inliers = pcd.segment_plane(
                distance_threshold=float(distance_threshold),
                ransac_n=3,
                num_iterations=300,
            )
        except RuntimeError:
            break
        inliers = np.asarray(inliers, dtype=np.int64)
        if inliers.size < min_plane_points:
            break
        inlier_pts = remain[inliers]
        bbox = inlier_pts.max(axis=0) - inlier_pts.min(axis=0)
        extent_xy = float(np.linalg.norm(bbox[:2]))
        extent_z = float(bbox[2])
        if extent_xy < float(min_extent_xy):
            mask = np.ones(remain.shape[0], dtype=bool)
            mask[inliers] = False
            remain = remain[mask]
            continue
        abc = np.asarray(model[:3], dtype=float)
        norm = float(np.linalg.norm(abc))
        if norm < 1e-8:
            break
        normal = abc / norm
        d = float(model[3] / norm)
        centroid = inlier_pts.mean(axis=0)
        planes.append(
            PlaneModel(
                normal=normal,
                offset=d,
                centroid=centroid,
                inlier_count=int(inliers.size),
                extent_xy=extent_xy,
                extent_z=extent_z,
            )
        )
        mask = np.ones(remain.shape[0], dtype=bool)
        mask[inliers] = False
        remain = remain[mask]
    return planes


def point_to_plane_distance(points: np.ndarray, plane: PlaneModel) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return np.abs(pts @ plane.normal + float(plane.offset))


def snap_points_to_plane(points: np.ndarray, plane: PlaneModel) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    signed = pts @ plane.normal + float(plane.offset)
    return pts - signed[:, None] * plane.normal[None, :]


def nearest_plane_assignment(points: np.ndarray, planes: List[PlaneModel]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0 or not planes:
        return np.full((pts.shape[0],), -1, dtype=np.int64), np.full((pts.shape[0],), np.inf, dtype=float)
    dists = np.stack([point_to_plane_distance(pts, plane) for plane in planes], axis=1)
    assign = np.argmin(dists, axis=1)
    dist = dists[np.arange(pts.shape[0]), assign]
    return assign.astype(np.int64), dist.astype(float)


def front_boundary_scores(front_points: np.ndarray, query_points: np.ndarray, *, radius: float = 0.08, k: int = 16) -> np.ndarray:
    front = np.asarray(front_points, dtype=float)
    query = np.asarray(query_points, dtype=float)
    if front.shape[0] == 0 or query.shape[0] == 0:
        return np.zeros((query.shape[0],), dtype=float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(front)
    tree = o3d.geometry.KDTreeFlann(pcd)
    scores = np.zeros((query.shape[0],), dtype=float)
    rad = float(max(1e-3, radius))
    for i, point in enumerate(query):
        count, idx, dist2 = tree.search_hybrid_vector_3d(point.astype(float), rad, int(max(4, k)))
        if count <= 0:
            scores[i] = 1.0
            continue
        mean_d = float(np.sqrt(np.mean(dist2[:count]))) if count > 0 else rad
        density = float(np.clip(count / max(4.0, float(k)), 0.0, 1.0))
        spread = float(np.clip(mean_d / rad, 0.0, 1.0))
        scores[i] = float(np.clip(0.55 * (1.0 - density) + 0.45 * spread, 0.0, 1.0))
    return scores


def plane_layout_mask(points: np.ndarray, assignments: np.ndarray, planes: List[PlaneModel]) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    keep = np.ones((pts.shape[0],), dtype=bool)
    if pts.shape[0] == 0 or not planes:
        return keep
    z_vals = pts[:, 2]
    floor_ref = float(np.quantile(z_vals, 0.15))
    ceil_ref = float(np.quantile(z_vals, 0.90))
    for i, a in enumerate(assignments):
        if a < 0:
            continue
        plane = planes[int(a)]
        nz = abs(float(plane.normal[2]))
        z = float(pts[i, 2])
        if nz > 0.9:
            if z > ceil_ref + 0.05:
                keep[i] = False
        elif nz < 0.3:
            keep[i] = keep[i] and bool(plane.extent_xy >= 0.35)
        else:
            keep[i] = False
    return keep
