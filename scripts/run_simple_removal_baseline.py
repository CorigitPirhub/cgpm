from __future__ import annotations

import argparse
import json
import tarfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import open3d as o3d
import requests
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.data.tum_rgbd import TUMRGBDStream
from egf_dhmap3d.eval.metrics import compute_reconstruction_metrics


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


def to_voxel_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(points / float(voxel_size)).astype(np.int32)


def _neighbor_offsets(radius: int) -> List[Tuple[int, int, int]]:
    radius = max(0, int(radius))
    out: List[Tuple[int, int, int]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                out.append((dx, dy, dz))
    return out


def _temporal_support(
    uniq_voxels: np.ndarray,
    history: Sequence[set[Tuple[int, int, int]]],
    offsets: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    if uniq_voxels.shape[0] == 0 or not history:
        return np.zeros((uniq_voxels.shape[0],), dtype=np.float32)
    support = np.zeros((uniq_voxels.shape[0],), dtype=np.float32)
    for i, v in enumerate(uniq_voxels):
        vx, vy, vz = int(v[0]), int(v[1]), int(v[2])
        s = 0.0
        for frame_set in history:
            found = False
            for ox, oy, oz in offsets:
                if (vx + ox, vy + oy, vz + oz) in frame_set:
                    found = True
                    break
            if found:
                s += 1.0
        support[i] = s
    return support


def _estimate_normals(points: np.ndarray, radius: float, max_nn: int = 40) -> np.ndarray:
    if points.shape[0] < 32:
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=float), (points.shape[0], 1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    normals = np.asarray(pcd.normals, dtype=float)
    if normals.shape[0] != points.shape[0]:
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=float), (points.shape[0], 1))
    nn = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(nn, 1e-9, None)


def _save_point_cloud(path: Path, points: np.ndarray, normals: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(float))
    if normals is not None and normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(float))
    o3d.io.write_point_cloud(str(path), pcd)


def _downsample_points(points: np.ndarray, voxel: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ds = pcd.voxel_down_sample(max(1e-3, float(voxel)))
    return np.asarray(ds.points, dtype=float)


def _save_poisson_mesh(
    out_path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    voxel_size: float,
    depth: int,
    min_points: int,
) -> Dict[str, float]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if points.shape[0] < int(min_points):
        _save_point_cloud(out_path, points, normals)
        return {"mode": "pointcloud", "surface_points": float(points.shape[0]), "vertices": 0.0, "triangles": 0.0}

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd = pcd.voxel_down_sample(max(0.5 * float(voxel_size), 0.01))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    densities = np.asarray(densities, dtype=float)
    if densities.size > 0:
        remove_mask = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(remove_mask)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    mesh = mesh.crop(bbox)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    return {
        "mode": "mesh",
        "surface_points": float(points.shape[0]),
        "vertices": float(np.asarray(mesh.vertices).shape[0]),
        "triangles": float(np.asarray(mesh.triangles).shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/tum")
    parser.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg3_walking_xyz")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_points_per_frame", type=int, default=3000)
    parser.add_argument("--surface_eval_thresh", type=float, default=0.05)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--out", type=str, default="output/benchmark_results/rgbd_dataset_freiburg3_walking_xyz/simple_removal")
    parser.add_argument("--temporal_window", type=int, default=6)
    parser.add_argument("--neighbor_cells", type=int, default=1)
    parser.add_argument("--min_temporal_support", type=int, default=2)
    parser.add_argument("--min_lifetime_hits", type=int, default=4)
    parser.add_argument("--warmup_frames", type=int, default=4)
    parser.add_argument("--max_saved_points", type=int, default=600000)
    parser.add_argument("--mesh_poisson_depth", type=int, default=8)
    parser.add_argument("--mesh_min_points", type=int, default=800)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.download:
        seq_dir = download_tum_sequence(dataset_root, args.sequence)
    else:
        seq_dir = dataset_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(f"sequence not found: {seq_dir}")

    cfg = EGF3DConfig()
    cfg.map3d.voxel_size = float(args.voxel_size)
    cfg.map3d.truncation = max(0.08, 3.0 * cfg.map3d.voxel_size)
    stream = TUMRGBDStream(
        sequence_dir=seq_dir,
        cfg=cfg,
        max_frames=args.frames,
        stride=args.stride,
        max_points=args.max_points_per_frame,
        assoc_max_diff=0.02,
        normal_radius=0.08,
    )

    history: deque[set[Tuple[int, int, int]]] = deque(maxlen=max(1, int(args.temporal_window)))
    offsets = _neighbor_offsets(int(args.neighbor_cells))
    lifetime_hits: defaultdict[Tuple[int, int, int], int] = defaultdict(int)

    filtered_points: List[np.ndarray] = []
    gt_refs: List[np.ndarray] = []
    rng = np.random.default_rng(7)
    raw_points_total = 0
    kept_points_total = 0
    support_mean_trace: List[float] = []
    keep_ratio_trace: List[float] = []

    total = len(stream)
    print(f"[run-simple] sequence={seq_dir.name} frames={total}")
    for i, frame in enumerate(stream):
        points = frame.points_world
        raw_points_total += int(points.shape[0])
        if points.shape[0] == 0:
            continue

        voxel_idx = to_voxel_indices(points, cfg.map3d.voxel_size)
        uniq_voxels, inv = np.unique(voxel_idx, axis=0, return_inverse=True)
        support_per_uniq = _temporal_support(uniq_voxels, list(history), offsets)
        support_per_point = support_per_uniq[inv]
        hit_per_uniq = np.array(
            [float(lifetime_hits[(int(v[0]), int(v[1]), int(v[2]))]) for v in uniq_voxels],
            dtype=np.float32,
        )
        hit_per_point = hit_per_uniq[inv]

        keep_mask = (support_per_point >= float(args.min_temporal_support)) | (hit_per_point >= float(args.min_lifetime_hits))
        if i < int(args.warmup_frames):
            keep_mask[:] = True

        kept = points[keep_mask]
        kept_points_total += int(kept.shape[0])
        support_mean_trace.append(float(np.mean(support_per_point)) if support_per_point.size > 0 else 0.0)
        keep_ratio_trace.append(float(kept.shape[0]) / float(points.shape[0]))
        filtered_points.append(kept)

        for v in uniq_voxels:
            key = (int(v[0]), int(v[1]), int(v[2]))
            lifetime_hits[key] += 1
        history.append({(int(v[0]), int(v[1]), int(v[2])) for v in uniq_voxels})

        ref = points
        if ref.shape[0] > 2500:
            keep = rng.choice(ref.shape[0], size=2500, replace=False)
            ref = ref[keep]
        gt_refs.append(ref)

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == total:
            print(
                f"  frame={i + 1:04d}/{total:04d} "
                f"keep={keep_ratio_trace[-1]:.3f} "
                f"support={support_mean_trace[-1]:.2f}"
            )

    pred_points = np.vstack(filtered_points) if filtered_points else np.zeros((0, 3), dtype=float)
    if pred_points.shape[0] > int(args.max_saved_points):
        keep = rng.choice(pred_points.shape[0], size=int(args.max_saved_points), replace=False)
        pred_points = pred_points[keep]
    pred_points = _downsample_points(pred_points, voxel=max(0.5 * cfg.map3d.voxel_size, 0.01))
    pred_normals = _estimate_normals(pred_points, radius=max(3.0 * cfg.map3d.voxel_size, 0.05))

    gt_points = np.vstack(gt_refs) if gt_refs else np.zeros((0, 3), dtype=float)
    metrics = compute_reconstruction_metrics(pred_points, gt_points, threshold=args.surface_eval_thresh)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_point_cloud(out_dir / "surface_points.ply", pred_points, pred_normals)
    _save_point_cloud(out_dir / "reference_points.ply", gt_points, None)
    mesh_info = _save_poisson_mesh(
        out_path=out_dir / "surface_mesh.ply",
        points=pred_points,
        normals=pred_normals,
        voxel_size=cfg.map3d.voxel_size,
        depth=int(args.mesh_poisson_depth),
        min_points=int(args.mesh_min_points),
    )

    summary = {
        "sequence": seq_dir.name,
        "frames_used": int(total),
        "stride": int(args.stride),
        "voxel_size": float(cfg.map3d.voxel_size),
        "surface_points": int(pred_points.shape[0]),
        "reference_points": int(gt_points.shape[0]),
        "filtering": {
            "raw_points_total": int(raw_points_total),
            "kept_points_total": int(kept_points_total),
            "keep_ratio_global": float(kept_points_total) / max(1.0, float(raw_points_total)),
            "keep_ratio_mean": float(np.mean(keep_ratio_trace)) if keep_ratio_trace else 0.0,
            "support_mean": float(np.mean(support_mean_trace)) if support_mean_trace else 0.0,
            "temporal_window": int(args.temporal_window),
            "neighbor_cells": int(args.neighbor_cells),
            "min_temporal_support": int(args.min_temporal_support),
            "min_lifetime_hits": int(args.min_lifetime_hits),
        },
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
    print("[done-simple] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
