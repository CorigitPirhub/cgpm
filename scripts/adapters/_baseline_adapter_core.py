from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.data.tum_rgbd import TUMRGBDStream
from egf_dhmap3d.eval.metrics import compute_reconstruction_metrics
from data.bonn_rgbd import BonnRGBDStream


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    nrm = np.asarray(normals, dtype=float)
    if nrm.size == 0:
        return nrm
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    return nrm / np.clip(norm, 1e-9, None)


def _estimate_normals(points: np.ndarray, radius: float = 0.08, max_nn: int = 40) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    if pts.shape[0] < 16:
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=float), (pts.shape[0], 1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    return _normalize_normals(np.asarray(pcd.normals, dtype=float))


def _load_points_and_normals(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=float)
    if pts.shape[0] > 0:
        normals = _normalize_normals(np.asarray(pcd.normals, dtype=float))
        if normals.shape[0] != pts.shape[0]:
            normals = _estimate_normals(pts)
        return pts, normals

    mesh = o3d.io.read_triangle_mesh(str(path))
    v = np.asarray(mesh.vertices, dtype=float)
    n = _normalize_normals(np.asarray(mesh.vertex_normals, dtype=float))
    if n.shape[0] != v.shape[0]:
        n = _estimate_normals(v)
    return v, n


def _save_pointcloud(path: Path, points: np.ndarray, normals: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals is not None and np.asarray(normals).shape == np.asarray(points).shape:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd)


def _build_reference_points(
    dataset_kind: str,
    sequence_dir: Path,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = EGF3DConfig()
    if dataset_kind == "bonn":
        stream = BonnRGBDStream(
            sequence_dir=sequence_dir,
            cfg=cfg,
            max_frames=frames,
            stride=stride,
            max_points=max_points_per_frame,
            assoc_max_diff=0.02,
            normal_radius=0.08,
            seed=seed,
        )
    else:
        stream = TUMRGBDStream(
            sequence_dir=sequence_dir,
            cfg=cfg,
            max_frames=frames,
            stride=stride,
            max_points=max_points_per_frame,
            assoc_max_diff=0.02,
            normal_radius=0.08,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    refs = []
    norms = []
    for frame in stream:
        pts = frame.points_world
        nrm = frame.normals_world
        if pts.shape[0] > 2500:
            keep = rng.choice(pts.shape[0], size=2500, replace=False)
            pts = pts[keep]
            nrm = nrm[keep]
        refs.append(pts)
        norms.append(nrm)
    if not refs:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    return np.vstack(refs), np.vstack(norms)


def _write_skipped_summary(out_dir: Path, method: str, reason: str, args: argparse.Namespace) -> None:
    summary = {
        "method": method,
        "status": "skipped",
        "reason": reason,
        "sequence": args.sequence,
        "seed": int(args.seed),
        "dataset_kind": args.dataset_kind,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_named_adapter(method_name: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dataset_kind", type=str, default="tum", choices=["tum", "bonn", "auto"])
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_points_per_frame", type=int, default=3000)
    parser.add_argument("--surface_eval_thresh", type=float, default=0.05)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred_points", type=str, default="")
    parser.add_argument("--pred_mesh", type=str, default="")
    parser.add_argument("--runner_cmd", type=str, default="")
    parser.add_argument("--reference_points", type=str, default="")
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    sequence_dir = dataset_root / args.sequence
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

    dataset_kind = args.dataset_kind
    if dataset_kind == "auto":
        dataset_kind = "bonn" if args.sequence.startswith("rgbd_bonn_") else "tum"

    if args.runner_cmd.strip():
        cmd = args.runner_cmd.strip()
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=str(PROJECT_ROOT))
        except subprocess.CalledProcessError as e:
            if args.allow_missing:
                _write_skipped_summary(out_dir, method_name, f"runner_cmd_failed: {e}", args)
                return
            raise

    pred_points_path = Path(args.pred_points) if args.pred_points.strip() else None
    pred_mesh_path = Path(args.pred_mesh) if args.pred_mesh.strip() else None

    src_path = None
    if pred_points_path is not None and pred_points_path.exists():
        src_path = pred_points_path
    elif pred_mesh_path is not None and pred_mesh_path.exists():
        src_path = pred_mesh_path

    if src_path is None:
        msg = "missing prediction source: provide --pred_points or --pred_mesh (or runner_cmd that generates one)."
        if args.allow_missing:
            _write_skipped_summary(out_dir, method_name, msg, args)
            return
        raise FileNotFoundError(msg)

    pred_points, pred_normals = _load_points_and_normals(src_path)

    ref_path = Path(args.reference_points) if args.reference_points.strip() else None
    if ref_path is not None and ref_path.exists():
        ref_points, ref_normals = _load_points_and_normals(ref_path)
    else:
        ref_points, ref_normals = _build_reference_points(
            dataset_kind=dataset_kind,
            sequence_dir=sequence_dir,
            frames=int(args.frames),
            stride=int(args.stride),
            max_points_per_frame=int(args.max_points_per_frame),
            seed=int(args.seed),
        )

    _save_pointcloud(out_dir / "surface_points.ply", pred_points, pred_normals)
    _save_pointcloud(out_dir / "reference_points.ply", ref_points, ref_normals)

    metrics = compute_reconstruction_metrics(
        pred_points=pred_points,
        gt_points=ref_points,
        threshold=float(args.surface_eval_thresh),
        pred_normals=pred_normals,
        gt_normals=ref_normals,
    )

    summary = {
        "method": method_name,
        "status": "ok",
        "sequence": args.sequence,
        "dataset_kind": dataset_kind,
        "frames_used": int(args.frames),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "source_path": str(src_path),
        "surface_points": int(pred_points.shape[0]),
        "reference_points": int(ref_points.shape[0]),
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
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done-{method_name}] summary written: {out_dir / 'summary.json'}")

