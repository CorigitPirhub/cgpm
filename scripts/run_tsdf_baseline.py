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


def points_to_depth_image(points_cam: np.ndarray, cfg: EGF3DConfig) -> np.ndarray:
    cam = cfg.camera
    h, w = cam.height, cam.width
    depth = np.zeros((h, w), dtype=np.float32)
    if points_cam.shape[0] == 0:
        return depth

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    valid = np.isfinite(z) & (z > cam.depth_min) & (z < cam.depth_max)
    if not np.any(valid):
        return depth

    x = x[valid]
    y = y[valid]
    z = z[valid]
    u = np.round(cam.fx * x / z + cam.cx).astype(np.int32)
    v = np.round(cam.fy * y / z + cam.cy).astype(np.int32)
    valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(valid_uv):
        return depth

    u = u[valid_uv]
    v = v[valid_uv]
    z = z[valid_uv]

    flat = v * w + u
    order = np.argsort(z)
    flat = flat[order]
    z = z[order]
    uniq_flat, first_idx = np.unique(flat, return_index=True)
    z_min = z[first_idx]
    vv = uniq_flat // w
    uu = uniq_flat % w
    depth[vv, uu] = z_min.astype(np.float32)
    return depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/tum")
    parser.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg1_xyz")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_points_per_frame", type=int, default=4000)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--sdf_trunc", type=float, default=None)
    parser.add_argument("--surface_eval_thresh", type=float, default=0.05)
    parser.add_argument("--out", type=str, default="output/baseline_compare/freiburg1_xyz/tsdf")
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
    cfg.map3d.truncation = float(args.sdf_trunc) if args.sdf_trunc is not None else max(0.08, 3.0 * cfg.map3d.voxel_size)

    stream = TUMRGBDStream(
        sequence_dir=seq_dir,
        cfg=cfg,
        max_frames=args.frames,
        stride=args.stride,
        max_points=args.max_points_per_frame,
        assoc_max_diff=0.02,
        normal_radius=0.08,
    )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.map3d.voxel_size,
        sdf_trunc=cfg.map3d.truncation,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        cfg.camera.width,
        cfg.camera.height,
        cfg.camera.fx,
        cfg.camera.fy,
        cfg.camera.cx,
        cfg.camera.cy,
    )

    gt_refs: List[np.ndarray] = []
    rng = np.random.default_rng(7)
    total = len(stream)
    print(f"[run-tsdf] sequence={seq_dir.name} frames={total}")
    for i, frame in enumerate(stream):
        depth = points_to_depth_image(frame.points_cam, cfg)
        color = np.zeros((cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8)
        depth_img = o3d.geometry.Image(depth.astype(np.float32))
        color_img = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            depth_scale=1.0,
            depth_trunc=cfg.camera.depth_max,
            convert_rgb_to_intensity=False,
        )
        extrinsic = np.linalg.inv(frame.pose_w_c)
        volume.integrate(rgbd, intrinsic, extrinsic)

        ref = frame.points_world
        if ref.shape[0] > 2500:
            keep = rng.choice(ref.shape[0], size=2500, replace=False)
            ref = ref[keep]
        gt_refs.append(ref)

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == total:
            print(f"  frame={i + 1:04d}/{total:04d}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    pred_points = np.asarray(pcd.points, dtype=float)
    gt_points = np.vstack(gt_refs) if gt_refs else np.zeros((0, 3), dtype=float)
    metrics = compute_reconstruction_metrics(pred_points, gt_points, threshold=args.surface_eval_thresh)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_dir / "surface_mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(out_dir / "surface_points.ply"), pcd)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    o3d.io.write_point_cloud(str(out_dir / "reference_points.ply"), gt_pcd)

    summary = {
        "sequence": seq_dir.name,
        "frames_used": int(total),
        "stride": int(args.stride),
        "voxel_size": float(cfg.map3d.voxel_size),
        "sdf_trunc": float(cfg.map3d.truncation),
        "surface_points": int(pred_points.shape[0]),
        "reference_points": int(gt_points.shape[0]),
        "metrics": {
            "chamfer": float(metrics.chamfer),
            "hausdorff": float(metrics.hausdorff),
            "precision": float(metrics.precision),
            "recall": float(metrics.recall),
            "fscore": float(metrics.fscore),
            "threshold": float(args.surface_eval_thresh),
        },
        "mesh": {
            "vertices": int(np.asarray(mesh.vertices).shape[0]),
            "triangles": int(np.asarray(mesh.triangles).shape[0]),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[done-tsdf] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
