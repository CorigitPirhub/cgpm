from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import open3d as o3d

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.data.tum_rgbd import TUMRGBDStream
from data.bonn_rgbd import BonnRGBDStream


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.zeros_like(v)
    return v / n


def save_pointcloud(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd)


def main() -> None:
    ap = argparse.ArgumentParser(description='StaticFusion-family local runnable representative')
    ap.add_argument('--dataset_root', type=str, required=True)
    ap.add_argument('--sequence', type=str, required=True)
    ap.add_argument('--dataset_kind', type=str, default='tum', choices=['tum', 'bonn'])
    ap.add_argument('--out_points', type=str, required=True)
    ap.add_argument('--out_meta', type=str, default='')
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=3000)
    ap.add_argument('--voxel_size', type=float, default=0.02)
    ap.add_argument('--min_hits', type=int, default=2)
    ap.add_argument('--max_normal_dev_deg', type=float, default=35.0)
    ap.add_argument('--max_point_dev_vox', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    cfg = EGF3DConfig()
    seq_dir = Path(args.dataset_root) / args.sequence
    if args.dataset_kind == 'bonn':
        stream = BonnRGBDStream(
            sequence_dir=seq_dir,
            cfg=cfg,
            max_frames=int(args.frames),
            stride=int(args.stride),
            max_points=int(args.max_points_per_frame),
            assoc_max_diff=0.02,
            normal_radius=0.08,
            seed=int(args.seed),
        )
    else:
        stream = TUMRGBDStream(
            sequence_dir=seq_dir,
            cfg=cfg,
            max_frames=int(args.frames),
            stride=int(args.stride),
            max_points=int(args.max_points_per_frame),
            assoc_max_diff=0.02,
            normal_radius=0.08,
            seed=int(args.seed),
        )

    voxel = float(args.voxel_size)
    max_dev = float(args.max_point_dev_vox) * voxel
    cos_min = float(np.cos(np.deg2rad(float(args.max_normal_dev_deg))))

    stats: Dict[Tuple[int, int, int], dict] = {}
    for frame_id, frame in enumerate(stream):
        pts = np.asarray(frame.points_world, dtype=float)
        nrms = np.asarray(frame.normals_world, dtype=float)
        if pts.shape[0] == 0:
            continue
        idxs = np.floor(pts / voxel).astype(np.int32)
        buckets = defaultdict(list)
        for i, idx in enumerate(idxs):
            buckets[(int(idx[0]), int(idx[1]), int(idx[2]))].append(i)
        for key, ids in buckets.items():
            p = np.mean(pts[ids], axis=0)
            n = normalize(np.mean(nrms[ids], axis=0))
            st = stats.get(key)
            if st is None:
                stats[key] = {
                    'count': 1,
                    'conflict': 0,
                    'mean_p': p,
                    'mean_n': n,
                    'last_seen': frame_id,
                }
                continue
            dist = float(np.linalg.norm(p - st['mean_p']))
            cos = float(np.clip(np.dot(n, st['mean_n']), -1.0, 1.0)) if float(np.linalg.norm(st['mean_n'])) > 1e-8 else 1.0
            if dist <= max_dev and cos >= cos_min:
                c = float(st['count'])
                st['mean_p'] = (c * st['mean_p'] + p) / (c + 1.0)
                st['mean_n'] = normalize(c * st['mean_n'] + n)
                st['count'] += 1
            else:
                st['conflict'] += 1
            st['last_seen'] = frame_id

    points = []
    normals = []
    kept = 0
    for st in stats.values():
        if int(st['count']) < int(args.min_hits):
            continue
        if int(st['conflict']) > max(1, int(0.5 * st['count'])):
            continue
        points.append(np.asarray(st['mean_p'], dtype=float))
        normals.append(normalize(np.asarray(st['mean_n'], dtype=float)))
        kept += 1

    out_points = Path(args.out_points)
    pts = np.asarray(points, dtype=float) if points else np.zeros((0, 3), dtype=float)
    nrms = np.asarray(normals, dtype=float) if normals else np.zeros((0, 3), dtype=float)
    save_pointcloud(out_points, pts, nrms)

    meta = {
        'method': 'staticfusion_family',
        'sequence': args.sequence,
        'dataset_kind': args.dataset_kind,
        'frames': int(args.frames),
        'stride': int(args.stride),
        'voxel_size': voxel,
        'min_hits': int(args.min_hits),
        'max_normal_dev_deg': float(args.max_normal_dev_deg),
        'max_point_dev_vox': float(args.max_point_dev_vox),
        'surface_points': int(pts.shape[0]),
        'raw_voxels': int(len(stats)),
    }
    meta_path = Path(args.out_meta) if args.out_meta else out_points.with_name('staticfusion_family_meta.json')
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'[done-staticfusion-family] points={pts.shape[0]} meta={meta_path}')


if __name__ == '__main__':
    main()
