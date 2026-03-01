#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def _parse_tum_text(path: Path) -> List[Tuple[float, List[str]]]:
    out: List[Tuple[float, List[str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 2:
                continue
            out.append((float(toks[0]), toks[1:]))
    return out


def _nearest_index(times: np.ndarray, t: float, max_diff: float) -> int:
    if times.size == 0:
        return -1
    j = int(np.searchsorted(times, t))
    cands: List[int] = []
    if j < times.size:
        cands.append(j)
    if j - 1 >= 0:
        cands.append(j - 1)
    if not cands:
        return -1
    best = min(cands, key=lambda k: abs(float(times[k]) - t))
    if abs(float(times[best]) - t) > max_diff:
        return -1
    return int(best)


def infer_tum_group(sequence_name: str) -> str:
    s = sequence_name.lower()
    if "freiburg1" in s or "fr1" in s:
        return "TUM1"
    if "freiburg2" in s or "fr2" in s:
        return "TUM2"
    return "TUM3"


def build_associations(
    sequence_dir: Path,
    assoc_out: Path,
    frames: int,
    stride: int,
    assoc_max_diff: float,
) -> List[Tuple[float, Path, Path]]:
    rgb_entries = _parse_tum_text(sequence_dir / "rgb.txt")
    depth_entries = _parse_tum_text(sequence_dir / "depth.txt")
    depth_times = np.asarray([t for t, _ in depth_entries], dtype=float)

    matches: List[Tuple[float, Path, Path, float]] = []
    for t_rgb, rgb_vals in rgb_entries:
        di = _nearest_index(depth_times, t_rgb, assoc_max_diff)
        if di < 0:
            continue
        rgb_rel = Path(rgb_vals[0])
        depth_t, depth_vals = depth_entries[di]
        depth_rel = Path(depth_vals[0])
        rgb_p = sequence_dir / rgb_rel
        depth_p = sequence_dir / depth_rel
        if not rgb_p.exists() or not depth_p.exists():
            continue
        matches.append((float(t_rgb), rgb_rel, depth_rel, float(depth_t)))

    matches = matches[:: max(1, int(stride))]
    if int(frames) > 0:
        matches = matches[: int(frames)]

    assoc_out.parent.mkdir(parents=True, exist_ok=True)
    with assoc_out.open("w", encoding="utf-8") as f:
        for t_rgb, rgb_rel, depth_rel, t_depth in matches:
            f.write(f"{t_rgb:.6f} {rgb_rel.as_posix()} {t_depth:.6f} {depth_rel.as_posix()}\n")

    return [(t_rgb, sequence_dir / rgb_rel, sequence_dir / depth_rel) for t_rgb, rgb_rel, depth_rel, _ in matches]


def load_trajectory(path: Path) -> List[Tuple[float, np.ndarray]]:
    traj: List[Tuple[float, np.ndarray]] = []
    if not path.exists():
        return traj
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 8:
                continue
            t = float(toks[0])
            tx, ty, tz = float(toks[1]), float(toks[2]), float(toks[3])
            qx, qy, qz, qw = float(toks[4]), float(toks[5]), float(toks[6]), float(toks[7])
            r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=float)
            T[:3, :3] = r
            T[:3, 3] = np.array([tx, ty, tz], dtype=float)
            traj.append((t, T))
    return traj


def nearest_pose(traj: Sequence[Tuple[float, np.ndarray]], t: float, max_diff: float) -> np.ndarray | None:
    if not traj:
        return None
    times = np.asarray([x[0] for x in traj], dtype=float)
    i = _nearest_index(times, t, max_diff)
    if i < 0:
        return None
    return traj[i][1]


def depth_to_points_cam(
    depth_path: Path,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    depth_min: float,
    depth_max: float,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    depth = imageio.imread(depth_path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    z = depth / float(depth_scale)
    mask = (z > float(depth_min)) & (z < float(depth_max))
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=float)
    if xs.size > int(max_points):
        keep = rng.choice(xs.size, size=int(max_points), replace=False)
        xs = xs[keep]
        ys = ys[keep]
    zz = z[ys, xs]
    xx = (xs.astype(np.float32) - float(cx)) * zz / float(fx)
    yy = (ys.astype(np.float32) - float(cy)) * zz / float(fy)
    return np.stack([xx, yy, zz], axis=1).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DynaSLAM RGB-D on TUM and export fused surface_points.ply")
    ap.add_argument("--sequence_dir", required=True, type=str)
    ap.add_argument("--out_points", required=True, type=str)
    ap.add_argument("--out_meta", type=str, default="")
    ap.add_argument("--dynaslam_root", type=str, default="third_party/DynaSLAM")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--assoc_max_diff", type=float, default=0.02)
    ap.add_argument("--traj_match_max_diff", type=float, default=0.03)
    ap.add_argument("--max_points_per_frame", type=int, default=2500)
    ap.add_argument("--voxel_downsample", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fx", type=float, default=535.4)
    ap.add_argument("--fy", type=float, default=539.2)
    ap.add_argument("--cx", type=float, default=320.1)
    ap.add_argument("--cy", type=float, default=247.6)
    ap.add_argument("--depth_scale", type=float, default=5000.0)
    ap.add_argument("--depth_min", type=float, default=0.2)
    ap.add_argument("--depth_max", type=float, default=4.5)
    ap.add_argument("--skip_run", action="store_true")
    args = ap.parse_args()

    sequence_dir = Path(args.sequence_dir).resolve()
    dynaslam_root = Path(args.dynaslam_root).resolve()
    out_points = Path(args.out_points).resolve()
    out_points.parent.mkdir(parents=True, exist_ok=True)

    run_dir = out_points.parent / "dynaslam_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    assoc_path = run_dir / "associations_eval.txt"

    selected = build_associations(
        sequence_dir=sequence_dir,
        assoc_out=assoc_path,
        frames=int(args.frames),
        stride=int(args.stride),
        assoc_max_diff=float(args.assoc_max_diff),
    )

    if not args.skip_run:
        sequence_name = sequence_dir.name
        tum_group = infer_tum_group(sequence_name)
        settings = dynaslam_root / "Examples" / "RGB-D" / f"{tum_group}.yaml"
        vocab = dynaslam_root / "Vocabulary" / "ORBvoc.txt"
        bin_path = dynaslam_root / "Examples" / "RGB-D" / "rgbd_tum"
        if not bin_path.exists():
            raise FileNotFoundError(f"DynaSLAM binary not found: {bin_path}")
        if not settings.exists():
            raise FileNotFoundError(f"DynaSLAM settings not found: {settings}")

        cmd = [
            str(bin_path),
            str(vocab),
            str(settings),
            str(sequence_dir),
            str(assoc_path),
        ]
        print("[dynaslam-cmd]", " ".join(cmd))
        env = os.environ.copy()
        env["DYNASLAM_REALTIME"] = "0"
        subprocess.run(cmd, check=True, cwd=str(run_dir), env=env)

    traj_path = run_dir / "CameraTrajectory.txt"
    traj = load_trajectory(traj_path)
    if not traj:
        raise RuntimeError(f"No trajectory found at {traj_path}")

    rng = np.random.default_rng(int(args.seed))
    all_pts: List[np.ndarray] = []
    used, missing_pose = 0, 0
    for ts, _rgb, depth_path in selected:
        T = nearest_pose(traj, ts, max_diff=float(args.traj_match_max_diff))
        if T is None:
            missing_pose += 1
            continue
        pc = depth_to_points_cam(
            depth_path=depth_path,
            fx=float(args.fx),
            fy=float(args.fy),
            cx=float(args.cx),
            cy=float(args.cy),
            depth_scale=float(args.depth_scale),
            depth_min=float(args.depth_min),
            depth_max=float(args.depth_max),
            max_points=int(args.max_points_per_frame),
            rng=rng,
        )
        if pc.shape[0] == 0:
            continue
        pw = (T[:3, :3] @ pc.T).T + T[:3, 3]
        all_pts.append(pw)
        used += 1

    if all_pts:
        pts = np.vstack(all_pts)
    else:
        pts = np.zeros((0, 3), dtype=float)

    if pts.shape[0] > 0 and float(args.voxel_downsample) > 0.0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(float(args.voxel_downsample))
        pts = np.asarray(pcd.points, dtype=float)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out_points), pcd)

    meta = {
        "sequence": sequence_dir.name,
        "frames_requested": int(args.frames),
        "stride": int(args.stride),
        "assoc_selected": int(len(selected)),
        "traj_entries": int(len(traj)),
        "used_frames": int(used),
        "missing_pose_frames": int(missing_pose),
        "output_points": int(pts.shape[0]),
        "out_points": str(out_points),
        "trajectory_path": str(traj_path),
        "association_path": str(assoc_path),
    }
    out_meta = Path(args.out_meta).resolve() if args.out_meta else (out_points.parent / "dynaslam_runner_meta.json")
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] points={pts.shape[0]} meta={out_meta}")


if __name__ == "__main__":
    main()
