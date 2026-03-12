from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from egf_dhmap3d.core.config import EGF3DConfig


@dataclass
class TUMFrame3D:
    timestamp: float
    rgb_path: Path
    depth_path: Path
    pose_w_c: np.ndarray
    points_cam: np.ndarray
    normals_cam: np.ndarray
    points_world: np.ndarray
    normals_world: np.ndarray
    dt: float


def _parse_tum_text(path: Path) -> List[Tuple[float, List[str]]]:
    out: List[Tuple[float, List[str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            out.append((float(tokens[0]), tokens[1:]))
    return out


def _nearest_index(times: np.ndarray, t: float, max_diff: float) -> int:
    if times.size == 0:
        return -1
    j = int(np.searchsorted(times, t))
    candidates: List[int] = []
    if j < times.size:
        candidates.append(j)
    if j - 1 >= 0:
        candidates.append(j - 1)
    if not candidates:
        return -1
    best = min(candidates, key=lambda k: abs(float(times[k]) - t))
    if abs(float(times[best]) - t) > max_diff:
        return -1
    return int(best)


def _pose_from_tum_entry(values: Sequence[str]) -> np.ndarray:
    # tx ty tz qx qy qz qw
    tx, ty, tz = float(values[0]), float(values[1]), float(values[2])
    qx, qy, qz, qw = float(values[3]), float(values[4]), float(values[5]), float(values[6])
    rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    t_wc = np.eye(4, dtype=float)
    t_wc[:3, :3] = rot
    t_wc[:3, 3] = np.array([tx, ty, tz], dtype=float)
    return t_wc


class TUMRGBDStream:
    def __init__(
        self,
        sequence_dir: str | Path,
        cfg: EGF3DConfig,
        max_frames: int | None = None,
        stride: int = 1,
        max_points: int = 5000,
        assoc_max_diff: float = 0.02,
        normal_radius: float = 0.08,
        normal_max_nn: int = 40,
        seed: int = 42,
        stress_occlusion_ratio: float = 0.0,
        stress_occlusion_mode: str = "moving_band",
        stress_occlusion_axis: str = "x",
    ):
        self.sequence_dir = Path(sequence_dir)
        self.cfg = cfg
        self.max_frames = max_frames
        self.stride = max(1, int(stride))
        self.max_points = max(128, int(max_points))
        self.assoc_max_diff = float(assoc_max_diff)
        self.normal_radius = float(normal_radius)
        self.normal_max_nn = int(normal_max_nn)
        self.rng = np.random.default_rng(seed)
        self.stress_occlusion_ratio = float(np.clip(stress_occlusion_ratio, 0.0, 0.95))
        self.stress_occlusion_mode = str(stress_occlusion_mode).strip().lower()
        self.stress_occlusion_axis = str(stress_occlusion_axis).strip().lower()

        if not self.sequence_dir.exists():
            raise FileNotFoundError(f"TUM sequence dir not found: {self.sequence_dir}")

        rgb_entries = _parse_tum_text(self.sequence_dir / "rgb.txt")
        depth_entries = _parse_tum_text(self.sequence_dir / "depth.txt")
        gt_entries = _parse_tum_text(self.sequence_dir / "groundtruth.txt")
        if not rgb_entries or not depth_entries or not gt_entries:
            raise RuntimeError("TUM sequence is missing rgb/depth/groundtruth entries")

        depth_times = np.array([t for t, _ in depth_entries], dtype=float)
        gt_times = np.array([t for t, _ in gt_entries], dtype=float)

        matches = []
        for t_rgb, rgb_values in rgb_entries:
            depth_idx = _nearest_index(depth_times, t_rgb, self.assoc_max_diff)
            gt_idx = _nearest_index(gt_times, t_rgb, self.assoc_max_diff)
            if depth_idx < 0 or gt_idx < 0:
                continue
            rgb_path = self.sequence_dir / rgb_values[0]
            depth_path = self.sequence_dir / depth_entries[depth_idx][1][0]
            pose_w_c = _pose_from_tum_entry(gt_entries[gt_idx][1])
            if not rgb_path.exists() or not depth_path.exists():
                continue
            matches.append((float(t_rgb), rgb_path, depth_path, pose_w_c))

        if not matches:
            raise RuntimeError("No valid rgb-depth-gt associations found for this sequence")

        matches = matches[:: self.stride]
        if self.max_frames is not None:
            matches = matches[: int(self.max_frames)]
        self.matches = matches

    def __len__(self) -> int:
        return len(self.matches)

    def _apply_occlusion_mask(self, depth: np.ndarray, frame_idx: int) -> np.ndarray:
        if self.stress_occlusion_ratio <= 1e-9:
            return depth
        h, w = int(depth.shape[0]), int(depth.shape[1])
        if h <= 1 or w <= 1:
            return depth

        out = depth.copy()
        axis_x = self.stress_occlusion_axis != "y"
        if axis_x:
            band = int(max(1, round(self.stress_occlusion_ratio * w)))
            if self.stress_occlusion_mode == "fixed_center":
                x0 = (w - band) // 2
            else:
                travel = max(1, w + band)
                x0 = int(((frame_idx * 37) % travel) - band)
            x1 = x0 + band
            x0 = max(0, x0)
            x1 = min(w, x1)
            if x1 > x0:
                out[:, x0:x1] = 0.0
        else:
            band = int(max(1, round(self.stress_occlusion_ratio * h)))
            if self.stress_occlusion_mode == "fixed_center":
                y0 = (h - band) // 2
            else:
                travel = max(1, h + band)
                y0 = int(((frame_idx * 29) % travel) - band)
            y1 = y0 + band
            y0 = max(0, y0)
            y1 = min(h, y1)
            if y1 > y0:
                out[y0:y1, :] = 0.0
        return out

    def _depth_to_points(self, depth_path: Path, frame_idx: int) -> np.ndarray:
        depth = imageio.imread(depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        depth = self._apply_occlusion_mask(depth, frame_idx=frame_idx)

        cam = self.cfg.camera
        z = depth / float(cam.depth_scale)
        mask = (z > cam.depth_min) & (z < cam.depth_max)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return np.zeros((0, 3), dtype=float)

        if xs.size > self.max_points:
            keep = self.rng.choice(xs.size, size=self.max_points, replace=False)
            xs = xs[keep]
            ys = ys[keep]

        zz = z[ys, xs]
        xx = (xs.astype(np.float32) - cam.cx) * zz / cam.fx
        yy = (ys.astype(np.float32) - cam.cy) * zz / cam.fy
        points = np.stack([xx, yy, zz], axis=1)
        return points.astype(float)

    def _estimate_normals(self, points_cam: np.ndarray) -> np.ndarray:
        if points_cam.shape[0] < 32:
            return np.tile(np.array([0.0, 0.0, -1.0], dtype=float), (points_cam.shape[0], 1))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_cam)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn,
            )
        )
        pcd.orient_normals_towards_camera_location(np.zeros(3, dtype=float))
        normals = np.asarray(pcd.normals, dtype=float)
        if normals.shape[0] != points_cam.shape[0]:
            normals = np.tile(np.array([0.0, 0.0, -1.0], dtype=float), (points_cam.shape[0], 1))
        nn = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(nn, 1e-9, None)
        return normals

    def __iter__(self):
        prev_t: float | None = None
        for frame_idx, (timestamp, rgb_path, depth_path, pose_w_c) in enumerate(self.matches):
            points_cam = self._depth_to_points(depth_path, frame_idx=frame_idx)
            if points_cam.shape[0] == 0:
                continue
            normals_cam = self._estimate_normals(points_cam)

            r_wc = pose_w_c[:3, :3]
            t_wc = pose_w_c[:3, 3]
            points_world = (r_wc @ points_cam.T).T + t_wc[None, :]
            normals_world = (r_wc @ normals_cam.T).T
            normals_world = normals_world / np.clip(np.linalg.norm(normals_world, axis=1, keepdims=True), 1e-9, None)

            if prev_t is None:
                dt = 1.0 / 30.0
            else:
                dt = max(1e-3, float(timestamp - prev_t))
            prev_t = float(timestamp)

            yield TUMFrame3D(
                timestamp=float(timestamp),
                rgb_path=rgb_path,
                depth_path=depth_path,
                pose_w_c=pose_w_c.copy(),
                points_cam=points_cam,
                normals_cam=normals_cam,
                points_world=points_world,
                normals_world=normals_world,
                dt=dt,
            )
