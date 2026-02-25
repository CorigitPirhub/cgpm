from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import open3d as o3d

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.core.voxel_hash import VoxelHashMap3D
from egf_dhmap3d.data.tum_rgbd import TUMFrame3D
from egf_dhmap3d.modules.associator import Associator3D
from egf_dhmap3d.modules.predictor import Predictor3D
from egf_dhmap3d.modules.updater import Updater3D


class EGFDHMap3D:
    def __init__(self, cfg: EGF3DConfig):
        self.cfg = cfg
        self.voxel_map = VoxelHashMap3D(cfg)
        self.predictor = Predictor3D(cfg)
        self.associator = Associator3D(cfg)
        self.updater = Updater3D(cfg)
        self.prev_gt_pose: np.ndarray | None = None
        self.trajectory: list[np.ndarray] = []
        self.dynamic_score: float = 0.0

    def _delta_from_gt(self, t_wc: np.ndarray) -> np.ndarray:
        if self.prev_gt_pose is None:
            self.prev_gt_pose = t_wc.copy()
            return np.eye(4, dtype=float)
        delta = np.linalg.inv(self.prev_gt_pose) @ t_wc
        self.prev_gt_pose = t_wc.copy()
        return delta

    def _transform_from_current_pose(
        self,
        points_cam: np.ndarray,
        normals_cam: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        t_wc = self.predictor.pose.as_matrix()
        r_wc = t_wc[:3, :3]
        t = t_wc[:3, 3]
        points_world = (r_wc @ points_cam.T).T + t[None, :]
        normals_world = (r_wc @ normals_cam.T).T
        normals_world = normals_world / np.clip(np.linalg.norm(normals_world, axis=1, keepdims=True), 1e-9, None)
        return points_world, normals_world

    def step(self, frame: TUMFrame3D, use_gt_pose: bool = True) -> Dict[str, float]:
        if use_gt_pose:
            delta = self._delta_from_gt(frame.pose_w_c)
            self.predictor.predict(delta_t=delta, dt=frame.dt)
            self.predictor.set_pose(frame.pose_w_c)
        else:
            self.predictor.predict(delta_t=None, dt=frame.dt)

        self.predictor.apply_field_prediction(self.voxel_map, dynamic_score=self.dynamic_score)

        if use_gt_pose:
            points_world = frame.points_world
            normals_world = frame.normals_world
        else:
            points_world, normals_world = self._transform_from_current_pose(frame.points_cam, frame.normals_cam)

        accepted, rejected, assoc_stats = self.associator.associate(self.voxel_map, points_world, normals_world)
        if use_gt_pose:
            sensor_origin = frame.pose_w_c[:3, 3]
        else:
            sensor_origin = self.predictor.pose.as_matrix()[:3, 3]
        update_stats = self.updater.update(self.voxel_map, accepted, rejected, sensor_origin=sensor_origin)

        # Global dynamic score (used only in legacy global forgetting mode + debug).
        alpha = float(np.clip(self.cfg.update.dyn_score_alpha, 0.01, 0.5))
        d2_ref = float(max(1e-6, self.cfg.update.dyn_d2_ref))
        score_obs = float(np.clip(assoc_stats.get("mean_d2", 0.0) / d2_ref, 0.0, 1.0))
        n_acc = float(assoc_stats.get("accepted", 0.0))
        n_rej = float(assoc_stats.get("rejected", 0.0))
        rej_ratio = n_rej / max(1.0, n_acc + n_rej)
        # In sparse voxel maps, raw reject ratio is naturally high; only use its extreme tail for dynamic cues.
        rej_tail = float(np.clip((rej_ratio - 0.985) / 0.015, 0.0, 1.0))
        target = 0.82 * score_obs + 0.18 * rej_tail
        self.dynamic_score = float((1.0 - alpha) * self.dynamic_score + alpha * target)
        self.dynamic_score = float(np.clip(self.dynamic_score, 0.0, 1.0))

        self.trajectory.append(self.predictor.pose.as_matrix())
        out = {}
        out.update(assoc_stats)
        out.update(update_stats)
        out["rejected"] = float(len(rejected))
        out["dynamic_score"] = float(self.dynamic_score)
        return out

    def extract_surface_points(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.voxel_map.extract_surface_points(
            phi_thresh=self.cfg.surface.phi_thresh,
            rho_thresh=self.cfg.surface.rho_thresh,
            min_weight=self.cfg.surface.min_weight,
            max_d_score=self.cfg.surface.max_d_score,
            max_free_ratio=self.cfg.surface.max_free_ratio,
            prune_free_min=self.cfg.surface.prune_free_min,
            prune_residual_min=self.cfg.surface.prune_residual_min,
            max_clear_hits=self.cfg.surface.max_clear_hits,
        )

    def save_surface_pointcloud(self, out_path: str | Path) -> int:
        points, normals = self.extract_surface_points()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals.shape[0] == points.shape[0]:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_path), pcd)
        return int(points.shape[0])

    def save_poisson_mesh(self, out_path: str | Path, min_points: int = 800) -> Dict[str, float]:
        points, normals = self.extract_surface_points()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if points.shape[0] < min_points:
            self.save_surface_pointcloud(out_path)
            return {"mode": "pointcloud", "surface_points": float(points.shape[0]), "vertices": 0.0, "triangles": 0.0}

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd = pcd.voxel_down_sample(max(0.5 * self.cfg.map3d.voxel_size, 0.01))

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=int(self.cfg.surface.poisson_depth),
        )
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

    def get_trajectory(self) -> np.ndarray:
        if not self.trajectory:
            return np.zeros((0, 4, 4), dtype=float)
        return np.asarray(self.trajectory, dtype=float)

    def export_dynamic_voxel_map(self, min_phi_w: float = 0.2):
        return self.voxel_map.export_voxel_arrays(min_phi_w=min_phi_w)
