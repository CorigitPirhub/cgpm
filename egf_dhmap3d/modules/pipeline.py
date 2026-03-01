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
        self.frame_counter: int = 0
        self.prev_cam_pcd: o3d.geometry.PointCloud | None = None
        self.prev_rgbd: o3d.geometry.RGBDImage | None = None
        cam = self.cfg.camera
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(cam.width),
            int(cam.height),
            float(cam.fx),
            float(cam.fy),
            float(cam.cx),
            float(cam.cy),
        )
        self.last_delta_cam: np.ndarray = np.eye(4, dtype=float)
        self.odom_valid_ratio: float = 0.0

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

    @staticmethod
    def _delta_stats(delta: np.ndarray) -> Tuple[float, float]:
        t = float(np.linalg.norm(delta[:3, 3]))
        tr = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
        r = float(np.degrees(np.arccos(tr)))
        return t, r

    def _build_cam_pcd(self, points_cam: np.ndarray, normals_cam: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points_cam, dtype=float))
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals_cam, dtype=float))
        voxel = float(max(1e-3, self.cfg.predict.icp_voxel_size))
        if voxel > 1e-6:
            pcd = pcd.voxel_down_sample(voxel)
            if len(pcd.points) >= 16:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=max(0.04, 2.0 * voxel),
                        max_nn=30,
                    )
                )
                pcd.orient_normals_towards_camera_location(np.zeros(3, dtype=float))
        return pcd

    def _load_rgbd(self, frame: TUMFrame3D) -> o3d.geometry.RGBDImage:
        color = o3d.io.read_image(str(Path(frame.rgb_path)))
        depth = o3d.io.read_image(str(Path(frame.depth_path)))
        cam = self.cfg.camera
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=float(cam.depth_scale),
            depth_trunc=float(cam.depth_max),
            convert_rgb_to_intensity=False,
        )

    def _delta_from_rgbd(self, frame: TUMFrame3D) -> Tuple[np.ndarray, Dict[str, float]]:
        curr = self._load_rgbd(frame)
        if self.prev_rgbd is None:
            self.prev_rgbd = curr
            return np.eye(4, dtype=float), {"odom_fitness": 1.0, "odom_rmse": 0.0, "odom_valid": 1.0, "odom_trans_norm": 0.0, "odom_rot_deg": 0.0, "odom_source": 2.0}

        jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        option = o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = float(max(0.03, self.cfg.predict.icp_max_corr))
        success, delta, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            self.prev_rgbd,
            curr,
            self.intrinsic,
            np.eye(4, dtype=float),
            jac,
            option,
        )
        delta = np.asarray(delta, dtype=float)
        trans_norm, rot_deg = self._delta_stats(delta)
        info_trace = float(np.trace(np.asarray(info, dtype=float))) if info is not None else 0.0
        # Convert information trace to a bounded confidence proxy [0, 1].
        fitness = float(np.clip(info_trace / 12000.0, 0.0, 1.0))
        valid = bool(
            success
            and np.all(np.isfinite(delta))
            and trans_norm <= float(self.cfg.predict.icp_max_trans_step)
            and rot_deg <= float(self.cfg.predict.icp_max_rot_deg_step)
        )
        # Open3D odometry returns source->target (prev->curr), which matches
        # predictor pose propagation convention (T_w_c,new = T_w_c,prev @ delta_prev_curr).
        # Do not invert here; inverting introduces systematic trajectory drift.
        trans_norm, rot_deg = self._delta_stats(delta)
        self.prev_rgbd = curr
        return delta, {
            "odom_fitness": fitness,
            "odom_rmse": 0.0,
            "odom_valid": 1.0 if valid else 0.0,
            "odom_trans_norm": trans_norm,
            "odom_rot_deg": rot_deg,
            "odom_source": 2.0,
        }

    def _delta_from_icp(self, points_cam: np.ndarray, normals_cam: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        curr_pcd = self._build_cam_pcd(points_cam, normals_cam)
        if self.prev_cam_pcd is None:
            self.prev_cam_pcd = curr_pcd
            return np.eye(4, dtype=float), {"odom_fitness": 1.0, "odom_rmse": 0.0, "odom_valid": 1.0, "odom_source": 1.0}

        prev_pcd = self.prev_cam_pcd
        if len(prev_pcd.points) < 16 or len(curr_pcd.points) < 16:
            self.prev_cam_pcd = curr_pcd
            return self.last_delta_cam.copy(), {"odom_fitness": 0.0, "odom_rmse": float("inf"), "odom_valid": 0.0, "odom_source": 1.0}

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(self.cfg.predict.icp_max_iters))
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        reg = o3d.pipelines.registration.registration_icp(
            prev_pcd,
            curr_pcd,
            float(max(1e-3, self.cfg.predict.icp_max_corr)),
            np.eye(4, dtype=float),
            estimation,
            criteria,
        )
        delta = np.asarray(reg.transformation, dtype=float)
        fitness = float(reg.fitness)
        rmse = float(reg.inlier_rmse)
        trans_norm, rot_deg = self._delta_stats(delta)
        valid = bool(
            np.all(np.isfinite(delta))
            and fitness >= float(self.cfg.predict.icp_min_fitness)
            and rmse <= float(self.cfg.predict.icp_max_rmse)
            and trans_norm <= float(self.cfg.predict.icp_max_trans_step)
            and rot_deg <= float(self.cfg.predict.icp_max_rot_deg_step)
        )
        if valid:
            delta_pose = delta.copy()
            self.last_delta_cam = delta_pose.copy()
        else:
            delta_pose = self.last_delta_cam.copy()
        trans_norm, rot_deg = self._delta_stats(delta_pose)

        self.prev_cam_pcd = curr_pcd
        return delta_pose, {
            "odom_fitness": fitness,
            "odom_rmse": rmse,
            "odom_valid": 1.0 if valid else 0.0,
            "odom_trans_norm": trans_norm,
            "odom_rot_deg": rot_deg,
            "odom_source": 1.0,
        }

    def _refine_pose_with_map(
        self,
        points_cam: np.ndarray,
        normals_cam: np.ndarray,
    ) -> Dict[str, float]:
        # Periodic map-to-frame ICP correction to suppress long-horizon drift.
        if self.frame_counter < 20 or (self.frame_counter % 5) != 0:
            return {"map_refine_applied": 0.0, "map_refine_fitness": 0.0, "map_refine_rmse": 0.0}
        if len(self.voxel_map) < 5000:
            return {"map_refine_applied": 0.0, "map_refine_fitness": 0.0, "map_refine_rmse": 0.0}

        map_pts, _ = self.voxel_map.extract_surface_points(
            phi_thresh=max(0.10, 2.5 * self.cfg.map3d.voxel_size),
            rho_thresh=max(0.05, self.cfg.surface.rho_thresh),
            min_weight=max(0.4, 0.25 * self.cfg.surface.min_weight),
            current_step=int(self.frame_counter),
            max_age_frames=int(min(max(30, self.cfg.surface.max_age_frames), 240)),
            max_d_score=min(0.75, self.cfg.surface.max_d_score + 0.15),
            max_free_ratio=max(0.9, self.cfg.surface.max_free_ratio),
            prune_free_min=self.cfg.surface.prune_free_min,
            prune_residual_min=self.cfg.surface.prune_residual_min,
            max_clear_hits=self.cfg.surface.max_clear_hits,
        )
        if map_pts.shape[0] < 1200:
            return {"map_refine_applied": 0.0, "map_refine_fitness": 0.0, "map_refine_rmse": 0.0}

        points_world, normals_world = self._transform_from_current_pose(points_cam, normals_cam)
        src = self._build_cam_pcd(points_world, normals_world)
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(np.asarray(map_pts, dtype=float))
        refine_voxel = float(max(0.05, 1.25 * self.cfg.predict.icp_voxel_size))
        src = src.voxel_down_sample(refine_voxel)
        tgt = tgt.voxel_down_sample(refine_voxel)
        if len(src.points) < 120 or len(tgt.points) < 300:
            return {"map_refine_applied": 0.0, "map_refine_fitness": 0.0, "map_refine_rmse": 0.0}

        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            float(max(0.12, 3.0 * self.cfg.predict.icp_max_corr)),
            np.eye(4, dtype=float),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25),
        )
        t_corr = np.asarray(reg.transformation, dtype=float)
        tr = float(np.linalg.norm(t_corr[:3, 3]))
        rot = self._delta_stats(t_corr)[1]
        fit = float(reg.fitness)
        rmse = float(reg.inlier_rmse)
        valid = bool(
            np.all(np.isfinite(t_corr))
            and fit >= 0.18
            and rmse <= 0.12
            and tr <= 0.18
            and rot <= 8.0
        )
        if not valid:
            return {"map_refine_applied": 0.0, "map_refine_fitness": fit, "map_refine_rmse": rmse}

        t_wc = self.predictor.pose.as_matrix()
        self.predictor.set_pose(t_corr @ t_wc)
        return {"map_refine_applied": 1.0, "map_refine_fitness": fit, "map_refine_rmse": rmse}

    def step(self, frame: TUMFrame3D, use_gt_pose: bool = True) -> Dict[str, float]:
        odom_stats: Dict[str, float] = {}
        if use_gt_pose:
            delta = self._delta_from_gt(frame.pose_w_c)
            self.predictor.predict(delta_t=delta, dt=frame.dt)
            self.predictor.set_pose(frame.pose_w_c)
            self.prev_cam_pcd = None
            self.prev_rgbd = None
            self.last_delta_cam = np.eye(4, dtype=float)
            odom_stats = {"odom_fitness": 1.0, "odom_rmse": 0.0, "odom_valid": 1.0, "odom_trans_norm": 0.0, "odom_rot_deg": 0.0, "odom_source": 0.0}
        else:
            if not self.trajectory and bool(self.cfg.predict.slam_anchor_with_first_gt):
                # SLAM mode uses a known start pose anchor; subsequent motion is pure ICP odometry.
                self.predictor.set_pose(frame.pose_w_c)
            if bool(self.cfg.predict.slam_use_gt_delta_odom):
                delta = self._delta_from_gt(frame.pose_w_c)
                d_t, d_r = self._delta_stats(delta)
                odom_stats = {
                    "odom_fitness": 1.0,
                    "odom_rmse": 0.0,
                    "odom_valid": 1.0,
                    "odom_trans_norm": d_t,
                    "odom_rot_deg": d_r,
                    "odom_source": 3.0,
                }
            else:
                # Use RGB-D odometry as primary front-end and fall back to geometric ICP
                # when RGB-D estimation is invalid.
                delta, odom_stats = self._delta_from_rgbd(frame)
                if odom_stats.get("odom_valid", 0.0) < 0.5:
                    delta, odom_stats = self._delta_from_icp(frame.points_cam, frame.normals_cam)
            self.predictor.predict(delta_t=delta, dt=frame.dt)

        self.predictor.apply_field_prediction(self.voxel_map, dynamic_score=self.dynamic_score)
        if (not use_gt_pose) and (not bool(self.cfg.predict.slam_use_gt_delta_odom)):
            odom_stats.update(self._refine_pose_with_map(frame.points_cam, frame.normals_cam))

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
        update_stats = self.updater.update(
            self.voxel_map,
            accepted,
            rejected,
            sensor_origin=sensor_origin,
            frame_id=self.frame_counter,
        )

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
        self.odom_valid_ratio = float((1.0 - alpha) * self.odom_valid_ratio + alpha * odom_stats.get("odom_valid", 0.0))

        self.trajectory.append(self.predictor.pose.as_matrix())
        self.frame_counter += 1
        out = {}
        out.update(assoc_stats)
        out.update(update_stats)
        out.update(odom_stats)
        out["odom_valid_ratio"] = float(self.odom_valid_ratio)
        out["rejected"] = float(len(rejected))
        out["dynamic_score"] = float(self.dynamic_score)
        return out

    def extract_surface_points(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.voxel_map.extract_surface_points(
            phi_thresh=self.cfg.surface.phi_thresh,
            rho_thresh=self.cfg.surface.rho_thresh,
            min_weight=self.cfg.surface.min_weight,
            current_step=int(self.frame_counter),
            max_age_frames=int(self.cfg.surface.max_age_frames),
            max_d_score=self.cfg.surface.max_d_score,
            max_free_ratio=self.cfg.surface.max_free_ratio,
            prune_free_min=self.cfg.surface.prune_free_min,
            prune_residual_min=self.cfg.surface.prune_residual_min,
            max_clear_hits=self.cfg.surface.max_clear_hits,
            use_zero_crossing=bool(self.cfg.surface.use_zero_crossing),
            zero_crossing_max_offset=float(self.cfg.surface.zero_crossing_max_offset),
            zero_crossing_phi_gate=float(self.cfg.surface.zero_crossing_phi_gate),
            consistency_enable=bool(self.cfg.surface.consistency_enable),
            consistency_radius=int(self.cfg.surface.consistency_radius),
            consistency_min_neighbors=int(self.cfg.surface.consistency_min_neighbors),
            consistency_normal_cos=float(self.cfg.surface.consistency_normal_cos),
            consistency_phi_diff=float(self.cfg.surface.consistency_phi_diff),
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
