import numpy as np
from typing import Dict, List, Optional, Tuple

from entity.entity import Entity
from entity.evidence import EvidencePoint
from .projection import ProjectionOperator, PolylineProjectionOperator
import time

class AssociationResult:
    def __init__(self, obs_index: int, entity_index: int, mahalanobis_sq: float, s_star: float, z_pred: np.ndarray):
        self.obs_index = obs_index
        self.entity_index = entity_index
        self.mahalanobis_sq = mahalanobis_sq
        self.s_star = s_star
        self.z_pred = z_pred

    def __repr__(self):
        return f"Assoc(obs={self.obs_index}->ent={self.entity_index}, d2={self.mahalanobis_sq:.2f})"

# TODO: 动态门限、自适应观测噪声等策略
class Associator:
    def __init__(
        self,
        chi2_threshold: float = 0.99,
        obs_sigma: float = 0.05,
        projector: Optional[ProjectionOperator] = None,
    ):
        self.chi2_threshold = chi2_threshold
        self.obs_sigma = obs_sigma
        self.projector = projector if projector is not None else ProjectionOperator()

    # 最近邻贪婪 + 马氏距离门限
    def process(self, observations: np.ndarray, entities: List[Entity], timestamp: float = 0.0) -> List[AssociationResult]:
        if observations.shape[1] == 0 or not entities:
            return []

        candidates: List[AssociationResult] = []

        # 遍历每个观测点
        for obs_idx in range(observations.shape[1]):
            z_obs = observations[:, obs_idx]  # 观测点的全局坐标
            best: Optional[AssociationResult] = None
            
            # Step 1: 矢量化粗筛
            coarse_list = []
            for ent_idx, ent in enumerate(entities):
                # 转换到局部坐标系
                g = ent.pose.to_matrix()
                R_T = g[:2, :2].T
                t = g[:2, 2]
                z_local = R_T @ (z_obs - t)
                
                # 矢量化粗搜索 (极快)
                s_c, d2_c = self.projector.coarse_search(ent, z_local)
                coarse_list.append((d2_c, ent_idx, ent, z_local, s_c))
            
            # 按欧氏距离排序，选出前 3 个候选进行精修
            coarse_list.sort(key=lambda x: x[0])
            
            # Step 2: 稀疏精筛 (Top-K)
            for i in range(min(len(coarse_list), 3)):
                _, ent_idx, ent, z_local, s_coarse = coarse_list[i]
                
                # 牛顿法精优化
                s_star, p_star_local = self.projector.newton_refine(ent, s_coarse, z_local)
                
                # 转换回全局
                g = ent.pose.to_matrix()
                z_pred_global = (g @ np.append(p_star_local, 1.0))[:2]
                
                residual = z_obs - z_pred_global

                # 计算不确定性
                sigma_total = self._compute_total_cov(ent, p_star_local)
                
                # 计算马氏距离 (法向误差分解)
                # 1. 切线方向
                tangent_local = ent.model.derivative(s_star)
                if np.linalg.norm(tangent_local) < 1e-9:
                    tangent_local = np.array([1.0, 0.0])

                # 2. 全局切线方向
                theta = ent.pose.theta
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])
                tangent_global = R @ tangent_local
                tangent_global /= (np.linalg.norm(tangent_global) + 1e-9)

                # 3. 法向量（旋转90度）
                n_vec = np.array([-tangent_global[1], tangent_global[0]])

                # 4. 投影到法向量方向的残差和协方差
                sigma_n = float(n_vec.T @ sigma_total @ n_vec)
                res_n = float(np.dot(residual, n_vec))
                
                # 避免数值问题
                if sigma_n < 1e-9:
                    continue

                # 5. 计算平方马氏距离
                d2 = (res_n ** 2) / sigma_n
                
                # 门限判断与贪婪更新
                if d2 < self.chi2_threshold:
                    candidate = AssociationResult(obs_idx, ent_idx, d2, s_star, z_pred_global)
                    if best is None or d2 < best.mahalanobis_sq:
                        best = candidate

            # 如果找到了匹配，添加证据到实体
            if best is not None:
                candidates.append(best)
                ent = entities[best.entity_index]
                
                # 计算权重：基于观测噪声
                # 注意：这里也可以根据马氏距离动态衰减权重
                weight = 1.0 / (self.obs_sigma ** 2)
                
                ev = EvidencePoint(
                    s=best.s_star,
                    z_global=observations[:, best.obs_index],
                    weight=weight,
                    timestamp=timestamp,
                    sigma=self.obs_sigma,
                )
                ent.evidence.add(ev)

        return candidates

    # --------- Helpers ---------

    def _compute_total_cov(self, entity: Entity, p_local: np.ndarray) -> np.ndarray:
        # 1. 观测噪声协方差
        sigma_obs = (self.obs_sigma ** 2) * np.eye(2)

        # 2. 位姿协方差 (取前3维)
        sigma_g = entity.covariance.matrix
        if sigma_g.shape[0] > 3:
            sigma_g = sigma_g[:3, :3]

        # 3. 计算 Jacobian: dz_global / d_pose
        theta = entity.pose.theta
        c = np.cos(theta)
        s = np.sin(theta)
        px, py = p_local[0], p_local[1]
        
        # z_global = R * p_local + t
        # d/dx = I
        # d/dy = I
        # d/dtheta = R' * p_local = [-sin, -cos; cos, -sin] * [px; py]
        J_pose = np.array([
            [1.0, 0.0, -s * px - c * py],
            [0.0, 1.0,  c * px - s * py],
        ])

        # 4. 总协方差：Sigma_total = Sigma_obs + J * Sigma_pose * J.T
        sigma_total = sigma_obs + J_pose @ sigma_g @ J_pose.T
        return sigma_total


class PolylineAssociator:
    def __init__(
        self,
        chi2_threshold: float = 0.99,
        obs_sigma: float = 0.05,
        epsilon_max_ext: float = 0.5,
        extension_residual_threshold: float = 0.5,
        reuse_threshold: float = 0.05,
        newton_max_iter: int = 5,
        curvature_threshold: float = 0.05,
        local_search_ratio: float = 0.1,
        local_search_samples: int = 7,
        fallback_residual_threshold: float = 0.2,
        projector: Optional[PolylineProjectionOperator] = None,
        top_k: int = 3,
        tangential_weight: float = 0.15,
        tangential_sigma_scale: float = 4.0,
        boundary_s_ratio: float = 0.08,
        boundary_proximity_ratio: float = 0.12,
        boundary_lateral_ratio: float = 0.35,
        boundary_min_outward: float = 0.02,
        boundary_override_margin: float = 0.02,
        boundary_seed_nearest_margin: float = 0.08,
        prefit_min_points: int = 3,
        prefit_back_tolerance: float = 0.08,
        prefit_lateral_base: float = 0.10,
        prefit_lateral_growth: float = 0.35,
        prefit_residual_scale: float = 1.25,
        prefit_max_range_margin: float = 0.6,
        prefit_min_progress: float = 0.05,
    ):
        self.chi2_threshold = chi2_threshold
        self.obs_sigma = obs_sigma
        self.epsilon_max_ext = epsilon_max_ext
        self.extension_residual_threshold = extension_residual_threshold
        self.top_k = top_k
        self.tangential_weight = float(max(0.0, tangential_weight))
        self.tangential_sigma_scale = float(max(1.0, tangential_sigma_scale))
        self.boundary_s_ratio = float(max(0.0, boundary_s_ratio))
        self.boundary_proximity_ratio = float(max(0.0, boundary_proximity_ratio))
        self.boundary_lateral_ratio = float(max(0.1, boundary_lateral_ratio))
        self.boundary_min_outward = float(max(0.0, boundary_min_outward))
        self.boundary_override_margin = float(max(0.0, boundary_override_margin))
        self.boundary_seed_nearest_margin = float(max(0.0, boundary_seed_nearest_margin))
        self.prefit_min_points = int(max(2, prefit_min_points))
        self.prefit_back_tolerance = float(max(0.0, prefit_back_tolerance))
        self.prefit_lateral_base = float(max(0.01, prefit_lateral_base))
        self.prefit_lateral_growth = float(max(0.0, prefit_lateral_growth))
        self.prefit_residual_scale = float(max(1.0, prefit_residual_scale))
        self.prefit_max_range_margin = float(max(0.0, prefit_max_range_margin))
        self.prefit_min_progress = float(max(0.0, prefit_min_progress))
        self.prefit_tail_local_window = 0.25
        self.projector = projector if projector is not None else PolylineProjectionOperator(
            reuse_threshold=reuse_threshold,
            newton_max_iter=newton_max_iter,
            fallback_residual_threshold=fallback_residual_threshold,
            boundary_min_outward=boundary_min_outward,
            boundary_override_margin=boundary_override_margin,
        )

    def process(
        self,
        observations: np.ndarray,
        entities: List[Entity],
        timestamp: float = 0.0,
        sensor_origin: Optional[np.ndarray] = None,
    ) -> Tuple[List[AssociationResult], List[Dict[str, object]]]:
        if observations.shape[1] == 0 or not entities:
            return [], []

        if sensor_origin is None:
            sensor_origin = np.zeros(2)
        sensor_origin = np.array(sensor_origin, dtype=float).reshape(2,)

        candidates: List[AssociationResult] = []
        events: List[Dict[str, object]] = []

        for obs_idx in range(observations.shape[1]):
            z_obs = observations[:, obs_idx]
            best: Optional[AssociationResult] = None
            ray_direction_global = z_obs - sensor_origin
            if np.linalg.norm(ray_direction_global) < 1e-9:
                ray_direction_global = None

            coarse_list = []
            for ent_idx, ent in enumerate(entities):
                g = ent.pose.to_matrix()
                R_T = g[:2, :2].T
                t = g[:2, 2]
                z_local = R_T @ (z_obs - t)
                s_c, _, d2_c = self.projector.coarse_search(ent, z_local)
                coarse_list.append((d2_c, s_c, ent_idx, ent))

            coarse_list.sort(key=lambda x: x[0])

            for i in range(min(len(coarse_list), self.top_k)):
                _, s_coarse, ent_idx, ent = coarse_list[i]
                s_clamped, s_raw, p_star_local, z_pred_global, _, _ = self.projector.project(
                    ent,
                    z_obs,
                    ray_direction_global=ray_direction_global,
                    s_init=s_coarse,
                )

                L = float(ent.model.domain_limit)
                if s_raw < -self.epsilon_max_ext or s_raw > (L + self.epsilon_max_ext):
                    continue

                residual = z_obs - z_pred_global
                residual_norm = float(np.linalg.norm(residual))
                boundary_vote = self._boundary_vote(
                    ent,
                    z_obs,
                    s_raw,
                    residual_norm,
                    ray_direction_global,
                )

                if s_raw < 0.0 or s_raw > L:
                    if residual_norm <= self.extension_residual_threshold:
                        side = "head" if s_raw < 0.0 else "tail"
                        s_cache = float(s_raw)
                        if boundary_vote is not None and boundary_vote["side"] == side:
                            s_cache = float(boundary_vote["s_ext"])
                        weight = 1.0 / (self.obs_sigma ** 2)
                        ev = EvidencePoint(
                            s=s_cache,
                            z_global=observations[:, obs_idx],
                            weight=weight,
                            timestamp=timestamp,
                            sigma=self.obs_sigma,
                        )
                        ent.evidence.cache_out_of_domain(ev, side, merge_spatial=False)
                        events.append({
                            "type": "cache",
                            "side": side,
                            "obs_index": obs_idx,
                            "entity_index": ent_idx,
                            "reason": "raw_out_of_domain",
                        })
                    continue

                if boundary_vote is not None and boundary_vote["prefer_cache"]:
                    cache_thresh = self.extension_residual_threshold * float(boundary_vote.get("residual_scale", 1.0))
                    if residual_norm <= cache_thresh:
                        side = str(boundary_vote["side"])
                        if not self._prefit_should_accept(ent, side, z_obs):
                            continue
                        weight = 1.0 / (self.obs_sigma ** 2)
                        ev = EvidencePoint(
                            s=float(boundary_vote["s_ext"]),
                            z_global=observations[:, obs_idx],
                            weight=weight,
                            timestamp=timestamp,
                            sigma=self.obs_sigma,
                        )
                        kind = str(boundary_vote.get("kind", "outward"))
                        if kind.endswith("seed"):
                            ent.evidence.cache_prefit(ev, side, merge_spatial=False)
                        else:
                            ent.evidence.cache_out_of_domain(ev, side, merge_spatial=False)
                        events.append({
                            "type": "cache",
                            "side": side,
                            "obs_index": obs_idx,
                            "entity_index": ent_idx,
                            "reason": f"boundary_{kind}",
                        })
                        continue

                prefit_vote = self._prefit_vote(
                    ent,
                    z_obs,
                    s_raw,
                    ray_direction_global,
                )
                if prefit_vote is not None:
                    cache_thresh = self.extension_residual_threshold * float(prefit_vote.get("residual_scale", self.prefit_residual_scale))
                    if residual_norm > cache_thresh:
                        continue
                    if not bool(prefit_vote.get("near_boundary", False)):
                        continue
                    side = str(prefit_vote["side"])
                    if not self._prefit_should_accept(ent, side, z_obs):
                        continue
                    weight = 1.0 / (self.obs_sigma ** 2)
                    ev = EvidencePoint(
                        s=float(prefit_vote["s_ext"]),
                        z_global=observations[:, obs_idx],
                        weight=weight,
                        timestamp=timestamp,
                        sigma=self.obs_sigma,
                    )
                    ent.evidence.cache_prefit(ev, side, merge_spatial=False)
                    events.append({
                        "type": "cache",
                        "side": side,
                        "obs_index": obs_idx,
                        "entity_index": ent_idx,
                        "reason": "boundary_prefit",
                    })
                    continue

                sigma_total = self._compute_total_cov(ent, p_star_local)

                tangent_local = ent.model.derivative(s_clamped)
                if np.linalg.norm(tangent_local) < 1e-9:
                    tangent_local = np.array([1.0, 0.0])

                theta = ent.pose.theta
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])
                tangent_global = R @ tangent_local
                tangent_global /= (np.linalg.norm(tangent_global) + 1e-9)

                n_vec = np.array([-tangent_global[1], tangent_global[0]])

                sigma_n = float(n_vec.T @ sigma_total @ n_vec)
                res_n = float(np.dot(residual, n_vec))
                if sigma_n < 1e-9:
                    continue

                sigma_t = float(tangent_global.T @ sigma_total @ tangent_global)
                sigma_t += (self.tangential_sigma_scale * self.obs_sigma) ** 2
                sigma_t = max(sigma_t, 1e-9)
                res_t = float(np.dot(residual, tangent_global))

                d2_n = (res_n ** 2) / sigma_n
                d2_t = (res_t ** 2) / sigma_t
                d2 = d2_n + self.tangential_weight * d2_t

                if d2 < self.chi2_threshold:
                    candidate = AssociationResult(obs_idx, ent_idx, d2, s_clamped, z_pred_global)
                    if best is None or d2 < best.mahalanobis_sq:
                        best = candidate

            if best is not None:
                candidates.append(best)
                ent = entities[best.entity_index]

                weight = 1.0 / (self.obs_sigma ** 2)

                ev = EvidencePoint(
                    s=best.s_star,
                    z_global=observations[:, best.obs_index],
                    weight=weight,
                    timestamp=timestamp,
                    sigma=self.obs_sigma,
                )
                ent.evidence.add(ev)

        return candidates, events

    def _boundary_vote(
        self,
        entity: Entity,
        z_obs: np.ndarray,
        s_raw: float,
        residual_norm: float,
        ray_direction_global: Optional[np.ndarray],
    ) -> Optional[Dict[str, object]]:
        L = float(entity.model.domain_limit)
        if L <= 1e-9:
            return None

        g = entity.pose.to_matrix()
        R = g[:2, :2]

        p0_global = (g @ np.append(entity.model.evaluate(0.0), 1.0))[:2]
        pL_global = (g @ np.append(entity.model.evaluate(L), 1.0))[:2]
        dist_head = float(np.linalg.norm(z_obs - p0_global))
        dist_tail = float(np.linalg.norm(z_obs - pL_global))

        t0_global = R @ entity.model.derivative(0.0)
        tL_global = R @ entity.model.derivative(L)

        t0_global = self._normalize(t0_global)
        tL_global = self._normalize(tL_global)
        if t0_global is None or tL_global is None:
            return None

        head_out = -t0_global
        tail_out = tL_global

        ray_unit = None
        if ray_direction_global is not None:
            ray_unit = self._normalize(ray_direction_global)

        prox_thresh = max(0.2, self.boundary_proximity_ratio * L)
        lat_thresh = max(0.08, self.boundary_lateral_ratio * prox_thresh)
        s_margin = max(0.05, self.boundary_s_ratio * L)
        seed_back_tol = max(0.04, 0.4 * self.boundary_min_outward)
        seed_lat_max = max(0.12, 1.2 * prox_thresh)
        seed_dist_max = max(0.18, prox_thresh)
        seed_s_offset = max(self.boundary_min_outward, 0.25 * seed_dist_max)

        candidates = []
        for side, p_boundary, outward in (
            ("head", p0_global, head_out),
            ("tail", pL_global, tail_out),
        ):
            v = z_obs - p_boundary
            along = float(np.dot(v, outward))
            lateral = abs(self._cross2d(outward, v))
            dist_to_boundary = float(np.linalg.norm(v))
            if dist_to_boundary > (prox_thresh + max(0.0, along) + seed_dist_max):
                continue

            align = 0.0
            if ray_unit is not None:
                align = float(np.dot(ray_unit, outward))
                if align < -0.35:
                    continue

            near_s = (side == "head" and s_raw <= s_margin) or (side == "tail" and s_raw >= (L - s_margin))
            nearest_boundary = (
                dist_to_boundary <= (dist_tail + self.boundary_seed_nearest_margin)
                if side == "head"
                else dist_to_boundary <= (dist_head + self.boundary_seed_nearest_margin)
            )
            outward_ok = (
                along > self.boundary_min_outward
                and lateral <= (lat_thresh + 0.3 * along)
                and dist_to_boundary <= (prox_thresh + along)
            )
            seed_ok = (
                (near_s or nearest_boundary)
                and dist_to_boundary <= seed_dist_max
                and along >= -seed_back_tol
                and lateral >= 0.6 * self.boundary_min_outward
                and lateral <= seed_lat_max
            )

            corner_seed_ok = (
                near_s
                and dist_to_boundary <= (prox_thresh + 0.15)
                and along >= (-0.8 * prox_thresh)
                and lateral >= max(0.08, 0.45 * lat_thresh, 0.9 * self.boundary_min_outward)
                and lateral >= 0.55 * abs(along)
            )

            if not (outward_ok or seed_ok or corner_seed_ok):
                continue

            strong_outward = (
                outward_ok
                and dist_to_boundary <= prox_thresh
                and lateral <= 0.6 * lat_thresh
                and along >= 1.5 * self.boundary_min_outward
            )
            if outward_ok:
                lower_residual = lateral + self.boundary_override_margin < residual_norm
                prefer_cache = bool(near_s or strong_outward or lower_residual)
                score = float(lateral + 0.2 * max(0.0, -align) + 0.05 * dist_to_boundary)
                s_ext = (-along) if side == "head" else (L + along)
                residual_scale = 1.2 if strong_outward else 1.0
                kind = "outward"
            elif seed_ok:
                prefer_cache = True
                score = float(0.9 * lateral + 0.2 * dist_to_boundary + 0.25)
                signed_lat = float(self._cross2d(outward, v))
                s_offset = max(
                    seed_s_offset,
                    0.2 * dist_to_boundary + 0.75 * abs(signed_lat) + 0.15 * max(0.0, along),
                )
                s_ext = (-s_offset) if side == "head" else (L + s_offset)
                residual_scale = 1.1
                kind = "seed"
            else:
                prefer_cache = True
                score = float(0.7 * lateral + 0.15 * dist_to_boundary + 0.18)
                signed_lat = float(self._cross2d(outward, v))
                s_offset = max(
                    seed_s_offset,
                    0.25 * dist_to_boundary + 0.9 * abs(signed_lat) + 0.2 * max(0.0, -along),
                )
                s_ext = (-s_offset) if side == "head" else (L + s_offset)
                residual_scale = 1.15
                kind = "corner_seed"

            candidates.append({
                "side": side,
                "s_ext": float(s_ext),
                "prefer_cache": prefer_cache,
                "score": score,
                "residual_scale": residual_scale,
                "kind": kind,
            })

        if not candidates:
            return None

        candidates.sort(key=lambda item: item["score"])
        return candidates[0]

    def _prefit_vote(
        self,
        entity: Entity,
        z_obs: np.ndarray,
        s_raw: float,
        ray_direction_global: Optional[np.ndarray],
    ) -> Optional[Dict[str, object]]:
        L = float(entity.model.domain_limit)
        if L <= 1e-9:
            return None

        g = entity.pose.to_matrix()
        R = g[:2, :2]
        p0_global = (g @ np.append(entity.model.evaluate(0.0), 1.0))[:2]
        pL_global = (g @ np.append(entity.model.evaluate(L), 1.0))[:2]

        t0_global = self._normalize(R @ entity.model.derivative(0.0))
        tL_global = self._normalize(R @ entity.model.derivative(L))
        if t0_global is None or tL_global is None:
            return None

        ray_unit = self._normalize(ray_direction_global) if ray_direction_global is not None else None
        prox_thresh = max(0.2, self.boundary_proximity_ratio * L)
        s_margin = max(0.05, self.boundary_s_ratio * L)

        best: Optional[Dict[str, object]] = None
        for side, p_boundary, outward in (
            ("head", p0_global, -t0_global),
            ("tail", pL_global, tL_global),
        ):
            cache = list(entity.evidence.get_prefit_cache(side))
            cache.extend(entity.evidence.get_cache(side))
            if len(cache) < self.prefit_min_points:
                continue

            cache_pts = np.array([ev.z_global for ev in cache])
            rel = cache_pts - p_boundary
            norms = np.linalg.norm(rel, axis=1)
            valid_mask = norms > max(0.5 * self.boundary_min_outward, 0.01)
            if np.count_nonzero(valid_mask) < self.prefit_min_points:
                continue

            rel_valid = rel[valid_mask]
            mean_vec = np.mean(rel_valid, axis=0)
            mean_norm = float(np.linalg.norm(mean_vec))
            if mean_norm < 1e-9:
                continue

            if len(rel_valid) >= 3:
                cov = np.cov(rel_valid.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                dir_prefit = eigvecs[:, int(np.argmax(eigvals))]
            else:
                dir_prefit = mean_vec

            dir_prefit = self._normalize(dir_prefit)
            if dir_prefit is None:
                continue
            if float(np.dot(dir_prefit, mean_vec)) < 0.0:
                dir_prefit = -dir_prefit

            progress = float(np.max(rel_valid @ dir_prefit))
            if progress < self.prefit_min_progress:
                continue

            v = z_obs - p_boundary
            along = float(np.dot(v, dir_prefit))
            lateral = abs(self._cross2d(dir_prefit, v))
            dist = float(np.linalg.norm(v))

            near_s = (side == "head" and s_raw <= s_margin) or (side == "tail" and s_raw >= (L - s_margin))
            if along < -self.prefit_back_tolerance:
                continue

            spread = np.percentile(
                np.abs(np.array([self._cross2d(dir_prefit, vec) for vec in rel_valid])),
                75,
            )
            lateral_thresh = self.prefit_lateral_base + self.prefit_lateral_growth * max(0.0, along) + float(spread)
            if lateral > lateral_thresh:
                continue

            max_range = progress + self.prefit_max_range_margin + 0.5 * prox_thresh
            if (not near_s) and dist > max_range:
                continue

            align = 0.0
            if ray_unit is not None:
                align = float(np.dot(ray_unit, dir_prefit))
                if align < -0.45:
                    continue

            signed_lat = float(self._cross2d(dir_prefit, v))
            s_offset = max(
                self.boundary_min_outward,
                min(max_range, max(0.0, along) + 0.65 * abs(signed_lat)),
            )
            s_ext = -s_offset if side == "head" else (L + s_offset)
            score = float(lateral + 0.08 * max(0.0, dist - progress) + 0.1 * max(0.0, -align))

            candidate = {
                "side": side,
                "s_ext": float(s_ext),
                "score": score,
                "residual_scale": self.prefit_residual_scale,
                "near_boundary": bool(near_s or dist <= (prox_thresh + 0.35 * progress)),
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

        return best

    def _prefit_should_accept(self, entity: Entity, side: str, z_obs: np.ndarray) -> bool:
        if side != "tail":
            return True

        cache = list(entity.evidence.get_prefit_cache(side))
        cache.extend(entity.evidence.get_cache(side))
        if len(cache) < 3:
            return True

        recent = sorted(cache, key=lambda ev: float(ev.timestamp), reverse=True)[:8]
        pts = np.array([ev.z_global for ev in recent])
        center = np.mean(pts, axis=0)
        spread = float(np.max(np.linalg.norm(pts - center, axis=1)))
        if spread < 0.03:
            return False

        if np.linalg.norm(z_obs - center) < max(0.04, self.prefit_tail_local_window * spread):
            return False

        return True

    def _normalize(self, vec: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        return vec / norm

    def _cross2d(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _compute_total_cov(self, entity: Entity, p_local: np.ndarray) -> np.ndarray:
        sigma_obs = (self.obs_sigma ** 2) * np.eye(2)

        sigma_g = entity.covariance.matrix
        if sigma_g.shape[0] > 3:
            sigma_g = sigma_g[:3, :3]

        theta = entity.pose.theta
        c = np.cos(theta)
        s = np.sin(theta)
        px, py = p_local[0], p_local[1]

        J_pose = np.array([
            [1.0, 0.0, -s * px - c * py],
            [0.0, 1.0,  c * px - s * py],
        ])

        sigma_total = sigma_obs + J_pose @ sigma_g @ J_pose.T
        return sigma_total
