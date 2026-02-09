import numpy as np
from typing import List, Optional, Tuple

from entity.entity import Entity, CubicBSplineModel, GeometricModel, PiecewiseCatmullRomModel, CatmullRomSegment
from operators.associator.associator import AssociationResult
from entity.evidence import EvidencePoint

# TODO: 面对稀疏点云时的鲁棒性改进；面对大幅度噪声时的稳定性改进
class GeometryFuser:
    """
    Implements the "What does it look like?" part of the update step.
    Refits the curve shape using Regularized Least Squares with Inertia and Robustness.
    """
    def __init__(self, regularization_lambda: float = 5.0, geo_lr: float = 0.1, max_shift: float = 0.2):
        """
        Args:
            regularization_lambda: Smoothness weight.
            geo_lr: Geometry learning rate (0.0 ~ 1.0). 
                    Low value = high inertia (resists noise/pose error).
                    High value = fast adaptation (reacts quickly to shape change).
            max_shift: Maximum allowed displacement for a single control point in one update (Robustness).
        """
        self.reg_lambda = regularization_lambda
        self.geo_lr = geo_lr
        self.max_shift = max_shift

    def fuse(self, entity: Entity, associations: List[AssociationResult], observations: np.ndarray) -> Entity:
        # 1. Gather All Evidence Points
        all_z_global = []
        all_weights = []
        all_s = []

        evidences = entity.evidence.get_all()
        if not evidences:
            return entity

        for ev in evidences:
            all_z_global.append(ev.z_global)
            all_weights.append(ev.weight)
            all_s.append(ev.s)

        Z_global = np.array(all_z_global)
        W = np.array(all_weights)
        S = np.array(all_s)
        
        # 2. Transform to Local Coordinates
        g = entity.pose.to_matrix()
        R_T = g[:2, :2].T
        t = g[:2, 2]
        Z_local = (R_T @ (Z_global - t).T).T
        
        # 3. Dispatch
        if isinstance(entity.model, CubicBSplineModel):
            return self._fuse_piecewise_bezier(entity, Z_local, W, S)
        elif isinstance(entity.model, GeometricModel):
            return self._fuse_linear(entity, Z_local)
            
        return entity

    def _fuse_linear(self, entity: Entity, Z_local: np.ndarray) -> Entity:
        """
        对于直线模型，通常不需要复杂的鲁棒性处理，简单更新端点即可。
        """
        xs = Z_local[:, 0]
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        current_len = max_x - min_x
        if current_len < 0.1: current_len = 0.1
        
        # 直线模型也可以加一点惯性
        old_len = entity.model.domain_limit
        new_len = old_len + self.geo_lr * (current_len - old_len)
        
        # 保持以原点为中心或根据策略调整，这里简单更新长度
        half_len = new_len / 2.0
        entity.model.control_points = np.array([[-half_len, 0.0], [half_len, 0.0]])
        entity.model.domain_limit = new_len
        entity.estimated_length = new_len
        return entity

    def _fuse_piecewise_bezier(self, entity: Entity, Z: np.ndarray, W: np.ndarray, S: np.ndarray) -> Entity:
        """
        Global Least Squares optimization with Inertia (Low-pass filtering).
        """
        model = entity.model
        cumulative = model.cumulative 
        segments_count = len(cumulative) - 1
        
        if segments_count < 1:
            return entity

        # -----------------------
        # Step 0: Extract Old Control Points (Prior) for Stability
        # -----------------------
        num_vars = 3 * segments_count + 1
        X_old = np.zeros((num_vars, 2))
        if len(model.segments) > 0:
            X_old[0] = model.segments[0][0]
            for i, seg in enumerate(model.segments):
                base = 3 * i
                # seg is [p0, p1, p2, p3]
                X_old[base+1] = seg[1]
                X_old[base+2] = seg[2]
                X_old[base+3] = seg[3]

        # -----------------------
        # Step 1: Solve for Fitted Control Points (X_new)
        # -----------------------
        
        # Build Design Matrix A
        S_clamped = np.clip(S, 0.0, model.domain_limit)
        seg_indices = np.searchsorted(cumulative, S_clamped, side='right') - 1
        seg_indices = np.clip(seg_indices, 0, segments_count - 1)
        
        starts = cumulative[seg_indices]
        lengths = np.array(model.segment_lengths)[seg_indices]
        lengths = np.where(lengths < 1e-9, 1.0, lengths)
        u = np.clip((S_clamped - starts) / lengths, 0.0, 1.0)
        
        m_u = 1.0 - u
        b0 = m_u**3
        b1 = 3 * u * m_u**2
        b2 = 3 * u**2 * m_u
        b3 = u**3
        
        num_obs = len(Z)
        A = np.zeros((num_obs, num_vars))
        col_indices = 3 * seg_indices
        rows = np.arange(num_obs)
        A[rows, col_indices]   = b0
        A[rows, col_indices+1] = b1
        A[rows, col_indices+2] = b2
        A[rows, col_indices+3] = b3
        
        sqrt_w = np.sqrt(W)[:, np.newaxis]
        A_w = A * sqrt_w
        Z_w = Z * sqrt_w
        
        L_reg = np.zeros((num_vars - 2, num_vars))
        for i in range(num_vars - 2):
            L_reg[i, i] = 1
            L_reg[i, i+1] = -2
            L_reg[i, i+2] = 1
            
        ATA = A_w.T @ A_w
        LTL = L_reg.T @ L_reg
        MAT = ATA + self.reg_lambda * LTL
        rhs = A_w.T @ Z_w
        
        # Add stability term to prevent divergence in sparse data regions
        # This keeps the shape close to X_old when data is missing
        stability_lambda = 0.5 
        MAT += stability_lambda * np.eye(num_vars)
        rhs += stability_lambda * X_old
        
        try:
            X_fitted = np.linalg.solve(MAT, rhs) # Shape: (N_vars, 2)
        except np.linalg.LinAlgError:
            return entity

        # -----------------------
        # Step 2: Apply Robustness & Inertia
        # -----------------------
        
        # X_old is already extracted in Step 0
        pass 
            
        # 2.1 计算位移向量
        diff = X_fitted - X_old
        
        # 2.2 鲁棒性：限幅
        # 如果某个控制点移动距离超过 max_shift，则将其截断
        dist = np.linalg.norm(diff, axis=1)
        # 找到超出阈值的点
        outlier_mask = dist > self.max_shift
        if np.any(outlier_mask):
            # 计算缩放因子
            scale = np.ones_like(dist)
            scale[outlier_mask] = self.max_shift / (dist[outlier_mask] + 1e-6)
            # 应用缩放
            diff = diff * scale[:, np.newaxis]
            
        # 2.3 惯性：低通滤波
        # 最终位移 = 学习率 * (截断后的拟合位移)
        # 相当于：X_final = X_old + lr * (X_fitted_clamped - X_old)
        delta_final = self.geo_lr * diff
        X_final = X_old + delta_final

        # -----------------------
        # Step 3: Update Entity Model
        # -----------------------
        new_segments = []
        for i in range(segments_count):
            base = 3 * i
            pts = X_final[base : base+4]
            new_segments.append(pts)
            
        entity.model.segments = new_segments
        entity.model.control_points = np.vstack(new_segments)
        entity.model.segments_array = np.array(new_segments)
        
        # 更新长度元数据 (保持逻辑不变，但建议这里也可以做平滑)
        new_lens = [float(np.linalg.norm(s[-1] - s[0])) for s in new_segments]
        entity.model.segment_lengths = new_lens
        entity.model.cumulative = np.cumsum([0.0] + new_lens)
        entity.model.domain_limit = float(entity.model.cumulative[-1])
        entity.estimated_length = entity.model.domain_limit
        
        return entity


class CatmullRomGeometryFuser:
    """
    Geometry fuser for PiecewiseCatmullRomModel.

    Rules:
    1) Never modify existing control points (only append/prepend new ones).
    2) Trigger fitting when cache size reaches a threshold.
    3) Denoise cached points before fitting (outlier removal).
    4) If the angle between cache direction and boundary tangent is sharp,
       create a new segment starting at the boundary point;
       otherwise, extend the existing curve with one new control point.
    """

    def __init__(
        self,
        cache_min_points: int = 5,
        cache_force_points: int = 12,
        angle_threshold: float = 0.7,
        mad_scale: float = 2.5,
        max_segment_points: int = 6,
        min_extend_spacing: float = 0.2,
        max_extend_spacing: float = 0.7,
        min_outward_distance: float = 0.02,
        max_cache_points: int = 160,
        new_segment_min_points: int = 4,
        prefit_only_min_points: int = 8,
        max_new_controls_per_segment: int = 2,
    ) -> None:
        self.cache_min_points = int(cache_min_points)
        self.cache_force_points = int(cache_force_points)
        self.angle_threshold = float(angle_threshold)
        self.mad_scale = float(mad_scale)
        self.max_segment_points = int(max_segment_points)
        self.min_extend_spacing = float(min_extend_spacing)
        self.max_extend_spacing = float(max_extend_spacing)
        self.min_outward_distance = float(min_outward_distance)
        self.max_cache_points = int(max_cache_points)
        self.new_segment_min_points = int(new_segment_min_points)
        self.prefit_only_min_points = int(max(self.cache_min_points + 1, prefit_only_min_points))
        self.max_new_controls_per_segment = int(max(1, max_new_controls_per_segment))
        self.prefit_spacing_scale = 0.6
        self.prefit_progress_min = 0.02

    def fuse(
        self,
        entity: Entity,
        associations: List[AssociationResult],
        observations: np.ndarray,
    ) -> Entity:
        if not isinstance(entity.model, PiecewiseCatmullRomModel):
            return entity

        for side in ("head", "tail"):
            cache = list(entity.evidence.get_cache(side))
            prefit_cache = list(entity.evidence.get_prefit_cache(side))

            ext_ready = len(cache) >= self.cache_min_points
            prefit_ready = len(prefit_cache) >= self.prefit_only_min_points
            if not (ext_ready or prefit_ready):
                self._set_trimmed_caches(entity, side, cache, prefit_cache)
                continue

            merged = [("cache", ev) for ev in cache]
            merged.extend(("prefit", ev) for ev in prefit_cache)
            cache_pts_global = np.array([item[1].z_global for item in merged])
            if len(cache_pts_global) < 2:
                self._set_trimmed_caches(entity, side, cache, prefit_cache)
                continue

            # Transform to local coordinates
            g = entity.pose.to_matrix()
            R_T = g[:2, :2].T
            t = g[:2, 2]
            cache_local = (R_T @ (cache_pts_global - t).T).T

            boundary_local, ref_dir = self._boundary_dir(entity.model, side)
            if ref_dir is None:
                self._set_trimmed_caches(entity, side, cache, prefit_cache)
                continue
            ref_dir = ref_dir / (np.linalg.norm(ref_dir) + 1e-9)

            prefit_dir = self._estimate_prefit_direction(cache_local, boundary_local, ref_dir)
            if prefit_dir is None:
                self._set_trimmed_caches(entity, side, cache, prefit_cache)
                continue

            mask = self._denoise_mask(cache_local, boundary_local, prefit_dir)
            if np.count_nonzero(mask) < max(2, self.cache_min_points):
                self._set_trimmed_caches(entity, side, cache, prefit_cache)
                continue

            valid_indices = np.where(mask)[0]
            valid_local = cache_local[valid_indices]
            rel_valid = valid_local - boundary_local
            along = np.dot(rel_valid, prefit_dir)

            order = np.argsort(along)
            ordered_along = along[order]
            ordered_local = valid_local[order]
            ordered_indices = valid_indices[order]

            if len(ordered_along) == 0:
                self._set_trimmed_caches(entity, side, cache, prefit_cache)
                continue

            farthest_along = float(ordered_along[-1])
            spacing_min, spacing_max = self._compute_spacing_targets(entity.model, side)
            trigger_progress = max(
                self.prefit_progress_min,
                self.prefit_spacing_scale * spacing_min,
            )
            if ext_ready:
                trigger_progress = max(trigger_progress, spacing_min)

            should_force = (
                len(cache) >= self.cache_force_points
                or len(prefit_cache) >= (self.cache_force_points + 2)
            )

            dir_to_cache = ordered_local[-1] - boundary_local
            if np.linalg.norm(dir_to_cache) < 1e-9:
                self._set_trimmed_caches(entity, side, cache, prefit_cache)
                continue

            cos_turn = float(np.clip(np.dot(prefit_dir, ref_dir), -1.0, 1.0))
            turn_angle = float(np.arccos(cos_turn))

            spread_angle = self._estimate_spread_angle(ordered_local, boundary_local, prefit_dir)
            effective_angle = max(turn_angle, spread_angle)
            lateral_spread = self._estimate_lateral_spread(ordered_local, boundary_local, ref_dir)

            if farthest_along < trigger_progress and not should_force:
                pass

            corner_bootstrap = (
                (not ext_ready)
                and (not prefit_ready)
                and (
                    effective_angle >= 1.2
                    or lateral_spread >= max(0.05, 2.5 * self.min_outward_distance)
                )
                and len(ordered_local) >= 2
                and farthest_along >= max(self.min_outward_distance, 0.015)
            )

            if farthest_along < trigger_progress and not should_force and not corner_bootstrap:
                self._set_trimmed_caches(entity, side, cache, prefit_cache)
                continue

            new_segment_angle_thresh = self.angle_threshold
            if ext_ready and not prefit_ready:
                new_segment_angle_thresh = self.angle_threshold + 0.2
            elif prefit_ready and not ext_ready:
                new_segment_angle_thresh = max(0.55, self.angle_threshold - 0.05)

            use_new_segment = (
                effective_angle > new_segment_angle_thresh
                and len(ordered_local) >= self.new_segment_min_points
                and farthest_along >= max(0.75 * trigger_progress, 1.2 * self.min_outward_distance)
            )
            if corner_bootstrap:
                use_new_segment = True

            growth_spacing_min = spacing_min
            if corner_bootstrap and farthest_along < spacing_min:
                growth_spacing_min = max(0.01, 0.8 * farthest_along)
            growth_spacing_max = max(spacing_max, growth_spacing_min + 1e-3)

            if use_new_segment:
                grown, consumed_limit = self._create_new_segment(
                    entity,
                    side,
                    boundary_local,
                    ordered_local,
                    ordered_along,
                    growth_spacing_min,
                    growth_spacing_max,
                )
            else:
                grown, consumed_limit = self._extend_existing(
                    entity,
                    side,
                    boundary_local,
                    ordered_local,
                    ordered_along,
                    growth_spacing_min,
                    growth_spacing_max,
                )

            if not grown:
                self._set_trimmed_caches(entity, side, cache, prefit_cache)
                continue

            consume_margin = max(0.08, 0.3 * trigger_progress)
            if corner_bootstrap:
                consume_margin = max(consume_margin, max(0.03, 1.1 * farthest_along))
            consumed_order_mask = ordered_along <= (consumed_limit + consume_margin)
            consumed_indices = set(int(idx) for idx in ordered_indices[consumed_order_mask])

            consumed_cache: List[EvidencePoint] = []
            remain_cache: List[EvidencePoint] = []
            remain_prefit: List[EvidencePoint] = []
            for idx, (source, ev) in enumerate(merged):
                if idx in consumed_indices:
                    consumed_cache.append(ev)
                elif source == "cache":
                    remain_cache.append(ev)
                else:
                    remain_prefit.append(ev)

            if consumed_cache:
                self._add_filtered_to_evidence(entity, consumed_cache)

            self._set_trimmed_caches(entity, side, remain_cache, remain_prefit)

        return entity

    def _set_trimmed_caches(
        self,
        entity: Entity,
        side: str,
        cache: List[EvidencePoint],
        prefit_cache: List[EvidencePoint],
    ) -> None:
        entity.evidence.set_cache(side, self._trim_cache(cache))
        entity.evidence.set_prefit_cache(side, self._trim_cache(prefit_cache))

    def _denoise_mask(self, points: np.ndarray, boundary: np.ndarray, ref_dir: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros(0, dtype=bool)

        rel = points - boundary
        along = np.dot(rel, ref_dir)
        lateral = np.abs(np.array([self._cross2d(ref_dir, r) for r in rel]))

        dynamic_forward = max(0.002, 0.15 * self.min_outward_distance)
        forward = along > dynamic_forward
        if np.count_nonzero(forward) < 2:
            fallback = np.linalg.norm(rel, axis=1) > max(0.02, 0.5 * self.min_outward_distance)
            if np.count_nonzero(fallback) >= 2:
                forward = fallback
            else:
                return forward

        lat_forward = lateral[forward]
        med = float(np.median(lat_forward))
        mad = float(np.median(np.abs(lat_forward - med))) + 1e-9
        lateral_thresh = med + self.mad_scale * mad + 0.03
        return forward & (lateral <= lateral_thresh)

    def _estimate_prefit_direction(
        self,
        points_local: np.ndarray,
        boundary_local: np.ndarray,
        fallback_dir: np.ndarray,
    ) -> Optional[np.ndarray]:
        if len(points_local) < 2:
            return fallback_dir

        rel = points_local - boundary_local
        norms = np.linalg.norm(rel, axis=1)
        valid = norms > max(0.01, 0.4 * self.min_outward_distance)
        if np.count_nonzero(valid) < 2:
            return fallback_dir

        rel_valid = rel[valid]
        # If we already have enough spread, prefer the farthest-point direction.
        # This is more robust for corner emergence where PCA tends to follow
        # the old boundary tangent due to anisotropic sampling density.
        far_idx = int(np.argmax(np.linalg.norm(rel_valid, axis=1)))
        far_vec = rel_valid[far_idx]
        far_norm = float(np.linalg.norm(far_vec))
        if far_norm > max(0.04, 2.0 * self.min_outward_distance):
            direction_far = far_vec / far_norm
            if np.dot(direction_far, fallback_dir) < -0.7:
                direction_far = -direction_far
            if np.linalg.norm(direction_far) > 1e-9:
                return direction_far

        mean_vec = np.mean(rel_valid, axis=0)
        if np.linalg.norm(mean_vec) < 1e-9:
            return fallback_dir

        if len(rel_valid) >= 3:
            cov = np.cov(rel_valid.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, int(np.argmax(eigvals))]
        else:
            direction = mean_vec

        direction = direction / (np.linalg.norm(direction) + 1e-9)
        if np.dot(direction, mean_vec) < 0.0:
            direction = -direction

        if np.linalg.norm(direction) < 1e-9:
            return fallback_dir
        return direction

    def _boundary_dir(
        self, model: PiecewiseCatmullRomModel, side: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if side == "head":
            s = 0.0
            boundary = model.evaluate(s)
            tangent = model.derivative(s)
            if np.linalg.norm(tangent) < 1e-9:
                tangent = self._fallback_tangent(model, side)
            ref_dir = -tangent
        else:
            s = model.domain_limit
            boundary = model.evaluate(s)
            tangent = model.derivative(s)
            if np.linalg.norm(tangent) < 1e-9:
                tangent = self._fallback_tangent(model, side)
            ref_dir = tangent
        if np.linalg.norm(ref_dir) < 1e-9:
            return boundary, None
        return boundary, ref_dir

    def _fallback_tangent(
        self, model: PiecewiseCatmullRomModel, side: str
    ) -> np.ndarray:
        if not model.segments:
            return np.zeros(2)
        if side == "head":
            cps = model.segments[0].control_points
            if len(cps) >= 2:
                return cps[1] - cps[0]
        else:
            cps = model.segments[-1].control_points
            if len(cps) >= 2:
                return cps[-1] - cps[-2]
        return np.zeros(2)

    def _create_new_segment(
        self,
        entity: Entity,
        side: str,
        boundary_local: np.ndarray,
        points_local: np.ndarray,
        points_along: np.ndarray,
        spacing_min: float,
        spacing_max: float,
    ) -> Tuple[bool, float]:
        selected_pts: List[np.ndarray] = []
        selected_along: List[float] = []
        max_new = self.max_new_controls_per_segment

        next_target = spacing_min
        for p, a in zip(points_local, points_along):
            if a >= next_target:
                selected_pts.append(p)
                selected_along.append(float(a))
                next_target = min(next_target + spacing_min, spacing_max)
            if len(selected_pts) >= max_new:
                break

        if not selected_pts:
            selected_pts = [points_local[-1]]
            selected_along = [float(points_along[-1])]
        elif selected_along[-1] < points_along[-1] and len(selected_pts) < max_new:
            selected_pts.append(points_local[-1])
            selected_along.append(float(points_along[-1]))

        point_gap_min = max(0.03, 0.35 * spacing_min, 0.5 * self.min_outward_distance)
        control_points = [boundary_local]
        for p in selected_pts:
            if np.linalg.norm(p - control_points[-1]) >= point_gap_min:
                control_points.append(p)

        if len(control_points) < 2:
            return False, 0.0

        segment = CatmullRomSegment(np.array(control_points))
        if segment.total_length < 0.5 * spacing_min:
            return False, 0.0

        if side == "head":
            entity.model.segments.insert(0, segment)
            entity.model._rebuild_cache()
            if segment.total_length > 1e-9:
                entity.evidence.shift_s(segment.total_length)
        else:
            entity.model.segments.append(segment)
            entity.model._rebuild_cache()

        entity.estimated_length = entity.model.domain_limit
        consumed_limit = float(selected_along[-1]) if selected_along else float(points_along[-1])
        return True, consumed_limit

    def _extend_existing(
        self,
        entity: Entity,
        side: str,
        boundary_local: np.ndarray,
        points_local: np.ndarray,
        points_along: np.ndarray,
        spacing_min: float,
        spacing_max: float,
    ) -> Tuple[bool, float]:
        if len(points_local) == 0:
            return False, 0.0

        farthest = float(points_along[-1])
        if farthest < spacing_min:
            return False, farthest

        target = float(np.clip(farthest, spacing_min, spacing_max))
        penalties = np.abs(points_along - target)
        idx = int(np.argmin(penalties))
        anchor = points_local[idx]
        consumed_limit = float(points_along[idx])

        if consumed_limit < 0.8 * spacing_min:
            anchor = points_local[-1]
            consumed_limit = farthest

        delta_L = entity.model.extend(side, anchor)
        if delta_L < 0.2 * spacing_min:
            return False, farthest

        entity.estimated_length = entity.model.domain_limit
        if side == "head" and delta_L > 1e-9:
            entity.evidence.shift_s(delta_L)
        return True, consumed_limit

    def _compute_spacing_targets(
        self,
        model: PiecewiseCatmullRomModel,
        side: str,
    ) -> Tuple[float, float]:
        base_spacing = self.min_extend_spacing
        if model.segments:
            if side == "head":
                cps = model.segments[0].control_points
                if len(cps) >= 2:
                    base_spacing = float(np.linalg.norm(cps[1] - cps[0]))
            else:
                cps = model.segments[-1].control_points
                if len(cps) >= 2:
                    base_spacing = float(np.linalg.norm(cps[-1] - cps[-2]))

        base_spacing = max(base_spacing, self.min_outward_distance)
        spacing_min = max(self.min_extend_spacing, 0.6 * base_spacing)
        spacing_max = max(self.max_extend_spacing, 1.8 * base_spacing, spacing_min + 1e-3)
        return spacing_min, spacing_max

    def _trim_cache(self, cache: List[EvidencePoint]) -> List[EvidencePoint]:
        if len(cache) <= self.max_cache_points:
            return cache
        sorted_cache = sorted(cache, key=lambda ev: float(ev.timestamp), reverse=True)
        return sorted_cache[: self.max_cache_points]

    def _cross2d(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _estimate_spread_angle(
        self,
        points_local: np.ndarray,
        boundary_local: np.ndarray,
        ref_dir: np.ndarray,
    ) -> float:
        if len(points_local) < 3:
            return 0.0

        rel = points_local - boundary_local
        norms = np.linalg.norm(rel, axis=1)
        mask = norms > max(0.03, self.min_outward_distance * 0.5)
        if np.count_nonzero(mask) < 3:
            return 0.0

        rel = rel[mask]
        norms = norms[mask]
        dirs = rel / (norms[:, np.newaxis] + 1e-9)

        angles = np.arctan2(
            np.array([self._cross2d(ref_dir, d) for d in dirs]),
            np.dot(dirs, ref_dir),
        )
        q_low = float(np.percentile(angles, 15))
        q_high = float(np.percentile(angles, 85))
        return float(max(0.0, q_high - q_low))

    def _estimate_lateral_spread(
        self,
        points_local: np.ndarray,
        boundary_local: np.ndarray,
        ref_dir: np.ndarray,
    ) -> float:
        if len(points_local) < 3:
            return 0.0

        rel = points_local - boundary_local
        lateral = np.abs(np.array([self._cross2d(ref_dir, r) for r in rel]))
        if len(lateral) == 0:
            return 0.0
        q = float(np.percentile(lateral, 80))
        return max(0.0, q)

    def _add_filtered_to_evidence(
        self, entity: Entity, filtered_cache: List[EvidencePoint]
    ) -> None:
        if not filtered_cache:
            return

        g = entity.pose.to_matrix()
        R_T = g[:2, :2].T
        t = g[:2, 2]
        pts_local = (R_T @ (np.array([ev.z_global for ev in filtered_cache]) - t).T).T

        seg_owner, local_param, _ = entity.model.project_to_polyline(pts_local)
        seg_owner = seg_owner.astype(int)
        offsets = entity.model.segment_offsets
        seg_len = offsets[seg_owner + 1] - offsets[seg_owner]
        s_vals = offsets[seg_owner] + local_param * seg_len

        for ev, s_new in zip(filtered_cache, s_vals):
            entity.evidence.add(
                EvidencePoint(
                    s=float(s_new),
                    z_global=ev.z_global,
                    weight=ev.weight,
                    timestamp=ev.timestamp,
                    sigma=ev.sigma,
                )
            )
