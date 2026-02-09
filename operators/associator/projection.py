import numpy as np
from typing import Optional, Tuple

from entity.entity import Entity
import time

# TODO: 自主调整采样数量和距离容差
# TODO: 基于证据集s获取过去数据点值的投影加速
class ProjectionOperator:
    """
    Implements the projection operator \Pi_\gamma: R^2 -> [0, L_hat].
    Finds s* that minimizes ||g_hat * gamma(s) - z||^2; if multiple minimizers
    exist (ray intersects curve at several points), chooses the one with the
    smallest tangent angle difference to the incoming ray.
    """

    def __init__(self, num_samples: int = 50, distance_tol: float = 1e-5):
        self.num_samples = num_samples
        self.distance_tol = distance_tol

    def project(
        self,
        entity: Entity,
        z_global: np.ndarray,
        ray_direction_global: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
        """
        Wrapper for coarse search + fine refinement.
        """
        g = entity.pose.to_matrix()
        R = g[:2, :2]
        t = g[:2, 2]

        z_local = R.T @ (z_global - t)
        
        # 1. Coarse Search
        start_coarse = time.time()
        s_coarse, _ = self.coarse_search(entity, z_local)
        end_coarse = time.time()
        
        # 2. Refine
        start_refine = time.time()
        s_refined, p_refined = self.newton_refine(entity, s_coarse, z_local)
        end_refine = time.time()
        
        p_star_global = self._to_global(g, p_refined)
        
        return s_refined, p_refined, p_star_global, (end_coarse - start_coarse), (end_refine - start_refine)

    def coarse_search(self, entity: Entity, z_local: np.ndarray) -> Tuple[float, float]:
        """
        Vectorized coarse search. 
        Returns (s_best, distance_sq_best).
        """
        # 1. 缓存采样点
        if not hasattr(entity, '_cached_samples_hash') or entity._cached_samples_hash != id(entity.model):
            s_grid = np.linspace(0.0, entity.model.domain_limit, self.num_samples)
            
            # 使用向量化 evaluate
            points = entity.model.evaluate(s_grid) # (N, 2)
            
            entity._cached_samples = (s_grid, points)
            entity._cached_samples_hash = id(entity.model)
        
        s_grid, points = entity._cached_samples
        
        # 2. 向量化计算距离
        diff = points - z_local # (N, 2)
        dists_sq = np.einsum('ij,ij->i', diff, diff) # (N,)
        
        min_idx = np.argmin(dists_sq)
        return s_grid[min_idx], dists_sq[min_idx]

    # 牛顿法精优化
    def newton_refine(self, entity: Entity, s_init: float, z_local: np.ndarray, max_iter: int = 5) -> Tuple[float, np.ndarray]:
        s = s_init
        for _ in range(max_iter):
            p = entity.model.evaluate(s)
            dp = entity.model.derivative(s)
            diff = p - z_local

            # 目标函数 0.5 * ||p(s) - z||^2 的梯度是 (p-z).T * dp
            g = np.dot(diff, dp)
            
            # 海森矩阵的二阶导数近似 dp.T * dp + (p-z).T * ddp
            ddp = entity.model.second_derivative(s)
            h = np.dot(dp, dp) + np.dot(diff, ddp)

            if abs(h) < 1e-9:
                break
                
            # 迭代更新 s_new = s - g / h
            delta = g / h
            s_new = s - delta
            
            # 边界处理 [0, L_hat]
            s_new = float(np.clip(s_new, 0.0, entity.model.domain_limit))
            
            # 终止条件 ||s_new - s|| < 1e-6
            if abs(s_new - s) < 1e-6:
                s = s_new
                break
            s = s_new
            
        p_final = entity.model.evaluate(s)
        return s, p_final

    # 解析或默认射线方向
    def _resolve_ray_direction(
        self,
        R: np.ndarray,
        z_local: np.ndarray,
        ray_direction_global: Optional[np.ndarray],
    ) -> np.ndarray:
        if ray_direction_global is not None:
            ray_local = R.T @ ray_direction_global
        else:
            ray_local = z_local
        norm = np.linalg.norm(ray_local)
        if norm < 1e-9:
            return np.array([1.0, 0.0])
        return ray_local / norm

    def _to_global(self, g: np.ndarray, p_local: np.ndarray) -> np.ndarray:
        homog = np.append(p_local, 1.0)
        return (g @ homog)[:2]


class PolylineProjectionOperator:
    """
    Fast projection operator for PiecewiseCatmullRomModel.
    Uses cached global samples for coarse search, evidence reuse for init,
    and supports boundary tangent extension.
    """

    def __init__(
        self,
        num_samples: int = 50,
        reuse_threshold: float = 0.05,
        newton_max_iter: int = 5,
        fallback_residual_threshold: float = 0.2,
        boundary_proximity_threshold: Optional[float] = None,
        boundary_s_threshold: Optional[float] = None,
        boundary_lateral_threshold: Optional[float] = None,
        boundary_min_outward: float = 0.02,
        boundary_override_margin: float = 0.02,
        boundary_alignment_weight: float = 0.25,
    ):
        self.num_samples = num_samples
        self.reuse_threshold = reuse_threshold
        self.newton_max_iter = newton_max_iter
        self.fallback_residual_threshold = fallback_residual_threshold
        self.boundary_proximity_threshold = boundary_proximity_threshold
        self.boundary_s_threshold = boundary_s_threshold
        self.boundary_lateral_threshold = boundary_lateral_threshold
        self.boundary_min_outward = boundary_min_outward
        self.boundary_override_margin = boundary_override_margin
        self.boundary_alignment_weight = boundary_alignment_weight

    def project(
        self,
        entity: Entity,
        z_global: np.ndarray,
        ray_direction_global: Optional[np.ndarray] = None,
        s_init: Optional[float] = None,
    ) -> Tuple[float, float, np.ndarray, np.ndarray, float, float]:
        g = entity.pose.to_matrix()
        R = g[:2, :2]
        t = g[:2, 2]

        z_local = R.T @ (z_global - t)

        if s_init is None:
            s_init, p_init, _ = self.coarse_search(entity, z_local)
        else:
            p_init = entity.model.evaluate(float(np.clip(s_init, 0.0, entity.model.domain_limit)))

        ray_local = self._resolve_ray_direction(R, z_local, ray_direction_global)

        s_init = self._reuse_evidence(entity, z_global, s_init)

        s_refined, p_refined = self._newton_refine(entity, s_init, z_local, self.newton_max_iter)

        residual = float(np.linalg.norm(p_refined - z_local))
        if residual > self.fallback_residual_threshold:
            s_refined = float(s_init)
            p_refined = self._eval_extended(entity, s_refined)[0]

        s_refined, p_refined = self._boundary_adjust(
            entity,
            z_local,
            s_refined,
            p_refined,
            ray_local,
        )

        s_raw = float(s_refined)
        if s_raw < 0.0 or s_raw > entity.model.domain_limit:
            p_star_global = self._to_global(g, p_refined)
            return s_raw, s_raw, p_refined, p_star_global, 0.0, 0.0

        s_clamped = float(s_raw)
        p_clamped = entity.model.evaluate(s_clamped)
        p_star_global = self._to_global(g, p_clamped)
        return s_clamped, s_raw, p_clamped, p_star_global, 0.0, 0.0

    def _resolve_ray_direction(
        self,
        R: np.ndarray,
        z_local: np.ndarray,
        ray_direction_global: Optional[np.ndarray],
    ) -> np.ndarray:
        if ray_direction_global is not None:
            ray_local = R.T @ ray_direction_global
        else:
            ray_local = z_local
        norm = np.linalg.norm(ray_local)
        if norm < 1e-9:
            return np.array([1.0, 0.0])
        return ray_local / norm

    def coarse_search(self, entity: Entity, z_local: np.ndarray) -> Tuple[float, np.ndarray, float]:
        s_grid, points = self._get_cached_samples(entity)
        diff = points - z_local
        dists_sq = np.einsum('ij,ij->i', diff, diff)
        min_idx = int(np.argmin(dists_sq))
        return float(s_grid[min_idx]), points[min_idx], float(dists_sq[min_idx])

    def _get_cached_samples(self, entity: Entity) -> Tuple[np.ndarray, np.ndarray]:
        model = entity.model
        version = getattr(model, "_cache_version", None)
        cache_key = (id(model), version, self.num_samples)
        if not hasattr(entity, "_cached_poly_samples") or getattr(entity, "_cached_poly_samples_key", None) != cache_key:
            s_grid = np.linspace(0.0, model.domain_limit, self.num_samples)
            points = model.evaluate(s_grid)
            entity._cached_poly_samples = (s_grid, points)
            entity._cached_poly_samples_key = cache_key
        return entity._cached_poly_samples

    def _reuse_evidence(self, entity: Entity, z_global: np.ndarray, s_init: float) -> float:
        candidates = entity.evidence.find_nearest_by_s(s_init)
        if not candidates:
            return float(s_init)
        best_ev = None
        best_dist = None
        for ev in candidates:
            d = float(np.linalg.norm(z_global - ev.z_global))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_ev = ev
        if best_ev is not None and best_dist is not None and best_dist < self.reuse_threshold:
            return float(best_ev.s)
        return float(s_init)

    def _newton_refine(self, entity: Entity, s_init: float, z_local: np.ndarray, max_iter: int) -> Tuple[float, np.ndarray]:
        s = float(s_init)
        for _ in range(max_iter):
            p, dp, ddp = self._eval_extended(entity, s)
            diff = p - z_local
            g = np.dot(diff, dp)
            h = np.dot(dp, dp) + np.dot(diff, ddp)
            if abs(h) < 1e-9:
                break
            s_new = s - g / h
            if abs(s_new - s) < 1e-6:
                s = s_new
                break
            s = s_new
        p_final = self._eval_extended(entity, s)[0]
        return s, p_final

    def _boundary_adjust(
        self,
        entity: Entity,
        z_local: np.ndarray,
        s_refined: float,
        p_refined: np.ndarray,
        ray_direction_local: Optional[np.ndarray],
    ) -> Tuple[float, np.ndarray]:
        current_residual = float(np.linalg.norm(z_local - p_refined))

        head_candidate = self._build_boundary_candidate(
            entity,
            z_local,
            side="head",
            ray_direction_local=ray_direction_local,
        )
        tail_candidate = self._build_boundary_candidate(
            entity,
            z_local,
            side="tail",
            ray_direction_local=ray_direction_local,
        )

        valid_candidates = [cand for cand in (head_candidate, tail_candidate) if cand is not None]
        if not valid_candidates:
            return s_refined, p_refined

        best = min(valid_candidates, key=lambda c: c["adjusted_residual"])

        if best["force"]:
            return float(best["s"]), best["p"]

        if best["adjusted_residual"] + self.boundary_override_margin < current_residual:
            return float(best["s"]), best["p"]

        # Backward compatibility: keep original near-boundary extrapolation behavior.
        L = float(entity.model.domain_limit)
        s_thresh = self.boundary_s_threshold
        if s_thresh is None:
            s_thresh = 0.1 * L

        if s_refined <= s_thresh and head_candidate is not None:
            return float(head_candidate["s"]), head_candidate["p"]
        if s_refined >= (L - s_thresh) and tail_candidate is not None:
            return float(tail_candidate["s"]), tail_candidate["p"]

        return s_refined, p_refined

    def _build_boundary_candidate(
        self,
        entity: Entity,
        z_local: np.ndarray,
        side: str,
        ray_direction_local: Optional[np.ndarray],
    ) -> Optional[dict]:
        L = float(entity.model.domain_limit)
        if L <= 1e-9:
            return None

        if side == "head":
            s_boundary = 0.0
            p_boundary = entity.model.evaluate(0.0)
            tangent = entity.model.derivative(0.0)
            sign = -1.0
        elif side == "tail":
            s_boundary = L
            p_boundary = entity.model.evaluate(L)
            tangent = entity.model.derivative(L)
            sign = 1.0
        else:
            raise ValueError("side must be 'head' or 'tail'")

        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-9:
            return None

        tangent_unit = tangent / tangent_norm
        outward_unit = sign * tangent_unit

        v = z_local - p_boundary
        along = float(np.dot(v, outward_unit))
        if along <= self.boundary_min_outward:
            return None

        dist_to_boundary = float(np.linalg.norm(v))
        prox_thresh = self.boundary_proximity_threshold
        if prox_thresh is None:
            prox_thresh = max(0.25, 0.12 * L)

        if dist_to_boundary > (prox_thresh + along):
            return None

        lateral_thresh = self.boundary_lateral_threshold
        if lateral_thresh is None:
            lateral_thresh = max(0.08, 0.35 * prox_thresh)

        lateral = abs(self._cross2d(outward_unit, v))
        if lateral > (lateral_thresh + 0.35 * along):
            return None

        p_ext = p_boundary + outward_unit * along
        s_ext = s_boundary + sign * along

        adjusted_residual = float(lateral)
        if ray_direction_local is not None:
            ray_norm = float(np.linalg.norm(ray_direction_local))
            if ray_norm > 1e-9:
                ray_unit = ray_direction_local / ray_norm
                align = float(np.clip(np.dot(ray_unit, outward_unit), -1.0, 1.0))
                if align >= 0.0:
                    adjusted_residual *= max(0.2, 1.0 - self.boundary_alignment_weight * align)
                else:
                    adjusted_residual *= 1.0 + self.boundary_alignment_weight * (-align)

        force = (
            dist_to_boundary <= prox_thresh
            and lateral <= 0.6 * lateral_thresh
            and along >= 1.5 * self.boundary_min_outward
        )

        return {
            "side": side,
            "s": float(s_ext),
            "p": p_ext,
            "residual": float(lateral),
            "adjusted_residual": float(adjusted_residual),
            "force": force,
        }

    def _cross2d(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _eval_extended(self, entity: Entity, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        L = float(entity.model.domain_limit)
        if s < 0.0:
            p0 = entity.model.evaluate(0.0)
            t0 = entity.model.derivative(0.0)
            p = p0 + t0 * s
            return p, t0, np.zeros(2)
        if s > L:
            pL = entity.model.evaluate(L)
            tL = entity.model.derivative(L)
            p = pL + tL * (s - L)
            return p, tL, np.zeros(2)
        p = entity.model.evaluate(s)
        dp = entity.model.derivative(s)
        ddp = entity.model.second_derivative(s)
        return p, dp, ddp

    def _to_global(self, g: np.ndarray, p_local: np.ndarray) -> np.ndarray:
        homog = np.append(p_local, 1.0)
        return (g @ homog)[:2]
