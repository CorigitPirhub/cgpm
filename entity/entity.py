import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from .state import SE2Pose, StateVector, StateCovariance
from .evidence import EvidenceSet, EvidencePoint

class CurveModel(ABC):
    """
    Base class for parametric curve models used by entities.
    Subclasses implement evaluate() and derivative().
    """

    def __init__(self, domain_limit: float):
        self.domain_limit = float(domain_limit)
        self.control_points = None

    @abstractmethod
    def evaluate(self, s: float) -> np.ndarray:
        ...

    @abstractmethod
    def derivative(self, s: float) -> np.ndarray:
        ...

    @abstractmethod
    def second_derivative(self, s: float) -> np.ndarray:
        ...

# TODO: 目前仅实现了线性和三次贝塞尔曲线模型，后续可扩展更多曲线类型
class GeometricModel(CurveModel):
    """
    Definition 2.1: Local Geometric Model M = (P, Phi, theta)
    P: Control Points
    Phi: Basis Functions (Implicitly defined by the evaluation method)
    theta: Shape parameters (Here we treat P as the primary parameters)
    """
    def __init__(self, control_points: np.ndarray, domain_limit: float):
        super().__init__(domain_limit)
        self.control_points = np.array(control_points)
    
    # 计算曲线在 s 处的坐标点
    def evaluate(self, s: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluates the curve gamma(s) at parameter s.
        gamma(s) = sum p_i * phi_i(s)
        
        Default implementation: Piecewise linear interpolation (Degree 1 B-Spline)
        to ensure functionality without external curve libraries.
        """
        # Clamp s
        s = np.atleast_1d(s)
        s = np.clip(s, 0, self.domain_limit)
        
        N = len(self.control_points)
        if N < 2:
             res = self.control_points[0] if N > 0 else np.zeros(2)
             # Handle vector output
             return np.tile(res, (len(s), 1)) if len(s) > 1 else res

        # Mapping s to index. Assuming uniform knot vector for now.
        if self.domain_limit <= 1e-9:
             res = self.control_points[0]
             return np.tile(res, (len(s), 1)) if len(s) > 1 else res

        seg_len = self.domain_limit / (N - 1)
        # Avoid division by zero if N=1 (handled above)
        
        idx = (s // seg_len).astype(int)
        idx = np.clip(idx, 0, N - 2)
        
        # Local parameter t in [0, 1]
        t = (s - idx * seg_len) / seg_len
        t = t[:, np.newaxis]
        
        p0 = self.control_points[idx]
        p1 = self.control_points[idx+1]
        
        res = (1 - t) * p0 + t * p1
        return res if res.shape[0] > 1 else res[0]

    # 计算曲线在 s 处的一阶导数
    def derivative(self, s: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluates gamma'(s).
        """
        s = np.atleast_1d(s)
        s = np.clip(s, 0, self.domain_limit)
        N = len(self.control_points)
        if N < 2: 
            res = np.zeros(2)
            return np.tile(res, (len(s), 1)) if len(s) > 1 else res
        
        seg_len = self.domain_limit / (N - 1)
        idx = (s // seg_len).astype(int)
        idx = np.clip(idx, 0, N - 2)
        
        p0 = self.control_points[idx]
        p1 = self.control_points[idx+1]
        
        res = (p1 - p0) / (seg_len + 1e-9)
        return res if res.shape[0] > 1 else res[0]

    # 计算曲线在 s 处的二阶导数
    def second_derivative(self, s: float) -> np.ndarray:
        """
        Evaluates gamma''(s). For linear segments, this is zero (almost everywhere).
        """
        return np.zeros(2)

class CubicBSplineModel(CurveModel):
    """
    Piecewise Cubic Bezier Curve model.
    虽然类名包含 BSpline，但为了配合 Fitter 的插值需求，这里实现的是贝塞尔形式。
    """
    def __init__(self, control_point_segments: List[np.ndarray]):
        if not control_point_segments:
            raise ValueError("control_point_segments must be non-empty")

        self.segments = [np.array(seg) for seg in control_point_segments]
        # 计算每段的弦长作为参数区间的长度
        self.segment_lengths = [float(np.linalg.norm(seg[-1] - seg[0])) for seg in self.segments]
        self.cumulative = np.cumsum([0.0] + self.segment_lengths)
        super().__init__(domain_limit=float(self.cumulative[-1] if len(self.cumulative) > 0 else 0.0))
        # 展平所有控制点用于可视化或其他用途
        self.control_points = np.vstack(self.segments)
        # Store segments as array for vectorization (N, 4, 2)
        self.segments_array = np.array(self.segments)

    # 将 s 映射到具体段和局部参数 u
    def _select_segment(self, s: Union[float, np.ndarray]) -> Tuple[Union[int, np.ndarray], Union[float, np.ndarray]]:
        s_arr = np.atleast_1d(s)
        s_clamped = np.clip(s_arr, 0.0, self.domain_limit)
        
        # 搜索 s 落在哪个段
        idx = np.searchsorted(self.cumulative, s_clamped, side="right") - 1
        idx = np.clip(idx, 0, len(self.segment_lengths) - 1).astype(int)
        
        start = self.cumulative[idx]
        # Use simple list indexing or array indexing for segment_lengths
        seg_lens = np.array(self.segment_lengths)[idx]
        seg_lens = np.where(seg_lens > 1e-9, seg_lens, 1.0)
        
        # 局部参数 u in [0, 1]
        u = (s_clamped - start) / seg_lens
        u = np.clip(u, 0.0, 1.0)
        
        if np.isscalar(s):
             return int(idx[0]), float(u[0])
        return idx, u

    def evaluate(self, s: Union[float, np.ndarray]) -> np.ndarray:
        s = np.atleast_1d(s)
        idx, u = self._select_segment(s)
        
        # Ensure idx is integer array for fancy indexing
        idx = idx.astype(int)
        
        # Fetch control points: (N, 4, 2)
        cp = self.segments_array[idx]
        
        # Expand u for broadcasting: (N, 1)
        u = u[:, np.newaxis]
        
        u2 = u * u
        u3 = u2 * u
        m_u = 1.0 - u
        m_u2 = m_u * m_u
        m_u3 = m_u2 * m_u

        b0 = m_u3
        b1 = 3.0 * u * m_u2
        b2 = 3.0 * u2 * m_u
        b3 = u3
        
        # Result shape (N, 2)
        res = b0 * cp[:, 0, :] + b1 * cp[:, 1, :] + b2 * cp[:, 2, :] + b3 * cp[:, 3, :]
        return res if res.shape[0] > 1 else res[0]

    def derivative(self, s: Union[float, np.ndarray]) -> np.ndarray:
        s = np.atleast_1d(s)
        idx, u = self._select_segment(s)
        idx = idx.astype(int)
        
        cp = self.segments_array[idx]
        
        # Handle segment lengths
        lens = np.array(self.segment_lengths)[idx][:, np.newaxis]
        lens = np.where(lens > 1e-9, lens, 1.0)
        
        u = u[:, np.newaxis]
        u2 = u * u
        m_u = 1.0 - u
        m_u2 = m_u * m_u

        db0 = -3.0 * m_u2
        db1 = 3.0 * m_u2 - 6.0 * u * m_u
        db2 = 6.0 * u * m_u - 3.0 * u2
        db3 = 3.0 * u2
        
        d_gamma_du = db0 * cp[:, 0, :] + db1 * cp[:, 1, :] + db2 * cp[:, 2, :] + db3 * cp[:, 3, :]
        
        res = d_gamma_du / lens
        return res if res.shape[0] > 1 else res[0]

    def second_derivative(self, s: Union[float, np.ndarray]) -> np.ndarray:
        s = np.atleast_1d(s)
        idx, u = self._select_segment(s)
        idx = idx.astype(int)
        
        cp = self.segments_array[idx]
        
        lens = np.array(self.segment_lengths)[idx][:, np.newaxis]
        lens = np.where(lens > 1e-9, lens, 1.0)

        u = u[:, np.newaxis]
        m_u = 1.0 - u
        
        ddb0 = 6.0 * m_u
        ddb1 = 6.0 * (3.0 * u - 2.0)
        ddb2 = 6.0 * (1.0 - 3.0 * u)
        ddb3 = 6.0 * u
        
        d2_gamma_du2 = ddb0 * cp[:, 0, :] + ddb1 * cp[:, 1, :] + ddb2 * cp[:, 2, :] + ddb3 * cp[:, 3, :]
        
        res = d2_gamma_du2 / (lens * lens)
        return res if res.shape[0] > 1 else res[0]


class CatmullRomUtils:
    @staticmethod
    def catmull_rom_basis(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(u)
        u = u[:, np.newaxis]
        u2 = u * u
        u3 = u2 * u
        term1 = 2.0 * p1
        term2 = (-p0 + p2) * u
        term3 = (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        term4 = (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
        res = 0.5 * (term1 + term2 + term3 + term4)
        return res

    @staticmethod
    def catmull_rom_derivative(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(u)
        u = u[:, np.newaxis]
        u2 = u * u
        term1 = (-p0 + p2)
        term2 = 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u
        term3 = 3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u2
        res = 0.5 * (term1 + term2 + term3)
        return res

    @staticmethod
    def catmull_rom_second_derivative(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(u)
        u = u[:, np.newaxis]
        term1 = 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
        term2 = 6.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u
        res = 0.5 * (term1 + term2)
        return res


class CatmullRomSegment:
    def __init__(self, points: np.ndarray):
        if points is None or len(points) == 0:
            raise ValueError("points must be non-empty")
        self.control_points = np.array(points, dtype=float)
        self._rebuild_lengths()

    def _rebuild_lengths(self) -> None:
        if len(self.control_points) < 2:
            self.cumulative_lengths = np.array([0.0])
            self.total_length = 0.0
            return
        diffs = self.control_points[1:] - self.control_points[:-1]
        dists = np.linalg.norm(diffs, axis=1)
        self.cumulative_lengths = np.concatenate(([0.0], np.cumsum(dists)))
        self.total_length = float(self.cumulative_lengths[-1])

    def add_control_point(self, point: np.ndarray) -> None:
        point = np.array(point, dtype=float)
        if self.control_points.size == 0:
            self.control_points = point.reshape(1, 2)
        else:
            self.control_points = np.vstack([self.control_points, point])
        self._rebuild_lengths()

    def _select_local(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.control_points) < 2 or self.total_length <= 1e-9:
            idx = np.zeros_like(u, dtype=int)
            t = np.zeros_like(u, dtype=float)
            return idx, t

        s = np.clip(u, 0.0, 1.0) * self.total_length
        idx = np.searchsorted(self.cumulative_lengths, s, side="right") - 1
        idx = np.clip(idx, 0, len(self.control_points) - 2)
        seg_len = self.cumulative_lengths[idx + 1] - self.cumulative_lengths[idx]
        seg_len = np.where(seg_len > 1e-9, seg_len, 1.0)
        t = (s - self.cumulative_lengths[idx]) / seg_len
        t = np.clip(t, 0.0, 1.0)
        return idx, t

    def _neighbors(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.control_points)
        i0 = np.clip(idx - 1, 0, n - 1)
        i1 = np.clip(idx, 0, n - 1)
        i2 = np.clip(idx + 1, 0, n - 1)
        i3 = np.clip(idx + 2, 0, n - 1)
        p0 = self.control_points[i0]
        p1 = self.control_points[i1]
        p2 = self.control_points[i2]
        p3 = self.control_points[i3]
        return p0, p1, p2, p3

    def evaluate(self, u: Union[float, np.ndarray]) -> np.ndarray:
        u_arr = np.atleast_1d(u)
        idx, t = self._select_local(u_arr)
        p0, p1, p2, p3 = self._neighbors(idx)
        res = CatmullRomUtils.catmull_rom_basis(p0, p1, p2, p3, t)
        return res if res.shape[0] > 1 else res[0]

    def derivative(self, u: Union[float, np.ndarray]) -> np.ndarray:
        u_arr = np.atleast_1d(u)
        idx, t = self._select_local(u_arr)
        p0, p1, p2, p3 = self._neighbors(idx)

        if self.total_length <= 1e-9:
            res = np.zeros((len(u_arr), 2))
            return res if res.shape[0] > 1 else res[0]

        seg_len = self.cumulative_lengths[idx + 1] - self.cumulative_lengths[idx]
        seg_len = np.where(seg_len > 1e-9, seg_len, 1.0)
        dt_du = self.total_length / seg_len

        d = CatmullRomUtils.catmull_rom_derivative(p0, p1, p2, p3, t)
        res = d * dt_du[:, np.newaxis]
        return res if res.shape[0] > 1 else res[0]

    def second_derivative(self, u: Union[float, np.ndarray]) -> np.ndarray:
        u_arr = np.atleast_1d(u)
        idx, t = self._select_local(u_arr)
        p0, p1, p2, p3 = self._neighbors(idx)

        if self.total_length <= 1e-9:
            res = np.zeros((len(u_arr), 2))
            return res if res.shape[0] > 1 else res[0]

        seg_len = self.cumulative_lengths[idx + 1] - self.cumulative_lengths[idx]
        seg_len = np.where(seg_len > 1e-9, seg_len, 1.0)
        dt_du = self.total_length / seg_len

        d2 = CatmullRomUtils.catmull_rom_second_derivative(p0, p1, p2, p3, t)
        res = d2 * (dt_du ** 2)[:, np.newaxis]
        return res if res.shape[0] > 1 else res[0]


class PiecewiseCatmullRomModel(CurveModel):
    def __init__(self, segments: List[CatmullRomSegment]):
        if not segments:
            raise ValueError("segments must be non-empty")
        self.segments = segments
        self._rebuild_cache()
        super().__init__(domain_limit=float(self.segment_offsets[-1]))

    def _rebuild_cache(self) -> None:
        lengths = [seg.total_length for seg in self.segments]
        self.segment_offsets = np.cumsum([0.0] + lengths)
        # Build a polyline without duplicating shared endpoints between segments
        poly_points = []
        for idx, seg in enumerate(self.segments):
            cps = seg.control_points
            if len(cps) == 0:
                continue
            if idx == 0:
                poly_points.append(cps)
            else:
                poly_points.append(cps[1:])
        self.global_control_points = np.vstack(poly_points) if poly_points else np.zeros((0, 2))
        # Map each polyline segment to its owning CatmullRom segment and local param range
        owners = []
        local_offsets = []
        local_lengths = []
        for seg_idx, seg in enumerate(self.segments):
            m = len(seg.control_points)
            if m < 2:
                continue
            step = 1.0 / (m - 1)
            for j in range(m - 1):
                owners.append(seg_idx)
                local_offsets.append(j * step)
                local_lengths.append(step)
        self.polyline_segment_owner = np.array(owners, dtype=int)
        self.polyline_segment_local_offset = np.array(local_offsets, dtype=float)
        self.polyline_segment_local_length = np.array(local_lengths, dtype=float)
        self.domain_limit = float(self.segment_offsets[-1])
        self._cache_version = getattr(self, "_cache_version", 0) + 1

    def _select_segment(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s_arr = np.atleast_1d(s)
        s_clamped = np.clip(s_arr, 0.0, self.domain_limit)
        idx = np.searchsorted(self.segment_offsets, s_clamped, side="right") - 1
        idx = np.clip(idx, 0, len(self.segments) - 1)

        starts = self.segment_offsets[idx]
        seg_len = self.segment_offsets[idx + 1] - self.segment_offsets[idx]
        seg_len = np.where(seg_len > 1e-9, seg_len, 1.0)
        u = (s_clamped - starts) / seg_len
        u = np.clip(u, 0.0, 1.0)
        return idx, u

    def evaluate(self, s: Union[float, np.ndarray]) -> np.ndarray:
        s_arr = np.atleast_1d(s)
        idx, u = self._select_segment(s_arr)

        res = np.zeros((len(s_arr), 2))
        for seg_idx in np.unique(idx):
            mask = idx == seg_idx
            res[mask] = self.segments[int(seg_idx)].evaluate(u[mask])
        return res if res.shape[0] > 1 else res[0]

    def derivative(self, s: Union[float, np.ndarray]) -> np.ndarray:
        s_arr = np.atleast_1d(s)
        idx, u = self._select_segment(s_arr)

        res = np.zeros((len(s_arr), 2))
        for seg_idx in np.unique(idx):
            mask = idx == seg_idx
            res[mask] = self.segments[int(seg_idx)].derivative(u[mask])
        return res if res.shape[0] > 1 else res[0]

    def second_derivative(self, s: Union[float, np.ndarray]) -> np.ndarray:
        s_arr = np.atleast_1d(s)
        idx, u = self._select_segment(s_arr)

        res = np.zeros((len(s_arr), 2))
        for seg_idx in np.unique(idx):
            mask = idx == seg_idx
            res[mask] = self.segments[int(seg_idx)].second_derivative(u[mask])
        return res if res.shape[0] > 1 else res[0]

    def project_to_polyline(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must be shaped (N, 2)")

        poly = self.global_control_points
        if len(poly) < 2:
            raise ValueError("not enough control points to form polyline")

        A = poly[:-1]
        B = poly[1:]
        AB = B - A
        AB_len2 = np.sum(AB * AB, axis=1)
        AB_len2 = np.where(AB_len2 > 1e-9, AB_len2, 1.0)

        AP = pts[:, np.newaxis, :] - A[np.newaxis, :, :]
        t = np.sum(AP * AB[np.newaxis, :, :], axis=2) / AB_len2[np.newaxis, :]
        t = np.clip(t, 0.0, 1.0)
        proj = A[np.newaxis, :, :] + t[:, :, np.newaxis] * AB[np.newaxis, :, :]
        diff = pts[:, np.newaxis, :] - proj
        dist2 = np.sum(diff * diff, axis=2)

        seg_idx = np.argmin(dist2, axis=1)
        t_best = t[np.arange(len(pts)), seg_idx]
        proj_best = proj[np.arange(len(pts)), seg_idx]
        if len(self.polyline_segment_owner) == 0:
            raise ValueError("invalid polyline mapping for segments")

        seg_owner = self.polyline_segment_owner[seg_idx]
        local_offset = self.polyline_segment_local_offset[seg_idx]
        local_len = self.polyline_segment_local_length[seg_idx]
        local_param = local_offset + t_best * local_len
        local_param = np.clip(local_param, 0.0, 1.0)
        return seg_owner, local_param, proj_best

    def extend(self, side: str, point_local: np.ndarray) -> float:
        point_local = np.array(point_local, dtype=float)
        if side not in ("head", "tail"):
            raise ValueError("side must be 'head' or 'tail'")

        if not self.segments:
            self.segments.append(CatmullRomSegment(np.array([point_local])))
            self._rebuild_cache()
            return 0.0

        if side == "tail":
            last_seg = self.segments[-1]
            last_point = last_seg.control_points[-1]
            delta_L = float(np.linalg.norm(point_local - last_point))
            last_seg.add_control_point(point_local)
        else:
            first_seg = self.segments[0]
            first_point = first_seg.control_points[0]
            delta_L = float(np.linalg.norm(first_point - point_local))
            first_seg.control_points = np.vstack([point_local, first_seg.control_points])
            first_seg._rebuild_lengths()

        self._rebuild_cache()
        return delta_L

    def check_continuity(self, new_point: np.ndarray, angle_threshold: float) -> bool:
        if not self.segments:
            return True
        last_seg = self.segments[-1]
        if len(last_seg.control_points) < 2:
            return True

        p_end = last_seg.control_points[-1]
        p_prev = last_seg.control_points[-2]
        v1 = p_end - p_prev
        v2 = np.array(new_point, dtype=float) - p_end

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return True

        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angle = float(np.arccos(cosang))
        return angle < angle_threshold

class Entity:
    """
    Definition 2.3: Hypothesis Entity H = (M, g_hat, Sigma_g, E, L_hat, t_last)
    """
    def __init__(self, 
                 model: CurveModel, 
                 pose: SE2Pose, 
                 covariance: StateCovariance,
                 timestamp: float,
                 state_vector: StateVector):
        self.model = model             # M
        self.pose = pose               # g_hat
        self.covariance = covariance   # Sigma_g
        self.evidence = EvidenceSet()  # E
        # L_hat is tracked in model, but also exposed here
        self.estimated_length = model.domain_limit
        self.last_update_time = timestamp # t_last
        self.state_vector = state_vector # Full state including kinematics
        self.is_splitting = False

    def get_open_boundary(self, bandwidth_c: float = 2.0) -> List[np.ndarray]:
        """
        Definition 2.4: Endpoint Set B.
        Returns the global coordinates of the boundaries.
        """
        boundary_points = []
        
        # Always add physical endpoints of the model domain
        p_start_local = self.model.evaluate(0)
        p_end_local = self.model.evaluate(self.model.domain_limit)
        
        # Apply pose g
        g_mat = self.pose.to_matrix()
        
        def transform(p_local):
            return (g_mat @ np.append(p_local, 1.0))[:2]

        boundary_points.append(transform(p_start_local))
        boundary_points.append(transform(p_end_local))
        
        # Note: Detection of internal boundaries based on density gradient 
        # requires finding s where d(mu)/ds < delta. 
        # This is implementation dependent and omitted for basic structure.
        
        return boundary_points

    def is_geometrically_closed(self, epsilon_gap=0.1, epsilon_pos=0.1, epsilon_tan=0.1) -> bool:
        """
        Definition 4.2: Geometric Closure check.
        """
        # 1. High Coverage: m(I_covered) / L > 1 - eps
        sigma_bar = self.evidence.get_average_sigma()
        h = 2.0 * sigma_bar if sigma_bar > 0 else 0.1
        
        s_samples = np.linspace(0, self.model.domain_limit, 50)
        covered_count = 0
        density_thresh = 0.01 
        
        for s in s_samples:
             if self.evidence.get_density_at(s, h) > density_thresh:
                 covered_count += 1
                 
        coverage_ratio = covered_count / len(s_samples)
        if coverage_ratio <= 1.0 - epsilon_gap:
            return False
            
        # 2. Position Closure: |gamma(0) - gamma(L)| < eps
        p0 = self.model.evaluate(0)
        pL = self.model.evaluate(self.model.domain_limit)
        if np.linalg.norm(p0 - pL) >= epsilon_pos:
            return False
            
        # 3. Tangent Closure: |gamma'(0) - gamma'(L)| < eps
        t0 = self.model.derivative(0)
        tL = self.model.derivative(self.model.domain_limit)
        if np.linalg.norm(t0 - tL) >= epsilon_tan:
            return False
            
        return True
