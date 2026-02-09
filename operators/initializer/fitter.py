import numpy as np
from typing import Dict, List, Union
from sklearn.decomposition import PCA
from entity.entity import GeometricModel, CubicBSplineModel, PiecewiseCatmullRomModel, CatmullRomSegment

class ParamRangeTransformer:
    """
    Utility to shift a symmetric parameter range (e.g., [-L/2, L/2]) so that it
    starts at 0 for components that assume non-negative s.

    This keeps the original length and provides reversible mapping:
    - to_zero_based: maps any s in [s_min, s_max] to [0, L]
    - from_zero_based: inverse mapping back to the original centered interval.
    """

    def __init__(self, s_min: float, s_max: float):
        if s_max <= s_min:
            raise ValueError("s_max must be greater than s_min")
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.length = self.s_max - self.s_min

    def to_zero_based(self, s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Map s from [s_min, s_max] to [0, length], clamped to bounds."""
        s_clamped = np.clip(s, self.s_min, self.s_max)
        return s_clamped - self.s_min

    def from_zero_based(self, s_zero: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Inverse of to_zero_based: map from [0, length] back to [s_min, s_max]."""
        s_clamped = np.clip(s_zero, 0.0, self.length)
        return s_clamped + self.s_min

class PCABasedFitter:
    """
    Fits a linear geometric model using PCA: returns center, heading, length,
    projections, control points, and range transformer for parameter mapping.
    """

    def fit(self, points: np.ndarray) -> Dict[str, Union[np.ndarray, float, ParamRangeTransformer]]:
        if points.shape[0] != 2:
            raise ValueError("points must be shaped (2, N)")

        X = points.T
        pca = PCA(n_components=2)
        pca.fit(X)

        center = pca.mean_
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        idx_major = int(np.argmax(eigenvalues))
        major_axis_vec = eigenvectors[idx_major]
        orientation = float(np.arctan2(major_axis_vec[1], major_axis_vec[0]))

        vecs = X - center
        projections = np.dot(vecs, major_axis_vec)

        min_s = float(np.min(projections))
        max_s = float(np.max(projections))
        range_transformer = ParamRangeTransformer(min_s, max_s)
        length = range_transformer.length

        half_length = length / 2.0
        control_points = np.array([[-half_length, 0.0], [half_length, 0.0]])

        model = GeometricModel(control_points=control_points, domain_limit=length)

        return {
            "center": center,
            "orientation": orientation,
            "length": length,
            "projections": projections,
            "range_transformer": range_transformer,
            "control_points": control_points,
            "major_axis": major_axis_vec,
            "model": model
        }

# TODO: 确定分段点的逻辑还可以优化（目前时常会在不需要分段的地方分段，该分段的地方不分段）
# TODO: 自主调整 epsilon, window, theta_thresh 参数的逻辑有待实现
class RDPBSplineFitter:
    """
    Multi-stage fitter: RDP corner proposal -> PCA tangent validation ->
    corner refinement -> cubic B-spline segment stitching.
    
    修正点：
    1. 增加了局部坐标变换，确保 model 围绕原点 (0,0)。
    2. 使用全局 PCA 计算稳健的 orientation。
    3. 计算真实的投影参数 s 用于初始化 evidence。
    """

    def __init__(self, epsilon: float = 0.05, window: int = 3, theta_thresh: float = 0.5):
        self.epsilon = epsilon
        self.window = window
        self.theta_thresh = theta_thresh

    # -------- Stage 1: RDP candidate extraction --------
    def _rdp_candidates(self, pts: np.ndarray) -> List[int]:
        candidates: List[int] = []

        def recurse(start: int, end: int):
            if end - start <= 1:
                return
            p_start = pts[start]
            p_end = pts[end]
            seg = p_end - p_start
            seg_len_sq = float(np.dot(seg, seg))
            max_dist = -1.0
            max_idx = -1
            for idx in range(start + 1, end):
                vec = pts[idx] - p_start
                if seg_len_sq < 1e-9:
                    dist = float(np.linalg.norm(vec))
                else:
                    proj = float(np.dot(vec, seg) / seg_len_sq)
                    proj_point = p_start + proj * seg
                    dist = float(np.linalg.norm(pts[idx] - proj_point))
                if dist > max_dist:
                    max_dist = dist
                    max_idx = idx
            if max_dist > self.epsilon and max_idx != -1:
                candidates.append(max_idx)
                recurse(start, max_idx)
                recurse(max_idx, end)

        recurse(0, len(pts) - 1)
        return sorted(set(candidates))

    # -------- Stage 2: PCA tangent validation --------
    def _validate_corners(self, pts: np.ndarray, candidates: List[int]) -> List[Dict[str, np.ndarray]]:
        validated: List[Dict[str, np.ndarray]] = []
        w = self.window
        n = len(pts)
        for c in candidates:
            l0 = max(0, c - w)
            l1 = c + 1
            r0 = c
            r1 = min(n, c + w + 1)
            if l1 - l0 < 2 or r1 - r0 < 2:
                continue
            L_pts = pts[l0:l1]
            R_pts = pts[r0:r1]
            c_L, v_L = self._pca_dir(L_pts)
            c_R, v_R = self._pca_dir(R_pts)
            dot = float(np.clip(abs(np.dot(v_L, v_R)), -1.0, 1.0))
            alpha = float(np.arccos(dot))
            if alpha > self.theta_thresh:
                validated.append({
                    "index": c,
                    "C_L": c_L,
                    "Dir_L": v_L,
                    "C_R": c_R,
                    "Dir_R": v_R,
                })
        return validated

    def _pca_dir(self, pts: np.ndarray) -> tuple:
        pts_center = np.mean(pts, axis=0)
        cov = np.cov((pts - pts_center).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = int(np.argmax(eigvals))
        direction = eigvecs[:, idx]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / norm
        return pts_center, direction

    # -------- Stage 3: corner refinement --------
    def _refine_corners(self, pts: np.ndarray, corners: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        Q = [pts[0]]
        for item in corners:
            c_L, v_L = item["C_L"], item["Dir_L"]
            c_R, v_R = item["C_R"], item["Dir_R"]
            M = np.stack([v_L, -v_R], axis=1)
            rhs = c_R - c_L
            try:
                params = np.linalg.solve(M, rhs)
                t = params[0]
                p_int = c_L + t * v_L
            except np.linalg.LinAlgError:
                p_c = pts[item["index"]]
                p_int = 0.5 * (c_L + c_R) if np.linalg.norm(c_L - c_R) < np.linalg.norm(p_c - c_L) else p_c
            Q.append(p_int)
        Q.append(pts[-1])
        return Q

    # -------- Stage 4: segment-wise cubic B-spline stitch --------
    # 用于计算两点距离（用于参数化）
    def _chord_length_parameterize(self, points: np.ndarray) -> np.ndarray:
        """计算点集的弦长参数化 t (0 到 1)"""
        if len(points) < 2:
            return np.array([0.0])
        
        diffs = points[1:] - points[:-1]
        dists = np.linalg.norm(diffs, axis=1)
        u = np.concatenate(([0.0], np.cumsum(dists)))
        return u / u[-1]

    # 拟合单段贝塞尔曲线的控制点
    def _fit_bezier_segment(self, p0: np.ndarray, p3: np.ndarray, segment_points: np.ndarray):
        """
        给定起点 p0, 终点 p3 和中间的一组点 segment_points，
        使用最小二乘法求解最优控制点 cp1, cp2。
        """
        if len(segment_points) < 2:
            # 点太少，退化为直线
            return p0 + (p3 - p0) / 3.0, p0 + 2.0 * (p3 - p0) / 3.0

        # 1. 参数化：为每个点分配一个 t 值 (0 到 1)
        t = self._chord_length_parameterize(segment_points)
        
        # 2. 构建最小二乘矩阵 A 和 目标向量 b
        # 贝塞尔基函数:
        # B0(t) = (1-t)^3
        # B1(t) = 3(1-t)^2 * t
        # B2(t) = 3(1-t) * t^2
        # B3(t) = t^3
        # 方程: P(t) = B0*p0 + B1*cp1 + B2*cp2 + B3*p3
        # 移项得: P(t) - (B0*p0 + B3*p3) = B1*cp1 + B2*cp2
        
        t_vals = t
        b0 = (1 - t_vals) ** 3
        b3 = t_vals ** 3
        b1 = 3 * (1 - t_vals) ** 2 * t_vals
        b2 = 3 * (1 - t_vals) * t_vals ** 2
        
        # 矩阵 A: shape (N, 2)，列为 B1 和 B2
        A = np.column_stack((b1, b2))
        
        # 目标向量 rhs: shape (N, 2)，即 P(t) - (B0*p0 + B3*p3)
        # 注意 numpy 广播机制
        rhs = segment_points - (np.outer(b0, p0) + np.outer(b3, p3))
        
        # 3. 求解最小二乘问题: A * [cp1, cp2]^T ≈ rhs
        # 使用 numpy.linalg.lstsq 求解
        # result 的 shape 将是 (2, 2)，第一行是 cp1，第二行是 cp2
        try:
            result, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            cp1 = result[0]
            cp2 = result[1]
        except np.linalg.LinAlgError:
            # 如果求解失败（罕见），退化为直线
            cp1 = p0 + (p3 - p0) / 3.0
            cp2 = p0 + 2.0 * (p3 - p0) / 3.0
            
        return cp1, cp2

    # 分段确定控制点
    def _build_segments(self, Q: List[np.ndarray], all_points: np.ndarray) -> List[np.ndarray]:
        segments: List[np.ndarray] = []
        
        # 为了将 Q 中的点映射回 all_points 的索引，我们需要找最近点
        # 注意：Q 中的点是经过 refine_corners 微调过的，可能不精确等于 all_points 中的点
        indices_in_Q = []
        for q_point in Q:
            # 计算距离
            dists = np.linalg.norm(all_points - q_point, axis=1)
            # 找到最近点的索引
            nearest_idx = int(np.argmin(dists))
            indices_in_Q.append(nearest_idx)
            
        # 确保索引是递增的（防止拟合出错）
        # 简单的冒泡或直接排序
        indices_in_Q = sorted(indices_in_Q)

        for i in range(len(Q) - 1):
            p0 = Q[i]       # 使用精炼后的端点作为曲线端点
            p3 = Q[i + 1]
            
            start_idx = indices_in_Q[i]
            end_idx = indices_in_Q[i + 1]
            
            # 提取该段内的所有点
            if end_idx - start_idx < 2:
                # 如果中间没有点或点太少，使用直线段
                chord = p3 - p0
                cp1 = p0 + chord / 3.0
                cp2 = p0 + 2.0 * chord / 3.0
            else:
                segment_points = all_points[start_idx : end_idx + 1]
                # !!! 核心修改：使用拟合算法计算控制点，而不是强行取直线上的点 !!!
                cp1, cp2 = self._fit_bezier_segment(p0, p3, segment_points)
            
            segments.append(np.stack([p0, cp1, cp2, p3], axis=0))
            
        return segments

    def fit(self, points: np.ndarray) -> Dict[str, Union[np.ndarray, float, ParamRangeTransformer]]:
        if points.shape[0] != 2:
            raise ValueError("points must be shaped (2, N)")
        if points.shape[1] < 2:
            raise ValueError("need at least 2 points to fit")

        # 1. 全局 PCA 计算：确定真实的物理主轴方向和中心
        #    Entity Pose = (center_global, orientation)
        pca = PCA(n_components=2)  # 要降维到 2 维
        pca.fit(points.T)
        center_global = pca.mean_  # 质心
        eigenvectors = pca.components_  # 主成分向量（是一个 n_components x n_features 矩阵，其中 n_components 是主成分数量，n_features 是输入点的坐标维数（x 和 y 就是 2 维）。以 2 维 n_components 为例，这个矩阵一共两行，每行是一个主成分方向）
        eigenvalues = pca.explained_variance_  # 每个主成分方向上的数据方差

        # 选取最大特征值对应的主轴
        idx_major = int(np.argmax(eigenvalues))
        orientation_vec = eigenvectors[idx_major]
        orientation = float(np.arctan2(orientation_vec[1], orientation_vec[0]))  # 主轴相对于 x 轴的角度

        # 2. 坐标系对齐：将点云转换到"模型局部坐标系"
        #    模型坐标系的 X 轴对齐主轴 (orientation)，原点为中心 (center_global)
        #    P_local = R(-theta) * (P_global - Center)
        c, s = np.cos(orientation), np.sin(orientation)
        # 旋转矩阵 R(theta) = [[c, -s], [s, c]]，用于将局部坐标转换到全局坐标
        # 逆旋转 R(-theta) = R(theta).T = [[c, s], [-s, c]]，用于将全局坐标转换到局部坐标
        R_inv = np.array([[c, s], [-s, c]])  # 逆旋转矩阵
        
        # 本质上是“平移”操作
        vecs_global = points.T - center_global  # 计算所有点相对于质心的偏移向量 (N, 2)
        # 本质上是“旋转”操作
        # 转置以便右乘或者做矩阵乘法，这里 vecs_global 是 (N,2)，我们需要 (2,N) 做 matrix alg
        pts_local_aligned = (R_inv @ vecs_global.T).T # 这个就是局部对齐坐标系下的点 (N, 2)

        # 3. 在对齐后的局部空间执行 RDP -> B样条
        #    如果点云是无序的，通常建议按 x 轴排序 (即主轴投影) 
        #    这里假设输入已经有序或直接按序处理，如果需要排序可在此处增加:
        #    order = np.argsort(pts_local_aligned[:, 0])
        #    pts_local_aligned = pts_local_aligned[order]
        
        candidates = self._rdp_candidates(pts_local_aligned)  # Stage 1: 获取 RDP 拟合的候选拐点索引
        corners = self._validate_corners(pts_local_aligned, candidates)  # Stage 2: 获取有效的拐点索引
        refined = self._refine_corners(pts_local_aligned, corners)  # Stage 3: 获取精炼后的拐点坐标列表
        segments = self._build_segments(refined, pts_local_aligned)  # Stage 4: 构建分段贝塞尔曲线控制点列表

        # 4. 构建 Model (Control Points 已经在局部对齐坐标系中)
        model = CubicBSplineModel(segments)

        # 5. 计算投影参数 s
        #    在对齐坐标系下，投影 s 即为 x 坐标 (忽略 y 的偏差)
        projections = pts_local_aligned[:, 0]
        min_s = float(np.min(projections))
        max_s = float(np.max(projections))
        
        # 6. 元数据构建
        #    RangeTransformer 将 [min_s, max_s] 映射到 [0, length]
        length = max_s - min_s
        range_transformer = ParamRangeTransformer(min_s, max_s)

        return {
            "center": center_global,
            "orientation": orientation,
            "length": length,
            "projections": projections,
            "range_transformer": range_transformer,
            "control_points": model.control_points,
            "model": model,
        }


class RDPCatmullRomFitter:
    """
    RDP-based fitter for Piecewise Catmull-Rom model (interpolating spline).
    Reuses RDP + PCA corner validation + corner refinement, but constructs
    Catmull-Rom segments directly from downsampled points.
    """

    def __init__(self, epsilon: float = 0.05, window: int = 3, theta_thresh: float = 0.5, sample_dist: float = 0.2):
        self.epsilon = epsilon
        self.window = window
        self.theta_thresh = theta_thresh
        self.sample_dist = sample_dist

    # -------- Stage 1: RDP candidate extraction --------
    def _rdp_candidates(self, pts: np.ndarray) -> List[int]:
        candidates: List[int] = []

        def recurse(start: int, end: int):
            if end - start <= 1:
                return
            p_start = pts[start]
            p_end = pts[end]
            seg = p_end - p_start
            seg_len_sq = float(np.dot(seg, seg))
            max_dist = -1.0
            max_idx = -1
            for idx in range(start + 1, end):
                vec = pts[idx] - p_start
                if seg_len_sq < 1e-9:
                    dist = float(np.linalg.norm(vec))
                else:
                    proj = float(np.dot(vec, seg) / seg_len_sq)
                    proj_point = p_start + proj * seg
                    dist = float(np.linalg.norm(pts[idx] - proj_point))
                if dist > max_dist:
                    max_dist = dist
                    max_idx = idx
            if max_dist > self.epsilon and max_idx != -1:
                candidates.append(max_idx)
                recurse(start, max_idx)
                recurse(max_idx, end)

        recurse(0, len(pts) - 1)
        return sorted(set(candidates))

    # -------- Stage 2: PCA tangent validation --------
    def _validate_corners(self, pts: np.ndarray, candidates: List[int]) -> List[Dict[str, np.ndarray]]:
        validated: List[Dict[str, np.ndarray]] = []
        w = self.window
        n = len(pts)
        for c in candidates:
            l0 = max(0, c - w)
            l1 = c + 1
            r0 = c
            r1 = min(n, c + w + 1)
            if l1 - l0 < 2 or r1 - r0 < 2:
                continue
            L_pts = pts[l0:l1]
            R_pts = pts[r0:r1]
            c_L, v_L = self._pca_dir(L_pts)
            c_R, v_R = self._pca_dir(R_pts)
            dot = float(np.clip(abs(np.dot(v_L, v_R)), -1.0, 1.0))
            alpha = float(np.arccos(dot))
            if alpha > self.theta_thresh:
                validated.append({
                    "index": c,
                    "C_L": c_L,
                    "Dir_L": v_L,
                    "C_R": c_R,
                    "Dir_R": v_R,
                })
        return validated

    def _pca_dir(self, pts: np.ndarray) -> tuple:
        pts_center = np.mean(pts, axis=0)
        cov = np.cov((pts - pts_center).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = int(np.argmax(eigvals))
        direction = eigvecs[:, idx]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / norm
        return pts_center, direction

    # -------- Stage 3: corner refinement --------
    def _refine_corners(self, pts: np.ndarray, corners: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        Q = [pts[0]]
        for item in corners:
            c_L, v_L = item["C_L"], item["Dir_L"]
            c_R, v_R = item["C_R"], item["Dir_R"]
            M = np.stack([v_L, -v_R], axis=1)
            rhs = c_R - c_L
            try:
                params = np.linalg.solve(M, rhs)
                t = params[0]
                p_int = c_L + t * v_L
            except np.linalg.LinAlgError:
                p_c = pts[item["index"]]
                p_int = 0.5 * (c_L + c_R) if np.linalg.norm(c_L - c_R) < np.linalg.norm(p_c - c_L) else p_c
            Q.append(p_int)
        Q.append(pts[-1])
        return Q

    def _downsample_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        kept = [points[0]]
        last = points[0]
        for p in points[1:]:
            if np.linalg.norm(p - last) >= self.sample_dist:
                kept.append(p)
                last = p
        if not np.allclose(kept[-1], points[-1]):
            kept.append(points[-1])
        return np.array(kept)

    def _build_segments(self, Q: List[np.ndarray], all_points: np.ndarray) -> List[CatmullRomSegment]:
        segments: List[CatmullRomSegment] = []

        indices_in_Q = []
        for q_point in Q:
            dists = np.linalg.norm(all_points - q_point, axis=1)
            nearest_idx = int(np.argmin(dists))
            indices_in_Q.append(nearest_idx)

        indices_in_Q = sorted(indices_in_Q)

        for i in range(len(Q) - 1):
            p0 = Q[i]
            p1 = Q[i + 1]
            start_idx = indices_in_Q[i]
            end_idx = indices_in_Q[i + 1]

            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx

            segment_points = all_points[start_idx:end_idx + 1]
            if len(segment_points) == 0:
                ctrl = np.vstack([p0, p1])
            else:
                middle = self._downsample_points(segment_points)
                ctrl = np.vstack([p0, middle, p1])

            segments.append(CatmullRomSegment(ctrl))

        return segments

    def fit(self, points: np.ndarray) -> Dict[str, Union[np.ndarray, float, ParamRangeTransformer]]:
        if points.shape[0] != 2:
            raise ValueError("points must be shaped (2, N)")
        if points.shape[1] < 2:
            raise ValueError("need at least 2 points to fit")

        pca = PCA(n_components=2)
        pca.fit(points.T)
        center_global = pca.mean_
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        idx_major = int(np.argmax(eigenvalues))
        orientation_vec = eigenvectors[idx_major]
        orientation = float(np.arctan2(orientation_vec[1], orientation_vec[0]))

        c, s = np.cos(orientation), np.sin(orientation)
        R_inv = np.array([[c, s], [-s, c]])
        vecs_global = points.T - center_global
        pts_local_aligned = (R_inv @ vecs_global.T).T

        candidates = self._rdp_candidates(pts_local_aligned)
        corners = self._validate_corners(pts_local_aligned, candidates)
        refined = self._refine_corners(pts_local_aligned, corners)
        segments = self._build_segments(refined, pts_local_aligned)

        model = PiecewiseCatmullRomModel(segments)

        seg_idx, t_vals, _ = model.project_to_polyline(pts_local_aligned)
        seg_lengths = model.segment_offsets[1:] - model.segment_offsets[:-1]
        s_vals = model.segment_offsets[seg_idx] + t_vals * seg_lengths[seg_idx]

        min_s = float(np.min(s_vals)) if len(s_vals) > 0 else 0.0
        max_s = float(np.max(s_vals)) if len(s_vals) > 0 else 0.0
        length = max_s - min_s
        range_transformer = ParamRangeTransformer(min_s, max_s) if max_s > min_s else ParamRangeTransformer(0.0, 1.0)

        return {
            "center": center_global,
            "orientation": orientation,
            "length": length,
            "projections": s_vals,
            "range_transformer": range_transformer,
            "control_points": model.global_control_points,
            "model": model,
        }
