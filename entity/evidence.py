import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

# TODO: EvidenceMeasure

@dataclass
class EvidencePoint:
    """
    Definition 2.2: Evidence e_k = (s_k, z_k, w_k, t_k)
    This represents a point on the model that has been observed.
    """
    s: float              # s_k: parameter coordinate on local model (after projection)
    z_global: np.ndarray  # z_k: global cartesian coordinate (x, y)
    weight: float         # w_k: weight, typically propto 1 / sigma^2
    timestamp: float      # t_k
    sigma: float          # Original std deviation (sqrt(variance)) for bandwidth calculation

    def __post_init__(self):
        # Ensure z_global is a numpy array
        if not isinstance(self.z_global, np.ndarray):
            self.z_global = np.array(self.z_global)

class EvidenceSet:
    def __init__(self, cell_size: float = 0.1, delta_s: float = 0.005, delta_xy: float = 0.08):
        self._evidence: List[EvidencePoint] = []
        self._delta_s = float(delta_s) if delta_s > 0 else 0.005
        self._delta_xy = float(delta_xy) if delta_xy > 0 else 0.08
        self._cell_size = float(cell_size) if cell_size > 0 else 0.1  # 网格索引单元大小
        if self._cell_size < self._delta_s:
            self._cell_size = self._delta_s
        self._s_grid: Dict[int, List[EvidencePoint]] = {}
        self._head_cache: List[EvidencePoint] = []
        self._tail_cache: List[EvidencePoint] = []
        self._head_prefit_cache: List[EvidencePoint] = []
        self._tail_prefit_cache: List[EvidencePoint] = []

    def get_cache_size(self, side: Optional[str] = None) -> int:
        if side is None:
            return len(self._head_cache) + len(self._tail_cache)
        if side == "head":
            return len(self._head_cache)
        if side == "tail":
            return len(self._tail_cache)
        raise ValueError("side must be 'head' or 'tail'")

    def get_prefit_cache_size(self, side: Optional[str] = None) -> int:
        if side is None:
            return len(self._head_prefit_cache) + len(self._tail_prefit_cache)
        if side == "head":
            return len(self._head_prefit_cache)
        if side == "tail":
            return len(self._tail_prefit_cache)
        raise ValueError("side must be 'head' or 'tail'")
        
    def add(self, point: EvidencePoint):
        candidate = self._find_merge_candidate(point.s)  # 检查是否存在参数距离较近的点
        # 如果找到了合并候选点，则进行加权平均合并
        if candidate is not None:
            self._merge_evidence(candidate, point)
            return
        # 如果没有找到合并候选点，则直接添加到证据集中
        self._evidence.append(point)
        key = int(point.s / self._cell_size)  # 网格索引键
        if key not in self._s_grid:
            self._s_grid[key] = []
        self._s_grid[key].append(point)

    def cache_out_of_domain(self, point: EvidencePoint, side: str, merge_spatial: bool = True) -> None:
        if side not in ("head", "tail"):
            raise ValueError("side must be 'head' or 'tail'")
        existing = self._find_merge_in_cache(point, side, merge_spatial=merge_spatial, prefit=False)
        if existing is not None:
            self._merge_evidence(existing, point)
            return
        if side == "head":
            self._head_cache.append(point)
        else:
            self._tail_cache.append(point)

    def cache_prefit(self, point: EvidencePoint, side: str, merge_spatial: bool = True) -> None:
        if side not in ("head", "tail"):
            raise ValueError("side must be 'head' or 'tail'")
        existing = self._find_merge_in_cache(point, side, merge_spatial=merge_spatial, prefit=True)
        if existing is not None:
            self._merge_evidence(existing, point)
            return
        if side == "head":
            self._head_prefit_cache.append(point)
        else:
            self._tail_prefit_cache.append(point)

    def flush_extension_cache(self, side: Optional[str] = None) -> List[EvidencePoint]:
        if side is None:
            cached = self._head_cache + self._tail_cache
            for ev in cached:
                self.add(ev)
            self._head_cache = []
            self._tail_cache = []
            return cached
        if side == "head":
            cached = self._head_cache
            for ev in cached:
                self.add(ev)
            self._head_cache = []
            return cached
        if side == "tail":
            cached = self._tail_cache
            for ev in cached:
                self.add(ev)
            self._tail_cache = []
            return cached
        raise ValueError("side must be 'head' or 'tail'")

    def get_cache(self, side: str) -> List[EvidencePoint]:
        if side == "head":
            return self._head_cache
        if side == "tail":
            return self._tail_cache
        raise ValueError("side must be 'head' or 'tail'")

    def get_prefit_cache(self, side: str) -> List[EvidencePoint]:
        if side == "head":
            return self._head_prefit_cache
        if side == "tail":
            return self._tail_prefit_cache
        raise ValueError("side must be 'head' or 'tail'")

    def clear_cache(self, side: Optional[str] = None) -> None:
        if side is None:
            self._head_cache = []
            self._tail_cache = []
            return
        if side == "head":
            self._head_cache = []
            return
        if side == "tail":
            self._tail_cache = []
            return
        raise ValueError("side must be 'head' or 'tail'")

    def clear_prefit_cache(self, side: Optional[str] = None) -> None:
        if side is None:
            self._head_prefit_cache = []
            self._tail_prefit_cache = []
            return
        if side == "head":
            self._head_prefit_cache = []
            return
        if side == "tail":
            self._tail_prefit_cache = []
            return
        raise ValueError("side must be 'head' or 'tail'")

    def set_cache(self, side: str, points: List[EvidencePoint]) -> None:
        if side == "head":
            self._head_cache = list(points)
            return
        if side == "tail":
            self._tail_cache = list(points)
            return
        raise ValueError("side must be 'head' or 'tail'")

    def set_prefit_cache(self, side: str, points: List[EvidencePoint]) -> None:
        if side == "head":
            self._head_prefit_cache = list(points)
            return
        if side == "tail":
            self._tail_prefit_cache = list(points)
            return
        raise ValueError("side must be 'head' or 'tail'")
        
    def get_all(self) -> List[EvidencePoint]:
        return self._evidence
    
    def __len__(self):
        return len(self._evidence)

    def find_nearest_by_s(self, s_query: float) -> List[EvidencePoint]:
        if not self._evidence:
            return []
        key = int(s_query / self._cell_size)
        candidates: List[EvidencePoint] = []
        for k in (key - 1, key, key + 1):
            bucket = self._s_grid.get(k)
            if bucket:
                candidates.extend(bucket)
        return candidates

    def _find_merge_candidate(self, s_query: float) -> Optional[EvidencePoint]:
        if not self._evidence:
            return None
        key = int(s_query / self._cell_size)
        best = None
        best_dist = None
        for k in (key - 1, key, key + 1):
            bucket = self._s_grid.get(k)
            if not bucket:
                continue
            for ev in bucket:
                d = abs(ev.s - s_query)
                if d < self._delta_s and (best_dist is None or d < best_dist):
                    best = ev
                    best_dist = d
        return best

    def _find_merge_in_cache(
        self,
        point: EvidencePoint,
        side: str,
        merge_spatial: bool = True,
        prefit: bool = False,
    ) -> Optional[EvidencePoint]:
        if side == "head":
            cache = self._head_prefit_cache if prefit else self._head_cache
        elif side == "tail":
            cache = self._tail_prefit_cache if prefit else self._tail_cache
        else:
            raise ValueError("side must be 'head' or 'tail'")
        if not cache:
            return None

        s_query = float(point.s)
        z_query = np.array(point.z_global)

        best = None
        best_dist = None
        for ev in cache:
            d = abs(ev.s - s_query)
            d_xy = float(np.linalg.norm(ev.z_global - z_query))
            if prefit:
                if merge_spatial:
                    cond = d_xy < (0.6 * self._delta_xy)
                else:
                    cond = d < (0.5 * self._delta_s)
            else:
                cond = (d < self._delta_s) or (merge_spatial and d_xy < self._delta_xy)
            if cond and (best_dist is None or d < best_dist):
                best = ev
                best_dist = d
        return best

    def _merge_evidence(self, target: EvidencePoint, incoming: EvidencePoint) -> None:
        w_old = float(target.weight)
        w_new = float(incoming.weight)
        w_sum = w_old + w_new
        if w_sum <= 1e-12:
            return
        target.z_global = (w_old * target.z_global + w_new * incoming.z_global) / w_sum
        target.weight = w_sum
        target.timestamp = incoming.timestamp
        target.sigma = (w_old * target.sigma + w_new * incoming.sigma) / w_sum

    def shift_s(self, delta: float) -> None:
        if abs(delta) < 1e-12:
            return
        for ev in self._evidence:
            ev.s += delta
        for ev in self._head_cache:
            ev.s += delta
        for ev in self._tail_cache:
            ev.s += delta
        for ev in self._head_prefit_cache:
            ev.s += delta
        for ev in self._tail_prefit_cache:
            ev.s += delta
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        self._s_grid = {}
        for ev in self._evidence:
            key = int(ev.s / self._cell_size)
            if key not in self._s_grid:
                self._s_grid[key] = []
            self._s_grid[key].append(ev)

    def get_average_sigma(self) -> float:
        """Helper for Definition 4.1: Compute mean sigma for bandwidth selection."""
        if not self._evidence:
            return 1.0 # Default fallback
        sigmas = [e.sigma for e in self._evidence]
        return np.mean(sigmas)

    def get_density_at(self, s_query: float, h: float) -> float:
        """
        Definition 4.1: Evidence Density Measure mu_E(s)
        mu_E(s) = sum( w_k * K_h(s - s_k) )
        """
        if not self._evidence or h <= 0:
            return 0.0
            
        density = 0.0
        # Normalization constant for Gaussian kernel
        norm_const = 1.0 / (np.sqrt(2 * np.pi) * h)
        
        # In an optimized implementation, we would use grid evaluation or KD-trees.
        # Here we do direct summation as per mathematical definition.
        for e in self._evidence:
            s_k = e.s
            w_k = e.weight
            
            diff = s_query - s_k
            # Gaussian Kernel
            k_val = norm_const * np.exp(-0.5 * (diff / h)**2)
            
            density += w_k * k_val
            
        return density

    def get_density_derivative_at(self, s_query: float, h: float) -> float:
        """
        Computes the spatial derivative of the evidence density function: d(mu_E)/ds.
        Useful for Definition 2.4 (Endpoint Set detection).
        """
        if not self._evidence or h <= 0:
            return 0.0
            
        deriv = 0.0
        norm_const = 1.0 / (np.sqrt(2 * np.pi) * h)
        
        for e in self._evidence:
            s_k = e.s
            w_k = e.weight
            diff = s_query - s_k
            
            # K_h(x)
            k_val = norm_const * np.exp(-0.5 * (diff / h)**2)
            
            # d/ds K_h(s - s_k) = K_h(s - s_k) * (-(s - s_k) / h^2)
            k_prime = k_val * (-diff / (h**2))
            
            deriv += w_k * k_prime
            
        return deriv

