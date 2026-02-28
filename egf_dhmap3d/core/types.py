from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VoxelCell3D:
    phi: float = 0.0
    phi_w: float = 0.0
    rho: float = 0.0
    g_mean: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    g_cov: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    c_rho: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    # Local dynamic memory for selective forgetting.
    d_score: float = 0.0
    surf_evidence: float = 0.0
    free_evidence: float = 0.0
    residual_evidence: float = 0.0
    rho_prev: float = 0.0
    rho_osc: float = 0.0
    clear_hits: float = 0.0
    # Frontier activation score (cold-start growth support).
    frontier_score: float = 0.0
    # Last frame index when the voxel was updated by a measurement.
    last_seen: int = 0


@dataclass
class PoseSE3State:
    t_wc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    r_wc: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    cov: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=float) * 0.01)

    def as_matrix(self) -> np.ndarray:
        out = np.eye(4, dtype=float)
        out[:3, :3] = self.r_wc
        out[:3, 3] = self.t_wc
        return out

    def set_matrix(self, t_wc: np.ndarray) -> None:
        self.r_wc = np.asarray(t_wc[:3, :3], dtype=float)
        self.t_wc = np.asarray(t_wc[:3, 3], dtype=float)
