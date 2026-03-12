from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class DesignCConfig:
    epsilon: float = 0.05


def phase_logit_distance(u: np.ndarray, epsilon: float) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), -1 + 1e-8, 1 - 1e-8)
    return np.sqrt(2.0) * float(epsilon) * np.arctanh(u)
