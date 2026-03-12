from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class DesignBConfig:
    epsilon: float = 0.05


def quasipotential_transform(rho: np.ndarray, epsilon: float) -> np.ndarray:
    rho = np.clip(np.asarray(rho, dtype=float), 1e-12, None)
    return -float(epsilon) * np.log(rho)
