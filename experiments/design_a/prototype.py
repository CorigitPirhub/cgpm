from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class DesignAConfig:
    epsilon: float = 0.05


def log_amplitude_distance(psi: np.ndarray, epsilon: float) -> np.ndarray:
    psi = np.clip(np.asarray(psi, dtype=float), 1e-12, None)
    return -float(epsilon) * np.log(psi)
