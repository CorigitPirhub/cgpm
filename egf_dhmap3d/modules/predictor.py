from __future__ import annotations

import numpy as np

from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.core.types import PoseSE3State
from egf_dhmap3d.core.voxel_hash import VoxelHashMap3D


class Predictor3D:
    def __init__(self, cfg: EGF3DConfig):
        self.cfg = cfg
        self.pose = PoseSE3State()
        self._frame_counter = 0

    def set_pose(self, t_wc: np.ndarray) -> None:
        self.pose.set_matrix(t_wc)

    def predict(self, delta_t: np.ndarray | None, dt: float) -> None:
        if delta_t is not None:
            t_prev = self.pose.as_matrix()
            t_new = t_prev @ np.asarray(delta_t, dtype=float)
            self.pose.set_matrix(t_new)

        q_t = max(1e-6, self.cfg.predict.process_noise_trans * dt)
        q_r = max(1e-6, self.cfg.predict.process_noise_rot * dt)
        q = np.diag([q_t * q_t, q_t * q_t, q_t * q_t, q_r * q_r, q_r * q_r, q_r * q_r])
        self.pose.cov = self.pose.cov + q

    def apply_field_prediction(self, voxel_map: VoxelHashMap3D, dynamic_score: float = 0.0) -> None:
        mode = str(self.cfg.update.forget_mode).lower()
        dyn_gain = float(np.clip(self.cfg.update.dyn_forget_gain, 0.0, 1.0))
        self._frame_counter += 1
        decay_interval = int(max(1, getattr(self.cfg.update, "decay_interval_frames", 1)))
        if (self._frame_counter % decay_interval) != 0:
            return
        voxel_map.decay_fields(
            mode=mode,
            global_dynamic_score=float(np.clip(dynamic_score, 0.0, 1.0)),
            dyn_forget_gain=dyn_gain,
            decay_steps=decay_interval,
        )
