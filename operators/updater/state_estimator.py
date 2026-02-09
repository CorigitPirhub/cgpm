import numpy as np
from typing import List
from entity.entity import Entity
from entity.state import StateVector, SE2Pose
from operators.associator.associator import AssociationResult

class StateEstimator:
    """
    Implements the "Where is it?" part of the update step (Robust EKF).
    Features:
    1. Mahalanobis Distance Gating: Rejects/Down-weights outliers caused by model mismatch.
    2. Velocity Damping: Prevents ghost motion for static objects by suppressing velocity updates.
    """
    def __init__(self, obs_sigma: float = 0.05, gate_threshold: float = 5.99, vel_damping: float = 0.2):
        """
        Args:
            obs_sigma: Base observation noise std.
            gate_threshold: Chi-squared threshold (2 DOF) for gating. 
                            5.99 corresponds to 95% confidence interval.
                            Higher = more permissive, Lower = more robust.
            vel_damping: Factor (0.0-1.0) to dampen velocity (v, omega) updates.
                         Lower = more resistance to ghost motion.
        """
        self.obs_sigma = obs_sigma
        self.gate_threshold = gate_threshold
        self.vel_damping = vel_damping

    def estimate(self, entity: Entity, associations: List[AssociationResult], observations: np.ndarray) -> Entity:
        if not associations:
            return entity

        # 1. State vector and Covariance
        x_vec = entity.state_vector.vector.copy()  # 状态向量 [x, y, theta, v, omega]
        P = entity.covariance.matrix.copy()  # 协方差矩阵 (5 x 5)

        # Observation Noise Covariance R (2x2)
        R_cov = (self.obs_sigma ** 2) * np.eye(2)  # 观测噪声协方差矩阵 (2x2)

        for assoc in associations:
            # 计算观测值和预测值之间的残差
            z_obs = observations[:, assoc.obs_index]
            s_star = assoc.s_star
            
            # --- 1. Build Observation Model ---
            theta = x_vec[2]
            c, s = np.cos(theta), np.sin(theta)
            R_mat = np.array([[c, -s], [s, c]])
            t_vec = x_vec[:2]
            
            p_local = entity.model.evaluate(s_star)
            z_pred = R_mat @ p_local + t_vec
            
            # --- 2. Calculate Innovation (Residual) ---
            nu = z_obs - z_pred
            
            # 计算雅可比矩阵 H
            # --- 3. Calculate Jacobian H ---
            px, py = p_local
            dz_dtheta = np.array([
                -s * px - c * py,
                 c * px - s * py
            ])
            
            H = np.zeros((2, 5))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            H[:, 2] = dz_dtheta
            
            # 计算卡尔曼增益 K
            # --- 4. Kalman Gain & Robustness Check ---
            S = H @ P @ H.T + R_cov
            
            try:
                S_inv = np.linalg.inv(S)
                K = P @ H.T @ S_inv
            except np.linalg.LinAlgError:
                continue
            
            # 鲁棒性检查：基于马氏距离的 gating
            # --- 5. Robustness: Gating based on Mahalanobis Distance ---
            # d^2 = nu.T * S_inv * nu
            # This measures how "likely" this observation is given the prediction uncertainty.
            md_squared = nu @ S_inv @ nu
            
            # Scale factor for the update
            # If the residual is too large (unlikely), scale down the gain K
            scale = 1.0
            if md_squared > self.gate_threshold:
                # Huber-like approach or simple clamping
                scale = self.gate_threshold / md_squared
            
            # Apply robustness scaling to the Kalman Gain
            # If observation is an outlier, K becomes smaller -> update is smaller
            K_robust = K * scale
            
            # --- 6. State Update ---
            delta = K_robust @ nu
            
            # --- 7. Velocity Damping (Fix Ghost Motion) ---
            # Even if the position updates, we resist changing velocity/omega 
            # based on single-frame noise.
            delta[3] *= self.vel_damping # v
            delta[4] *= self.vel_damping # omega
            
            # Apply update
            x_vec = x_vec + delta
            P = (np.eye(5) - K_robust @ H) @ P # Joseph form for stability with variable K
            
            # Normalize Theta
            x_vec[2] = (x_vec[2] + np.pi) % (2 * np.pi) - np.pi
            
        # Update Entity
        entity.state_vector = StateVector(*x_vec)
        entity.pose = SE2Pose(entity.state_vector.x, entity.state_vector.y, entity.state_vector.theta)
        entity.covariance.matrix = P
        entity.last_update_time = float(np.max([0] + [entity.last_update_time])) # Should be updated by caller actually
        
        return entity
