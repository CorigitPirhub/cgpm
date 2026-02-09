import numpy as np
from enum import Enum
from typing import Protocol, Tuple, Dict
from entity.entity import Entity
from entity.state import StateVector, SE2Pose

class MotionModelType(Enum):
    STATIC = "STATIC"
    CV = "CV"
    CTRV = "CTRV"

class MotionModel(Protocol):
    def predict(self, state: StateVector, cov: np.ndarray, dt: float, params: Dict[str, float]) -> Tuple[StateVector, np.ndarray]:
        ...

# TODO: 实现不同的运动模型预测逻辑
# 静态模型
class StaticModel:
    def predict(self, state: StateVector, cov: np.ndarray, dt: float, params: Dict[str, float]) -> Tuple[StateVector, np.ndarray]:
        new_state = StateVector(state.x, state.y, state.theta, 0.0, 0.0)
        J = np.eye(5)
        Q = self._build_noise(dt, params)
        new_cov = J @ cov @ J.T + Q
        return new_state, new_cov

    def _build_noise(self, dt: float, params: Dict[str, float]) -> np.ndarray:
        std_acc = params.get('std_acc', 0.01)
        std_yawdd = params.get('std_yawdd', 0.01)
        q_pos = (std_acc * dt) ** 2
        q_theta = (std_yawdd * dt) ** 2
        q_v = (std_acc * dt) ** 2
        q_omega = (std_yawdd * dt) ** 2
        return np.diag([q_pos, q_pos, q_theta, q_v, q_omega])

# 匀速直线模型
class CVModel:
    def predict(self, state: StateVector, cov: np.ndarray, dt: float, params: Dict[str, float]) -> Tuple[StateVector, np.ndarray]:
        new_state = self._cv_function(state, dt)
        J = self._cv_jacobian(state, dt)
        Q = self._build_noise(dt, params)
        new_cov = J @ cov @ J.T + Q
        return new_state, new_cov

    def _cv_function(self, state: StateVector, dt: float) -> StateVector:
        x = state.x
        y = state.y
        theta = state.theta
        v = state.v
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta
        v_new = v
        omega_new = state.omega
        return StateVector(x_new, y_new, theta_new, v_new, omega_new)

    def _cv_jacobian(self, state: StateVector, dt: float) -> np.ndarray:
        theta = state.theta
        v = state.v
        J = np.eye(5)
        J[0, 2] = -v * np.sin(theta) * dt
        J[0, 3] =  np.cos(theta) * dt
        J[1, 2] =  v * np.cos(theta) * dt
        J[1, 3] =  np.sin(theta) * dt
        return J

    def _build_noise(self, dt: float, params: Dict[str, float]) -> np.ndarray:
        std_acc = params.get('std_acc', 0.5)
        std_yawdd = params.get('std_yawdd', 0.1)
        q_pos = (std_acc * dt) ** 2
        q_theta = (std_yawdd * dt) ** 2
        q_v = (std_acc * dt) ** 2
        q_omega = (std_yawdd * dt) ** 2
        return np.diag([q_pos, q_pos, q_theta, q_v, q_omega])

# 匀速转弯模型
class CTRVModel:
    def predict(self, state: StateVector, cov: np.ndarray, dt: float, params: Dict[str, float]) -> Tuple[StateVector, np.ndarray]:
        new_state = self._ctrv_function(state, dt)
        J = self._ctrv_jacobian(state, dt)
        Q = self._build_noise(dt, params)
        new_cov = J @ cov @ J.T + Q
        return new_state, new_cov

    def _ctrv_function(self, state: StateVector, dt: float) -> StateVector:
        x = state.x
        y = state.y
        theta = state.theta
        v = state.v
        omega = state.omega
        if abs(omega) > 1e-6:
            theta_new = theta + omega * dt
            x_new = x + v / omega * (np.sin(theta_new) - np.sin(theta))
            y_new = y + v / omega * (-np.cos(theta_new) + np.cos(theta))
        else:
            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt
            theta_new = theta
        v_new = v
        omega_new = omega
        return StateVector(x_new, y_new, theta_new, v_new, omega_new)

    def _ctrv_jacobian(self, state: StateVector, dt: float) -> np.ndarray:
        theta = state.theta
        v = state.v
        omega = state.omega
        J = np.eye(5)
        if abs(omega) > 1e-6:
            theta1 = theta + omega * dt
            s_th = np.sin(theta)
            c_th = np.cos(theta)
            s_th1 = np.sin(theta1)
            c_th1 = np.cos(theta1)
            A = s_th1 - s_th
            B = c_th - c_th1
            J[0, 2] = v / omega * (c_th1 - c_th)
            J[0, 3] = A / omega
            J[0, 4] = v * (omega * dt * c_th1 - A) / (omega ** 2)
            J[1, 2] = v / omega * (-np.sin(theta) + np.sin(theta1))
            J[1, 3] = B / omega
            J[1, 4] = v * (omega * dt * np.sin(theta1) - B) / (omega ** 2)
            J[2, 4] = dt
        else:
            J = self._cv_jacobian(state, dt)
        J[3, 3] = 1.0
        J[4, 4] = 1.0
        return J

    def _cv_jacobian(self, state: StateVector, dt: float) -> np.ndarray:
        theta = state.theta
        v = state.v
        J = np.eye(5)
        J[0, 2] = -v * np.sin(theta) * dt
        J[0, 3] =  np.cos(theta) * dt
        J[1, 2] =  v * np.cos(theta) * dt
        J[1, 3] =  np.sin(theta) * dt
        return J

    def _build_noise(self, dt: float, params: Dict[str, float]) -> np.ndarray:
        std_acc = params.get('std_acc', 0.5)
        std_yawdd = params.get('std_yawdd', 0.5)
        q_pos = (std_acc * dt) ** 2
        q_theta = (std_yawdd * dt) ** 2
        q_v = (std_acc * dt) ** 2
        q_omega = (std_yawdd * dt) ** 2
        return np.diag([q_pos, q_pos, q_theta, q_v, q_omega])

class Predictor:
    def __init__(self):
        # 定义不同模型的噪声参数
        self.noise_params = {
            MotionModelType.STATIC: {'std_acc': 0.01, 'std_yawdd': 0.01},
            MotionModelType.CV:     {'std_acc': 0.5,  'std_yawdd': 0.1},
            MotionModelType.CTRV:   {'std_acc': 0.5,  'std_yawdd': 0.5},
        }
        # 运动模型实例
        self.models: Dict[MotionModelType, MotionModel] = {
            MotionModelType.STATIC: StaticModel(),
            MotionModelType.CV: CVModel(),
            MotionModelType.CTRV: CTRVModel(),
        }

    def process(self, entity: Entity, dt: float) -> Entity:
        """
        主流程：选模型 -> 计算预测 -> 更新实体
        """
        # 1. 基于当前状态选择运动模型
        model_type = self._select_model_by_state(entity.state_vector)
        
        # 2. 获取该模型对应的噪声参数
        params = self.noise_params[model_type]
        
        # 3. 基于模型预测，核心是更新运动状态和协方差矩阵
        model = self.models[model_type]
        pred_state, pred_cov = model.predict(entity.state_vector, entity.covariance.matrix, dt, params)
        
        # 4. 输出更新后的实体 (修改 entity 属性)
        entity.state_vector = pred_state
        entity.pose = SE2Pose(pred_state.x, pred_state.y, pred_state.theta)
        entity.covariance.matrix = pred_cov

        return entity

    def _select_model_by_state(self, state: StateVector) -> MotionModelType:
        """第2点：模型选择逻辑"""
        if abs(state.v) < 0.1:
            return MotionModelType.STATIC
        elif abs(state.omega) < 0.1:
            return MotionModelType.CV
        else:
            return MotionModelType.CTRV

