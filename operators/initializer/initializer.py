import numpy as np
from typing import List
import sys
import os
import time

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from entity.entity import Entity, GeometricModel
from entity.state import SE2Pose, StateCovariance, StateVector
from entity.evidence import EvidencePoint, EvidenceSet
from .fitter import PCABasedFitter, RDPBSplineFitter, RDPCatmullRomFitter

# TODO: 随流程的进展动态更新噪声方差
class EntityInitializer:
    """
    Operator responsible for converting raw clustered points into initialized Entity hypotheses.
    Corresponds to the "Initial Injection" phase.
    """
    
    def __init__(self):
        # self.fitter = PCABasedFitter()
        # self.fitter = RDPBSplineFitter(epsilon=0.2, window=3, theta_thresh=0.7)
        self.fitter = RDPCatmullRomFitter(epsilon=0.2, window=3, theta_thresh=0.7, sample_dist=0.4)

    def process(self, points: np.ndarray, labels: np.ndarray, timestamp: float, sensor_pos: np.ndarray = None) -> List[Entity]:
        """
        Converts labeled point cloud data into a list of Entity objects.
        
        Args:
            points: (2, N) numpy array of global coordinates.
            labels: (N,) numpy array of cluster labels. -1 indicates noise.
            timestamp: Current simulation time.
            
        Returns:
            List[Entity]: List of newly created entities.
        """
        
        entities = []

        # 把与 points 数量相同的 labels 转换为仅含有效簇标签的集合
        unique_labels = set(labels)
        # 忽略噪声标签
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        for label in unique_labels:
            # 仅提取当前簇的点
            mask = (labels == label)
            cluster_points = points[:, mask] # (2, M)
            # 若簇中点数小于2，则跳过
            if cluster_points.shape[1] < 2:
                continue
                
            entity = self._initialize_single_entity(cluster_points, timestamp, sensor_pos)
            if entity:
                entities.append(entity)
                
        return entities

    def _initialize_single_entity(self, points: np.ndarray, timestamp: float, sensor_pos: np.ndarray = None) -> Entity:
        """
        Creates an Entity from a single cluster of points.
        REVISION: Local Frame Origin is set to PCA Center (Center of Mass).
        """
        fit = self.fitter.fit(points)

        center = fit["center"]  # 质心坐标
        orientation = fit["orientation"]  # 主轴方向
        length = fit["length"]
        projections = fit["projections"]
        range_transformer = fit["range_transformer"]
        control_points = fit["control_points"]
        model = fit.get("model")

        # 创建初始状态向量
        x, y, theta = center[0], center[1], orientation
        v, w = 0.0, 0.0
        state_vector = StateVector(x, y, theta, v, w)
        
        # 确定 Pose（即 g^）
        pose = SE2Pose(center[0], center[1], orientation)
        
        # 初始化协方差矩阵 Σ
        cov_matrix = np.diag([0.1, 0.1, 0.1, 100.0, 100.0]) # 增加 vx, w 的大方差
        covariance = StateCovariance(cov_matrix)
        
        # 创建实体
        entity = Entity(model, pose, covariance, timestamp, state_vector)
        # 使用带网格索引的证据集
        entity.evidence = EvidenceSet(cell_size=0.1, delta_s=0.005)
        
        # 构建证据集
        sensor_sigma = 0.01
        sensor_pos = np.zeros(2) if sensor_pos is None else np.array(sensor_pos).reshape(2,)
        for i in range(len(projections)):
            s_val = projections[i] 
            mapped_s = range_transformer.to_zero_based(s_val)
            
            pt_global = points[:, i]
            r = np.linalg.norm(pt_global - sensor_pos)
            sigma_r = 0.01 + 0.005 * r
            w = 1.0 / (sigma_r ** 2)
            
            ev_point = EvidencePoint(
                s=mapped_s, # 或者直接存 s_val，这取决于你后续如何 evaluate
                z_global=pt_global,
                weight=w,
                timestamp=timestamp,
                sigma=sensor_sigma
            )
            entity.evidence.add(ev_point)
            
        return entity
