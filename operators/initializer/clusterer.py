import numpy as np
from sklearn.cluster import DBSCAN
import sys
import os
import time

# Add current directory to path so we can import samples if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from simulator.samples import get_sample_data

# TODO: 更高效更鲁棒的聚类方法
def cluster_lidar_data(points, eps=0.5, min_samples=3):
    """
    Cluster LiDAR points using DBSCAN.
    
    Args:
        points (np.ndarray): 2xN array of points.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        
    Returns:
        labels (np.ndarray): Cluster labels for each point. Noisy samples are given the label -1.
    """
    if points.shape[1] == 0:
        return np.array([])

    X = points.T  # (N, 2) 即 points 的转置
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    return clustering.labels_  # 一个与 points 数量相同的数组，表示每个点的簇标签

