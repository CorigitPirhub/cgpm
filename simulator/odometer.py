import numpy as np
from typing import Tuple


def get_robot_pose(robot) -> Tuple[float, float, float]:
	"""
	Get robot pose (x, y, theta) from irsim robot state.
	"""
	state = getattr(robot, "state", [0.0, 0.0, 0.0])
	x = float(state[0])
	y = float(state[1])
	theta = float(state[2]) if len(state) > 2 else 0.0
	return x, y, theta


def points_to_global(robot, lidar_points: np.ndarray) -> np.ndarray:
	"""
	Transform LiDAR points from robot frame to global frame: P_g = R * P_r + t.

	Args:
		robot: irsim robot object, expected to have state [x, y, theta, ...]
		lidar_points: 2xN or Nx2 array in robot/local frame

	Returns:
		2xN array in global frame
	"""
	if lidar_points is None:
		return lidar_points

	pts = np.asarray(lidar_points)
	if pts.size == 0:
		return pts

	# Normalize to shape (2, N)
	if pts.shape[0] == 2:
		pts_local = pts
	elif pts.shape[1] == 2:
		pts_local = pts.T
	else:
		raise ValueError("lidar_points must be shaped (2, N) or (N, 2)")

	x, y, theta = get_robot_pose(robot)
	c, s = np.cos(theta), np.sin(theta)
	R = np.array([[c, -s], [s, c]])
	t = np.array([[x], [y]])

	pts_global = (R @ pts_local) + t
	return pts_global


__all__ = ["get_robot_pose", "points_to_global"]
