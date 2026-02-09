import numpy as np
from typing import Dict, List
from sklearn.decomposition import PCA

from entity.entity import Entity
from entity.state import SE2Pose, StateVector
from operators.associator.associator import AssociationResult

# TODO: 此部分逻辑暂未集成，后续可以把这个策略融入
class GlobalUpdater:
	"""
	Global-frame update: refit a local frame from newly associated observations
	(center + major-axis heading), compare it against the current pose, and
	apply a Kalman correction on the pose portion of the state. Geometry is
	then re-centered and resized along the inferred major axis.
	"""

	def __init__(self, obs_sigma: float = 0.05):
		self.obs_sigma = obs_sigma

	def process(
		self,
		entities: List[Entity],
		associations: List[AssociationResult],
		observations: np.ndarray,
	) -> List[Entity]:
		if not associations:
			return entities

		# Group observations by entity
		assoc_map: Dict[int, List[int]] = {}
		for assoc in associations:
			assoc_map.setdefault(assoc.entity_index, []).append(assoc.obs_index)

		for ent_idx, obs_indices in assoc_map.items():
			ent = entities[ent_idx]
			obs_points = observations[:, obs_indices]  # shape (2, M)
			if obs_points.shape[1] < 2:
				# Need at least two points to infer orientation
				continue

			center, theta_new, length = self._fit_local_frame(obs_points)

			# Measurement: pose (x, y, theta) from refit frame
			z_meas = np.array([center[0], center[1], theta_new])

			x_vec = ent.state_vector.vector  # [x, y, theta, v, omega]
			P = ent.covariance.matrix

			# Ensure 5x5 covariance
			if P.shape != (5, 5):
				if P.shape[0] >= 5:
					P = P[:5, :5]
				else:
					pad = np.eye(5) * 1e-3
					pad[: P.shape[0], : P.shape[1]] = P
					P = pad

			# H maps state to measurement (pose only)
			H = np.zeros((3, 5))
			H[:3, :3] = np.eye(3)

			# Observation noise: smaller on position, looser on heading
			R = np.diag([
				self.obs_sigma ** 2,
				self.obs_sigma ** 2,
				(2.0 * self.obs_sigma) ** 2,
			])

			# Residual
			y = z_meas - x_vec[:3]
			y[2] = self._wrap_angle(y[2])

			# Innovation covariance and gain
			S = H @ P @ H.T + R
			try:
				K = P @ H.T @ np.linalg.inv(S)
			except np.linalg.LinAlgError:
				continue

			# State update
			delta = K @ y
			x_new = x_vec + delta
			x_new[2] = self._wrap_angle(x_new[2])
			P_new = (np.eye(5) - K @ H) @ P

			ent.state_vector = StateVector(*x_new)
			ent.pose = SE2Pose(ent.state_vector.x, ent.state_vector.y, ent.state_vector.theta)
			ent.covariance.matrix = P_new

			# Refit geometry in the updated local frame using all evidence
			self._refit_geometry(ent)
			ent.estimated_length = length

		return entities

	# ----------------- Helpers -----------------
	def _fit_local_frame(self, obs_points: np.ndarray) -> tuple:
		"""Fit center, heading, and length from global observations (2xM)."""
		X = obs_points.T  # (M, 2)
		pca = PCA(n_components=2)
		pca.fit(X)

		center = pca.mean_
		eigenvectors = pca.components_
		eigenvalues = pca.explained_variance_

		idx_major = int(np.argmax(eigenvalues))
		major_axis = eigenvectors[idx_major]
		theta = float(np.arctan2(major_axis[1], major_axis[0]))

		# Project points onto major axis to estimate length
		vecs = X - center
		projections = np.dot(vecs, major_axis)
		min_s = float(np.min(projections))
		max_s = float(np.max(projections))
		length = max(max_s - min_s, 1e-3)

		return center, theta, length

	def _wrap_angle(self, ang: float) -> float:
		return (ang + np.pi) % (2 * np.pi) - np.pi

	def _refit_geometry(self, entity: Entity):
		evidences = entity.evidence.get_all()
		if len(evidences) < 2:
			return

		g = entity.pose.to_matrix()
		R = g[:2, :2]
		t = g[:2, 2]

		pts_local = []
		for ev in evidences:
			z = ev.z_global
			z_local = R.T @ (z - t)
			pts_local.append(z_local)

		pts_local = np.stack(pts_local, axis=0)
		xs = pts_local[:, 0]
		min_x, max_x = float(np.min(xs)), float(np.max(xs))
		length = max(max_x - min_x, 1e-3)
		half_len = length / 2.0

		entity.model.control_points = np.array([[-half_len, 0.0], [half_len, 0.0]])
		entity.model.domain_limit = length
		entity.estimated_length = length
