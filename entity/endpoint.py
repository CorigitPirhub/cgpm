import numpy as np
from typing import List

from .entity import Entity


def compute_endpoints(entity: Entity, delta: float = 0.0, num_samples: int = 200, bandwidth_c: float = 2.0) -> List[np.ndarray]:
	"""
	Definition 2.4: Endpoint Set B.

	B = { g_hat * gamma(0), g_hat * gamma(L) } 
		∪ { g_hat * gamma(s) | s in [0, L], d(mu_E)/ds < delta }

	Args:
		entity: target Entity
		delta: threshold for density derivative
		num_samples: sampling resolution along [0, L]
		bandwidth_c: bandwidth multiplier for evidence density

	Returns:
		List of global 2D points for endpoints
	"""
	if entity is None or entity.model is None:
		return []

	L = float(entity.model.domain_limit)
	if L <= 0.0:
		return []

	g_mat = entity.pose.to_matrix()

	def to_global(p_local: np.ndarray) -> np.ndarray:
		return (g_mat @ np.append(p_local, 1.0))[:2]

	endpoints: List[np.ndarray] = []

	# Always include physical endpoints
	endpoints.append(to_global(entity.model.evaluate(0.0)))
	endpoints.append(to_global(entity.model.evaluate(L)))

	# Evidence density derivative-based endpoints
	sigma_bar = entity.evidence.get_average_sigma()
	h = bandwidth_c * sigma_bar if sigma_bar > 0 else 0.1

	s_vals = np.linspace(0.0, L, num_samples)
	for s in s_vals:
		dmu = entity.evidence.get_density_derivative_at(s, h)
		if dmu < delta:
			endpoints.append(to_global(entity.model.evaluate(s)))

	return endpoints

