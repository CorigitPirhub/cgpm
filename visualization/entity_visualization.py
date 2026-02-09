"""
Utility to visualize a list of Entity objects in 2D and save to an image file.
"""
import matplotlib
# Non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_entities(entities, output_path: str = "output/entity.png", num_samples: int = 100):
	"""
	Visualize entities in 2D and save as an image.

	Args:
		entities: List[Entity]
		output_path: File path to save the image
		num_samples: Number of samples along each curve for plotting
	"""
	if not entities:
		raise ValueError("No entities to visualize")

	fig, ax = plt.subplots(figsize=(8, 8))
	colors = plt.cm.tab10(np.linspace(0, 1, max(len(entities), 3)))

	for idx, ent in enumerate(entities):
		color = colors[idx % len(colors)]

		# Sample curve in local frame, then transform to global
		s_vals = np.linspace(0.0, ent.model.domain_limit, num_samples)
		pts_local = ent.model.evaluate(s_vals)  # (N,2)

		g = ent.pose.to_matrix()
		R = g[:2, :2]
		t = g[:2, 2]
		pts_global = (R @ pts_local.T).T + t  # (N,2)

		ax.plot(pts_global[:, 0], pts_global[:, 1], color=color, lw=2, label=f"Entity {idx}")

		# Control points (global)
		cps_local = None
		if hasattr(ent.model, "control_points") and ent.model.control_points is not None:
			cps_local = ent.model.control_points
		elif hasattr(ent.model, "global_control_points"):
			cps_local = ent.model.global_control_points
		elif hasattr(ent.model, "segments"):
			try:
				cps_local = np.vstack([np.array(seg) for seg in ent.model.segments])
			except Exception:
				cps_local = None

		if cps_local is not None and len(cps_local) > 0:
			cps_global = (R @ cps_local.T).T + t
			ax.scatter(cps_global[:, 0], cps_global[:, 1], color=color, marker='s', s=30, alpha=0.8)

		# Evidence points (global)
		evs = ent.evidence.get_all()
		if evs:
			ev_global = np.array([e.z_global for e in evs])
			ax.scatter(ev_global[:, 0], ev_global[:, 1], color=color, marker='.', alpha=0.4, s=10)

		# Text annotation near first control point
		if cps_local is not None and len(cps_local) > 0:
			cps_global = (R @ cps_local.T).T + t
			anchor = cps_global[0]
		else:
			anchor = pts_global[0]
		text = (
			# f"Entity {idx}\n"
			# f"Pose: ({ent.pose.x:.3f}, {ent.pose.y:.3f}, {ent.pose.theta:.3f})\n"
			# f"Cov det: {np.linalg.det(ent.covariance.matrix):.3e}\n"
			# f"Evidence: {len(ent.evidence)}\n"
			# f"Length: {ent.estimated_length:.3f}\n"
			# f"State: ({ent.state_vector.x:.2f}, {ent.state_vector.y:.2f}, {ent.state_vector.theta:.2f}, {ent.state_vector.v:.2f}, {ent.state_vector.omega:.2f})"
		)
		ax.annotate(
			text,
			xy=(anchor[0], anchor[1]),
			xytext=(5, 5),
			textcoords="offset points",
			fontsize=8,
			color=color,
			bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"),
		)

	ax.set_aspect('equal')
	ax.grid(True, linestyle='--', alpha=0.5)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.legend()
	ax.set_title('Entities Visualization')

	# Expand limits using curves + control points + evidence
	x_all = []
	y_all = []
	for ent in entities:
		g = ent.pose.to_matrix()
		R = g[:2, :2]
		t = g[:2, 2]
		# control points
		cps_local = None
		if hasattr(ent.model, "control_points") and ent.model.control_points is not None:
			cps_local = ent.model.control_points
		elif hasattr(ent.model, "global_control_points"):
			cps_local = ent.model.global_control_points
		elif hasattr(ent.model, "segments"):
			try:
				cps_local = np.vstack([np.array(seg) for seg in ent.model.segments])
			except Exception:
				cps_local = None

		if cps_local is not None and len(cps_local) > 0:
			cps_global = (R @ cps_local.T).T + t
			x_all.extend(cps_global[:, 0].tolist())
			y_all.extend(cps_global[:, 1].tolist())
		# curve samples
		s_vals = np.linspace(0.0, ent.model.domain_limit, num_samples)
		pts_local = np.array([ent.model.evaluate(s) for s in s_vals])
		pts_global = (R @ pts_local.T).T + t
		x_all.extend(pts_global[:, 0].tolist())
		y_all.extend(pts_global[:, 1].tolist())
		# evidences
		evs = ent.evidence.get_all()
		if evs:
			ev_global = np.array([e.z_global for e in evs])
			x_all.extend(ev_global[:, 0].tolist())
			y_all.extend(ev_global[:, 1].tolist())

	if x_all and y_all:
		x_min, x_max = min(x_all), max(x_all)
		y_min, y_max = min(y_all), max(y_all)
		dx = max(1.0, 0.15 * (x_max - x_min + 1e-3))
		dy = max(1.0, 0.15 * (y_max - y_min + 1e-3))
		ax.set_xlim(x_min - dx, x_max + dx)
		ax.set_ylim(y_min - dy, y_max + dy)

	os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
	fig.savefig(output_path, dpi=150, bbox_inches='tight')
	plt.close(fig)

	return output_path

