"""
Minimal test pipeline: sample -> cluster -> initialize entities -> predict one step.
"""

import os
import sys
import time
from tracemalloc import start
import numpy as np

# Ensure project root is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from simulator.samples import get_sample_data
from operators.initializer.clusterer import cluster_lidar_data
from operators.initializer.initializer import EntityInitializer
from operators.predictor.predictor import Predictor
from operators.associator.associator import Associator, PolylineAssociator
from operators.updater.updater_local import LocalUpdater
from operators.updater.updater_global import GlobalUpdater
from visualization.entity_visualization import visualize_entities
from visualization.clusterer_visualization import visualize_clusters
from visualization.points_visulization import visualize_point_cloud

def main():
	data = get_sample_data('config/obstacles_fixed.yaml')
	points = data["lidar_points"]

	start_time = time.time()
	labels = cluster_lidar_data(points, eps=0.5, min_samples=5)
	end_time = time.time()
	print(f"Clustered {points.shape[1]} points in {end_time - start_time:.4f} seconds")

	visualize_point_cloud(data, output_path="output/scan.png")

	visualize_clusters(data, labels, output_path="output/clusterer.png")

	# 聚类初始化实体
	initializer = EntityInitializer()
	start_time = time.time()
	entities = initializer.process(points, labels, timestamp=0.0, sensor_pos=data.get("sensor_pos"))
	end_time = time.time()
	print(f"Initialized {len(entities)} entities in {end_time - start_time:.4f} seconds")

	visualize_entities(entities, output_path="output/entity_ini.png")

	print("\n=== Initialization step ===")
	for i, e in enumerate(entities):
		print(f'Entity {i}')
		print(f'  Pose: {e.pose}')
		print(f'  Covariance Determinant (Volume): {np.linalg.det(e.covariance.matrix):.4e}')
		print(f'  Evidence Count: {len(e.evidence)}')
		print(f'  Length: {e.estimated_length:.3f}')
		print(f'  Timestamp: {e.last_update_time:.3f}')
		print(f'  state_vector: {e.state_vector}')
		print(f'  Closed? {e.is_geometrically_closed()}')

	for batch in range(5):
		print(f"\n=== Batch {batch} ===\n")

		# 预测
		pre_entities = []
		predictor = Predictor()
		dt = 0.1
		start_time = time.time()
		for i, e in enumerate(entities):
			ent = predictor.process(e, dt)
			pre_entities.append(ent)
		end_time = time.time()
		print(f"Predicted {len(entities)} entities in {end_time - start_time:.4f} seconds")

		visualize_entities(pre_entities, output_path=f"output/entity_pre{batch}.png")

		# print("\n=== Prediction step ===")
		# for i, e in enumerate(pre_entities):
		# 	print(f'Entity {i}')
		# 	print(f'  Pose: {e.pose}')
		# 	print(f'  Covariance Determinant (Volume): {np.linalg.det(e.covariance.matrix):.4e}')
		# 	print(f'  Evidence Count: {len(e.evidence)}')
		# 	print(f'  Length: {e.estimated_length:.3f}')
		# 	print(f'  Timestamp: {e.last_update_time:.3f}')
		# 	print(f'  state_vector: {e.state_vector}')
		# 	print(f'  Closed? {e.is_geometrically_closed()}')

		# 关联
		associator = PolylineAssociator(chi2_threshold=0.99, obs_sigma=0.01)
		start_time = time.time()
		results, _events = associator.process(points, pre_entities, timestamp=dt)
		end_time = time.time()
		print(f"Associated {len(results)} results in {end_time - start_time:.4f} seconds")

		print(f"\nAssociated {len(results)} observations")
		for r in results:
			print(f"  obs {r.obs_index} -> entity {r.entity_index}, d2={r.mahalanobis_sq:.3f}, s*={r.s_star:.3f}")

		# 更新
		updater = LocalUpdater(obs_sigma=0.01)
		start_time = time.time()
		entities = updater.process(pre_entities, results, points)
		end_time = time.time()
		print(f"Updated {len(entities)} entities in {end_time - start_time:.4f} seconds")

		print("\n=== Update step ===")
		for i, e in enumerate(pre_entities):
			print(f'Entity {i}')
			print(f'  Pose: {e.pose}')
			print(f'  Covariance Determinant (Volume): {np.linalg.det(e.covariance.matrix):.4e}')
			print(f'  Evidence Count: {len(e.evidence)}')
			print(f'  Length: {e.estimated_length:.3f}')
			print(f'  Timestamp: {e.last_update_time:.3f}')
			print(f'  state_vector: {e.state_vector}')
			print(f'  Closed? {e.is_geometrically_closed()}')

		visualize_entities(entities, output_path=f"output/entity_upd{batch}.png")

if __name__ == "__main__":
	main()
