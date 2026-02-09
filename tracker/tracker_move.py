import os
import sys
import time
import numpy as np

# Ensure project root is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from simulator.samples_move import LidarDataStream
from operators.initializer.clusterer import cluster_lidar_data
from operators.initializer.initializer import EntityInitializer
from operators.predictor.predictor import Predictor
from operators.associator.associator import Associator, PolylineAssociator
from operators.updater.updater_local import LocalUpdater
from visualization.entity_visualization import visualize_entities
from visualization.points_visulization import visualize_point_cloud
from visualization.clusterer_visualization import visualize_clusters

def main():
    # Use obstacles_move.yaml for dynamic scenario
    # You can change max_frames to run longer
    stream = LidarDataStream(config_file='config/robot_move.yaml', max_frames=100)
    
    entities = []
    
    # Initialize operators
    initializer = EntityInitializer()
    predictor = Predictor()
    associator = PolylineAssociator(chi2_threshold=0.99, obs_sigma=0.01)
    updater = LocalUpdater(obs_sigma=0.01)
    
    output_dir = "output/move"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting continuous tracking simulation with {stream.max_frames} frames...")

    for data in stream:
        frame_idx = data['frame_index']
        points = data['lidar_points']
        dt = data['dt']
        sensor_pos = data['sensor_pos']
        
        visualize_point_cloud(data, output_path=os.path.join(output_dir, f"scan_{frame_idx:03d}.png"))

        print(f"\n--- Frame {frame_idx} (t={data['timestamp']:.2f}s) ---")
        
        start_time = time.time()
        
        if frame_idx == 1: # First frame (1-based index from our stream implementation)
            # Step 1: Cluster
            print("Processing First Frame: Clustering + Initialization")
            if points.shape[1] == 0:
                print("Warning: No points in first frame, skipping initialization.")
                continue
                
            labels = cluster_lidar_data(points, eps=0.5, min_samples=3)
            
            # Step 2: Initialize
            entities = initializer.process(points, labels, timestamp=data['timestamp'], sensor_pos=sensor_pos)
            print(f"Initialized {len(entities)} entities.")
            
        else:
            # Subsequent frames: Predict -> Associate -> Update
            if not entities:
                print("No entities to track. Waiting for restart or re-init (not implemented).")
                continue
                
            # Step 1: Predict
            pre_entities = []
            for e in entities:
                ent = predictor.process(e, dt)
                pre_entities.append(ent)
            
            # Step 2: Associate
            # associator expects timestamp as dt usually for motion consistency checks if any, 
            # or absolute time. tracker.py passed `dt`. We'll pass `dt`.
            results, _events = associator.process(
                points,
                pre_entities,
                timestamp=dt,
                sensor_origin=sensor_pos,
            )
            for i, ev in enumerate(entities):
                print(f"Entity {i} - Evidence size: {len(ev.evidence)}, Cache size: {ev.evidence.get_cache_size()}")
            print(f"Associated {len(results)} observations.")

            # Step 3: Update
            entities = updater.process(pre_entities, results, points)
            print(f"Updated {len(entities)} entities.")
            
        proc_time = time.time() - start_time
        print(f"Frame processing time: {proc_time:.4f}s")
        
        # Visualization
        # Save to output/move/frame_XXX.png
        out_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
        if entities:
            visualize_entities(entities, output_path=out_path)
        else:
            print("No entities to visualize.")

if __name__ == "__main__":
    main()
