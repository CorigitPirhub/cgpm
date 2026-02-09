import irsim
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Generator

# Import helpers from the existing samples module
from .samples import get_sensor_position
from .border_filter import filter_lidar_data
from .odometer import points_to_global

class LidarDataStream:
    def __init__(self, config_file: str = 'config/obstacles_move.yaml', max_frames: int = 100):
        """
        Initialize the LiDAR data stream.
        
        Args:
            config_file: Relative path to the config file from project root.
            max_frames: Maximum number of frames to yield.
        """
        current_dir = Path(__file__).resolve()
        # Assuming folder structure: Preception/CGPM/simulator/samples_move.py
        # Project root is Preception/CGPM
        self.project_root = current_dir.parent.parent
        self.config_path = self.project_root / config_file
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        self.env = irsim.make(str(self.config_path))
        self.max_frames = max_frames
        self.frame_count = 0
        
    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.frame_count >= self.max_frames:
            # Clean up if needed, though irsim might not require explicit close
            # self.env.close() 
            raise StopIteration

        # Step the environment
        self.env.step()
        self.frame_count += 1
        
        robot = self.env.robot
        
        # 1. Get LiDAR data
        scan_ranges = robot.lidar.get_scan()['ranges']
        lidar_points = robot.lidar.get_points()
        
        # 2. Filter LiDAR data
        lidar_std = getattr(robot.lidar, 'std', 0.01)
        scan_ranges, lidar_points_filtered = filter_lidar_data(robot, scan_ranges, lidar_points, threshold=lidar_std)

        # 2.1 Convert to global frame (robot is moving)
        lidar_points_global = points_to_global(robot, lidar_points_filtered)
        
        # 3. Get sensor position
        sensor_pos = get_sensor_position(robot)
        
        # 4. Extract obstacles for visualization
        obstacles_data = self._extract_obstacles()
        
        return {
            'lidar_ranges': scan_ranges,
            'lidar_points': lidar_points_global,
            'obstacles': obstacles_data,
            'sensor_pos': sensor_pos,
            'timestamp': self.env.time,   # Current simulation time
            'dt': self.env.step_time if hasattr(self.env, 'step_time') else 0.1,
            'frame_index': self.frame_count
        }

    def _extract_obstacles(self):
        """Helper to extract obstacle info similar to samples.py"""
        obstacles_data = []
        
        # Inspect environment for obstacles
        candidate_lists = [
            getattr(self.env, 'object_list', []),
            getattr(self.env.world, 'object_list', []) if hasattr(self.env, 'world') else [],
            getattr(self.env, 'obstacle_list', [])
        ]
        
        all_objects = []
        for lst in candidate_lists:
            if lst:
                all_objects = lst
                break
                
        for obj in all_objects:
            # Check role or type
            role = getattr(obj, 'role', '')
            # Sometimes obstacles might not have 'role' set to 'obstacle' explicitly in all versions of irsim, 
            # but usually they do.
            if role == 'obstacle':
                obs_info = {}
                if hasattr(obj, 'vertices') and obj.vertices is not None:
                    obs_info['type'] = 'polygon'
                    # Assuming obj.vertices are global vertices
                    obs_info['vertices'] = obj.vertices
                elif hasattr(obj, 'radius'):
                    obs_info['type'] = 'circle'
                    obs_info['center'] = obj.state[:2].flatten()
                    obs_info['radius'] = obj.radius
                
                obstacles_data.append(obs_info)
        return obstacles_data

def get_continuous_samples(config_file: str = 'config/obstacles_move.yaml', max_frames: int = 100):
    """
    Convenience function to get the generator.
    """
    return LidarDataStream(config_file, max_frames)
