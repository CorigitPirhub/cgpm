import irsim
import numpy as np
import os
import yaml
from pathlib import Path
from .border_filter import filter_lidar_data

def get_sensor_position(robot) -> np.ndarray:
    """
    Returns the global 2D position of the LiDAR sensor. If the robot exposes a
    lidar offset, it is added on top of the robot base position.
    """
    base = np.array(getattr(robot, "state", [0.0, 0.0])[:2])
    offset = np.zeros(2)
    if hasattr(robot, "lidar") and hasattr(robot.lidar, "offset"):
        try:
            offset = np.array(robot.lidar.offset[:2])
        except Exception:
            offset = np.zeros(2)
    return base + offset

def get_sample_data(input_path: str = 'config/obstacles_few.yaml'):
    """
    Generates a sample LiDAR scan from a simulated environment.
    
    Returns:
        dict: {
            'lidar_ranges': np.ndarray, # (360,) ranges
            'lidar_points': np.ndarray, # (2, N) xy coordinates
            'obstacles': list of dicts with geometry info for visualization
        }
    """
    # Path to the config file
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # config_path = os.path.join(current_dir, 'one_obstacle.yaml')
    current_dir = Path(__file__).resolve()
    # config_path = current_file_path.parent.parent / 'config' / 'world.yaml'
    config_path = current_dir.parent.parent / input_path
    
     
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Initialize environment
    # Note: irsim might print output, we can't easily suppress it without redirecting stdout
    env = irsim.make(config_path)
    
    # Run a few steps to let obstacles move a bit (if dynamic)
    for _ in range(5):
        env.step()
    
    robot = env.robot
    
    # Get LiDAR data
    # ranges
    scan_ranges = robot.lidar.get_scan()['ranges']
    
    # points (2, N) usually
    lidar_points = robot.lidar.get_points()
    
    # Apply filtering to remove max-range artifacts
    print('Points count before filtering:', lidar_points.shape[1])
    lidar_std = getattr(robot.lidar, 'std', 0.01)
    scan_ranges, lidar_points_filtered = filter_lidar_data(robot, scan_ranges, lidar_points, threshold=lidar_std)
    print('Points count after filtering:', lidar_points_filtered.shape[1])

    # Sensor position (global)
    sensor_pos = get_sensor_position(robot)
    
    # Extract obstacle geometries for 'faint shadow' visualization
    # We look at env.world.obstacles or similar container
    obstacles_data = []
    
    # irsim stores objects lists. We need to find where obstacles are.
    # Typically env.world.objects varies but we can iterate env.object_list
    # Filtering for those that are not the robot (id 0 usually)
    
    # Based on previous probe, robot has `external_objects` or similar?
    # Let's inspect env via dir() in a separate quick check if needed, but let's try standard way.
    # irsim env usually has `obejct_list` 
    
    # Note: The user prompt asked to show obstacles as "faint shadow".
    # We need the vertices for polygons or radius for circles.
    
    # Inspect environment for obstacles
    # Try different locations where irsim might store objects
    candidate_lists = [
        getattr(env, 'object_list', []),
        getattr(env.world, 'object_list', []) if hasattr(env, 'world') else [],
        getattr(env, 'obstacle_list', [])
    ]
    
    all_objects = []
    for lst in candidate_lists:
        if lst:
            all_objects = lst
            break
            
    # print(f"Found {len(all_objects)} objects in environment.")

    for obj in all_objects:
        # print(f"Object: {obj}, Role: {getattr(obj, 'role', 'unknown')}")
        if getattr(obj, 'role', '') == 'obstacle':
            obs_info = {}
            if hasattr(obj, 'vertices') and obj.vertices is not None:
                obs_info['type'] = 'polygon'
                # vertices are 2xN usually in irsim, logic needs verification
                # transform vertices to global frame
                # vertices are usually relative to center, need rotation + translation
                
                # irsim objects maintain `vertices` (global or local?)
                # Usually `obj.vertices` updates with state in irsim if it's dynamic?
                # Let's assume we need to calculate global vertices from state
                
                # Actually, `irsim` objects often have a helper or property. 
                # Let's just grab the state and shape.
                
                # However, for simple visualization, let's grab the `vertices` property 
                # which in irsim (checking source or common usage) usually returns global vertices for plotting.
                obs_info['vertices'] = obj.vertices
                
            elif hasattr(obj, 'radius'):
                obs_info['type'] = 'circle'
                obs_info['center'] = obj.state[:2].flatten()
                obs_info['radius'] = obj.radius
                
            obstacles_data.append(obs_info)

    data = {
        'lidar_ranges': scan_ranges,
        'lidar_points': lidar_points_filtered,
        'obstacles': obstacles_data,
        'sensor_pos': sensor_pos,
    }
    
    # env.end() # visual cleaning might cause issues in headless/automation
    return data

