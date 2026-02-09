import numpy as np

def filter_lidar_data(robot, scan_ranges, lidar_points, threshold=0.01, points_frame: str = "auto"):
    """
    Filters out LiDAR points that are near the maximum range (likely no obstacle).
    
    Args:
        robot: The robot object.
        scan_ranges: 1D array of ranges.
        lidar_points: 2xN array of points.
        threshold: Distance threshold to subtract from range_max (should match lidar std).
    
    Returns:
        Tuple of (filtered_scan_ranges, filtered_lidar_points)
    """
    if len(scan_ranges) == 0:
        return scan_ranges, lidar_points

    # 1. Determine range_max
    # Try to access it from the robot's sensor configuration
    if hasattr(robot, 'lidar') and hasattr(robot.lidar, 'range_max'):
        r_max = robot.lidar.range_max
    else:
        # Fallback: assume the max observed value is the limit
        r_max = np.max(scan_ranges)

    # 2. Filter scan_ranges
    # Keep ranges strictly less than range_max - threshold
    valid_ranges_mask = scan_ranges < (r_max - threshold * 3) 
    filtered_ranges = scan_ranges[valid_ranges_mask]
    
    # 3. Filter lidar_points by range mask first (frame-invariant)
    if lidar_points is None:
        return filtered_ranges, lidar_points

    pts = np.asarray(lidar_points)
    if pts.size == 0:
        return filtered_ranges, lidar_points

    transposed = False
    if pts.shape[0] == 2:
        pts_use = pts
    elif pts.shape[1] == 2:
        pts_use = pts.T
        transposed = True
    else:
        return filtered_ranges, lidar_points

    if len(scan_ranges) == pts_use.shape[1]:
        filtered_points = pts_use[:, valid_ranges_mask]
    else:
        # Fallback: compute distance in chosen frame
        if points_frame == "global":
            robot_pos = np.array(robot.state[:2]).reshape(2, 1)
            dists = np.linalg.norm(pts_use - robot_pos, axis=0)
        else:
            # "local" or "auto" defaults to origin
            dists = np.linalg.norm(pts_use, axis=0)

        valid_points_mask = dists < (r_max - threshold * 3)
        filtered_points = pts_use[:, valid_points_mask]

    if transposed:
        filtered_points = filtered_points.T
        
    return filtered_ranges, filtered_points
