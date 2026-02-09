import matplotlib
# Non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_point_cloud(data, output_path: str = "output/scan.png"):
    """
    Visualize LiDAR points and obstacles.
    
    Args:
        data: Dict containing 'lidar_points' (2xN), 'obstacles', 'sensor_pos'
        output_path: Path to save the image
    """
    lidar_points = data['lidar_points'] # 2xN
    obstacles = data.get('obstacles', [])
    sensor_pos = data.get('sensor_pos', np.zeros(2))

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot obstacles
    # Use a flag to avoid repetitive legend entries
    obs_label_added = False
    
    for obs in obstacles:
        if obs.get('type') == 'polygon':
            verts = obs['vertices']
            # Matplotlib requires (N, 2) vertices
            if verts.shape[0] == 2 and verts.shape[1] > 2:
                 verts = verts.T
            
            label = 'Obstacle' if not obs_label_added else None
            poly = plt.Polygon(verts, facecolor='gray', alpha=0.3, edgecolor='black', label=label)
            ax.add_patch(poly)
            obs_label_added = True
            
        elif obs.get('type') == 'circle':
            center = obs['center']
            radius = obs['radius']
            label = 'Obstacle' if not obs_label_added else None
            circle = plt.Circle(center, radius, facecolor='gray', alpha=0.3, edgecolor='black', label=label)
            ax.add_patch(circle)
            obs_label_added = True
            
    # Plot Sensor Position
    ax.plot(sensor_pos[0], sensor_pos[1], 'rx', markersize=10, markeredgewidth=2, label='Sensor')

    # Plot LiDAR points
    if lidar_points.shape[1] > 0:
        ax.scatter(lidar_points[0, :], lidar_points[1, :], s=5, c='blue', alpha=0.6, label='LiDAR Points')

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Handle Legend deduplication manually if needed, but matplotlib handles None labels well
    ax.legend(loc='upper right')
    
    ax.set_title(f"LiDAR Scan (Points: {lidar_points.shape[1]})")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    
    # Set limits based on points and sensor
    all_x = [sensor_pos[0]]
    all_y = [sensor_pos[1]]
    
    if lidar_points.shape[1] > 0:
        all_x.extend(lidar_points[0, :].tolist())
        all_y.extend(lidar_points[1, :].tolist())
        
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        margin = 2.0
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

