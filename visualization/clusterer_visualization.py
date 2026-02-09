import matplotlib
# Non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time 

def visualize_clusters(data, labels, output_path: str = 'output/clusterer.png'):
    points = data['lidar_points']
    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot obstacles for context
    obstacles = data.get('obstacles', [])
    for obs in obstacles:
        if obs.get('type') == 'polygon':
            verts = obs['vertices'].T
            poly = plt.Polygon(verts, facecolor='gray', alpha=0.1, edgecolor='none')
            ax.add_patch(poly)
        elif obs.get('type') == 'circle':
            center = obs['center']
            radius = obs['radius']
            circle = plt.Circle(center, radius, facecolor='gray', alpha=0.1, edgecolor='none')
            ax.add_patch(circle)

    # Plot clusters
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        
        xy = points[:, class_member_mask]
        
        ax.plot(xy[0], xy[1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6 if k != -1 else 3, 
                label=f'Cluster {k}' if k != -1 else 'Noise')

        if k != -1:
            centroid = np.mean(xy, axis=1)
            ax.text(centroid[0], centroid[1], str(k), fontsize=12, 
                    fontweight='bold', color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_title('Estimated number of clusters: %d' % (len(unique_labels) - (1 if -1 in labels else 0)))
    ax.set_aspect('equal')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)

