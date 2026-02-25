from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ReconMetrics3D:
    chamfer: float
    hausdorff: float
    precision: float
    recall: float
    fscore: float


def compute_reconstruction_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.05,
) -> ReconMetrics3D:
    pred_points = np.asarray(pred_points, dtype=float)
    gt_points = np.asarray(gt_points, dtype=float)
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return ReconMetrics3D(
            chamfer=float("inf"),
            hausdorff=float("inf"),
            precision=0.0,
            recall=0.0,
            fscore=0.0,
        )

    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    d_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
    d_gt_to_pred, _ = pred_tree.query(gt_points, k=1)

    chamfer = float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))
    haus = float(max(np.max(d_pred_to_gt), np.max(d_gt_to_pred)))

    precision = float(np.mean(d_pred_to_gt < threshold))
    recall = float(np.mean(d_gt_to_pred < threshold))
    fscore = 0.0 if (precision + recall) <= 1e-9 else float(2.0 * precision * recall / (precision + recall))

    return ReconMetrics3D(
        chamfer=chamfer,
        hausdorff=haus,
        precision=precision,
        recall=recall,
        fscore=fscore,
    )
