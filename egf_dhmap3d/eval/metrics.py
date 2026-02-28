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
    normal_consistency: float
    precision_2cm: float
    recall_2cm: float
    fscore_2cm: float
    precision_5cm: float
    recall_5cm: float
    fscore_5cm: float
    precision_10cm: float
    recall_10cm: float
    fscore_10cm: float


@dataclass
class TrajectoryMetrics:
    ate_rmse: float
    ate_mean: float
    ate_median: float
    ate_max: float
    rpe_trans_rmse: float
    rpe_trans_mean: float
    rpe_rot_deg_rmse: float
    rpe_rot_deg_mean: float
    frame_count: int
    valid_pair_count: int
    finite_ratio: float


def _safe_unit(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float)
    if vv.size == 0:
        return vv
    n = np.linalg.norm(vv, axis=1, keepdims=True)
    return vv / np.clip(n, 1e-9, None)


def _fscore_at(d_pred_to_gt: np.ndarray, d_gt_to_pred: np.ndarray, threshold: float) -> tuple[float, float, float]:
    precision = float(np.mean(d_pred_to_gt < threshold))
    recall = float(np.mean(d_gt_to_pred < threshold))
    if (precision + recall) <= 1e-9:
        fscore = 0.0
    else:
        fscore = float(2.0 * precision * recall / (precision + recall))
    return precision, recall, fscore


def compute_reconstruction_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.05,
    pred_normals: np.ndarray | None = None,
    gt_normals: np.ndarray | None = None,
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
            normal_consistency=0.0,
            precision_2cm=0.0,
            recall_2cm=0.0,
            fscore_2cm=0.0,
            precision_5cm=0.0,
            recall_5cm=0.0,
            fscore_5cm=0.0,
            precision_10cm=0.0,
            recall_10cm=0.0,
            fscore_10cm=0.0,
        )

    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    d_pred_to_gt, nn_idx = gt_tree.query(pred_points, k=1)
    d_gt_to_pred, _ = pred_tree.query(gt_points, k=1)

    chamfer = float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))
    haus = float(max(np.max(d_pred_to_gt), np.max(d_gt_to_pred)))

    precision, recall, fscore = _fscore_at(d_pred_to_gt, d_gt_to_pred, float(threshold))
    precision_2cm, recall_2cm, fscore_2cm = _fscore_at(d_pred_to_gt, d_gt_to_pred, 0.02)
    precision_5cm, recall_5cm, fscore_5cm = _fscore_at(d_pred_to_gt, d_gt_to_pred, 0.05)
    precision_10cm, recall_10cm, fscore_10cm = _fscore_at(d_pred_to_gt, d_gt_to_pred, 0.10)

    normal_consistency = 0.0
    if pred_normals is not None and gt_normals is not None:
        pred_normals = np.asarray(pred_normals, dtype=float)
        gt_normals = np.asarray(gt_normals, dtype=float)
        if pred_normals.shape[0] == pred_points.shape[0] and gt_normals.shape[0] == gt_points.shape[0]:
            n_pred = _safe_unit(pred_normals)
            n_gt = _safe_unit(gt_normals)
            n_gt_nn = n_gt[np.asarray(nn_idx, dtype=np.int64)]
            dots = np.sum(n_pred * n_gt_nn, axis=1)
            normal_consistency = float(np.mean(np.abs(np.clip(dots, -1.0, 1.0))))

    return ReconMetrics3D(
        chamfer=chamfer,
        hausdorff=haus,
        precision=precision,
        recall=recall,
        fscore=fscore,
        normal_consistency=normal_consistency,
        precision_2cm=precision_2cm,
        recall_2cm=recall_2cm,
        fscore_2cm=fscore_2cm,
        precision_5cm=precision_5cm,
        recall_5cm=recall_5cm,
        fscore_5cm=fscore_5cm,
        precision_10cm=precision_10cm,
        recall_10cm=recall_10cm,
        fscore_10cm=fscore_10cm,
    )


def _rot_deg_from_matrix(r: np.ndarray) -> float:
    tr = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(tr)))


def compute_trajectory_metrics(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    step: int = 1,
    align_first: bool = True,
) -> TrajectoryMetrics:
    pred = np.asarray(pred_poses, dtype=float)
    gt = np.asarray(gt_poses, dtype=float)
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[1:] != (4, 4) or gt.shape[1:] != (4, 4):
        return TrajectoryMetrics(
            ate_rmse=float("inf"),
            ate_mean=float("inf"),
            ate_median=float("inf"),
            ate_max=float("inf"),
            rpe_trans_rmse=float("inf"),
            rpe_trans_mean=float("inf"),
            rpe_rot_deg_rmse=float("inf"),
            rpe_rot_deg_mean=float("inf"),
            frame_count=0,
            valid_pair_count=0,
            finite_ratio=0.0,
        )

    n = int(min(pred.shape[0], gt.shape[0]))
    if n <= 0:
        return TrajectoryMetrics(
            ate_rmse=float("inf"),
            ate_mean=float("inf"),
            ate_median=float("inf"),
            ate_max=float("inf"),
            rpe_trans_rmse=float("inf"),
            rpe_trans_mean=float("inf"),
            rpe_rot_deg_rmse=float("inf"),
            rpe_rot_deg_mean=float("inf"),
            frame_count=0,
            valid_pair_count=0,
            finite_ratio=0.0,
        )

    pred = pred[:n]
    gt = gt[:n]
    finite_mask = np.isfinite(pred).all(axis=(1, 2)) & np.isfinite(gt).all(axis=(1, 2))
    finite_ratio = float(np.mean(finite_mask.astype(float))) if n > 0 else 0.0
    if align_first:
        try:
            t_align = gt[0] @ np.linalg.inv(pred[0])
            pred_aligned = np.einsum("ij,njk->nik", t_align, pred)
        except np.linalg.LinAlgError:
            pred_aligned = pred.copy()
    else:
        pred_aligned = pred.copy()

    ate_t = np.linalg.norm(pred_aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    ate_rmse = float(np.sqrt(np.mean(np.square(ate_t)))) if ate_t.size else float("inf")
    ate_mean = float(np.mean(ate_t)) if ate_t.size else float("inf")
    ate_median = float(np.median(ate_t)) if ate_t.size else float("inf")
    ate_max = float(np.max(ate_t)) if ate_t.size else float("inf")

    d = max(1, int(step))
    trans_err = []
    rot_err = []
    for i in range(0, n - d):
        if not (finite_mask[i] and finite_mask[i + d]):
            continue
        try:
            d_gt = np.linalg.inv(gt[i]) @ gt[i + d]
            d_pr = np.linalg.inv(pred_aligned[i]) @ pred_aligned[i + d]
        except np.linalg.LinAlgError:
            continue
        d_err = np.linalg.inv(d_gt) @ d_pr
        trans_err.append(float(np.linalg.norm(d_err[:3, 3])))
        rot_err.append(_rot_deg_from_matrix(d_err[:3, :3]))
    trans_err = np.asarray(trans_err, dtype=float)
    rot_err = np.asarray(rot_err, dtype=float)
    if trans_err.size > 0:
        rpe_trans_rmse = float(np.sqrt(np.mean(np.square(trans_err))))
        rpe_trans_mean = float(np.mean(trans_err))
    else:
        rpe_trans_rmse = float("inf")
        rpe_trans_mean = float("inf")
    if rot_err.size > 0:
        rpe_rot_deg_rmse = float(np.sqrt(np.mean(np.square(rot_err))))
        rpe_rot_deg_mean = float(np.mean(rot_err))
    else:
        rpe_rot_deg_rmse = float("inf")
        rpe_rot_deg_mean = float("inf")

    return TrajectoryMetrics(
        ate_rmse=ate_rmse,
        ate_mean=ate_mean,
        ate_median=ate_median,
        ate_max=ate_max,
        rpe_trans_rmse=rpe_trans_rmse,
        rpe_trans_mean=rpe_trans_mean,
        rpe_rot_deg_rmse=rpe_rot_deg_rmse,
        rpe_rot_deg_mean=rpe_rot_deg_mean,
        frame_count=n,
        valid_pair_count=int(trans_err.size),
        finite_ratio=finite_ratio,
    )
