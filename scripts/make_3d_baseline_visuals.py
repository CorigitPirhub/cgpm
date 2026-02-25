from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree


def load_mesh_vertices(path: str | Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    pts = np.asarray(mesh.vertices, dtype=float)
    if pts.shape[0] == 0:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=float)
    return pts


def _auto_ghost_roi(tsdf_pts: np.ndarray, egf_pts: np.ndarray, min_dist: float = 0.08) -> Tuple[float, float, float, float]:
    if tsdf_pts.shape[0] == 0:
        return (0.0, 0.0, 0.0, 0.0)
    if egf_pts.shape[0] == 0:
        mins = np.min(tsdf_pts[:, :2], axis=0)
        maxs = np.max(tsdf_pts[:, :2], axis=0)
        return float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1])

    tree = cKDTree(egf_pts)
    d, _ = tree.query(tsdf_pts, k=1)
    ghost = tsdf_pts[d > min_dist]
    if ghost.shape[0] < 20:
        mins = np.min(tsdf_pts[:, :2], axis=0)
        maxs = np.max(tsdf_pts[:, :2], axis=0)
        cx = 0.5 * (mins[0] + maxs[0])
        cy = 0.5 * (mins[1] + maxs[1])
        return cx - 0.4, cx + 0.4, cy - 0.4, cy + 0.4

    xy = ghost[:, :2]
    x, y = xy[:, 0], xy[:, 1]
    xbins = np.linspace(np.min(x), np.max(x), 28)
    ybins = np.linspace(np.min(y), np.max(y), 28)
    if xbins.size < 2 or ybins.size < 2:
        mins = np.min(xy, axis=0)
        maxs = np.max(xy, axis=0)
        return float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1])
    h, xe, ye = np.histogram2d(x, y, bins=[xbins, ybins])
    ix, iy = np.unravel_index(np.argmax(h), h.shape)
    x0, x1 = xe[ix], xe[ix + 1]
    y0, y1 = ye[iy], ye[iy + 1]
    mx = max(0.12, 0.8 * (x1 - x0))
    my = max(0.12, 0.8 * (y1 - y0))
    return float(x0 - mx), float(x1 + mx), float(y0 - my), float(y1 + my)


def _count_in_roi(points: np.ndarray, roi: Tuple[float, float, float, float]) -> int:
    if points.shape[0] == 0:
        return 0
    x0, x1, y0, y1 = roi
    m = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    return int(np.count_nonzero(m))


def _outside_roi(points: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    x0, x1, y0, y1 = roi
    m = (points[:, 0] < x0) | (points[:, 0] > x1) | (points[:, 1] < y0) | (points[:, 1] > y1)
    return points[m]


def _fscore(pred: np.ndarray, gt: np.ndarray, thr: float = 0.05) -> Dict[str, float]:
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "fscore": 0.0}
    t_gt = cKDTree(gt)
    t_pr = cKDTree(pred)
    d_pr, _ = t_gt.query(pred, k=1)
    d_gt, _ = t_pr.query(gt, k=1)
    p = float(np.mean(d_pr < thr))
    r = float(np.mean(d_gt < thr))
    f = 0.0 if (p + r) <= 1e-9 else float(2.0 * p * r / (p + r))
    return {"precision": p, "recall": r, "fscore": f}


def draw_compare(
    egf_pts: np.ndarray,
    tsdf_pts: np.ndarray,
    egf_summary: Dict,
    tsdf_summary: Dict,
    out_path: Path,
    mode: str,
    forced_roi: Tuple[float, float, float, float] | None = None,
) -> Dict[str, float]:
    roi = None
    roi_metrics: Dict[str, float] = {}
    if mode == "dynamic":
        roi = forced_roi if forced_roi is not None else _auto_ghost_roi(tsdf_pts, egf_pts, min_dist=0.08)
        tsdf_n = _count_in_roi(tsdf_pts, roi)
        egf_n = _count_in_roi(egf_pts, roi)
        roi_metrics = {
            "roi_xmin": float(roi[0]),
            "roi_xmax": float(roi[1]),
            "roi_ymin": float(roi[2]),
            "roi_ymax": float(roi[3]),
            "roi_points_tsdf": float(tsdf_n),
            "roi_points_egf": float(egf_n),
            "roi_points_reduction": float(tsdf_n - egf_n),
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    panels = [
        ("EGF-DHMap", egf_pts, egf_summary, "#1f77b4"),
        ("TSDF", tsdf_pts, tsdf_summary, "#ff7f0e"),
    ]
    for ax, (name, pts, summary, color) in zip(axes, panels):
        if pts.shape[0] > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=0.25, c=color, alpha=0.75)
        met = summary.get("metrics", {})
        mesh = summary.get("mesh", {})
        t = (
            f"{name}\n"
            f"Chamfer={met.get('chamfer', float('nan')):.4f}, "
            f"F={met.get('fscore', float('nan')):.4f}\n"
            f"Vertices={mesh.get('vertices', 0)}"
        )
        ax.set_title(t)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
        if roi is not None:
            x0, x1, y0, y1 = roi
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.0, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

    if mode == "dynamic" and roi_metrics:
        fig.suptitle(
            "Dynamic Ghost ROI (red): "
            f"TSDF={int(roi_metrics['roi_points_tsdf'])}, "
            f"EGF={int(roi_metrics['roi_points_egf'])}, "
            f"Δ={int(roi_metrics['roi_points_reduction'])}"
        )
    else:
        fig.suptitle("Static Scene Mesh Top-View Comparison")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return roi_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--egf_mesh", type=str, required=True)
    parser.add_argument("--tsdf_mesh", type=str, required=True)
    parser.add_argument("--egf_summary", type=str, required=True)
    parser.add_argument("--tsdf_summary", type=str, required=True)
    parser.add_argument("--out_png", type=str, required=True)
    parser.add_argument("--mode", type=str, default="static", choices=["static", "dynamic"])
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--reference_points", type=str, default=None)
    parser.add_argument("--bg_eval_thresh", type=float, default=0.05)
    parser.add_argument("--roi_xmin", type=float, default=None)
    parser.add_argument("--roi_xmax", type=float, default=None)
    parser.add_argument("--roi_ymin", type=float, default=None)
    parser.add_argument("--roi_ymax", type=float, default=None)
    args = parser.parse_args()

    egf_pts = load_mesh_vertices(args.egf_mesh)
    tsdf_pts = load_mesh_vertices(args.tsdf_mesh)
    with open(args.egf_summary, "r", encoding="utf-8") as f:
        egf_summary = json.load(f)
    with open(args.tsdf_summary, "r", encoding="utf-8") as f:
        tsdf_summary = json.load(f)

    forced_roi = None
    if (
        args.roi_xmin is not None
        and args.roi_xmax is not None
        and args.roi_ymin is not None
        and args.roi_ymax is not None
    ):
        forced_roi = (
            float(args.roi_xmin),
            float(args.roi_xmax),
            float(args.roi_ymin),
            float(args.roi_ymax),
        )

    roi_metrics = draw_compare(
        egf_pts=egf_pts,
        tsdf_pts=tsdf_pts,
        egf_summary=egf_summary,
        tsdf_summary=tsdf_summary,
        out_path=Path(args.out_png),
        mode=args.mode,
        forced_roi=forced_roi,
    )
    if args.reference_points and args.mode == "dynamic":
        ref = np.asarray(o3d.io.read_point_cloud(args.reference_points).points, dtype=float)
        if ref.shape[0] > 0 and roi_metrics:
            roi = (
                float(roi_metrics["roi_xmin"]),
                float(roi_metrics["roi_xmax"]),
                float(roi_metrics["roi_ymin"]),
                float(roi_metrics["roi_ymax"]),
            )
            ref_bg = _outside_roi(ref, roi)
            egf_bg = _outside_roi(egf_pts, roi)
            tsdf_bg = _outside_roi(tsdf_pts, roi)
            roi_metrics["background_metrics"] = {
                "egf": _fscore(egf_bg, ref_bg, thr=float(args.bg_eval_thresh)),
                "tsdf": _fscore(tsdf_bg, ref_bg, thr=float(args.bg_eval_thresh)),
            }
    if args.out_json:
        out = {"mode": args.mode, "roi_metrics": roi_metrics}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
