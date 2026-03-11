from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from egf_dhmap3d.P10_method.rps_plane_attribution import extract_dominant_planes, point_to_plane_distance, snap_points_to_plane
from run_benchmark import (
    build_dynamic_references,
    compute_dynamic_metrics,
    compute_recon_metrics,
    load_points_with_normals,
    rigid_align_for_eval,
)


CONTROL_ROOT = PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "99_manhattan_plane_completion"
DEEP_COMPARE = PROJECT_ROOT / "processes" / "s2" / "S2_RPS_DEEP_EXPLORE_COMPARE.csv"
TSDF_TABLE = PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "80_ray_penetration_consistency" / "bonn_slam" / "slam" / "tables" / "dynamic_metrics.csv"
BONN_ALL3 = ["rgbd_bonn_balloon2", "rgbd_bonn_balloon", "rgbd_bonn_crowd2"]


def load_control_tum_metrics() -> Tuple[float, float]:
    rows = list(csv.DictReader(DEEP_COMPARE.open("r", encoding="utf-8")))
    row = next(r for r in rows if r["variant"] == "99_manhattan_plane_completion")
    return float(row["tum_acc_cm"]), float(row["tum_comp_r_5cm"])


def load_control_tsdf_ghosts() -> Dict[str, float]:
    rows = list(csv.DictReader(TSDF_TABLE.open("r", encoding="utf-8")))
    return {str(r["sequence"]): float(r["ghost_ratio"]) for r in rows if str(r.get("method", "")).lower() == "tsdf"}


def load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    out: List[dict] = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            if value in (None, ""):
                parsed[key] = 0.0
            else:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
        out.append(parsed)
    return out


def save_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["x", "y", "z"])
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_point_cloud(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd)


def classify_points(sequence: str, points: np.ndarray, *, frames: int, stride: int, max_points: int, seed: int) -> Dict[str, float]:
    import open3d as o3d

    stable_bg, _tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
        sequence_dir=PROJECT_ROOT / "data" / "bonn" / sequence,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points,
        seed=seed,
    )
    pts = np.asarray(points, dtype=float)
    bg_tree = None
    if stable_bg.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stable_bg)
        bg_tree = o3d.geometry.KDTreeFlann(pcd)
    out = {"tb": 0.0, "ghost": 0.0, "noise": 0.0}
    for point in pts:
        voxel = tuple(np.floor(point / float(dynamic_voxel)).astype(np.int32).tolist())
        if voxel in dynamic_region:
            out["ghost"] += 1.0
            continue
        if bg_tree is not None:
            _, idx, dist2 = bg_tree.search_knn_vector_3d(point.astype(float), 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < 0.05:
                out["tb"] += 1.0
                continue
        out["noise"] += 1.0
    return out


def rear_plane_distances(points: np.ndarray, planes: List[object]) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0 or not planes:
        return np.zeros((pts.shape[0],), dtype=float)
    d = np.stack([point_to_plane_distance(pts, plane) for plane in planes], axis=1)
    return d[np.arange(pts.shape[0]), np.argmin(d, axis=1)]


def orthogonality_drift(planes: List[object]) -> float:
    if len(planes) < 2:
        return 0.0
    vals = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            vals.append(abs(float(np.dot(planes[i].normal, planes[j].normal))))
    return float(np.mean(vals)) if vals else 0.0


def build_affine_from_planes(planes: List[object], gamma: float = 0.25) -> np.ndarray | None:
    if len(planes) < 2:
        return None
    a1 = np.asarray(planes[0].normal, dtype=float)
    a2 = np.asarray(planes[1].normal, dtype=float)
    a1 = a1 / max(1e-9, float(np.linalg.norm(a1)))
    a2 = a2 / max(1e-9, float(np.linalg.norm(a2)))
    a3 = np.cross(a1, a2)
    if float(np.linalg.norm(a3)) < 1e-6:
        return None
    a3 = a3 / float(np.linalg.norm(a3))
    current = np.stack([a1, a2, a3], axis=1)
    b1 = a1
    b2 = a2 - float(np.dot(a2, b1)) * b1
    if float(np.linalg.norm(b2)) < 1e-6:
        return None
    b2 = b2 / float(np.linalg.norm(b2))
    b3 = np.cross(b1, b2)
    b3 = b3 / max(1e-9, float(np.linalg.norm(b3)))
    target = np.stack([b1, b2, b3], axis=1)
    return (1.0 - gamma) * np.eye(3, dtype=float) + gamma * (target @ np.linalg.inv(current))


def transform_variant(
    *,
    variant: str,
    rear_points: np.ndarray,
    rear_normals: np.ndarray,
    front_points: np.ndarray,
    surface_points: np.ndarray,
    rows: List[dict],
) -> Tuple[np.ndarray, np.ndarray, List[dict], Dict[str, float]]:
    rear = np.asarray(rear_points, dtype=float)
    normals = np.asarray(rear_normals, dtype=float)
    out_rows = [dict(row) for row in rows]
    planes = extract_dominant_planes(surface_points, distance_threshold=0.03, min_plane_points=40, max_planes=6, min_extent_xy=0.25)
    before_dist = rear_plane_distances(rear, planes)
    mean_before = float(np.mean(before_dist)) if before_dist.size > 0 else 0.0
    drift_before = orthogonality_drift(planes)

    if variant == "99_manhattan_plane_completion":
        for i, row in enumerate(out_rows):
            row["mean_distance_to_plane_before"] = float(before_dist[i]) if before_dist.size else 0.0
            row["mean_distance_to_plane_after"] = float(before_dist[i]) if before_dist.size else 0.0
        return rear, normals, out_rows, {
            "snapped_points": 0.0,
            "drift_before": drift_before,
            "drift_after": drift_before,
            "mean_distance_to_plane_before": mean_before,
            "mean_distance_to_plane_after": mean_before,
        }

    transformed = rear.copy()
    transformed_normals = normals.copy()
    snapped_mask = np.zeros((rear.shape[0],), dtype=bool)

    if variant == "101_manhattan_plane_projection_hard_snapping":
        if planes:
            d = np.stack([point_to_plane_distance(transformed, plane) for plane in planes], axis=1)
            assign = np.argmin(d, axis=1)
            mind = d[np.arange(transformed.shape[0]), assign]
            snapped_mask = mind <= 0.03
            for i in np.where(snapped_mask)[0]:
                plane = planes[int(assign[i])]
                transformed[i : i + 1] = snap_points_to_plane(transformed[i : i + 1], plane)
                if transformed_normals.shape == rear.shape:
                    nn = np.asarray(plane.normal, dtype=float).copy()
                    if float(np.dot(nn, transformed_normals[i])) < 0.0:
                        nn = -nn
                    transformed_normals[i] = nn
    elif variant == "102_scale_drift_correction":
        affine = build_affine_from_planes(planes, gamma=0.25)
        if affine is not None:
            center = transformed.mean(axis=0)
            transformed = ((affine @ (transformed - center).T).T) + center
            if transformed_normals.shape == rear.shape:
                transformed_normals = (affine @ transformed_normals.T).T
                transformed_normals = transformed_normals / np.clip(np.linalg.norm(transformed_normals, axis=1, keepdims=True), 1e-9, None)
            snapped_mask[:] = True
    elif variant == "103_local_cluster_refinement":
        patch_mask = np.asarray([float(row.get("completion_is_patch", 0.0)) > 0.5 for row in out_rows], dtype=bool)
        if patch_mask.any():
            patch_points = transformed[patch_mask]
            tree = cKDTree(patch_points)
            patch_indices = np.where(patch_mask)[0]
            for local_i, global_i in enumerate(patch_indices):
                k = min(6, patch_points.shape[0])
                _dist, idx = tree.query(transformed[global_i], k=k)
                idx = np.atleast_1d(idx)[1:]
                if idx.size <= 0:
                    continue
                mean = patch_points[idx].mean(axis=0)
                transformed[global_i] = 0.75 * transformed[global_i] + 0.25 * mean
                snapped_mask[global_i] = True
    else:
        raise ValueError(f"Unknown variant: {variant}")

    after_dist = rear_plane_distances(transformed, planes)
    mean_after = float(np.mean(after_dist)) if after_dist.size > 0 else mean_before
    drift_after = orthogonality_drift(planes)
    for i, row in enumerate(out_rows):
        row["mean_distance_to_plane_before"] = float(before_dist[i]) if before_dist.size else 0.0
        row["mean_distance_to_plane_after"] = float(after_dist[i]) if after_dist.size else 0.0
        row["geometry_tightened"] = 1.0 if snapped_mask[i] else 0.0
    return transformed, transformed_normals, out_rows, {
        "snapped_points": float(np.count_nonzero(snapped_mask)),
        "drift_before": drift_before,
        "drift_after": drift_after,
        "mean_distance_to_plane_before": mean_before,
        "mean_distance_to_plane_after": mean_after,
    }


def evaluate_variant(
    *,
    variant_name: str,
    root_out: Path,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    seed: int,
    tsdf_ghosts: Dict[str, float],
    tum_acc_cm: float,
    tum_comp_r_5cm: float,
) -> dict:
    bonn_accs: List[float] = []
    bonn_comps: List[float] = []
    bonn_ghost_reds: List[float] = []
    rear_sum = tb_sum = ghost_sum = noise_sum = 0.0
    snapped_sum = 0.0
    plane_before_sum = plane_after_sum = 0.0
    drift_before_sum = drift_after_sum = 0.0
    seq_pairs: List[Tuple[float, float]] = []

    for seq in BONN_ALL3:
        base = CONTROL_ROOT / "bonn_slam" / "slam" / seq / "egf"
        seq_out = root_out / "bonn_slam" / "slam" / seq / "egf"
        seq_out.parent.mkdir(parents=True, exist_ok=True)
        if variant_name != "99_manhattan_plane_completion":
            shutil.copytree(base, seq_out, dirs_exist_ok=True)
        else:
            seq_out.mkdir(parents=True, exist_ok=True)

        rear_points, rear_normals = load_points_with_normals(base / "rear_surface_points.ply")
        front_points, front_normals = load_points_with_normals(base / "front_surface_points.ply")
        surface_points, _surface_normals = load_points_with_normals(base / "surface_points.ply")
        ref_points, ref_normals = load_points_with_normals(base / "reference_points.ply")
        rows = load_rows(base / "rear_surface_features.csv")

        transformed_rear, transformed_normals, rows_out, tighten_stats = transform_variant(
            variant=variant_name,
            rear_points=rear_points,
            rear_normals=rear_normals,
            front_points=front_points,
            surface_points=surface_points,
            rows=rows,
        )

        pred_points = np.vstack([front_points, transformed_rear]) if front_points.shape[0] > 0 else transformed_rear
        pred_normals = np.vstack([front_normals, transformed_normals]) if front_points.shape[0] > 0 else transformed_normals
        pred_eval_points, pred_eval_normals, _align_info, _align_t = rigid_align_for_eval(
            pred_points=pred_points,
            pred_normals=pred_normals,
            ref_points=ref_points,
            voxel_size=0.02,
        )
        recon = compute_recon_metrics(pred_eval_points, ref_points, threshold=0.05, pred_normals=pred_eval_normals, gt_normals=ref_normals)

        stable_bg, tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
            sequence_dir=PROJECT_ROOT / "data" / "bonn" / seq,
            frames=frames,
            stride=stride,
            max_points_per_frame=max_points_per_frame,
            seed=seed,
            stable_voxel=max(0.03, 0.02),
            stable_ratio=0.25,
            tail_frames=12,
            dynamic_voxel=max(0.05, 0.04),
            min_dynamic_hits=2,
            max_dynamic_ratio=0.35,
        )
        dyn = compute_dynamic_metrics(
            pred_points=pred_eval_points,
            stable_bg_points=stable_bg,
            tail_points=tail_points,
            dynamic_region=dynamic_region,
            dynamic_voxel=dynamic_voxel,
            ghost_thresh=0.08,
            bg_thresh=0.05,
        )
        ghost_red = ((tsdf_ghosts[seq] - float(dyn["ghost_ratio"])) / max(1e-9, tsdf_ghosts[seq])) * 100.0

        bonn_accs.append(float(recon["accuracy"]) * 100.0)
        bonn_comps.append(float(recon["recall_5cm"]) * 100.0)
        bonn_ghost_reds.append(ghost_red)
        classes = classify_points(seq, transformed_rear, frames=frames, stride=stride, max_points=max_points_per_frame, seed=seed)
        rear_sum += float(transformed_rear.shape[0])
        tb_sum += classes["tb"]
        ghost_sum += classes["ghost"]
        noise_sum += classes["noise"]
        seq_pairs.append((float(classes["tb"]), float(classes["noise"])))
        snapped_sum += float(tighten_stats["snapped_points"])
        plane_before_sum += float(tighten_stats["mean_distance_to_plane_before"])
        plane_after_sum += float(tighten_stats["mean_distance_to_plane_after"])
        drift_before_sum += float(tighten_stats["drift_before"])
        drift_after_sum += float(tighten_stats["drift_after"])

        save_point_cloud(seq_out / "rear_surface_points.ply", transformed_rear, transformed_normals)
        save_point_cloud(seq_out / "surface_points.ply", pred_points, pred_normals)
        save_rows(seq_out / "rear_surface_features.csv", rows_out)
        with (seq_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"variant": variant_name, "tighten_stats": tighten_stats, "recon": recon, "dynamic": dyn}, f, indent=2)

    pair_arr = np.asarray(seq_pairs, dtype=float) if seq_pairs else np.zeros((0, 2), dtype=float)
    tb_noise_corr = 0.0
    if pair_arr.shape[0] >= 2 and np.std(pair_arr[:, 0]) > 1e-9 and np.std(pair_arr[:, 1]) > 1e-9:
        tb_noise_corr = float(np.corrcoef(pair_arr[:, 0], pair_arr[:, 1])[0, 1])

    return {
        "variant": variant_name,
        "tum_acc_cm": tum_acc_cm,
        "tum_comp_r_5cm": tum_comp_r_5cm,
        "bonn_acc_cm": float(np.mean(bonn_accs)),
        "bonn_comp_r_5cm": float(np.mean(bonn_comps)),
        "bonn_ghost_reduction_vs_tsdf": float(np.mean(bonn_ghost_reds)),
        "bonn_rear_points_sum": float(rear_sum),
        "bonn_rear_true_background_sum": float(tb_sum),
        "bonn_rear_ghost_sum": float(ghost_sum),
        "bonn_rear_hole_or_noise_sum": float(noise_sum),
        "snapped_points_sum": float(snapped_sum),
        "mean_distance_to_plane_before": float(plane_before_sum / len(BONN_ALL3)),
        "mean_distance_to_plane_after": float(plane_after_sum / len(BONN_ALL3)),
        "orthogonality_drift_before": float(drift_before_sum / len(BONN_ALL3)),
        "orthogonality_drift_after": float(drift_after_sum / len(BONN_ALL3)),
        "tb_noise_correlation": float(tb_noise_corr),
        "decision": "pending",
    }


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r["variant"] == "99_manhattan_plane_completion")
    for row in rows:
        if row is control:
            row["decision"] = "control"
            continue
        acc_ok = row["bonn_acc_cm"] <= 3.10
        comp_ok = row["bonn_comp_r_5cm"] >= 70.0
        corr_ok = row["tb_noise_correlation"] < 0.0
        if acc_ok and comp_ok and corr_ok:
            row["decision"] = "iterate"
        elif row["bonn_acc_cm"] < 3.5 and comp_ok and corr_ok:
            row["decision"] = "progress"
        else:
            row["decision"] = "abandon"


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# S2 Acc tightening compare",
        "",
        "| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | tb_noise_correlation | mean_distance_to_plane_before | mean_distance_to_plane_after | orthogonality_drift_before | orthogonality_drift_after | snapped_points_sum | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_acc_cm']:.3f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['tb_noise_correlation']:.3f} | {row['mean_distance_to_plane_before']:.4f} | {row['mean_distance_to_plane_after']:.4f} | {row['orthogonality_drift_before']:.4f} | {row['orthogonality_drift_after']:.4f} | {row['snapped_points_sum']:.0f} | {row['decision']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_distribution(rows: List[dict], path_md: Path) -> None:
    lines = [
        "# S2 Acc tightening distribution report",
        "",
        "日期：`2026-03-11`",
        "协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "",
        "| variant | bonn_acc_cm | comp_r | ghost_reduction | tb | ghost | noise | plane_dist_before | plane_dist_after | snapped_points_sum | tb_noise_correlation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_acc_cm']:.3f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['mean_distance_to_plane_before']:.4f} | {row['mean_distance_to_plane_after']:.4f} | {row['snapped_points_sum']:.0f} | {row['tb_noise_correlation']:.3f} |"
        )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r["variant"] == "99_manhattan_plane_completion")
    best = min(rows[1:], key=lambda r: r["bonn_acc_cm"])
    lines = [
        "# S2 Acc tightening analysis",
        "",
        "日期：`2026-03-11`",
        "协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "对比表：`processes/s2/S2_RPS_ACC_TIGHTENING_COMPARE.csv`",
        "",
        "## 1. Acc 改善幅度",
        f"- 控制组 `99`：Bonn `Acc={control['bonn_acc_cm']:.3f} cm`, `Comp-R={control['bonn_comp_r_5cm']:.2f}%`, `corr={control['tb_noise_correlation']:.3f}`。",
        f"- 最佳候选 `{best['variant']}`：Bonn `Acc={best['bonn_acc_cm']:.3f} cm`, `Comp-R={best['bonn_comp_r_5cm']:.2f}%`, `corr={best['tb_noise_correlation']:.3f}`。",
        "",
        "## 2. 投影/校正/紧化谁起作用",
        "- 若 `mean_distance_to_plane_after` 明显下降但 `Acc` 几乎不变，说明当前瓶颈不再是平面厚度噪声，而是更深层的几何畸变或前景/背景的系统偏差。",
        "- 若 `orthogonality_drift_before` 与 `orthogonality_drift_after` 基本一致，说明没有明显的 geometry-only scale drift 证据。",
        "",
        "## 3. 结论",
        "- 若 `Acc` 没有从 `4.31 cm` 显著下降，则本轮 `hard snapping / drift correction / local refinement` 只能视为负结果链。",
        "- 若 `Comp-R` 与 `TB/corr` 基本保持，而 `Acc` 仍顽固不动，则说明瓶颈已从“局部噪声”转移到“几何畸变 / front-side bias”。",
        "- 只要 `Acc` 仍高于 `3.10 cm`，`S2` 仍未通过，且绝对不能进入 `S3`。",
    ]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 Acc tightening runner.")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_points_per_frame", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tum_acc_cm, tum_comp_r_5cm = load_control_tum_metrics()
    tsdf_ghosts = load_control_tsdf_ghosts()
    variants = [
        ("99_manhattan_plane_completion", CONTROL_ROOT),
        ("101_manhattan_plane_projection_hard_snapping", PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "101_manhattan_plane_projection_hard_snapping"),
        ("102_scale_drift_correction", PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "102_scale_drift_correction"),
        ("103_local_cluster_refinement", PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "103_local_cluster_refinement"),
    ]
    for _, root in variants[1:]:
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(CONTROL_ROOT / "tum_oracle" / "oracle", root / "tum_oracle" / "oracle", dirs_exist_ok=True)

    rows = [
        evaluate_variant(
            variant_name=name,
            root_out=root,
            frames=args.frames,
            stride=args.stride,
            max_points_per_frame=args.max_points_per_frame,
            seed=args.seed,
            tsdf_ghosts=tsdf_ghosts,
            tum_acc_cm=tum_acc_cm,
            tum_comp_r_5cm=tum_comp_r_5cm,
        )
        for name, root in variants
    ]
    decide(rows)
    out_dir = PROJECT_ROOT / "processes" / "s2"
    write_compare(rows, out_dir / "S2_RPS_ACC_TIGHTENING_COMPARE.csv", out_dir / "S2_RPS_ACC_TIGHTENING_COMPARE.md")
    write_distribution(rows, out_dir / "S2_RPS_ACC_TIGHTENING_DISTRIBUTION.md")
    write_analysis(rows, out_dir / "S2_RPS_ACC_TIGHTENING_ANALYSIS.md")
    print("[done]", out_dir / "S2_RPS_ACC_TIGHTENING_COMPARE.csv")


if __name__ == "__main__":
    main()
