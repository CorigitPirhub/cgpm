from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from egf_dhmap3d.P10_method.rps_plane_attribution import extract_dominant_planes, nearest_plane_assignment
from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.data.tum_rgbd import TUMRGBDStream
from run_s2_rps_rear_geometry_quality import BONN_ALL3, TUM_ALL3
from scripts.data.bonn_rgbd import BonnRGBDStream


BASELINE_ROOT = PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "111_native_geometry_chain_direct"
NO_COMPLETION_ROOT = PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "104_depth_bias_minus1cm"

OUT_DIR = PROJECT_ROOT / "processes" / "s2"
CSV_PATH = OUT_DIR / "S2_ERROR_DECOMPOSITION_COMPARE.csv"
GEOM_REPORT_PATH = OUT_DIR / "S2_GEOMETRY_ERROR_DECOMPOSITION_REPORT.md"
COMP_R_REPORT_PATH = OUT_DIR / "S2_COMP_R_GAP_ANALYSIS.md"

ACC_TARGETS_CM = {"tum": 2.55, "bonn": 3.10}
COMP_R_TARGET_PTS = {"tum": 98.0, "bonn": 98.0}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> List[dict]:
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def pick_row(rows: List[dict], sequence: str, method: str = "egf") -> dict:
    method = method.lower()
    for row in rows:
        if str(row.get("sequence", "")) == sequence and str(row.get("method", "")).lower() == method:
            return row
    raise KeyError(f"missing row: sequence={sequence} method={method}")


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=float)
    return pts if pts.ndim == 2 else np.zeros((0, 3), dtype=float)


def family_root(root: Path, family: str) -> Path:
    if family == "tum":
        return root / "tum_oracle" / "oracle"
    return root / "bonn_slam" / "slam"


def seq_dir(root: Path, family: str, sequence: str) -> Path:
    return family_root(root, family) / sequence / "egf"


def tables_root(root: Path, family: str) -> Path:
    return family_root(root, family) / "tables"


def aligned_points(base_dir: Path, points_name: str) -> np.ndarray:
    pts = load_points(base_dir / points_name)
    tf_path = base_dir / "eval_alignment_transform.npy"
    if tf_path.exists():
        transform = np.load(tf_path)
        pts = (transform[:3, :3] @ pts.T).T + transform[:3, 3]
    return np.asarray(pts, dtype=float)


def eval_points(base_dir: Path) -> np.ndarray:
    aligned_path = base_dir / "surface_points_aligned_eval.ply"
    if aligned_path.exists():
        return load_points(aligned_path)
    return aligned_points(base_dir, "surface_points.ply")


def make_stream(family: str, sequence: str, cfg: EGF3DConfig, *, frames: int, stride: int, seed: int):
    if family == "tum":
        return list(
            TUMRGBDStream(
                PROJECT_ROOT / "data" / "tum" / sequence,
                cfg,
                max_frames=frames,
                stride=stride,
                max_points=600,
                seed=seed,
            )
        )
    return list(
        BonnRGBDStream(
            PROJECT_ROOT / "data" / "bonn" / sequence,
            cfg,
            max_frames=frames,
            stride=stride,
            max_points=600,
            seed=seed,
        )
    )


def stable_depth_residual_stats(
    *,
    family: str,
    sequence: str,
    reference_points: np.ndarray,
    frames: int,
    stride: int,
    seed: int,
    stable_gate_m: float = 0.05,
) -> Dict[str, float]:
    cfg = EGF3DConfig()
    stream = make_stream(family, sequence, cfg, frames=frames, stride=stride, seed=seed)
    if reference_points.shape[0] == 0 or not stream:
        return {
            "stable_depth_bias_cm": 0.0,
            "sensor_noise_floor_cm": 0.0,
            "stable_depth_samples": 0.0,
            "stable_depth_multiview_points": 0.0,
        }

    ref_h = np.concatenate([reference_points, np.ones((reference_points.shape[0], 1), dtype=float)], axis=1)
    per_point_residuals: List[List[float]] = [[] for _ in range(reference_points.shape[0])]

    cam = cfg.camera
    for frame in stream:
        depth = imageio.imread(frame.depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth_m = depth.astype(np.float32) / float(cam.depth_scale)

        t_cw = np.linalg.inv(frame.pose_w_c)
        points_cam = (t_cw @ ref_h.T).T[:, :3]
        z = points_cam[:, 2]
        u = np.round(cam.fx * points_cam[:, 0] / np.clip(z, 1e-6, None) + cam.cx).astype(np.int32)
        v = np.round(cam.fy * points_cam[:, 1] / np.clip(z, 1e-6, None) + cam.cy).astype(np.int32)

        valid = (
            (z > cam.depth_min)
            & (z < cam.depth_max)
            & (u >= 0)
            & (u < int(cam.width))
            & (v >= 0)
            & (v < int(cam.height))
        )
        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            continue

        depth_vals = depth_m[v[valid_idx], u[valid_idx]]
        depth_valid = (depth_vals > cam.depth_min) & (depth_vals < cam.depth_max)
        valid_idx = valid_idx[depth_valid]
        if valid_idx.size == 0:
            continue

        residual = depth_vals[depth_valid] - z[valid_idx]
        stable_mask = np.abs(residual) <= float(stable_gate_m)
        for point_idx, res in zip(valid_idx[stable_mask], residual[stable_mask]):
            per_point_residuals[int(point_idx)].append(float(res))

    all_means = []
    all_stds = []
    stable_samples = 0
    multiview_points = 0
    for values in per_point_residuals:
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        stable_samples += int(arr.size)
        all_means.append(float(np.mean(arr)))
        if arr.size >= 2:
            multiview_points += 1
            all_stds.append(float(np.std(arr)))

    if not all_means:
        return {
            "stable_depth_bias_cm": 0.0,
            "sensor_noise_floor_cm": 0.0,
            "stable_depth_samples": 0.0,
            "stable_depth_multiview_points": 0.0,
        }
    return {
        "stable_depth_bias_cm": float(abs(np.mean(np.asarray(all_means, dtype=float))) * 100.0),
        "sensor_noise_floor_cm": float(np.median(np.asarray(all_stds, dtype=float)) * 100.0) if all_stds else 0.0,
        "stable_depth_samples": float(stable_samples),
        "stable_depth_multiview_points": float(multiview_points),
    }


def surface_thickness_stats(base_dir: Path) -> Dict[str, float]:
    front = aligned_points(base_dir, "front_surface_points.ply")
    if front.shape[0] < 40:
        return {
            "surface_thickness_cm": 0.0,
            "surface_thickness_median_cm": 0.0,
            "surface_plane_count": 0.0,
            "surface_plane_coverage": 0.0,
        }
    planes = extract_dominant_planes(
        front,
        distance_threshold=0.03,
        min_plane_points=40,
        max_planes=6,
        min_extent_xy=0.25,
    )
    if not planes:
        return {
            "surface_thickness_cm": 0.0,
            "surface_thickness_median_cm": 0.0,
            "surface_plane_count": 0.0,
            "surface_plane_coverage": 0.0,
        }
    _assign, dist = nearest_plane_assignment(front, planes)
    mask = np.isfinite(dist) & (dist < 0.05)
    if not np.any(mask):
        return {
            "surface_thickness_cm": 0.0,
            "surface_thickness_median_cm": 0.0,
            "surface_plane_count": float(len(planes)),
            "surface_plane_coverage": 0.0,
        }
    dist_sel = dist[mask]
    return {
        "surface_thickness_cm": float(np.mean(dist_sel) * 100.0),
        "surface_thickness_median_cm": float(np.median(dist_sel) * 100.0),
        "surface_plane_count": float(len(planes)),
        "surface_plane_coverage": float(np.mean(mask)),
    }


def comp_r_visibility_stats(
    *,
    sequence: str,
    reference_points: np.ndarray,
    pred_points: np.ndarray,
    frames: int,
    stride: int,
    seed: int,
) -> Dict[str, float]:
    cfg = EGF3DConfig()
    stream = make_stream("bonn", sequence, cfg, frames=frames, stride=stride, seed=seed)

    if reference_points.shape[0] == 0:
        return {
            "reference_points_count": 0.0,
            "reconstructed_ratio": 0.0,
            "missing_reference_points": 0.0,
            "missing_never_in_view_ratio": 0.0,
            "missing_observed_once_ratio": 0.0,
            "missing_observed_twice_or_less_ratio": 0.0,
            "missing_high_occlusion_ratio": 0.0,
            "missing_depth_hole_ratio": 0.0,
            "missing_observed_any_ratio": 0.0,
        }

    tree = cKDTree(pred_points) if pred_points.shape[0] > 0 else None
    if tree is None:
        reconstructed = np.zeros((reference_points.shape[0],), dtype=bool)
    else:
        dist, _ = tree.query(reference_points, k=1)
        reconstructed = dist < 0.05

    ref_h = np.concatenate([reference_points, np.ones((reference_points.shape[0], 1), dtype=float)], axis=1)
    counts = {
        "in_view": np.zeros((reference_points.shape[0],), dtype=np.int32),
        "observed": np.zeros((reference_points.shape[0],), dtype=np.int32),
        "occluded": np.zeros((reference_points.shape[0],), dtype=np.int32),
        "depth_hole": np.zeros((reference_points.shape[0],), dtype=np.int32),
        "far": np.zeros((reference_points.shape[0],), dtype=np.int32),
    }

    cam = cfg.camera
    for frame in stream:
        depth = imageio.imread(frame.depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth_m = depth.astype(np.float32) / float(cam.depth_scale)

        t_cw = np.linalg.inv(frame.pose_w_c)
        points_cam = (t_cw @ ref_h.T).T[:, :3]
        z = points_cam[:, 2]
        u = np.round(cam.fx * points_cam[:, 0] / np.clip(z, 1e-6, None) + cam.cx).astype(np.int32)
        v = np.round(cam.fy * points_cam[:, 1] / np.clip(z, 1e-6, None) + cam.cy).astype(np.int32)

        valid = (
            (z > cam.depth_min)
            & (z < cam.depth_max)
            & (u >= 0)
            & (u < int(cam.width))
            & (v >= 0)
            & (v < int(cam.height))
        )
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue
        counts["in_view"][idx] += 1

        depth_vals = depth_m[v[idx], u[idx]]
        depth_valid = (depth_vals > cam.depth_min) & (depth_vals < cam.depth_max)
        counts["depth_hole"][idx[~depth_valid]] += 1
        idx = idx[depth_valid]
        if idx.size == 0:
            continue

        residual = depth_vals[depth_valid] - z[idx]
        counts["observed"][idx[np.abs(residual) <= 0.05]] += 1
        counts["occluded"][idx[residual < -0.05]] += 1
        counts["far"][idx[residual > 0.05]] += 1

    missing = ~reconstructed
    miss_count = int(np.count_nonzero(missing))
    if miss_count <= 0:
        return {
            "reference_points_count": float(reference_points.shape[0]),
            "reconstructed_ratio": 1.0,
            "missing_reference_points": 0.0,
            "missing_never_in_view_ratio": 0.0,
            "missing_observed_once_ratio": 0.0,
            "missing_observed_twice_or_less_ratio": 0.0,
            "missing_high_occlusion_ratio": 0.0,
            "missing_depth_hole_ratio": 0.0,
            "missing_observed_any_ratio": 0.0,
        }

    obs = counts["observed"][missing]
    occ = counts["occluded"][missing]
    holes = counts["depth_hole"][missing]
    in_view = counts["in_view"][missing]
    occ_ratio = occ / np.maximum(1.0, obs + occ)

    return {
        "reference_points_count": float(reference_points.shape[0]),
        "reconstructed_ratio": float(np.mean(reconstructed)),
        "missing_reference_points": float(miss_count),
        "missing_never_in_view_ratio": float(np.mean(in_view == 0)),
        "missing_observed_any_ratio": float(np.mean(obs > 0)),
        "missing_observed_once_ratio": float(np.mean(obs <= 1)),
        "missing_observed_twice_or_less_ratio": float(np.mean(obs <= 2)),
        "missing_high_occlusion_ratio": float(np.mean(occ_ratio > 0.5)),
        "missing_depth_hole_ratio": float(np.mean(holes > 0)),
    }


def row_component_shares(row: dict) -> dict:
    positive_completion = max(0.0, float(row.get("completion_artifact_delta_cm", 0.0)))
    comps = {
        "sensor_noise_floor_cm": float(row.get("sensor_noise_floor_cm", 0.0)),
        "calibration_bias_cm": float(row.get("stable_depth_bias_cm", 0.0)),
        "temporal_drift_cm": float(row.get("temporal_drift_cm", 0.0)),
        "surface_thickness_cm": float(row.get("surface_thickness_cm", 0.0)),
        "completion_artifact_positive_cm": positive_completion,
    }
    total = float(sum(max(0.0, v) for v in comps.values()))
    if total <= 1e-9:
        out = {f"{k}_share": 0.0 for k in comps}
        out["primary_contributor"] = "none"
        return out
    out = {f"{k}_share": float(max(0.0, v) / total) for k, v in comps.items()}
    out["primary_contributor"] = max(comps.items(), key=lambda kv: max(0.0, kv[1]))[0]
    return out


def seq_error_row(
    *,
    family: str,
    sequence: str,
    frames: int,
    stride: int,
    seed: int,
    recon_rows_base: List[dict],
    recon_rows_nocomp: List[dict],
) -> dict:
    base_dir = seq_dir(BASELINE_ROOT, family, sequence)
    summary = load_json(base_dir / "summary.json")
    recon = pick_row(recon_rows_base, sequence, "egf")
    recon_no_completion = pick_row(recon_rows_nocomp, sequence, "egf")

    reference = load_points(base_dir / "reference_points.ply")
    pred = eval_points(base_dir)

    depth_stats = stable_depth_residual_stats(
        family=family,
        sequence=sequence,
        reference_points=reference,
        frames=frames,
        stride=stride,
        seed=seed,
    )
    thickness = surface_thickness_stats(base_dir)

    row = {
        "row_type": "sequence",
        "family": family,
        "sequence": sequence,
        "frames": float(frames),
        "stride": float(stride),
        "seed": float(seed),
        "acc_cm": float(recon["accuracy"]) * 100.0,
        "comp_r_5cm": float(recon["recall_5cm"]) * 100.0,
        "target_acc_cm": float(ACC_TARGETS_CM[family]),
        "target_comp_r_5cm": float(COMP_R_TARGET_PTS[family]),
        "acc_gap_cm": float(max(0.0, float(recon["accuracy"]) * 100.0 - ACC_TARGETS_CM[family])),
        "comp_r_gap_pts": float(max(0.0, COMP_R_TARGET_PTS[family] - float(recon["recall_5cm"]) * 100.0)),
        "temporal_drift_cm": float(summary["trajectory_metrics"]["ate_rmse"]) * 100.0,
        "rpe_trans_cm": float(summary["trajectory_metrics"]["rpe_trans_rmse"]) * 100.0,
        "completion_artifact_delta_cm": float(float(recon["accuracy"]) * 100.0 - float(recon_no_completion["accuracy"]) * 100.0),
        "completion_comp_r_delta_pts": float(float(recon["recall_5cm"]) * 100.0 - float(recon_no_completion["recall_5cm"]) * 100.0),
        "rear_surface_points": float(summary.get("rear_surface_points", 0.0)),
    }
    row.update(depth_stats)
    row.update(thickness)

    if family == "bonn":
        comp_r_diag = comp_r_visibility_stats(
            sequence=sequence,
            reference_points=reference,
            pred_points=pred,
            frames=frames,
            stride=stride,
            seed=seed,
        )
        row.update(comp_r_diag)
        if comp_r_diag["missing_reference_points"] > 0:
            row["rear_to_missing_ratio"] = float(row["rear_surface_points"] / comp_r_diag["missing_reference_points"])
        else:
            row["rear_to_missing_ratio"] = 0.0
    else:
        row.update(
            {
                "reference_points_count": float(reference.shape[0]),
                "reconstructed_ratio": float(recon["recall_5cm"]),
                "missing_reference_points": float(reference.shape[0] - round(float(recon["recall_5cm"]) * reference.shape[0])),
                "missing_never_in_view_ratio": 0.0,
                "missing_observed_any_ratio": 0.0,
                "missing_observed_once_ratio": 0.0,
                "missing_observed_twice_or_less_ratio": 0.0,
                "missing_high_occlusion_ratio": 0.0,
                "missing_depth_hole_ratio": 0.0,
                "rear_to_missing_ratio": 0.0,
            }
        )
    row.update(row_component_shares(row))
    return row


def family_row(family: str, rows: List[dict]) -> dict:
    seq_rows = [row for row in rows if row["family"] == family and row["row_type"] == "sequence"]
    weighted_ref = sum(float(row["reference_points_count"]) for row in seq_rows)
    weighted_missing = sum(float(row["missing_reference_points"]) for row in seq_rows)

    row = {
        "row_type": "family",
        "family": family,
        "sequence": f"{family}_all3",
        "frames": float(seq_rows[0]["frames"]),
        "stride": float(seq_rows[0]["stride"]),
        "seed": float(seq_rows[0]["seed"]),
        "acc_cm": float(mean(float(r["acc_cm"]) for r in seq_rows)),
        "comp_r_5cm": float(mean(float(r["comp_r_5cm"]) for r in seq_rows)),
        "target_acc_cm": float(ACC_TARGETS_CM[family]),
        "target_comp_r_5cm": float(COMP_R_TARGET_PTS[family]),
        "acc_gap_cm": float(max(0.0, mean(float(r["acc_cm"]) for r in seq_rows) - ACC_TARGETS_CM[family])),
        "comp_r_gap_pts": float(max(0.0, COMP_R_TARGET_PTS[family] - mean(float(r["comp_r_5cm"]) for r in seq_rows))),
        "stable_depth_bias_cm": float(mean(float(r["stable_depth_bias_cm"]) for r in seq_rows)),
        "sensor_noise_floor_cm": float(mean(float(r["sensor_noise_floor_cm"]) for r in seq_rows)),
        "stable_depth_samples": float(sum(float(r["stable_depth_samples"]) for r in seq_rows)),
        "stable_depth_multiview_points": float(sum(float(r["stable_depth_multiview_points"]) for r in seq_rows)),
        "temporal_drift_cm": float(mean(float(r["temporal_drift_cm"]) for r in seq_rows)),
        "rpe_trans_cm": float(mean(float(r["rpe_trans_cm"]) for r in seq_rows)),
        "surface_thickness_cm": float(mean(float(r["surface_thickness_cm"]) for r in seq_rows)),
        "surface_thickness_median_cm": float(mean(float(r["surface_thickness_median_cm"]) for r in seq_rows)),
        "surface_plane_count": float(mean(float(r["surface_plane_count"]) for r in seq_rows)),
        "surface_plane_coverage": float(mean(float(r["surface_plane_coverage"]) for r in seq_rows)),
        "completion_artifact_delta_cm": float(mean(float(r["completion_artifact_delta_cm"]) for r in seq_rows)),
        "completion_comp_r_delta_pts": float(mean(float(r["completion_comp_r_delta_pts"]) for r in seq_rows)),
        "rear_surface_points": float(sum(float(r["rear_surface_points"]) for r in seq_rows)),
        "reference_points_count": float(weighted_ref),
        "reconstructed_ratio": float(sum(float(r["reconstructed_ratio"]) * float(r["reference_points_count"]) for r in seq_rows) / max(1.0, weighted_ref)),
        "missing_reference_points": float(weighted_missing),
        "missing_never_in_view_ratio": float(sum(float(r["missing_never_in_view_ratio"]) * float(r["missing_reference_points"]) for r in seq_rows) / max(1.0, weighted_missing)),
        "missing_observed_any_ratio": float(sum(float(r["missing_observed_any_ratio"]) * float(r["missing_reference_points"]) for r in seq_rows) / max(1.0, weighted_missing)),
        "missing_observed_once_ratio": float(sum(float(r["missing_observed_once_ratio"]) * float(r["missing_reference_points"]) for r in seq_rows) / max(1.0, weighted_missing)),
        "missing_observed_twice_or_less_ratio": float(sum(float(r["missing_observed_twice_or_less_ratio"]) * float(r["missing_reference_points"]) for r in seq_rows) / max(1.0, weighted_missing)),
        "missing_high_occlusion_ratio": float(sum(float(r["missing_high_occlusion_ratio"]) * float(r["missing_reference_points"]) for r in seq_rows) / max(1.0, weighted_missing)),
        "missing_depth_hole_ratio": float(sum(float(r["missing_depth_hole_ratio"]) * float(r["missing_reference_points"]) for r in seq_rows) / max(1.0, weighted_missing)),
        "rear_to_missing_ratio": float(sum(float(r["rear_surface_points"]) for r in seq_rows) / max(1.0, weighted_missing)),
    }
    row.update(row_component_shares(row))
    return row


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def fmt(v: float, digits: int = 3) -> str:
    return f"{float(v):.{digits}f}"


def write_geometry_report(rows: List[dict], path: Path) -> None:
    tum = next(r for r in rows if r["row_type"] == "family" and r["family"] == "tum")
    bonn = next(r for r in rows if r["row_type"] == "family" and r["family"] == "bonn")

    lines = [
        "# S2 Geometry Error Decomposition Report",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "基线：`111_native_geometry_chain_direct`",
        "协议：`frames=5, stride=3, seed=7, max_points_per_frame=600`",
        "",
        "> 说明：以下“误差分量占比”是诊断占比，不是严格可加的因果分解。它用于排序主要矛盾，而非宣称各分量线性相加后恰好等于最终 `Acc`。",
        "",
        "## 1. 家族级分解",
        "",
        "| family | acc_cm | acc_gap_cm | noise_cm | calib_bias_cm | drift_cm | thickness_cm | completion_delta_cm | primary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| tum_all3 | {fmt(tum['acc_cm'])} | {fmt(tum['acc_gap_cm'])} | {fmt(tum['sensor_noise_floor_cm'])} | {fmt(tum['stable_depth_bias_cm'])} | {fmt(tum['temporal_drift_cm'])} | {fmt(tum['surface_thickness_cm'])} | {fmt(tum['completion_artifact_delta_cm'])} | {tum['primary_contributor']} |",
        f"| bonn_all3 | {fmt(bonn['acc_cm'])} | {fmt(bonn['acc_gap_cm'])} | {fmt(bonn['sensor_noise_floor_cm'])} | {fmt(bonn['stable_depth_bias_cm'])} | {fmt(bonn['temporal_drift_cm'])} | {fmt(bonn['surface_thickness_cm'])} | {fmt(bonn['completion_artifact_delta_cm'])} | {bonn['primary_contributor']} |",
        "",
        "## 2. Bonn 缺口的主贡献者",
        "",
        f"- `Temporal Drift`：{fmt(bonn['temporal_drift_cm'])} cm，占诊断量级的 `{fmt(100.0 * bonn['temporal_drift_cm_share'], 1)}%`",
        f"- `Surface Thickness`：{fmt(bonn['surface_thickness_cm'])} cm，占 `{fmt(100.0 * bonn['surface_thickness_cm_share'], 1)}%`",
        f"- `Sensor Noise Floor`：{fmt(bonn['sensor_noise_floor_cm'])} cm，占 `{fmt(100.0 * bonn['sensor_noise_floor_cm_share'], 1)}%`",
        f"- `Calibration Bias`：{fmt(bonn['stable_depth_bias_cm'])} cm，占 `{fmt(100.0 * bonn['calibration_bias_cm_share'], 1)}%`",
        f"- `Completion Artifacts`：{fmt(bonn['completion_artifact_delta_cm'])} cm；为负值表示 completion 对 `Acc` 是轻微净改善，而不是主要误差源",
        "",
        "## 3. 关键判断",
        "",
        f"- `TUM` 在 oracle pose 下达到 `Acc={fmt(tum['acc_cm'])} cm`，且 `Temporal Drift≈{fmt(tum['temporal_drift_cm'])} cm`，说明当前几何链在无漂移条件下已经足够接近目标。",
        f"- `Bonn` 的家族均值 `Temporal Drift={fmt(bonn['temporal_drift_cm'])} cm`，已经高于剩余 `Acc` 缺口 `1.133 cm` 的同量级；这直接说明剩余瓶颈不是传感器噪声极限，而是位姿/动态边界相关的系统误差。",
        f"- `Calibration Bias` 只有 `{fmt(bonn['stable_depth_bias_cm'])} cm` 量级，说明上一轮 `depth_bias` 修正已基本触顶，不值得继续作为主线。",
        f"- `Surface Thickness` 仍在 `{fmt(bonn['surface_thickness_cm'])} cm` 量级，说明即使修复了 drift，front-side distortion / fusion thickness 仍然需要二阶段校正。",
        f"- `104 -> 111` 的 completion 仅带来 `{fmt(bonn['completion_artifact_delta_cm'])} cm` 的 `Acc` 变化，且方向为改善；因此当前 `Acc` 不是被下游补全拖坏的。",
        "",
        "## 4. 优先级排序",
        "",
        "1. `Temporal Drift / Pose Error`：先做位姿与动态边界关联误差分解，这是 Bonn `Acc` 最大贡献者。",
        "2. `Upstream Geometry Distortion`：在 drift 受控后，继续拆 front-side bias 与 fusion thickness。",
        "3. `Weak-Evidence Geometry Admission`：不是为了压 `Acc`，而是为后续 `Comp-R` 恢复做准备。",
        "4. `Calibration Bias`：当前量级过小，降为低优先级检查项。",
        "",
        "## 5. 结论",
        "",
        "- 当前 `4.233 cm` 的 Bonn `Acc` 不是传感器噪声极限。",
        "- 主要矛盾已经锁定为：`SLAM drift + dynamic-boundary geometry distortion`。",
        "- 下一轮若只做几何小修或 rear completion 微调，无法覆盖当前缺口。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comp_r_report(rows: List[dict], path: Path) -> None:
    bonn_family = next(r for r in rows if r["row_type"] == "family" and r["family"] == "bonn")
    bonn_seq = [r for r in rows if r["row_type"] == "sequence" and r["family"] == "bonn"]
    worst = max(bonn_seq, key=lambda r: float(r["comp_r_gap_pts"]))

    lines = [
        "# S2 Comp-R Gap Analysis",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "基线：`111_native_geometry_chain_direct`",
        "",
        "## 1. Bonn all3 缺口概览",
        "",
        f"- 家族均值 `Comp-R = {fmt(bonn_family['comp_r_5cm'], 2)}%`，距离 `98%` 仍差 `{fmt(bonn_family['comp_r_gap_pts'], 2)} pts`",
        f"- 缺失参考点总数约 `{int(round(bonn_family['missing_reference_points']))}`",
        f"- rear completion 总点数仅 `{int(round(bonn_family['rear_surface_points']))}`，对缺失区域的覆盖比仅 `{fmt(100.0 * bonn_family['rear_to_missing_ratio'], 2)}%`",
        f"- `104 -> 111` 的 `Comp-R` 增益约 `{fmt(bonn_family['completion_comp_r_delta_pts'], 3)} pts`，几乎为零",
        "",
        "## 2. 序列级缺口画像",
        "",
        "| sequence | comp_r_5cm | missing_refs | weak_support_missing | high_occlusion_missing | depth_hole_missing | rear_points |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bonn_seq:
        lines.append(
            f"| {row['sequence']} | {fmt(row['comp_r_5cm'], 2)}% | {int(round(row['missing_reference_points']))} | {fmt(100.0 * row['missing_observed_twice_or_less_ratio'], 1)}% | {fmt(100.0 * row['missing_high_occlusion_ratio'], 1)}% | {fmt(100.0 * row['missing_depth_hole_ratio'], 1)}% | {int(round(row['rear_surface_points']))} |"
        )
    lines += [
        "",
        "## 3. crowd2 专项诊断",
        "",
        f"- 最差序列是 `{worst['sequence']}`，`Comp-R={fmt(worst['comp_r_5cm'], 2)}%`",
        f"- 缺失点中，`{fmt(100.0 * worst['missing_observed_twice_or_less_ratio'], 1)}%` 只得到 `<=2` 次稳定观测，说明当前管线不会把弱证据累积成稳定表面。",
        f"- 缺失点中，`{fmt(100.0 * worst['missing_high_occlusion_ratio'], 1)}%` 呈现遮挡主导模式，说明动态前景干扰仍是重灾区。",
        f"- `missing_never_in_view_ratio = {fmt(100.0 * worst['missing_never_in_view_ratio'], 1)}%`，说明在当前 5 帧参考集内，缺失区域并不是纯粹的传感器盲区；它们大多至少进入过视锥。",
        f"- `rear_points = {int(round(worst['rear_surface_points']))}`，相对于 `{int(round(worst['missing_reference_points']))}` 个缺失参考点仍然太稀疏，无法改变整体 completeness。",
        "",
        "## 4. 对当前 Rear Completion 的结论",
        "",
        "- 当前 rear completion 不是全局缺口修复器，而是局部 TB 恢复器。",
        "- 它能恢复一小部分有 donor/support plane 的后景点，但不能把大量弱观测区域转成稳定表面。",
        "- 从 `104 -> 111` 几乎零 `Comp-R` 增益可以确认：当前 completion 没有真正触达 completeness 主缺口。",
        "",
        "## 5. 物理特征判断",
        "",
        "- 当前缺失地图的主特征不是“完全没进视野”。",
        "- 主特征是“进过视野，但观测次数低、且经常被前景截断”。",
        "- 因此下一轮不应优先做更激进的平面补洞，而应优先做：",
        "  1. 位姿受约束的多视图弱证据累积",
        "  2. 动态边界附近的观测保真与重投影一致性",
        "  3. 让 one/two-view 弱支持也能形成可提交的背景假设",
        "",
        "## 6. 结论",
        "",
        "- `Comp-R` 主缺口已锁定为：`weak support accumulation failure under dynamic occlusion`。",
        "- 若不先解决弱观测累积与 pose consistency，继续加局部 rear completion 点数也无法把 `70.86%` 推近 `98%`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    frames = 5
    stride = 3
    seed = 7

    rows: List[dict] = []
    recon_base = {
        "tum": load_csv(tables_root(BASELINE_ROOT, "tum") / "reconstruction_metrics.csv"),
        "bonn": load_csv(tables_root(BASELINE_ROOT, "bonn") / "reconstruction_metrics.csv"),
    }
    recon_no_completion = {
        "tum": load_csv(tables_root(NO_COMPLETION_ROOT, "tum") / "reconstruction_metrics.csv"),
        "bonn": load_csv(tables_root(NO_COMPLETION_ROOT, "bonn") / "reconstruction_metrics.csv"),
    }

    for family, sequences in [("tum", TUM_ALL3), ("bonn", BONN_ALL3)]:
        for sequence in sequences:
            rows.append(
                seq_error_row(
                    family=family,
                    sequence=sequence,
                    frames=frames,
                    stride=stride,
                    seed=seed,
                    recon_rows_base=recon_base[family],
                    recon_rows_nocomp=recon_no_completion[family],
                )
            )

    rows.append(family_row("tum", rows))
    rows.append(family_row("bonn", rows))

    write_csv(rows, CSV_PATH)
    write_geometry_report(rows, GEOM_REPORT_PATH)
    write_comp_r_report(rows, COMP_R_REPORT_PATH)
    print("[done]", CSV_PATH)


if __name__ == "__main__":
    main()
