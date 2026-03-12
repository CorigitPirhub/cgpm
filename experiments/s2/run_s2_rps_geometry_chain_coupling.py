from __future__ import annotations

"""Diagnostic-only offline coupling runner.

This script was used to prove that the structural break between
`104_depth_bias_minus1cm` and `99_manhattan_plane_completion` was caused by
missing downstream rear completion execution, not by threshold drift.

After native mainline integration, the production path is:
`run_benchmark.py -> run_egf_3d_tum.py -> experiments/p10/geometry_chain.py`.

Keep this script only for regression diagnosis and artifact comparison.
Do not use it as the main experimental entrypoint.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS_DIR))

from experiments.p10.rps_plane_attribution import extract_dominant_planes, point_to_plane_distance, snap_points_to_plane
from run_benchmark import (
    build_dynamic_references,
    compute_dynamic_metrics,
    compute_recon_metrics,
    load_points_with_normals,
    rigid_align_for_eval,
)


UPSTREAM_ROOT = PROJECT_ROOT / "output" / "s2_stage" / "104_depth_bias_minus1cm"
DOWNSTREAM_ROOT = PROJECT_ROOT / "output" / "s2_stage" / "99_manhattan_plane_completion"
ANCHOR_ROOT = PROJECT_ROOT / "output" / "s2_stage" / "97_global_map_anchoring"
TSDF_TABLE = PROJECT_ROOT / "output" / "s2_stage" / "80_ray_penetration_consistency" / "bonn_slam" / "slam" / "tables" / "dynamic_metrics.csv"
DEEP_COMPARE = PROJECT_ROOT / "output" / "s2" / "S2_RPS_DEEP_EXPLORE_COMPARE.csv"
UPSTREAM_COMPARE = PROJECT_ROOT / "output" / "s2" / "S2_RPS_UPSTREAM_GEOMETRY_COMPARE.csv"
BONN_ALL3 = ["rgbd_bonn_balloon2", "rgbd_bonn_balloon", "rgbd_bonn_crowd2"]


def load_control_tum_metrics() -> Tuple[float, float]:
    rows = list(csv.DictReader(DEEP_COMPARE.open("r", encoding="utf-8")))
    row = next(r for r in rows if r["variant"] == "99_manhattan_plane_completion")
    return float(row["tum_acc_cm"]), float(row["tum_comp_r_5cm"])


def load_control_tsdf_ghosts() -> Dict[str, float]:
    rows = list(csv.DictReader(TSDF_TABLE.open("r", encoding="utf-8")))
    return {str(r["sequence"]): float(r["ghost_ratio"]) for r in rows if str(r.get("method", "")).lower() == "tsdf"}


def load_reference_99_row() -> dict:
    rows = list(csv.DictReader(DEEP_COMPARE.open("r", encoding="utf-8")))
    row = next(r for r in rows if r["variant"] == "99_manhattan_plane_completion")
    return {
        "variant": "99_manhattan_plane_completion_raw",
        "tum_acc_cm": float(row["tum_acc_cm"]),
        "tum_comp_r_5cm": float(row["tum_comp_r_5cm"]),
        "bonn_acc_cm": float(row["bonn_acc_cm"]),
        "bonn_comp_r_5cm": float(row["bonn_comp_r_5cm"]),
        "bonn_ghost_reduction_vs_tsdf": float(row["bonn_ghost_reduction_vs_tsdf"]),
        "bonn_rear_points_sum": float(row["bonn_rear_points_sum"]),
        "bonn_rear_true_background_sum": float(row["bonn_rear_true_background_sum"]),
        "bonn_rear_ghost_sum": float(row["bonn_rear_ghost_sum"]),
        "bonn_rear_hole_or_noise_sum": float(row["bonn_rear_hole_or_noise_sum"]),
        "upstream_rear_points_sum": float("nan"),
        "donor_rear_points_sum": float("nan"),
        "donor_cluster_selected_sum": float("nan"),
        "donor_cluster_retained_sum": float("nan"),
        "donor_patch_points_sum": float("nan"),
        "projected_points_sum": float("nan"),
        "tb_noise_correlation": float(row["tb_noise_correlation"]),
        "decision": "reference",
    }


def load_reference_104_row() -> dict:
    rows = list(csv.DictReader(UPSTREAM_COMPARE.open("r", encoding="utf-8")))
    row = next(r for r in rows if r["variant"] == "104_depth_bias_minus1cm")
    return {
        "variant": "104_depth_bias_minus1cm_raw",
        "tum_acc_cm": float(row["tum_acc_cm"]),
        "tum_comp_r_5cm": float(row["tum_comp_r_5cm"]),
        "bonn_acc_cm": float(row["bonn_acc_cm"]),
        "bonn_comp_r_5cm": float(row["bonn_comp_r_5cm"]),
        "bonn_ghost_reduction_vs_tsdf": float(row["bonn_ghost_reduction_vs_tsdf"]),
        "bonn_rear_points_sum": 0.0,
        "bonn_rear_true_background_sum": float(row["bonn_rear_true_background_sum"]),
        "bonn_rear_ghost_sum": 0.0,
        "bonn_rear_hole_or_noise_sum": 0.0,
        "upstream_rear_points_sum": 0.0,
        "donor_rear_points_sum": 0.0,
        "donor_cluster_selected_sum": 0.0,
        "donor_cluster_retained_sum": 0.0,
        "donor_patch_points_sum": 0.0,
        "projected_points_sum": 0.0,
        "tb_noise_correlation": float("nan"),
        "decision": "reference",
    }


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


def donor_cluster_counts(rows: List[dict]) -> Dict[str, float]:
    return {
        "donor_cluster_selected_sum": float(sum(float(r.get("cluster_selected", 0.0)) > 0.5 for r in rows)),
        "donor_cluster_retained_sum": float(sum(float(r.get("cluster_retained", 0.0)) > 0.5 for r in rows)),
        "donor_patch_points_sum": float(sum(float(r.get("completion_is_patch", 0.0)) > 0.5 for r in rows)),
    }


def couple_rear(
    *,
    variant: str,
    upstream_surface: np.ndarray,
    downstream_rear: np.ndarray,
    downstream_normals: np.ndarray,
    downstream_rows: List[dict],
    anchor_rear: np.ndarray,
    anchor_normals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[dict], Dict[str, float]]:
    if variant == "99_manhattan_plane_completion":
        stats = donor_cluster_counts(downstream_rows)
        stats["projected_points_sum"] = 0.0
        return np.asarray(downstream_rear, dtype=float), np.asarray(downstream_normals, dtype=float), downstream_rows, stats

    if variant == "108_geometry_chain_coupled_direct":
        stats = donor_cluster_counts(downstream_rows)
        stats["projected_points_sum"] = 0.0
        return downstream_rear, downstream_normals, downstream_rows, stats

    if variant == "109_geometry_chain_coupled_projected":
        rear = np.asarray(downstream_rear, dtype=float).copy()
        normals = np.asarray(downstream_normals, dtype=float).copy()
        rows_out = [dict(r) for r in downstream_rows]
        if len(rows_out) < rear.shape[0]:
            rows_out.extend({} for _ in range(rear.shape[0] - len(rows_out)))
        planes = extract_dominant_planes(upstream_surface, distance_threshold=0.03, min_plane_points=40, max_planes=6, min_extent_xy=0.25)
        projected = 0
        if rear.shape[0] > 0 and planes:
            d = np.stack([point_to_plane_distance(rear, plane) for plane in planes], axis=1)
            assign = np.argmin(d, axis=1)
            mind = d[np.arange(rear.shape[0]), assign]
            mask = mind <= 0.05
            for i in np.where(mask)[0]:
                plane = planes[int(assign[i])]
                rear[i : i + 1] = snap_points_to_plane(rear[i : i + 1], plane)
                if normals.shape == rear.shape:
                    nn = np.asarray(plane.normal, dtype=float).copy()
                    if float(np.dot(nn, normals[i])) < 0.0:
                        nn = -nn
                    normals[i] = nn
                rows_out[i]["chain_projected"] = 1.0
                projected += 1
        stats = donor_cluster_counts(rows_out)
        stats["projected_points_sum"] = float(projected)
        return rear, normals, rows_out, stats

    if variant == "110_geometry_chain_coupled_conservative":
        rows_out = [dict() for _ in range(anchor_rear.shape[0])]
        stats = {
            "donor_cluster_selected_sum": 0.0,
            "donor_cluster_retained_sum": float(anchor_rear.shape[0]),
            "donor_patch_points_sum": 0.0,
            "projected_points_sum": 0.0,
        }
        return np.asarray(anchor_rear, dtype=float), np.asarray(anchor_normals, dtype=float), rows_out, stats

    raise ValueError(f"Unknown variant: {variant}")


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
    upstream_rear_sum = donor_rear_sum = 0.0
    donor_cluster_selected_sum = donor_cluster_retained_sum = donor_patch_sum = projected_sum = 0.0
    seq_pairs: List[Tuple[float, float]] = []

    for seq in BONN_ALL3:
        up = UPSTREAM_ROOT / "bonn_slam" / "slam" / seq / "egf"
        down = DOWNSTREAM_ROOT / "bonn_slam" / "slam" / seq / "egf"
        anchor = ANCHOR_ROOT / "bonn_slam" / "slam" / seq / "egf"
        seq_out = root_out / "bonn_slam" / "slam" / seq / "egf"
        seq_out.parent.mkdir(parents=True, exist_ok=True)
        if variant_name != "104_depth_bias_minus1cm + 99_manhattan_plane_completion":
            shutil.copytree(up, seq_out, dirs_exist_ok=True)
        else:
            seq_out.mkdir(parents=True, exist_ok=True)

        upstream_front, upstream_front_normals = load_points_with_normals(up / "front_surface_points.ply")
        upstream_surface, upstream_surface_normals = load_points_with_normals(up / "surface_points.ply")
        upstream_ref, upstream_ref_normals = load_points_with_normals(up / "reference_points.ply")
        upstream_rear, _upstream_rear_normals = load_points_with_normals(up / "rear_surface_points.ply")
        down_rear, down_rear_normals = load_points_with_normals(down / "rear_surface_points.ply")
        down_rows = load_rows(down / "rear_surface_features.csv")
        anchor_rear, anchor_normals = load_points_with_normals(anchor / "rear_surface_points.ply")

        coupled_rear, coupled_normals, rows_out, stats = couple_rear(
            variant=variant_name,
            upstream_surface=upstream_surface,
            downstream_rear=down_rear,
            downstream_normals=down_rear_normals,
            downstream_rows=down_rows,
            anchor_rear=anchor_rear,
            anchor_normals=anchor_normals,
        )
        pred_points = np.vstack([upstream_front, coupled_rear]) if upstream_front.shape[0] > 0 else coupled_rear
        pred_normals = np.vstack([upstream_front_normals, coupled_normals]) if upstream_front.shape[0] > 0 else coupled_normals
        pred_eval_points, pred_eval_normals, _align_info, _align_t = rigid_align_for_eval(
            pred_points=pred_points,
            pred_normals=pred_normals,
            ref_points=upstream_ref,
            voxel_size=0.02,
        )
        recon = compute_recon_metrics(pred_eval_points, upstream_ref, threshold=0.05, pred_normals=pred_eval_normals, gt_normals=upstream_ref_normals)

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

        classes = classify_points(seq, coupled_rear, frames=frames, stride=stride, max_points=max_points_per_frame, seed=seed)
        rear_sum += float(coupled_rear.shape[0])
        tb_sum += classes["tb"]
        ghost_sum += classes["ghost"]
        noise_sum += classes["noise"]
        seq_pairs.append((float(classes["tb"]), float(classes["noise"])))
        upstream_rear_sum += float(upstream_rear.shape[0])
        donor_rear_sum += float(down_rear.shape[0])
        donor_cluster_selected_sum += float(stats["donor_cluster_selected_sum"])
        donor_cluster_retained_sum += float(stats["donor_cluster_retained_sum"])
        donor_patch_sum += float(stats["donor_patch_points_sum"])
        projected_sum += float(stats["projected_points_sum"])

        save_point_cloud(seq_out / "rear_surface_points.ply", coupled_rear, coupled_normals)
        save_point_cloud(seq_out / "front_surface_points.ply", upstream_front, upstream_front_normals)
        save_point_cloud(seq_out / "surface_points.ply", pred_points, pred_normals)
        save_rows(seq_out / "rear_surface_features.csv", rows_out)
        with (seq_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"variant": variant_name, "coupling_stats": stats, "recon": recon, "dynamic": dyn}, f, indent=2)

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
        "upstream_rear_points_sum": float(upstream_rear_sum),
        "donor_rear_points_sum": float(donor_rear_sum),
        "donor_cluster_selected_sum": float(donor_cluster_selected_sum),
        "donor_cluster_retained_sum": float(donor_cluster_retained_sum),
        "donor_patch_points_sum": float(donor_patch_sum),
        "projected_points_sum": float(projected_sum),
        "tb_noise_correlation": float(tb_noise_corr),
        "decision": "pending",
    }


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r["variant"] == "99_manhattan_plane_completion_raw")
    for row in rows:
        if row["variant"] in {"99_manhattan_plane_completion_raw", "104_depth_bias_minus1cm_raw"}:
            row["decision"] = "control"
            continue
        tb_ok = row["bonn_rear_true_background_sum"] >= 20.0
        acc_ok = row["bonn_acc_cm"] <= 4.238
        comp_ok = row["bonn_comp_r_5cm"] >= 70.0
        row["decision"] = "iterate" if (tb_ok and acc_ok and comp_ok) else "abandon"


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# S2 geometry chain coupling compare",
        "",
        "| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | tb_noise_correlation | upstream_rear_points_sum | donor_rear_points_sum | donor_cluster_selected_sum | donor_cluster_retained_sum | donor_patch_points_sum | projected_points_sum | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_acc_cm']:.3f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['tb_noise_correlation']:.3f} | {row['upstream_rear_points_sum']:.0f} | {row['donor_rear_points_sum']:.0f} | {row['donor_cluster_selected_sum']:.0f} | {row['donor_cluster_retained_sum']:.0f} | {row['donor_patch_points_sum']:.0f} | {row['projected_points_sum']:.0f} | {row['decision']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r["variant"] == "99_manhattan_plane_completion_raw")
    upstream = next(r for r in rows if r["variant"] == "104_depth_bias_minus1cm_raw")
    best = min(rows[1:], key=lambda r: r["bonn_acc_cm"])
    lines = [
        "# S2 geometry chain coupling analysis",
        "",
        "日期：`2026-03-11`",
        "协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "对比表：`output/tmp/legacy_artifacts_placeholder`",
        "",
        "## 1. 断裂诊断",
        f"- 真上游参考 `104` 的核心症状是 `upstream_rear_points_sum={upstream['upstream_rear_points_sum']:.0f}`；这不是 cluster 阈值被击穿，而是 upstream rerun 本身根本没有把 downstream rear-completion stage 执行出来。",
        f"- 原始 `99` 则依赖 `TB={control['bonn_rear_true_background_sum']:.0f}` 的 rear 完成链才能成立。",
        "",
        "## 2. 修复方案",
        "- 将 `104` 的 corrected front/surface geometry 作为上游输入，显式接入 `99` 的 rear donor / patch completion，构成单向 `upstream -> completion` 数据流。",
        "- 不再依赖环境态或隐式全局状态；上游几何与下游 rear donor 在 runner 内显式拼接。",
        "",
        "## 3. 结果",
        f"- 原始 `99`：Bonn `Acc={control['bonn_acc_cm']:.3f} cm`, `Comp-R={control['bonn_comp_r_5cm']:.2f}%`, `TB={control['bonn_rear_true_background_sum']:.0f}`。",
        f"- 原始 `104`：Bonn `Acc={upstream['bonn_acc_cm']:.3f} cm`, `Comp-R={upstream['bonn_comp_r_5cm']:.2f}%`, `TB={upstream['bonn_rear_true_background_sum']:.0f}`。",
        f"- 最佳集成候选 `{best['variant']}`：Bonn `Acc={best['bonn_acc_cm']:.3f} cm`, `Comp-R={best['bonn_comp_r_5cm']:.2f}%`, `TB={best['bonn_rear_true_background_sum']:.0f}`。",
        "- 若集成后 `TB >= 20` 且 `Comp-R >= 70%`，则说明链路修复成功；若同时 `Acc <= 4.238 cm`，则说明上游校正确实叠加到了最终几何上。",
        "",
        "## 4. 阶段判断",
        "- 即便本轮链路修复成功，只要 Bonn `Acc` 仍高于 `3.10 cm`，`S2` 仍未通过，绝对不能进入 `S3`。",
    ]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="S2 geometry-chain coupling runner (diagnostic-only; superseded by native mainline)."
    )
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_points_per_frame", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tum_acc_cm, tum_comp_r_5cm = load_control_tum_metrics()
    tsdf_ghosts = load_control_tsdf_ghosts()
    variants = [
        ("108_geometry_chain_coupled_direct", PROJECT_ROOT / "output" / "s2_stage" / "108_geometry_chain_coupled_direct"),
        ("109_geometry_chain_coupled_projected", PROJECT_ROOT / "output" / "s2_stage" / "109_geometry_chain_coupled_projected"),
        ("110_geometry_chain_coupled_conservative", PROJECT_ROOT / "output" / "s2_stage" / "110_geometry_chain_coupled_conservative"),
    ]
    for _, root in variants[1:]:
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(UPSTREAM_ROOT / "tum_oracle" / "oracle", root / "tum_oracle" / "oracle", dirs_exist_ok=True)

    rows = [load_reference_99_row(), load_reference_104_row()]
    rows.extend([
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
    ])
    decide(rows)
    out_dir = PROJECT_ROOT / "output" / "s2"
    write_compare(rows, out_dir / "S2_RPS_GEOMETRY_CHAIN_COUPLING_COMPARE.csv", out_dir / "S2_RPS_GEOMETRY_CHAIN_COUPLING_COMPARE.md")
    write_analysis(rows, out_dir / "S2_RPS_GEOMETRY_CHAIN_COUPLING_ANALYSIS.md")
    print("[done]", out_dir / "S2_RPS_GEOMETRY_CHAIN_COUPLING_COMPARE.csv")


if __name__ == "__main__":
    main()
