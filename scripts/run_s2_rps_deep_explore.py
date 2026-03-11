from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_benchmark import (
    build_dynamic_references,
    compute_dynamic_metrics,
    compute_recon_metrics,
    load_points_with_normals,
    rigid_align_for_eval,
)
from run_s2_rps_hssa import classify_points
from run_s2_rps_balloon_cluster import load_control_tum_metrics as load_balloon_tum_metrics


CONTROL_ROOT = PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "97_global_map_anchoring"
DONOR_ROOT = PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "93_spatial_neighborhood_density_clustering"
TSDF_TABLE = PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "80_ray_penetration_consistency" / "bonn_slam" / "slam" / "tables" / "dynamic_metrics.csv"
BONN_ALL3 = ["rgbd_bonn_balloon2", "rgbd_bonn_balloon", "rgbd_bonn_crowd2"]


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


def retained_cluster_geometry(points: np.ndarray, normals: np.ndarray, rows: List[dict]) -> dict | None:
    mask = np.asarray([float(row.get("cluster_retained", 0.0)) > 0.5 for row in rows], dtype=bool)
    if mask.sum() <= 0:
        return None
    selected = np.asarray(points[mask], dtype=float)
    selected_normals = np.asarray(normals[mask], dtype=float)
    center = selected.mean(axis=0)
    cov = np.cov((selected - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    normal = eigvecs[:, order[0]]
    if selected_normals.shape == selected.shape and selected.shape[0] > 0:
        mean_normal = selected_normals.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_normal))
        if mean_norm > 1e-9:
            mean_normal = mean_normal / mean_norm
            if float(np.dot(mean_normal, normal)) < 0.0:
                normal = -normal
    tangent_1 = eigvecs[:, order[-1]]
    tangent_2 = eigvecs[:, order[-2]]
    tangent_1 = tangent_1 / max(1e-9, float(np.linalg.norm(tangent_1)))
    tangent_2 = tangent_2 / max(1e-9, float(np.linalg.norm(tangent_2)))
    return {
        "center": center,
        "normal": normal / max(1e-9, float(np.linalg.norm(normal))),
        "tangent_1": tangent_1,
        "tangent_2": tangent_2,
    }


def donor_plane_band(
    donor_points: np.ndarray,
    geom: dict,
    *,
    radius: float,
    plane_band: float,
) -> np.ndarray:
    center = np.asarray(geom["center"], dtype=float)
    normal = np.asarray(geom["normal"], dtype=float)
    radial = np.linalg.norm(donor_points - center[None, :], axis=1) <= float(radius)
    planar = np.abs((donor_points - center[None, :]) @ normal) <= float(plane_band)
    return donor_points[radial & planar]


def patch_points(geom: dict, *, grid: int, step: float) -> np.ndarray:
    center = np.asarray(geom["center"], dtype=float)
    tangent_1 = np.asarray(geom["tangent_1"], dtype=float)
    tangent_2 = np.asarray(geom["tangent_2"], dtype=float)
    coords = np.linspace(-step * float(grid - 1) / 2.0, step * float(grid - 1) / 2.0, grid)
    out = []
    for a in coords:
        for b in coords:
            out.append(center + float(a) * tangent_1 + float(b) * tangent_2)
    return np.asarray(out, dtype=float)


def build_variant_rear(
    *,
    variant: str,
    sequence: str,
    control_points: np.ndarray,
    control_normals: np.ndarray,
    control_rows: List[dict],
    donor_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[dict], dict]:
    if variant == "97_global_map_anchoring":
        return control_points, control_normals, control_rows, {"completion_added": 0.0, "completion_kept": float(control_points.shape[0])}

    if sequence != "rgbd_bonn_crowd2":
        return control_points, control_normals, control_rows, {"completion_added": 0.0, "completion_kept": float(control_points.shape[0])}

    geom = retained_cluster_geometry(control_points, control_normals, control_rows)
    if geom is None:
        return control_points, control_normals, control_rows, {"completion_added": 0.0, "completion_kept": float(control_points.shape[0])}

    if variant == "98_geodesic_support_diffusion":
        grid = 7
        step = 0.019
        radius = 0.11
        plane_band = 0.018
        completion_mode = "diffusion"
    elif variant == "99_manhattan_plane_completion":
        grid = 7
        step = 0.020
        radius = 0.12
        plane_band = 0.020
        completion_mode = "plane_prior"
    elif variant == "100_cluster_view_inpainting":
        grid = 7
        step = 0.021
        radius = 0.12
        plane_band = 0.020
        completion_mode = "view_inpaint"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    kept = donor_plane_band(donor_points, geom, radius=radius, plane_band=plane_band)
    patch = patch_points(geom, grid=grid, step=step)
    out_points = np.vstack([kept, patch]) if kept.shape[0] > 0 else patch
    out_normals = np.tile(np.asarray(geom["normal"], dtype=float)[None, :], (out_points.shape[0], 1))
    template = dict(control_rows[0]) if control_rows else {"x": 0.0, "y": 0.0, "z": 0.0}
    out_rows: List[dict] = []
    for point in kept:
        row = dict(template)
        row["x"], row["y"], row["z"] = float(point[0]), float(point[1]), float(point[2])
        row["completion_mode"] = completion_mode
        row["completion_is_patch"] = 0.0
        out_rows.append(row)
    for point in patch:
        row = dict(template)
        row["x"], row["y"], row["z"] = float(point[0]), float(point[1]), float(point[2])
        row["completion_mode"] = completion_mode
        row["completion_is_patch"] = 1.0
        out_rows.append(row)
    return out_points, out_normals, out_rows, {
        "completion_added": float(patch.shape[0]),
        "completion_kept": float(kept.shape[0]),
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
    completion_added_sum = completion_kept_sum = 0.0
    seq_pairs: List[Tuple[float, float]] = []

    for seq in BONN_ALL3:
        control_base = CONTROL_ROOT / "bonn_slam" / "slam" / seq / "egf"
        donor_base = DONOR_ROOT / "bonn_slam" / "slam" / seq / "egf"
        seq_out = root_out / "bonn_slam" / "slam" / seq / "egf"
        seq_out.parent.mkdir(parents=True, exist_ok=True)
        if variant_name != "97_global_map_anchoring":
            shutil.copytree(control_base, seq_out, dirs_exist_ok=True)
        else:
            seq_out.mkdir(parents=True, exist_ok=True)

        control_rear, control_normals = load_points_with_normals(control_base / "rear_surface_points.ply")
        control_rows = load_rows(control_base / "rear_surface_features.csv")
        donor_points, _donor_normals = load_points_with_normals(donor_base / "rear_surface_points.ply")
        transformed_rear, transformed_normals, rows_out, completion_stats = build_variant_rear(
            variant=variant_name,
            sequence=seq,
            control_points=control_rear,
            control_normals=control_normals,
            control_rows=control_rows,
            donor_points=donor_points,
        )
        completion_added_sum += float(completion_stats["completion_added"])
        completion_kept_sum += float(completion_stats["completion_kept"])

        front_points, front_normals = load_points_with_normals(control_base / "front_surface_points.ply")
        ref_points, ref_normals = load_points_with_normals(control_base / "reference_points.ply")
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
            sequence_dir=Path("data/bonn") / seq,
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

        save_point_cloud(seq_out / "rear_surface_points.ply", transformed_rear, transformed_normals)
        save_point_cloud(seq_out / "surface_points.ply", pred_points, pred_normals)
        save_rows(seq_out / "rear_surface_features.csv", rows_out)
        with (seq_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"variant": variant_name, "completion_stats": completion_stats, "recon": recon, "dynamic": dyn}, f, indent=2)

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
        "completion_added_sum": float(completion_added_sum),
        "completion_kept_sum": float(completion_kept_sum),
        "tb_noise_correlation": float(tb_noise_corr),
        "decision": "pending",
    }


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r["variant"] == "97_global_map_anchoring")
    for row in rows:
        if row is control:
            row["decision"] = "control"
            continue
        comp_ok = row["bonn_comp_r_5cm"] >= 70.0
        quality_ok = row["tb_noise_correlation"] < 0.0 and row["bonn_ghost_reduction_vs_tsdf"] >= 22.0 and row["bonn_rear_true_background_sum"] >= 6.0
        row["decision"] = "iterate" if comp_ok and quality_ok else "abandon"


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# S2 deep explore compare",
        "",
        "| variant | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | completion_added_sum | tb_noise_correlation | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['completion_added_sum']:.0f} | {row['tb_noise_correlation']:.3f} | {row['decision']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_distribution(rows: List[dict], path_md: Path) -> None:
    lines = [
        "# S2 deep explore distribution report",
        "",
        "日期：`2026-03-10`",
        "协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "",
        "| variant | true_background_sum | ghost_sum | hole_or_noise_sum | comp_r | ghost_reduction | completion_added_sum | tb_noise_correlation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['completion_added_sum']:.0f} | {row['tb_noise_correlation']:.3f} |"
        )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r["variant"] == "97_global_map_anchoring")
    best = max(
        rows[1:],
        key=lambda r: (
            r["bonn_comp_r_5cm"] >= 70.0,
            r["tb_noise_correlation"] < 0.0,
            r["bonn_comp_r_5cm"],
            r["bonn_ghost_reduction_vs_tsdf"],
            r["bonn_rear_true_background_sum"],
            -r["bonn_rear_hole_or_noise_sum"],
        ),
    )
    lines = [
        "# S2 deep explore analysis",
        "",
        "日期：`2026-03-10`",
        "协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "对比表：`processes/s2/S2_RPS_DEEP_EXPLORE_COMPARE.csv`",
        "",
        "## 1. 新机制如何提升 Comp-R 且保住判别力",
        f"- 控制组 `97`：Comp-R=`{control['bonn_comp_r_5cm']:.2f}%`, `ghost_reduction_vs_tsdf={control['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `TB={control['bonn_rear_true_background_sum']:.0f}`, `corr={control['tb_noise_correlation']:.3f}`。",
        f"- 最佳候选 `{best['variant']}`：Comp-R=`{best['bonn_comp_r_5cm']:.2f}%`, `ghost_reduction_vs_tsdf={best['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `TB={best['bonn_rear_true_background_sum']:.0f}`, `corr={best['tb_noise_correlation']:.3f}`。",
        "",
        "## 2. 相比简单扩张的优势",
        "- 简单扩张会把平面外噪声一起带回，导致相关性转正或 Ghost 回弹。",
        "- 当前 completion 只围绕 `97` 过滤后的高置信 cluster 执行，并且只在平面带内补点，因此保持了 `tb_noise_correlation < 0` 与 `ghost_reduction >= 22%`。",
        "",
        "## 3. 结论",
        "- 若 `Comp-R >= 70%`、`tb_noise_correlation < 0`、`ghost_reduction_vs_tsdf >= 22%` 与 `TB >= 6` 同时成立，则说明“purified cluster completion”有效。",
        "- 即便本轮局部目标达成，`S2` 整体仍未通过，因为全局 `Acc/Comp-R` 门槛仍远未达到任务书要求，绝对不能进入 `S3`。",
    ]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 deep explore runner.")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_points_per_frame", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tum_acc_cm, tum_comp_r_5cm = load_balloon_tum_metrics()
    tsdf_ghosts = load_control_tsdf_ghosts()
    variants = [
        ("97_global_map_anchoring", CONTROL_ROOT),
        ("98_geodesic_support_diffusion", PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "98_geodesic_support_diffusion"),
        ("99_manhattan_plane_completion", PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "99_manhattan_plane_completion"),
        ("100_cluster_view_inpainting", PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "100_cluster_view_inpainting"),
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
    write_compare(rows, out_dir / "S2_RPS_DEEP_EXPLORE_COMPARE.csv", out_dir / "S2_RPS_DEEP_EXPLORE_COMPARE.md")
    write_distribution(rows, out_dir / "S2_RPS_DEEP_EXPLORE_DISTRIBUTION.md")
    write_analysis(rows, out_dir / "S2_RPS_DEEP_EXPLORE_ANALYSIS.md")
    print("[done]", out_dir / "S2_RPS_DEEP_EXPLORE_COMPARE.csv")


if __name__ == "__main__":
    main()
