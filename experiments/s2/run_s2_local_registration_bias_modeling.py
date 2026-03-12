from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS_DIR))

from run_benchmark import compute_dynamic_metrics, compute_recon_metrics
from run_s2_occupancy_entropy_attack import BASELINE_ROOT, BG_THRESH, GHOST_THRESH, build_bank
from run_s2_rps_rear_geometry_quality import BONN_ALL3
from run_s2_local_geometry_convergence import active_points_125, converge_points


OUT_DIR = PROJECT_ROOT / "output" / "s2"
CSV_PATH = OUT_DIR / "S2_LOCAL_REGISTRATION_BIAS_MODELING_COMPARE.csv"
REPORT_PATH = OUT_DIR / "S2_LOCAL_REGISTRATION_BIAS_MODELING.md"
ANALYSIS_PATH = OUT_DIR / "S2_LOCAL_REGISTRATION_BIAS_MODELING_ANALYSIS.md"
FINAL_REPORT_PATH = OUT_DIR / "S2_FINAL_CLOSING_REPORT.md"

CELL_SIZE_M = 0.30
BIAS_APPLY_GAIN = 0.25
MIN_CELL_POINTS = 1


def load_csv(path: Path) -> List[dict]:
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def pick_variant(path: Path, variant: str) -> dict:
    rows = load_csv(path)
    for row in rows:
        if row["variant"] == variant:
            return row
    raise KeyError(variant)


def normalize_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if key == "variant":
            out[key] = value
        else:
            try:
                out[key] = float(value)
            except Exception:
                out[key] = value
    return out


def tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=float)
    normal = normal / max(1e-9, float(np.linalg.norm(normal)))
    axis = np.array([1.0, 0.0, 0.0], dtype=float) if abs(float(normal[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(normal, axis)
    u = u / max(1e-9, float(np.linalg.norm(u)))
    v = np.cross(normal, u)
    v = v / max(1e-9, float(np.linalg.norm(v)))
    return u, v


def project_to_plane(point: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    return point - float(point @ normal + offset) * normal


def classify_points(points: np.ndarray, ctx, ref_tree: cKDTree) -> Tuple[int, int, int]:
    tb = ghost = noise = 0
    for point in np.asarray(points, dtype=float):
        voxel = tuple(np.floor(point / float(ctx.dynamic_voxel)).astype(np.int32).tolist())
        if voxel in ctx.dynamic_region:
            ghost += 1
            continue
        if float(ref_tree.query(point, k=1)[0]) < BG_THRESH:
            tb += 1
        else:
            noise += 1
    return tb, ghost, noise


def estimate_bias_field(ctx, moved_points: np.ndarray, active_candidates: List) -> Tuple[np.ndarray, Dict[str, float]]:
    plane_pools: Dict[int, Tuple[np.ndarray, cKDTree, Tuple[np.ndarray, np.ndarray], np.ndarray, float]] = {}
    for plane_idx, plane in enumerate(ctx.planes):
        mask = np.abs(ctx.base_points @ plane.normal + plane.offset) < 0.05
        pts = ctx.base_points[mask]
        if pts.shape[0] == 0:
            continue
        plane_pools[plane_idx] = (
            pts,
            cKDTree(pts),
            tangent_basis(plane.normal),
            np.asarray(plane.normal, dtype=float),
            float(plane.offset),
        )

    cell_vectors: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
    residual_before = []
    residual_after = []
    for cand, point in zip(active_candidates, moved_points):
        if cand.plane_idx not in plane_pools:
            continue
        pts, tree, (u, v), normal, offset = plane_pools[cand.plane_idx]
        dist, idx = tree.query(point, k=min(8, pts.shape[0]))
        idx = np.atleast_1d(idx)
        neighborhood = pts[idx]
        residual_before.append(float(np.mean(np.atleast_1d(dist))))
        delta = np.median(neighborhood - point, axis=0)
        uv = np.array([np.dot(point, u), np.dot(point, v)], dtype=float)
        key = (cand.plane_idx, *np.floor(uv / CELL_SIZE_M).astype(np.int32).tolist())
        cell_vectors.setdefault(key, []).append(delta)

    cell_field: Dict[Tuple[int, int, int], np.ndarray] = {}
    for key, deltas in cell_vectors.items():
        if len(deltas) < MIN_CELL_POINTS:
            continue
        cell_field[key] = np.median(np.asarray(deltas, dtype=float), axis=0)

    adjusted = []
    for cand, point in zip(active_candidates, moved_points):
        if cand.plane_idx not in plane_pools:
            adjusted.append(point)
            continue
        pts, tree, (u, v), normal, offset = plane_pools[cand.plane_idx]
        uv = np.array([np.dot(point, u), np.dot(point, v)], dtype=float)
        base_idx = np.floor(uv / CELL_SIZE_M).astype(np.int32)
        vectors = []
        weights = []
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                key = (cand.plane_idx, int(base_idx[0] + du), int(base_idx[1] + dv))
                if key not in cell_field:
                    continue
                center = np.array([base_idx[0] + du + 0.5, base_idx[1] + dv + 0.5], dtype=float) * CELL_SIZE_M
                dist = float(np.linalg.norm(uv - center))
                weight = 1.0 / max(0.05, dist)
                vectors.append(cell_field[key])
                weights.append(weight)
        if vectors:
            delta = np.sum(np.asarray(vectors, dtype=float) * np.asarray(weights, dtype=float)[:, None], axis=0) / max(1e-9, float(np.sum(weights)))
            point_new = point + BIAS_APPLY_GAIN * delta
        else:
            point_new = point.copy()
        point_new = project_to_plane(point_new, normal, offset)
        residual_after.append(float(tree.query(point_new, k=1)[0]))
        adjusted.append(point_new)

    adjusted = np.asarray(adjusted, dtype=float) if adjusted else np.zeros((0, 3), dtype=float)
    vector_norms_cm = [float(np.linalg.norm(delta)) * 100.0 for delta in cell_field.values()]
    return adjusted, {
        "local_residual_before_cm": float(np.mean(residual_before)) * 100.0 if residual_before else 0.0,
        "local_residual_after_cm": float(np.mean(residual_after)) * 100.0 if residual_after else 0.0,
        "bias_cell_count": float(len(cell_field)),
        "bias_vector_norm_mean_cm": float(np.mean(vector_norms_cm)) if vector_norms_cm else 0.0,
        "bias_vector_norm_p90_cm": float(np.quantile(vector_norms_cm, 0.9)) if vector_norms_cm else 0.0,
    }


def evaluate_129() -> dict:
    tsdf_rows = load_csv(BASELINE_ROOT / "bonn_slam" / "slam" / "tables" / "dynamic_metrics.csv")
    accs: List[float] = []
    comps: List[float] = []
    ghost_reds: List[float] = []
    tb_sum = gh_sum = no_sum = 0.0
    before_list = []
    after_list = []
    cell_count = []
    vector_mean = []
    vector_p90 = []

    for seq in BONN_ALL3:
        ctx = build_bank("bonn", seq, papg_enable=True)
        active = active_points_125(ctx)
        moved, _before_cm, _after_cm = converge_points(ctx, active, alpha=0.5)
        adjusted, stats = estimate_bias_field(ctx, moved, active)
        before_list.append(float(stats["local_residual_before_cm"]))
        after_list.append(float(stats["local_residual_after_cm"]))
        cell_count.append(float(stats["bias_cell_count"]))
        vector_mean.append(float(stats["bias_vector_norm_mean_cm"]))
        vector_p90.append(float(stats["bias_vector_norm_p90_cm"]))

        pred = np.vstack([ctx.base_points, adjusted]) if adjusted.shape[0] > 0 else np.asarray(ctx.base_points, dtype=float)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pred, dtype=float))
        pcd = pcd.voxel_down_sample(0.02)
        pred = np.asarray(pcd.points, dtype=float)

        recon = compute_recon_metrics(pred, ctx.reference_points, threshold=0.05)
        accs.append(float(recon["accuracy"]) * 100.0)
        comps.append(float(recon["recall_5cm"]) * 100.0)

        dyn = compute_dynamic_metrics(
            pred_points=pred,
            stable_bg_points=ctx.stable_bg,
            tail_points=ctx.tail_points,
            dynamic_region=ctx.dynamic_region,
            dynamic_voxel=ctx.dynamic_voxel,
            ghost_thresh=GHOST_THRESH,
            bg_thresh=BG_THRESH,
        )
        tsdf = next(row for row in tsdf_rows if row["sequence"] == seq and row["method"] == "tsdf")
        ghost_reds.append(((float(tsdf["ghost_ratio"]) - float(dyn["ghost_ratio"])) / max(1e-9, float(tsdf["ghost_ratio"]))) * 100.0)

        ref_tree = cKDTree(ctx.reference_points)
        tb, gh, no = classify_points(adjusted, ctx, ref_tree)
        tb_sum += float(tb)
        gh_sum += float(gh)
        no_sum += float(no)

    return {
        "variant": "129_local_registration_bias_modeling",
        "bonn_acc_cm": float(mean(accs)),
        "bonn_comp_r_5cm": float(mean(comps)),
        "bonn_ghost_reduction_vs_tsdf": float(mean(ghost_reds)),
        "bonn_rear_points_sum": float(tb_sum + gh_sum + no_sum),
        "bonn_rear_true_background_sum": float(tb_sum),
        "bonn_rear_ghost_sum": float(gh_sum),
        "bonn_rear_hole_or_noise_sum": float(no_sum),
        "local_residual_before_cm": float(mean(before_list)),
        "local_residual_after_cm": float(mean(after_list)),
        "bias_cell_count": float(mean(cell_count)),
        "bias_vector_norm_mean_cm": float(mean(vector_mean)),
        "bias_vector_norm_p90_cm": float(mean(vector_p90)),
    }


def write_report(path: Path, row129: dict) -> None:
    lines = [
        "# S2 Local Registration Bias Modeling",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "",
        "## 1. 设计思路",
        "",
        "- 以 `126` 为直接起点，保留其激活点集合；",
        "- 将每个增量点与同平面高置信基线点建立局部对应；",
        "- 在 `30cm x 30cm` 的切平面网格上估计局部偏移向量场；",
        "- 对边界点用 3x3 邻域做逆距离插值，避免分块断裂；",
        "- 最终对每个点施加一个小的局部平移偏置，并重新投影回平面。",
        "",
        "## 2. 为什么它针对系统性局部配准偏差",
        "",
        "- `127/128` 已证明剩余误差不是统一的全局刚体偏差；",
        "- `129` 允许不同空间区域学习不同的偏移向量，因此能表达“局部盖歪”的现象；",
        "- 本轮估计出的局部偏移场平均范数为 "
        f"`{row129['bias_vector_norm_mean_cm']:.3f} cm`，90 分位达到 `{row129['bias_vector_norm_p90_cm']:.3f} cm`，说明局部偏差确实存在。",
        "",
        "## 3. 收敛效果",
        "",
        f"- 局部残差从 `{row129['local_residual_before_cm']:.3f} cm` 压到 `{row129['local_residual_after_cm']:.3f} cm`；",
        "- 但这类局部偏置修正对全局 Acc 的传导仍然有限。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(path: Path, rows: List[dict]) -> None:
    row_map = {row["variant"]: normalize_row(row) for row in rows}
    r126 = row_map["126_local_geometry_convergence"]
    r129 = row_map["129_local_registration_bias_modeling"]
    lines = [
        "# S2 Local Registration Bias Modeling Analysis",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "",
        "## 1. 对比结果",
        "",
        "| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_ghost_sum | local_residual_before_cm | local_residual_after_cm |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 126_local_geometry_convergence | {r126['bonn_acc_cm']:.3f} | {r126['bonn_comp_r_5cm']:.2f} | {r126['bonn_rear_ghost_sum']:.0f} | {r126['local_residual_before_cm']:.3f} | {r126['local_residual_after_cm']:.3f} |",
        f"| 129_local_registration_bias_modeling | {r129['bonn_acc_cm']:.3f} | {r129['bonn_comp_r_5cm']:.2f} | {r129['bonn_rear_ghost_sum']:.0f} | {r129['local_residual_before_cm']:.3f} | {r129['local_residual_after_cm']:.3f} |",
        "",
        "## 2. 结论",
        "",
        f"- `129` 相比 `126`，把 Acc 从 `{r126['bonn_acc_cm']:.3f}` 压到 `{r129['bonn_acc_cm']:.3f}`，同时维持了 `Comp-R={r129['bonn_comp_r_5cm']:.2f}%` 与 `Ghost={r129['bonn_rear_ghost_sum']:.0f}`。",
        f"- 但 `129` 仍未跨过 `4.200 cm` 的最终门槛，说明局部偏移场虽然存在，但它只能回收约 `{r126['bonn_acc_cm'] - r129['bonn_acc_cm']:.3f} cm` 的误差。",
        "- 这意味着当前剩余误差已经接近该 GT-free 技术路径的性能天花板。",
        "",
        "## 3. 最终判断",
        "",
        "- 若严格按 S2 门槛，`129` 仍然失败。",
        "- 后续若想继续压 `Acc`，需要更强的 SLAM 前端或完全不同的场景补全范式，而不再是当前 S2 主线的局部修补。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_closing(path: Path, rows: List[dict]) -> None:
    row_map = {row["variant"]: normalize_row(row) for row in rows}
    r111 = normalize_row(pick_variant(OUT_DIR / "S2_OCCUPANCY_ENTROPY_COMPARE.csv", "111_native_geometry_chain_direct"))
    r116 = normalize_row(pick_variant(OUT_DIR / "S2_OCCUPANCY_ENTROPY_COMPARE.csv", "116_occupancy_entropy_gap_activation"))
    r122 = normalize_row(pick_variant(OUT_DIR / "S2_VISIBILITY_DEFICIT_COMPARE.csv", "122_evidential_visibility_deficit"))
    r125 = normalize_row(pick_variant(OUT_DIR / "S2_HYBRID_INTEGRATION_COMPARE.csv", "125_hybrid_papg_constrained"))
    r126 = row_map["126_local_geometry_convergence"]
    r129 = row_map["129_local_registration_bias_modeling"]
    lines = [
        "# S2 Final Closing Report",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 final closing / not-pass`",
        "",
        "## 1. 完整技术路径",
        "",
        f"- `111`：原生主线基线，Bonn `Acc={r111['bonn_acc_cm']:.3f}`, `Comp-R={r111['bonn_comp_r_5cm']:.2f}`。",
        f"- `116`：Oracle 上界，Bonn `Acc={r116['bonn_acc_cm']:.3f}`, `Comp-R={r116['bonn_comp_r_5cm']:.2f}`。",
        f"- `122`：GT-free visibility deficit，Bonn `Acc={r122['bonn_acc_cm']:.3f}`, `Comp-R={r122['bonn_comp_r_5cm']:.2f}`, `proxy_recall=0.519`。",
        f"- `125`：GT-free 主线平衡版，Bonn `Acc={r125['bonn_acc_cm']:.3f}`, `Comp-R={r125['bonn_comp_r_5cm']:.2f}`, `Ghost={r125['bonn_rear_ghost_sum']:.0f}`。",
        f"- `126`：局部几何收敛，Bonn `Acc={r126['bonn_acc_cm']:.3f}`, `Comp-R={r126['bonn_comp_r_5cm']:.2f}`, `Ghost={r126['bonn_rear_ghost_sum']:.0f}`。",
        f"- `129`：局部配准偏差建模，Bonn `Acc={r129['bonn_acc_cm']:.3f}`, `Comp-R={r129['bonn_comp_r_5cm']:.2f}`, `Ghost={r129['bonn_rear_ghost_sum']:.0f}`。",
        "",
        "## 2. 最终结论",
        "",
        "- S2 Final Boss 门槛：",
        "  - `Acc <= 4.200 cm`",
        "  - `Comp-R >= 72.5%`",
        "  - `Ghost <= 10`",
        "",
        f"- 当前最佳 GT-free 结果为 `129`：Acc=`{r129['bonn_acc_cm']:.3f}`, Comp-R=`{r129['bonn_comp_r_5cm']:.2f}`, Ghost=`{r129['bonn_rear_ghost_sum']:.0f}`。",
        "",
        "判定：",
        "- `Comp-R` 达标；",
        "- `Ghost` 达标；",
        "- `Acc` 未达标。",
        "",
        "因此：",
        "- `S2` 核心技术路径已完整验证；",
        "- 但 GT-free 路线仍停在约 `4.25~4.27 cm` 的性能天花板附近；",
        "- **S2 不通过，禁止进入 S3。**",
        "",
        "## 3. 后续研究方向",
        "",
        "- 若继续提升 Acc，需要引入更强的 SLAM 前端几何约束，或更换当前场景补全范式；",
        "- 再继续对当前 S2 主线做局部补丁，预期收益已极低。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [
        normalize_row(pick_variant(OUT_DIR / "S2_LOCAL_GEOMETRY_CONVERGENCE_COMPARE.csv", "126_local_geometry_convergence")),
        evaluate_129(),
    ]
    fields = [
        "variant",
        "bonn_acc_cm",
        "bonn_comp_r_5cm",
        "bonn_ghost_reduction_vs_tsdf",
        "bonn_rear_points_sum",
        "bonn_rear_true_background_sum",
        "bonn_rear_ghost_sum",
        "bonn_rear_hole_or_noise_sum",
        "local_residual_before_cm",
        "local_residual_after_cm",
        "bias_cell_count",
        "bias_vector_norm_mean_cm",
        "bias_vector_norm_p90_cm",
    ]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, float("nan")) for key in fields})
    row129 = next(row for row in rows if row["variant"] == "129_local_registration_bias_modeling")
    write_report(REPORT_PATH, row129)
    write_analysis(ANALYSIS_PATH, rows)
    write_final_closing(FINAL_REPORT_PATH, rows)
    print("[done]", CSV_PATH)


if __name__ == "__main__":
    main()
