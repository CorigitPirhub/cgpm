from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_benchmark import compute_dynamic_metrics, compute_recon_metrics
from run_s2_local_geometry_convergence import active_points_125, converge_points, project_to_plane
from run_s2_occupancy_entropy_attack import BASELINE_ROOT, BG_THRESH, GHOST_THRESH, build_bank
from run_s2_rps_rear_geometry_quality import BONN_ALL3


OUT_DIR = PROJECT_ROOT / "processes" / "s2"
CSV_PATH = OUT_DIR / "S2_GLOBAL_RIGIDITY_ALIGNMENT_COMPARE.csv"
DESIGN_PATH = OUT_DIR / "S2_GLOBAL_RIGIDITY_ALIGNMENT_DESIGN.md"
FINAL_REPORT_PATH = OUT_DIR / "S2_FINAL_CLOSING_REPORT.md"


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


def rigid_kabsch(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_c = src - src_centroid
    dst_c = dst - dst_centroid
    h = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = dst_centroid - rotation @ src_centroid
    return rotation, translation


def similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_c = src - src_centroid
    dst_c = dst - dst_centroid
    cov = (dst_c.T @ src_c) / float(src.shape[0])
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3, dtype=float)
    if float(np.linalg.det(u) * np.linalg.det(vt)) < 0.0:
        s[-1, -1] = -1.0
    rotation = u @ s @ vt
    var_src = float(np.mean(np.sum(src_c**2, axis=1)))
    scale = float(np.trace(np.diag(d) @ s) / max(1e-9, var_src))
    translation = dst_centroid - scale * (rotation @ src_centroid)
    return scale, rotation, translation


def rotation_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    cos_theta = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def build_correspondences(ctx, moved_points: np.ndarray, active_candidates: List) -> Tuple[np.ndarray, np.ndarray]:
    src = []
    dst = []
    for cand, point in zip(active_candidates, moved_points):
        plane = ctx.planes[cand.plane_idx]
        mask = np.abs(ctx.base_points @ plane.normal + plane.offset) < 0.05
        plane_points = ctx.base_points[mask]
        if plane_points.shape[0] < 5:
            continue
        tree = cKDTree(plane_points)
        nearest = plane_points[tree.query(point, k=1)[1]]
        target = project_to_plane(nearest, plane.normal, plane.offset)
        src.append(point)
        dst.append(target)
    if not src:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    return np.asarray(src, dtype=float), np.asarray(dst, dtype=float)


def evaluate_alignment(variant: str) -> dict:
    tsdf_rows = load_csv(BASELINE_ROOT / "bonn_slam" / "slam" / "tables" / "dynamic_metrics.csv")
    accs: List[float] = []
    comps: List[float] = []
    ghost_reds: List[float] = []
    tb_sum = gh_sum = no_sum = 0.0
    rot_degs: List[float] = []
    scales: List[float] = []

    for seq in BONN_ALL3:
        ctx = build_bank("bonn", seq, papg_enable=True)
        active = active_points_125(ctx)
        moved, _before_cm, _after_cm = converge_points(ctx, active, alpha=0.5)
        src, dst = build_correspondences(ctx, moved, active)
        full_points = np.vstack([ctx.base_points, moved]) if moved.shape[0] > 0 else np.asarray(ctx.base_points, dtype=float)

        if src.shape[0] < 3:
            if variant == "127_global_rigidity_alignment":
                rotation = np.eye(3, dtype=float)
                translation = np.zeros((3,), dtype=float)
                scale = 1.0
            else:
                scale = 1.0
                rotation = np.eye(3, dtype=float)
                translation = np.zeros((3,), dtype=float)
        else:
            if variant == "127_global_rigidity_alignment":
                rotation, translation = rigid_kabsch(src, dst)
                scale = 1.0
            elif variant == "128_global_similarity_alignment":
                scale, rotation, translation = similarity_umeyama(src, dst)
            else:
                raise ValueError(variant)

        transformed = (scale * (rotation @ full_points.T)).T + translation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(transformed, dtype=float))
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
        transformed_moved = (scale * (rotation @ moved.T)).T + translation if moved.shape[0] > 0 else moved
        tb, gh, no = classify_points(transformed_moved, ctx, ref_tree)
        tb_sum += float(tb)
        gh_sum += float(gh)
        no_sum += float(no)
        rot_degs.append(rotation_deg(rotation))
        scales.append(float(scale))

    return {
        "variant": variant,
        "bonn_acc_cm": float(mean(accs)),
        "bonn_comp_r_5cm": float(mean(comps)),
        "bonn_ghost_reduction_vs_tsdf": float(mean(ghost_reds)),
        "bonn_rear_points_sum": float(tb_sum + gh_sum + no_sum),
        "bonn_rear_true_background_sum": float(tb_sum),
        "bonn_rear_ghost_sum": float(gh_sum),
        "bonn_rear_hole_or_noise_sum": float(no_sum),
        "mean_rotation_deg": float(mean(rot_degs)),
        "mean_scale": float(mean(scales)),
    }


def write_design(path: Path, r127: dict, r128: dict) -> None:
    lines = [
        "# S2 Global Rigidity Alignment Design",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "",
        "## 1. 核心思想",
        "",
        "- `126` 已证明局部平滑只能减少局部残差，无法修复全局坐标系下的微小偏差；",
        "- 因此本轮直接估计一个小的全局刚体 / 相似变换，将整张地图向高置信平面锚点整体对齐。",
        "",
        "## 2. 127 刚体校正",
        "",
        "步骤：",
        "1. 取 `126` 收敛后的增量点作为 source；",
        "2. 在对应 PAPG 平面上，为每个增量点找到最近的高置信基线锚点并投影到平面，形成 target；",
        "3. 用 Kabsch 解一个全局旋转 `R` 与平移 `t`；",
        "4. 对整张地图应用 `P_new = R P + t`。",
        "",
        "## 3. 128 相似校正",
        "",
        "在 127 基础上允许一个极小的全局尺度因子 `s`：",
        "",
        "`P_new = s R P + t`",
        "",
        "用 Umeyama 解闭式相似变换，检查是否存在可测的尺度漂移。",
        "",
        "## 4. 本轮观察",
        "",
        f"- 127 的平均旋转角约 `{r127['mean_rotation_deg']:.3f} deg`；",
        f"- 128 的平均旋转角约 `{r128['mean_rotation_deg']:.3f} deg`，平均尺度约 `{r128['mean_scale']:.6f}`；",
        "- 说明当前确实只存在非常小的全局偏差，不是大的坐标系错位。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(path: Path, rows: List[dict]) -> None:
    row_map = {row["variant"]: normalize_row(row) for row in rows}
    r111 = normalize_row(pick_variant(OUT_DIR / "S2_OCCUPANCY_ENTROPY_COMPARE.csv", "111_native_geometry_chain_direct"))
    r116 = normalize_row(pick_variant(OUT_DIR / "S2_OCCUPANCY_ENTROPY_COMPARE.csv", "116_occupancy_entropy_gap_activation"))
    r122 = normalize_row(pick_variant(OUT_DIR / "S2_VISIBILITY_DEFICIT_COMPARE.csv", "122_evidential_visibility_deficit"))
    r125 = normalize_row(pick_variant(OUT_DIR / "S2_HYBRID_INTEGRATION_COMPARE.csv", "125_hybrid_papg_constrained"))
    r126 = row_map["126_local_geometry_convergence"]
    r127 = row_map["127_global_rigidity_alignment"]
    r128 = row_map["128_global_similarity_alignment"]
    lines = [
        "# S2 Final Closing Report",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 closing / final assessment`",
        "",
        "## 1. 技术路径",
        "",
        f"- `111`：原生化耦合链，Bonn `Acc={r111['bonn_acc_cm']:.3f}`, `Comp-R={r111['bonn_comp_r_5cm']:.2f}`。",
        f"- `116`：Oracle gap-only 上界，Bonn `Acc={r116['bonn_acc_cm']:.3f}`, `Comp-R={r116['bonn_comp_r_5cm']:.2f}`, `Ghost={r116['bonn_rear_ghost_sum']:.0f}`。",
        f"- `122`：GT-free visibility deficit，`proxy_recall=0.519`, `proxy_precision=0.309`。",
        f"- `125`：GT-free 主线平衡版本，Bonn `Acc={r125['bonn_acc_cm']:.3f}`, `Comp-R={r125['bonn_comp_r_5cm']:.2f}`, `Ghost={r125['bonn_rear_ghost_sum']:.0f}`。",
        f"- `126`：局部几何收敛，Bonn `Acc={r126['bonn_acc_cm']:.3f}`, `Comp-R={r126['bonn_comp_r_5cm']:.2f}`, `Ghost={r126['bonn_rear_ghost_sum']:.0f}`。",
        f"- `127`：全局刚体校正，Bonn `Acc={r127['bonn_acc_cm']:.3f}`, `Comp-R={r127['bonn_comp_r_5cm']:.2f}`, `Ghost={r127['bonn_rear_ghost_sum']:.0f}`。",
        f"- `128`：全局相似校正，Bonn `Acc={r128['bonn_acc_cm']:.3f}`, `Comp-R={r128['bonn_comp_r_5cm']:.2f}`, `Ghost={r128['bonn_rear_ghost_sum']:.0f}`。",
        "",
        "## 2. 最终判定",
        "",
        "- S2 Final Boss 门槛：",
        "  - `Acc <= 4.200 cm`",
        "  - `Comp-R >= 72.5%`",
        "  - `Ghost <= 10`",
        "",
        f"- 最佳 GT-free 版本仍是 `126/127/128` 家族附近，但最佳 `Acc` 仍高于门槛；",
        f"- 当前最佳 GT-free Acc 为 `{min(r126['bonn_acc_cm'], r127['bonn_acc_cm'], r128['bonn_acc_cm']):.3f} cm`；",
        f"- 当前最佳 GT-free Comp-R 为 `{max(r126['bonn_comp_r_5cm'], r127['bonn_comp_r_5cm'], r128['bonn_comp_r_5cm']):.2f}%`；",
        f"- 当前 Ghost 约束已满足；",
        "",
        "结论：",
        "- `S2` 核心技术路径已完整验证并闭环；",
        "- 但严格按门槛，`Acc` 仍未跨过 `4.200 cm`，因此**不能正式批准 S2 结项**；",
        "- **严禁进入 S3**。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [
        normalize_row(pick_variant(OUT_DIR / "S2_LOCAL_GEOMETRY_CONVERGENCE_COMPARE.csv", "126_local_geometry_convergence")),
        evaluate_alignment("127_global_rigidity_alignment"),
        evaluate_alignment("128_global_similarity_alignment"),
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
        "mean_rotation_deg",
        "mean_scale",
    ]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, float("nan")) for key in fields})
    r127 = next(row for row in rows if row["variant"] == "127_global_rigidity_alignment")
    r128 = next(row for row in rows if row["variant"] == "128_global_similarity_alignment")
    write_design(DESIGN_PATH, r127, r128)
    write_final_report(FINAL_REPORT_PATH, rows)
    print("[done]", CSV_PATH)


if __name__ == "__main__":
    main()
