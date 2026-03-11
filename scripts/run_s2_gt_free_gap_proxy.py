from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_benchmark import compute_dynamic_metrics, compute_recon_metrics
from run_s2_occupancy_entropy_attack import BG_THRESH, GHOST_THRESH, BASELINE_ROOT
from run_s2_rps_rear_geometry_quality import BONN_ALL3, TUM_ALL3
from scripts.run_s2_occupancy_entropy_attack import build_bank, union_points


OUT_DIR = PROJECT_ROOT / "processes" / "s2"
CSV_PATH = OUT_DIR / "S2_GT_FREE_GAP_PROXY_COMPARE.csv"
DESIGN_PATH = OUT_DIR / "S2_GT_FREE_GAP_PROXY_DESIGN.md"
ANALYSIS_PATH = OUT_DIR / "S2_GT_FREE_GAP_PROXY_ANALYSIS.md"


def load_csv(path: Path) -> List[dict]:
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def pick_variant_row(path: Path, variant: str) -> dict:
    rows = load_csv(path)
    for row in rows:
        if row["variant"] == variant:
            return row
    raise KeyError(variant)


def plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    n = n / max(1e-9, float(np.linalg.norm(n)))
    anchor = np.array([1.0, 0.0, 0.0], dtype=float) if abs(float(n[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n, anchor)
    u = u / max(1e-9, float(np.linalg.norm(u)))
    v = np.cross(n, u)
    v = v / max(1e-9, float(np.linalg.norm(v)))
    return u, v


def classify_extra(points: np.ndarray, ref_tree: cKDTree, dynamic_region: set, dynamic_voxel: float) -> Tuple[int, int, int]:
    tb = ghost = noise = 0
    for point in np.asarray(points, dtype=float):
        voxel = tuple(np.floor(point / float(dynamic_voxel)).astype(np.int32).tolist())
        if voxel in dynamic_region:
            ghost += 1
            continue
        if float(ref_tree.query(point, k=1)[0]) < BG_THRESH:
            tb += 1
        else:
            noise += 1
    return tb, ghost, noise


def oracle_proxy_metrics(ctx, active_points: np.ndarray) -> Tuple[float, float]:
    oracle = [cand for cand in ctx.candidates if cand.oracle_gap_dist < 0.05]
    if not oracle:
        return 0.0, 0.0
    active_keys = {tuple(np.round(point, 5).tolist()) for point in np.asarray(active_points, dtype=float)}
    hit = sum(1 for cand in oracle if tuple(np.round(cand.center, 5).tolist()) in active_keys)
    recall = float(hit / max(1, len(oracle)))
    precision = float(hit / max(1, len(active_keys))) if active_keys else 0.0
    return recall, precision


def proxy_frustum_unobserved(cand, ctx) -> bool:
    return cand.visible >= 4 and 0.01 <= cand.gap_dist_to_base <= 0.03 and cand.score > 0.04


def proxy_plane_extrapolation(cand, ctx, plane_occ: Dict[int, set], plane_centroid: Dict[int, np.ndarray], plane_uv: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> bool:
    plane_idx = cand.plane_idx
    u, v = plane_uv[plane_idx]
    centroid = plane_centroid[plane_idx]
    xy = np.array([np.dot(cand.center - centroid, u), np.dot(cand.center - centroid, v)], dtype=float)
    grid = tuple(np.floor(xy / 0.03).astype(np.int32).tolist())
    occupied = plane_occ[plane_idx]
    if grid in occupied:
        return False
    neighbors = 0
    for du in range(-2, 3):
        for dv in range(-2, 3):
            if du == 0 and dv == 0:
                continue
            if (grid[0] + du, grid[1] + dv) in occupied:
                neighbors += 1
    return neighbors >= 4 and cand.score > 0.04


def proxy_entropy_guided(cand, ctx) -> bool:
    return 0.008 <= cand.gap_dist_to_base <= 0.03 and cand.entropy > 0.60 and cand.score > 0.04


def evaluate_gt_free_variant(variant: str) -> dict:
    bonn_accs: List[float] = []
    bonn_comps: List[float] = []
    bonn_ghost_reds: List[float] = []
    bonn_tb: List[float] = []
    bonn_gh: List[float] = []
    bonn_no: List[float] = []
    proxy_recalls: List[float] = []
    proxy_precs: List[float] = []

    tsdf_rows = load_csv(BASELINE_ROOT / "bonn_slam" / "slam" / "tables" / "dynamic_metrics.csv")

    for seq in BONN_ALL3:
        ctx = build_bank("bonn", seq, papg_enable=True)
        ref_tree = cKDTree(ctx.reference_points)
        if variant == "117_frustum_unobserved_proxy":
            active = [cand.center for cand in ctx.candidates if proxy_frustum_unobserved(cand, ctx)]
        elif variant == "118_plane_extrapolation_closure":
            plane_occ: Dict[int, set] = {}
            plane_centroid: Dict[int, np.ndarray] = {}
            plane_uv: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for idx, plane in enumerate(ctx.planes):
                plane_centroid[idx] = np.asarray(plane.centroid, dtype=float)
                plane_uv[idx] = plane_basis(plane.normal)
                occ = set()
                for point in ctx.base_points:
                    if abs(float(point @ plane.normal + plane.offset)) < 0.05:
                        u, v = plane_uv[idx]
                        xy = np.array([np.dot(point - plane_centroid[idx], u), np.dot(point - plane_centroid[idx], v)], dtype=float)
                        occ.add(tuple(np.floor(xy / 0.03).astype(np.int32).tolist()))
                plane_occ[idx] = occ
            active = [
                cand.center
                for cand in ctx.candidates
                if proxy_plane_extrapolation(cand, ctx, plane_occ, plane_centroid, plane_uv)
            ]
        elif variant == "119_entropy_guided_proxy":
            active = [cand.center for cand in ctx.candidates if proxy_entropy_guided(cand, ctx)]
        else:
            raise ValueError(variant)

        active = np.asarray(active, dtype=float) if active else np.zeros((0, 3), dtype=float)
        pred = union_points(ctx.base_points, active)
        recon = compute_recon_metrics(pred, ctx.reference_points, threshold=0.05)
        bonn_accs.append(float(recon["accuracy"]) * 100.0)
        bonn_comps.append(float(recon["recall_5cm"]) * 100.0)

        dyn = compute_dynamic_metrics(
            pred_points=pred,
            stable_bg_points=ctx.stable_bg,
            tail_points=ctx.tail_points,
            dynamic_region=ctx.dynamic_region,
            dynamic_voxel=ctx.dynamic_voxel,
            ghost_thresh=GHOST_THRESH,
            bg_thresh=BG_THRESH,
        )
        tsdf_row = next(row for row in tsdf_rows if row["sequence"] == seq and row["method"] == "tsdf")
        bonn_ghost_reds.append(
            ((float(tsdf_row["ghost_ratio"]) - float(dyn["ghost_ratio"])) / max(1e-9, float(tsdf_row["ghost_ratio"]))) * 100.0
        )

        tb, ghost, noise = classify_extra(active, ref_tree, ctx.dynamic_region, ctx.dynamic_voxel)
        bonn_tb.append(float(tb))
        bonn_gh.append(float(ghost))
        bonn_no.append(float(noise))
        recall, precision = oracle_proxy_metrics(ctx, active)
        proxy_recalls.append(recall)
        proxy_precs.append(precision)

    return {
        "variant": variant,
        "tum_acc_cm": float("nan"),
        "tum_comp_r_5cm": float("nan"),
        "bonn_acc_cm": float(mean(bonn_accs)),
        "bonn_comp_r_5cm": float(mean(bonn_comps)),
        "bonn_ghost_reduction_vs_tsdf": float(mean(bonn_ghost_reds)),
        "bonn_rear_points_sum": float(sum(bonn_tb) + sum(bonn_gh) + sum(bonn_no)),
        "bonn_rear_true_background_sum": float(sum(bonn_tb)),
        "bonn_rear_ghost_sum": float(sum(bonn_gh)),
        "bonn_rear_hole_or_noise_sum": float(sum(bonn_no)),
        "proxy_recall": float(mean(proxy_recalls)),
        "proxy_precision": float(mean(proxy_precs)),
    }


def write_design(path: Path) -> None:
    lines = [
        "# S2 GT-Free Gap Proxy Design",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "",
        "## 1. 设计目标",
        "",
        "- 目标不是重新发明 116 的激活公式；目标是把 `oracle gap mask` 替换为 GT-free gap proxy。",
        "- 本轮三种 proxy 都严格禁止使用 `reference_points` 参与激活；`reference_points` 只用于离线评测 proxy recall / precision。",
        "",
        "## 2. 三种 GT-Free Proxy",
        "",
        "### 117 `Frustum-Based Unobserved Proxy`",
        "- 规则：`visible >= 4` 且 `0.01 <= gap_dist_to_base <= 0.03` 且 `score > 0.04`。",
        "- 直觉：若候选点在多帧视锥内长期存在，但基线地图在该位置仍有局部 gap，则它更像“被漏掉的表面”而不是随机噪声。",
        "",
        "### 118 `Plane Extrapolation & Closure Proxy`",
        "- 把 base map 投影到每个高置信平面上，构造 2D occupancy grid。",
        "- 若候选 cell 落在空网格，但周围 `5x5` 邻域已有足够 occupied cells，则视为 plane-hole closure 候选。",
        "- 直觉：缺口通常表现为平面中的局部断裂，而不是任意方向的散点。",
        "",
        "### 119 `Entropy-Guided Proxy`",
        "- 规则：`0.008 <= gap_dist_to_base <= 0.03` 且 `entropy > 0.60` 且 `score > 0.04`。",
        "- 直觉：高熵意味着状态尚不确定，但如果它又贴近当前 map 的 gap band，就值得被优先关注。",
        "",
        "## 3. Oracle 对比方式",
        "",
        "- Oracle gap 仍然定义为：candidate 到 `111` 与 reference 差集的最近距离 `< 0.05 m`。",
        "- `proxy_recall`：Oracle gap candidate 被 proxy 覆盖的比例。",
        "- `proxy_precision`：被 proxy 激活的 candidate 中，真实属于 Oracle gap 的比例。",
        "",
        "## 4. 结论预期",
        "",
        "- 若某个 GT-free proxy 不能同时做到 `Comp-R >= 75%` 与 `Ghost <= 50`，则它只能作为中间诊断，而不能替代 oracle 版本。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(path: Path, rows: List[dict]) -> None:
    row_map = {row["variant"]: row for row in rows}
    r114 = row_map["114_papg_plane_union"]
    r116 = row_map["116_occupancy_entropy_gap_activation"]
    r117 = row_map["117_frustum_unobserved_proxy"]
    r118 = row_map["118_plane_extrapolation_closure"]
    r119 = row_map["119_entropy_guided_proxy"]
    lines = [
        "# S2 GT-Free Gap Proxy Analysis",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "",
        "## 1. Oracle 上界 vs GT-Free 下界",
        "",
        f"- Oracle 上界 `116`：Bonn `Acc={r116['bonn_acc_cm']:.3f} cm`, `Comp-R={r116['bonn_comp_r_5cm']:.2f}%`, `Ghost={r116['bonn_rear_ghost_sum']:.0f}`",
        f"- GT-free 下界 `114`：Bonn `Acc={r114['bonn_acc_cm']:.3f} cm`, `Comp-R={r114['bonn_comp_r_5cm']:.2f}%`, `Ghost={r114['bonn_rear_ghost_sum']:.0f}`",
        "",
        "## 2. 三个 GT-Free Proxy",
        "",
        "| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_ghost_sum | proxy_recall | proxy_precision |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 117_frustum_unobserved_proxy | {r117['bonn_acc_cm']:.3f} | {r117['bonn_comp_r_5cm']:.2f} | {r117['bonn_rear_ghost_sum']:.0f} | {r117['proxy_recall']:.3f} | {r117['proxy_precision']:.3f} |",
        f"| 118_plane_extrapolation_closure | {r118['bonn_acc_cm']:.3f} | {r118['bonn_comp_r_5cm']:.2f} | {r118['bonn_rear_ghost_sum']:.0f} | {r118['proxy_recall']:.3f} | {r118['proxy_precision']:.3f} |",
        f"| 119_entropy_guided_proxy | {r119['bonn_acc_cm']:.3f} | {r119['bonn_comp_r_5cm']:.2f} | {r119['bonn_rear_ghost_sum']:.0f} | {r119['proxy_recall']:.3f} | {r119['proxy_precision']:.3f} |",
        "",
        "## 3. 结论",
        "",
        f"- 三个 GT-free proxy 中，`118_plane_extrapolation_closure` 最接近可用：`Acc={r118['bonn_acc_cm']:.3f}`, `Comp-R={r118['bonn_comp_r_5cm']:.2f}%`，但 `Ghost={r118['bonn_rear_ghost_sum']:.0f}` 仍高于安全线。",
        f"- `117` 与 `119` 都未复现 116 的 coverage 收益，`proxy_recall` 分别只有 `{r117['proxy_recall']:.3f}` 与 `{r119['proxy_recall']:.3f}`，说明单纯用 frustum 或 entropy 仍难以定位大部分真缺口。",
        f"- `118` 的 `proxy_recall={r118['proxy_recall']:.3f}` 虽然高于其他 GT-free proxy，但距离目标 `0.6` 仍有明显差距；这说明 plane-hole closure 只覆盖了局部结构化缺口。",
        "",
        "## 4. 是否找到可替代 Oracle 的方案",
        "",
        "- 结论：**尚未找到**。",
        "- 原因不是 occupancy+entropy 机制无效，而是 GT-free gap localization 仍然过弱。",
        "- 下一轮应继续设计：`visibility deficit + plane closure + entropy` 的联合 proxy，而不是退回到全图 union。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_design(DESIGN_PATH)
    rows = []
    upper = pick_variant_row(OUT_DIR / "S2_OCCUPANCY_ENTROPY_COMPARE.csv", "116_occupancy_entropy_gap_activation")
    lower = pick_variant_row(OUT_DIR / "S2_OCCUPANCY_ENTROPY_COMPARE.csv", "114_papg_plane_union")
    rows.append({k: (float(v) if k != "variant" else v) for k, v in lower.items()})
    rows.append({k: (float(v) if k != "variant" else v) for k, v in upper.items()})
    rows.append(evaluate_gt_free_variant("117_frustum_unobserved_proxy"))
    rows.append(evaluate_gt_free_variant("118_plane_extrapolation_closure"))
    rows.append(evaluate_gt_free_variant("119_entropy_guided_proxy"))
    save_fields = [
        "variant",
        "bonn_acc_cm",
        "bonn_comp_r_5cm",
        "bonn_ghost_reduction_vs_tsdf",
        "bonn_rear_points_sum",
        "bonn_rear_true_background_sum",
        "bonn_rear_ghost_sum",
        "bonn_rear_hole_or_noise_sum",
        "proxy_recall",
        "proxy_precision",
    ]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=save_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, float("nan")) for key in save_fields})
    write_analysis(ANALYSIS_PATH, rows)
    print("[done]", CSV_PATH)


if __name__ == "__main__":
    main()
