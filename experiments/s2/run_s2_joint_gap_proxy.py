from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
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
from run_s2_occupancy_entropy_attack import BG_THRESH, GHOST_THRESH, BASELINE_ROOT
from run_s2_rps_rear_geometry_quality import BONN_ALL3
from run_s2_occupancy_entropy_attack import build_bank, union_points


OUT_DIR = PROJECT_ROOT / "output" / "s2"
CSV_PATH = OUT_DIR / "S2_JOINT_GAP_PROXY_COMPARE.csv"
DESIGN_PATH = OUT_DIR / "S2_JOINT_GAP_PROXY_DESIGN.md"
ANALYSIS_PATH = OUT_DIR / "S2_JOINT_GAP_PROXY_ANALYSIS.md"


def load_csv(path: Path) -> List[dict]:
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def pick_variant(path: Path, variant: str) -> dict:
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


def prepare(seq: str):
    ctx = build_bank("bonn", seq, papg_enable=True)
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
    ref_tree = cKDTree(ctx.reference_points)
    return ctx, plane_occ, plane_centroid, plane_uv, ref_tree


def classify_active(active: np.ndarray, ctx, ref_tree: cKDTree) -> Tuple[int, int, int]:
    tb = ghost = noise = 0
    for point in np.asarray(active, dtype=float):
        voxel = tuple(np.floor(point / float(ctx.dynamic_voxel)).astype(np.int32).tolist())
        if voxel in ctx.dynamic_region:
            ghost += 1
            continue
        if float(ref_tree.query(point, k=1)[0]) < BG_THRESH:
            tb += 1
        else:
            noise += 1
    return tb, ghost, noise


def proxy_metrics(active: np.ndarray, ctx) -> Tuple[float, float]:
    oracle = [cand for cand in ctx.candidates if cand.oracle_gap_dist < 0.05]
    if not oracle:
        return 0.0, 0.0
    active_keys = {tuple(np.round(point, 5).tolist()) for point in np.asarray(active, dtype=float)}
    hit = sum(1 for cand in oracle if tuple(np.round(cand.center, 5).tolist()) in active_keys)
    recall = float(hit / max(1, len(oracle)))
    precision = float(hit / max(1, len(active_keys))) if active_keys else 0.0
    return recall, precision


def geometric_fusion_mask(cand, plane_occ, plane_centroid, plane_uv, planes) -> bool:
    u, v = plane_uv[cand.plane_idx]
    centroid = plane_centroid[cand.plane_idx]
    xy = np.array([np.dot(cand.center - centroid, u), np.dot(cand.center - centroid, v)], dtype=float)
    grid = tuple(np.floor(xy / 0.03).astype(np.int32).tolist())
    neighbors = 0
    for du in range(-2, 3):
        for dv in range(-2, 3):
            if du == 0 and dv == 0:
                continue
            if (grid[0] + du, grid[1] + dv) in plane_occ[cand.plane_idx]:
                neighbors += 1
    c_vis = cand.visible >= 4 and 0.008 <= cand.gap_dist_to_base <= 0.03
    c_plane = (grid not in plane_occ[cand.plane_idx]) and neighbors >= 4
    c_ent = cand.entropy > 0.60
    high_conf = c_vis and c_plane
    mid_conf = c_vis and c_ent and cand.score > 0.04
    return bool(high_conf or mid_conf)


def score_weighted_mask(cand, plane_occ, plane_centroid, plane_uv, planes) -> bool:
    u, v = plane_uv[cand.plane_idx]
    centroid = plane_centroid[cand.plane_idx]
    xy = np.array([np.dot(cand.center - centroid, u), np.dot(cand.center - centroid, v)], dtype=float)
    grid = tuple(np.floor(xy / 0.03).astype(np.int32).tolist())
    neighbors = 0
    for du in range(-2, 3):
        for dv in range(-2, 3):
            if du == 0 and dv == 0:
                continue
            if (grid[0] + du, grid[1] + dv) in plane_occ[cand.plane_idx]:
                neighbors += 1
    f_vis = 1.0 if (cand.visible >= 4 and 0.008 <= cand.gap_dist_to_base <= 0.03) else 0.0
    f_plane = 1.0 if ((grid not in plane_occ[cand.plane_idx]) and neighbors >= 4) else 0.0
    f_ent = 1.0 if cand.entropy > 0.60 else 0.0
    plane_good = abs(float(planes[cand.plane_idx].normal[2])) > 0.85
    gap_score = 0.25 * f_vis + 0.50 * f_plane + 0.25 * f_ent + 0.25 * cand.score
    return bool(gap_score > 0.75 and plane_good)


def evaluate_joint(variant: str) -> dict:
    tsdf_rows = load_csv(BASELINE_ROOT / "bonn_slam" / "slam" / "tables" / "dynamic_metrics.csv")
    accs: List[float] = []
    comps: List[float] = []
    reds: List[float] = []
    tb_sum = gh_sum = no_sum = 0.0
    recs: List[float] = []
    precs: List[float] = []
    for seq in BONN_ALL3:
        ctx, plane_occ, plane_centroid, plane_uv, ref_tree = prepare(seq)
        if variant == "120_joint_proxy_geometric_fusion":
            active = [
                cand.center
                for cand in ctx.candidates
                if geometric_fusion_mask(cand, plane_occ, plane_centroid, plane_uv, ctx.planes)
            ]
        elif variant == "121_joint_proxy_score_weighted":
            active = [
                cand.center
                for cand in ctx.candidates
                if score_weighted_mask(cand, plane_occ, plane_centroid, plane_uv, ctx.planes)
            ]
        else:
            raise ValueError(variant)

        active = np.asarray(active, dtype=float) if active else np.zeros((0, 3), dtype=float)
        pred = union_points(ctx.base_points, active)
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
        reds.append(((float(tsdf["ghost_ratio"]) - float(dyn["ghost_ratio"])) / max(1e-9, float(tsdf["ghost_ratio"]))) * 100.0)
        tb, gh, no = classify_active(active, ctx, ref_tree)
        tb_sum += float(tb)
        gh_sum += float(gh)
        no_sum += float(no)
        recall, precision = proxy_metrics(active, ctx)
        recs.append(recall)
        precs.append(precision)

    return {
        "variant": variant,
        "bonn_acc_cm": float(mean(accs)),
        "bonn_comp_r_5cm": float(mean(comps)),
        "bonn_ghost_reduction_vs_tsdf": float(mean(reds)),
        "bonn_rear_points_sum": float(tb_sum + gh_sum + no_sum),
        "bonn_rear_true_background_sum": float(tb_sum),
        "bonn_rear_ghost_sum": float(gh_sum),
        "bonn_rear_hole_or_noise_sum": float(no_sum),
        "proxy_recall": float(mean(recs)),
        "proxy_precision": float(mean(precs)),
    }


def write_design(path: Path) -> None:
    lines = [
        "# S2 Joint Gap Proxy Design",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "",
        "## 1. 设计动机",
        "",
        "- 单点 GT-free proxy 已证明：`Visibility` 太宽，`Plane Closure` 太窄，`Entropy` 缺乏方向性。",
        "- 因此本轮不再把三种信号单独当作 proxy，而是构造成联合规则。",
        "",
        "## 2. 变体 A：120 `Joint Proxy Geometric Fusion`",
        "",
        "定义三类信号：",
        "- `C_vis`：`visible >= 4` 且 `gap_dist_to_base` 落在局部 gap band (`0.008~0.03 m`)；",
        "- `C_plane`：平面 2D occupancy 中存在 closure hole；",
        "- `C_ent`：`entropy > 0.60`。",
        "",
        "融合规则：",
        "- 高置信：`C_vis ∧ C_plane`",
        "- 中置信：`C_vis ∧ C_ent ∧ score > 0.04`",
        "- 激活：`high_conf ∨ mid_conf`",
        "",
        "## 3. 变体 B：121 `Joint Proxy Score Weighted`",
        "",
        "特征：",
        "- `f1 = I(C_vis)`",
        "- `f2 = I(C_plane)`",
        "- `f3 = I(C_ent)`",
        "- `f4 = occupancy-confidence score`",
        "",
        "评分函数：",
        "",
        "`GapScore(v) = 0.25 f1 + 0.50 f2 + 0.25 f3 + 0.25 f4`",
        "",
        "激活规则：",
        "- `GapScore > 0.75`",
        "- 且候选所属平面满足 `|n_z| > 0.85`，用于抑制非曼哈顿噪声。",
        "",
        "## 4. 借鉴来源",
        "",
        "- `GO-SLAM` / `Loopy-SLAM`：先做全局一致性，再讨论 dense gap completion。",
        "- `PaSCo`：不确定性 / 熵应成为激活决策的一部分，而不是只做后验分析。",
        "- `VisFusion`：多视角 visibility 与结构几何需要联合，而不是独立阈值化。",
        "",
        "## 5. 评测方式",
        "",
        "- 系统指标：`Acc / Comp-R / Ghost`",
        "- Proxy 质量：`proxy_recall / proxy_precision`",
        "- Oracle `116` 只作为上界；`118` 作为最佳单点下界。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(path: Path, rows: List[dict]) -> None:
    row_map = {row["variant"]: row for row in rows}
    def normalize(row: dict) -> dict:
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
    r116 = normalize(row_map["116_occupancy_entropy_gap_activation"])
    r118 = normalize(row_map["118_plane_extrapolation_closure"])
    r120 = normalize(row_map["120_joint_proxy_geometric_fusion"])
    r121 = normalize(row_map["121_joint_proxy_score_weighted"])
    lines = [
        "# S2 Joint Gap Proxy Analysis",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "",
        "## 1. 对比结果",
        "",
        "| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_ghost_sum | proxy_recall | proxy_precision |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 118_plane_extrapolation_closure | {r118['bonn_acc_cm']:.3f} | {r118['bonn_comp_r_5cm']:.2f} | {r118['bonn_rear_ghost_sum']:.0f} | {r118['proxy_recall']:.3f} | {r118['proxy_precision']:.3f} |",
        f"| 120_joint_proxy_geometric_fusion | {r120['bonn_acc_cm']:.3f} | {r120['bonn_comp_r_5cm']:.2f} | {r120['bonn_rear_ghost_sum']:.0f} | {r120['proxy_recall']:.3f} | {r120['proxy_precision']:.3f} |",
        f"| 121_joint_proxy_score_weighted | {r121['bonn_acc_cm']:.3f} | {r121['bonn_comp_r_5cm']:.2f} | {r121['bonn_rear_ghost_sum']:.0f} | {r121['proxy_recall']:.3f} | {r121['proxy_precision']:.3f} |",
        f"| 116_oracle_upper_bound | {r116['bonn_acc_cm']:.3f} | {r116['bonn_comp_r_5cm']:.2f} | {r116['bonn_rear_ghost_sum']:.0f} | - | - |",
        "",
        "## 2. 结论",
        "",
        f"- `120` 没有超过 `118`：`proxy_recall={r120['proxy_recall']:.3f}` 低于 `118` 的 `{r118['proxy_recall']:.3f}`，同时 `Ghost={r120['bonn_rear_ghost_sum']:.0f}` 仍偏高。",
        f"- `121` 进一步压低了 recall 到 `{r121['proxy_recall']:.3f}`，说明 score-weighted 融合在当前特征质量下更像是强裁剪器，而不是高质量 proxy。",
        f"- 这意味着三种信号虽然互补，但当前可用的 `Visibility / Plane / Entropy` 估计仍然太粗糙，简单联合后仍无法逼近 `116`。",
        "",
        "## 3. 距离 Oracle 的剩余差距",
        "",
        f"- `116` 的 Bonn `Acc={r116['bonn_acc_cm']:.3f}`、`Comp-R={r116['bonn_comp_r_5cm']:.2f}`、`Ghost={r116['bonn_rear_ghost_sum']:.0f}`；",
        f"- 最接近的 GT-free 单点仍是 `118`，但 recall 只有 `{r118['proxy_recall']:.3f}`；",
        f"- `120/121` 进一步说明：当前差距不在激活规则，而在输入信号本身还不够强。",
        "",
        "## 4. 阶段判断",
        "",
        "- 联合 proxy 已经完成构建，但**未成功**达到工程可用门槛。",
        "- `S2` 仍不具备“无 GT 通关”的可能性；当前还缺一个更强的 GT-free gap localization 信号。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_design(DESIGN_PATH)
    rows = [
        pick_variant(OUT_DIR / "S2_GT_FREE_GAP_PROXY_COMPARE.csv", "118_plane_extrapolation_closure"),
        evaluate_joint("120_joint_proxy_geometric_fusion"),
        evaluate_joint("121_joint_proxy_score_weighted"),
        pick_variant(OUT_DIR / "S2_OCCUPANCY_ENTROPY_COMPARE.csv", "116_occupancy_entropy_gap_activation"),
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
        "proxy_recall",
        "proxy_precision",
    ]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, float("nan")) for key in fields})
    write_analysis(ANALYSIS_PATH, rows)
    print("[done]", CSV_PATH)


if __name__ == "__main__":
    main()
