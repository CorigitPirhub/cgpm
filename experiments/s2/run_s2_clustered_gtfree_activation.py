from __future__ import annotations

import csv
import math
import subprocess
import sys
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
S2_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _path in (PROJECT_ROOT, S2_DIR, ROOT_SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_benchmark import compute_dynamic_metrics, compute_recon_metrics
from run_s2_occupancy_entropy_attack import BASELINE_ROOT, BG_THRESH, GHOST_THRESH, build_bank, union_points
from run_s2_rps_rear_geometry_quality import BONN_ALL3
from run_s2_visibility_deficit_attack import candidate_ray_stats

OUT_DIR = PROJECT_ROOT / "output" / "s2"
CSV_PATH = OUT_DIR / "S2_CLUSTERED_GTFREE_ACTIVATION_COMPARE.csv"
REPORT_PATH = OUT_DIR / "S2_CLUSTERED_GTFREE_ACTIVATION_REPORT.md"
ANALYSIS_PATH = OUT_DIR / "S2_CLUSTERED_GTFREE_ACTIVATION_ANALYSIS.md"

PYTHON = sys.executable

VIS_THRESH = 0.08
RAY_THRESH = 0.145
CELL_SCORE_THRESH = 0.11
CLUSTER_SCORE_THRESH = 0.20
CLUSTER_RADIUS_M = 0.10
CLUSTER_MIN_SIZE = 3
CLUSTER_KEEP_TOP_FRAC = 0.85
PLANE_EXTENT_MIN = 1.0
GAP_MAX_M = 0.10
ENTROPY_MAX = 0.72
P_OCC_MIN = 0.22


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


def ensure_dependency(csv_path: Path, script_name: str) -> None:
    if csv_path.exists():
        return
    script = PROJECT_ROOT / "experiments" / "s2" / script_name
    subprocess.run([PYTHON, str(script)], cwd=str(PROJECT_ROOT), check=True)


def ensure_current_chain() -> None:
    if not BASELINE_ROOT.exists() or not (OUT_DIR / "S2_NATIVE_BASELINE_111.csv").exists():
        subprocess.run([PYTHON, str(PROJECT_ROOT / "experiments" / "s2" / "run_s2_native_geometry_chain.py")], cwd=str(PROJECT_ROOT), check=True)
    ensure_dependency(OUT_DIR / "S2_OCCUPANCY_ENTROPY_COMPARE.csv", "run_s2_occupancy_entropy_attack.py")
    ensure_dependency(OUT_DIR / "S2_VISIBILITY_DEFICIT_COMPARE.csv", "run_s2_visibility_deficit_attack.py")
    ensure_dependency(OUT_DIR / "S2_HYBRID_INTEGRATION_COMPARE.csv", "run_s2_hybrid_integration.py")


def make_support_score(history_anchor: float, observation_support: float, direct_count: float, occlusion_count: float, neighbor_count: float) -> float:
    history_reactivate = 0.65 * history_anchor + 0.35 * observation_support
    direct_boost = 1.0 + 0.25 * max(direct_count - 1.0, 0.0)
    occ_penalty = 1.0 + 0.35 * max(occlusion_count, 0.0)
    crowd_penalty = 1.0 + 0.08 * max(neighbor_count - 4.0, 0.0)
    return float(history_reactivate * direct_boost / max(1e-6, occ_penalty * crowd_penalty))


def cluster_indices(points: np.ndarray, *, radius: float, min_size: int) -> List[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return []
    tree = cKDTree(pts)
    seen = np.zeros((pts.shape[0],), dtype=bool)
    clusters: List[np.ndarray] = []
    for i in range(pts.shape[0]):
        if seen[i]:
            continue
        neigh = tree.query_ball_point(pts[i], radius)
        if len(neigh) < min_size:
            seen[i] = True
            continue
        queue = deque(neigh)
        for n in neigh:
            seen[n] = True
        comp: List[int] = []
        while queue:
            j = int(queue.popleft())
            comp.append(j)
            neigh2 = tree.query_ball_point(pts[j], radius)
            if len(neigh2) >= min_size:
                for k in neigh2:
                    if not seen[k]:
                        seen[k] = True
                        queue.append(k)
        clusters.append(np.asarray(sorted(set(comp)), dtype=np.int64))
    return clusters


def classify_active(points: np.ndarray, ctx, ref_tree: cKDTree) -> Tuple[int, int, int]:
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


def proxy_metrics(active: np.ndarray, ctx) -> Tuple[float, float]:
    oracle = [cand for cand in ctx.candidates if cand.oracle_gap_dist < 0.05]
    if not oracle:
        return 0.0, 0.0
    active_keys = {tuple(np.round(point, 5).tolist()) for point in np.asarray(active, dtype=float)}
    hit = sum(1 for cand in oracle if tuple(np.round(cand.center, 5).tolist()) in active_keys)
    recall = float(hit / max(1, len(oracle)))
    precision = float(hit / max(1, len(active_keys))) if active_keys else 0.0
    return recall, precision


def candidate_feature_rows(ctx, seq: str) -> List[dict]:
    stats = candidate_ray_stats(ctx, seq)
    centers = np.asarray([row["candidate"].center for row in stats], dtype=float) if stats else np.zeros((0, 3), dtype=float)
    if centers.shape[0] > 0:
        tree = cKDTree(centers)
        neighbor_counts = [max(0, len(tree.query_ball_point(point, CLUSTER_RADIUS_M)) - 1) for point in centers]
    else:
        neighbor_counts = []

    rows = []
    for idx, row in enumerate(stats):
        cand = row["candidate"]
        vis_norm = max(float(row["visibility_deficit_score"]) / max(1e-6, VIS_THRESH), float(row["ray_deficit_ratio"]) / max(1e-6, RAY_THRESH))
        vis_norm = float(np.clip(vis_norm, 0.0, 2.0))
        history_anchor = float(cand.p_occ)
        observation_support = float(0.5 * row["m_occ"] + 0.5 * max(0.0, 1.0 - row["m_unobs"]))
        support_score = make_support_score(
            history_anchor=history_anchor,
            observation_support=observation_support,
            direct_count=float(row["n_surface_hit"]),
            occlusion_count=float(row["n_unknown_traversal"]),
            neighbor_count=float(neighbor_counts[idx] if idx < len(neighbor_counts) else 0.0),
        )
        support_norm = float(np.clip(support_score / 0.25, 0.0, 2.0))
        occ_norm = float(np.clip(cand.score / 0.20, 0.0, 2.0))
        cell_score = float(0.45 * occ_norm + 0.35 * vis_norm + 0.20 * support_norm)
        rows.append(
            {
                "candidate": cand,
                "center": np.asarray(cand.center, dtype=float),
                "plane_idx": int(cand.plane_idx),
                "vis_norm": vis_norm,
                "support_score": float(support_score),
                "support_norm": support_norm,
                "occ_norm": occ_norm,
                "cell_score": cell_score,
                "neighbor_count": float(neighbor_counts[idx] if idx < len(neighbor_counts) else 0.0),
            }
        )
    return rows


def build_clusters(ctx, rows: List[dict], *, cluster_min_points: int = CLUSTER_MIN_SIZE, cluster_score_threshold: float = CLUSTER_SCORE_THRESH) -> List[dict]:
    clusters: List[dict] = []
    by_plane: Dict[int, List[int]] = {}
    for i, row in enumerate(rows):
        by_plane.setdefault(int(row["plane_idx"]), []).append(i)

    for plane_idx, indices in by_plane.items():
        plane_rows = [rows[i] for i in indices]
        points = np.asarray([r["center"] for r in plane_rows], dtype=float)
        local_clusters = cluster_indices(points, radius=CLUSTER_RADIUS_M, min_size=cluster_min_points)
        for cid, local_idx in enumerate(local_clusters):
            global_idx = np.asarray([indices[int(i)] for i in local_idx], dtype=np.int64)
            cand_rows = [rows[int(i)] for i in global_idx]
            cluster_score = float(np.mean([r["cell_score"] for r in cand_rows]))
            support_mean = float(np.mean([r["support_score"] for r in cand_rows]))
            vis_mean = float(np.mean([r["vis_norm"] for r in cand_rows]))
            occ_mean = float(np.mean([r["candidate"].score for r in cand_rows]))
            gap_mean = float(np.mean([r["candidate"].gap_dist_to_base for r in cand_rows]))
            retained = (
                len(cand_rows) >= cluster_min_points
                and cluster_score >= cluster_score_threshold
                and support_mean >= 0.08
                and vis_mean >= 0.90
                and gap_mean <= GAP_MAX_M
            )
            clusters.append(
                {
                    "cluster_id": len(clusters),
                    "plane_idx": plane_idx,
                    "indices": global_idx,
                    "cluster_score": cluster_score,
                    "support_mean": support_mean,
                    "vis_mean": vis_mean,
                    "occ_mean": occ_mean,
                    "gap_mean": gap_mean,
                    "retained": bool(retained),
                }
            )
    return clusters


def activate_clustered_gtfree(
    seq: str,
    *,
    cluster_min_points: int = CLUSTER_MIN_SIZE,
    activation_threshold: float = P_OCC_MIN,
    cluster_keep_top_frac: float = CLUSTER_KEEP_TOP_FRAC,
) -> Tuple[dict, dict]:
    ctx = build_bank("bonn", seq, papg_enable=True)
    ref_tree = cKDTree(ctx.reference_points)
    rows = candidate_feature_rows(ctx, seq)
    clusters = build_clusters(ctx, rows, cluster_min_points=cluster_min_points)

    cluster_membership: Dict[int, dict] = {}
    for cluster in clusters:
        for idx in cluster["indices"]:
            cluster_membership[int(idx)] = cluster

    active_points = []
    retained_cluster_count = 0
    for cluster in clusters:
        if cluster["retained"]:
            retained_cluster_count += 1
            member_rows = [rows[int(i)] for i in cluster["indices"]]
            member_rows.sort(key=lambda r: r["cell_score"], reverse=True)
            keep_n = max(1, int(math.ceil(cluster_keep_top_frac * len(member_rows))))
            for row in member_rows[:keep_n]:
                cand = row["candidate"]
                plane = ctx.planes[cand.plane_idx]
                if abs(float(plane.normal[2])) < 0.8:
                    continue
                if float(plane.extent_xy) < PLANE_EXTENT_MIN:
                    continue
                if float(cand.gap_dist_to_base) > GAP_MAX_M:
                    continue
                if float(cand.entropy) > ENTROPY_MAX:
                    continue
                if float(cand.p_occ) < activation_threshold:
                    continue
                if float(cand.score) < 0.08:
                    continue
                if float(row["vis_norm"]) < 0.90:
                    continue
                if float(row["cell_score"]) < CELL_SCORE_THRESH:
                    continue
                active_points.append(cand.center)

    active = np.asarray(active_points, dtype=float) if active_points else np.zeros((0, 3), dtype=float)
    pred = union_points(ctx.base_points, active)
    recon = compute_recon_metrics(pred, ctx.reference_points, threshold=0.05)
    dyn = compute_dynamic_metrics(
        pred_points=pred,
        stable_bg_points=ctx.stable_bg,
        tail_points=ctx.tail_points,
        dynamic_region=ctx.dynamic_region,
        dynamic_voxel=ctx.dynamic_voxel,
        ghost_thresh=GHOST_THRESH,
        bg_thresh=BG_THRESH,
    )
    tsdf_rows = load_csv(BASELINE_ROOT / "bonn_slam" / "slam" / "tables" / "dynamic_metrics.csv")
    tsdf = next(row for row in tsdf_rows if row["sequence"] == seq and row["method"] == "tsdf")
    ghost_red = ((float(tsdf["ghost_ratio"]) - float(dyn["ghost_ratio"])) / max(1e-9, float(tsdf["ghost_ratio"]))) * 100.0
    tb, gh, no = classify_active(active, ctx, ref_tree)
    recall, precision = proxy_metrics(active, ctx)
    diag = {
        "cluster_count": float(len(clusters)),
        "retained_cluster_count": float(retained_cluster_count),
        "activated_points_sum": float(active.shape[0]),
        "mean_cluster_score": float(mean([c["cluster_score"] for c in clusters])) if clusters else 0.0,
        "mean_support_mean": float(mean([c["support_mean"] for c in clusters])) if clusters else 0.0,
        "mean_vis_mean": float(mean([c["vis_mean"] for c in clusters])) if clusters else 0.0,
        "cluster_min_points": float(cluster_min_points),
        "activation_threshold": float(activation_threshold),
        "cluster_keep_top_frac": float(cluster_keep_top_frac),
    }
    row = {
        "variant": "130_clustered_gtfree_activation",
        "bonn_acc_cm": float(recon["accuracy"]) * 100.0,
        "bonn_comp_r_5cm": float(recon["recall_5cm"]) * 100.0,
        "bonn_ghost_reduction_vs_tsdf": float(ghost_red),
        "bonn_rear_points_sum": float(tb + gh + no),
        "bonn_rear_true_background_sum": float(tb),
        "bonn_rear_ghost_sum": float(gh),
        "bonn_rear_hole_or_noise_sum": float(no),
        "proxy_recall": float(recall),
        "proxy_precision": float(precision),
        **diag,
    }
    return row, {"clusters": clusters, "rows": rows}


def evaluate_config(
    *,
    cluster_min_points: int = CLUSTER_MIN_SIZE,
    activation_threshold: float = P_OCC_MIN,
    cluster_keep_top_frac: float = CLUSTER_KEEP_TOP_FRAC,
    variant_name: str = "130_clustered_gtfree_activation",
) -> dict:
    eval_rows = []
    for seq in BONN_ALL3:
        row, _detail = activate_clustered_gtfree(
            seq,
            cluster_min_points=cluster_min_points,
            activation_threshold=activation_threshold,
            cluster_keep_top_frac=cluster_keep_top_frac,
        )
        eval_rows.append(row)
    return {
        "variant": variant_name,
        "bonn_acc_cm": float(mean([r["bonn_acc_cm"] for r in eval_rows])),
        "bonn_comp_r_5cm": float(mean([r["bonn_comp_r_5cm"] for r in eval_rows])),
        "bonn_ghost_reduction_vs_tsdf": float(mean([r["bonn_ghost_reduction_vs_tsdf"] for r in eval_rows])),
        "bonn_rear_points_sum": float(sum(r["bonn_rear_points_sum"] for r in eval_rows)),
        "bonn_rear_true_background_sum": float(sum(r["bonn_rear_true_background_sum"] for r in eval_rows)),
        "bonn_rear_ghost_sum": float(sum(r["bonn_rear_ghost_sum"] for r in eval_rows)),
        "bonn_rear_hole_or_noise_sum": float(sum(r["bonn_rear_hole_or_noise_sum"] for r in eval_rows)),
        "proxy_recall": float(mean([r["proxy_recall"] for r in eval_rows])),
        "proxy_precision": float(mean([r["proxy_precision"] for r in eval_rows])),
        "cluster_count": float(sum(r["cluster_count"] for r in eval_rows)),
        "retained_cluster_count": float(sum(r["retained_cluster_count"] for r in eval_rows)),
        "activated_points_sum": float(sum(r["activated_points_sum"] for r in eval_rows)),
        "mean_cluster_score": float(mean([r["mean_cluster_score"] for r in eval_rows])),
        "mean_support_mean": float(mean([r["mean_support_mean"] for r in eval_rows])),
        "mean_vis_mean": float(mean([r["mean_vis_mean"] for r in eval_rows])),
        "cluster_min_points": float(cluster_min_points),
        "activation_threshold": float(activation_threshold),
        "cluster_keep_top_frac": float(cluster_keep_top_frac),
    }


def write_report(path: Path, rows: List[dict]) -> None:
    row_map = {row["variant"]: normalize_row(row) for row in rows}
    r111 = row_map["111_native_geometry_chain_direct"]
    r116 = row_map["116_occupancy_entropy_gap_activation"]
    r122 = row_map["122_evidential_visibility_deficit"]
    r123 = row_map["123_ray_deficit_accumulation"]
    r125 = row_map["125_hybrid_papg_constrained"]
    r130 = row_map["130_clustered_gtfree_activation"]
    lines = [
        "# S2 Clustered GT-free Activation Report",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / independent-closure / no-S3`",
        "",
        "## 1. 设计",
        "",
        "- 用 `122/123` 提供 GT-free gap 候选；",
        "- 用 `116` 的 occupancy+entropy 机制做最终激活，而不是直接用 proxy 放点；",
        "- 用旧 RPS 的簇级支持思想只保留高支持 cluster，再在 cluster 内做受控激活；",
        "- 整体链路以 `111` 为原始数据可重建的 native baseline，不再依赖历史 `80/93/97/99` 中间产物。",
        "",
        "## 2. 对比",
        "",
        f"- `111`：Acc=`{r111['bonn_acc_cm']:.3f}`, Comp-R=`{r111['bonn_comp_r_5cm']:.2f}`, Ghost=`{r111['bonn_rear_ghost_sum']:.0f}`",
        f"- `122`：Acc=`{r122['bonn_acc_cm']:.3f}`, Comp-R=`{r122['bonn_comp_r_5cm']:.2f}`, Ghost=`{r122['bonn_rear_ghost_sum']:.0f}`, Recall=`{r122['proxy_recall']:.3f}`",
        f"- `123`：Acc=`{r123['bonn_acc_cm']:.3f}`, Comp-R=`{r123['bonn_comp_r_5cm']:.2f}`, Ghost=`{r123['bonn_rear_ghost_sum']:.0f}`, Recall=`{r123['proxy_recall']:.3f}`",
        f"- `125`：Acc=`{r125['bonn_acc_cm']:.3f}`, Comp-R=`{r125['bonn_comp_r_5cm']:.2f}`, Ghost=`{r125['bonn_rear_ghost_sum']:.0f}`",
        f"- `130`：Acc=`{r130['bonn_acc_cm']:.3f}`, Comp-R=`{r130['bonn_comp_r_5cm']:.2f}`, Ghost=`{r130['bonn_rear_ghost_sum']:.0f}`, Recall=`{r130['proxy_recall']:.3f}`",
        f"- `116`：Acc=`{r116['bonn_acc_cm']:.3f}`, Comp-R=`{r116['bonn_comp_r_5cm']:.2f}`, Ghost=`{r116['bonn_rear_ghost_sum']:.0f}`",
        "",
        "## 3. 结论",
        "",
        "- 如果 `130` 优于 `125`，说明‘GT-free signal + occupancy activation + cluster support’ 的独立闭环成立；",
        "- 如果 `130` 仍弱于 `125`，说明簇支持思想可迁移，但当前阈值和 cluster budget 还不足以形成真正增益；",
        "- 无论结果正负，这条脚本都已经替代了历史依赖中间产物的 `80/93/97/99` 旧链。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(path: Path, rows: List[dict]) -> None:
    row_map = {row["variant"]: normalize_row(row) for row in rows}
    r125 = row_map["125_hybrid_papg_constrained"]
    r130 = row_map["130_clustered_gtfree_activation"]
    r116 = row_map["116_occupancy_entropy_gap_activation"]
    lines = [
        "# S2 Clustered GT-free Activation Analysis",
        "",
        "| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_rear_ghost_sum | proxy_recall | proxy_precision |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 125_hybrid_papg_constrained | {r125['bonn_acc_cm']:.3f} | {r125['bonn_comp_r_5cm']:.2f} | {r125['bonn_rear_ghost_sum']:.0f} | - | - |",
        f"| 130_clustered_gtfree_activation | {r130['bonn_acc_cm']:.3f} | {r130['bonn_comp_r_5cm']:.2f} | {r130['bonn_rear_ghost_sum']:.0f} | {r130['proxy_recall']:.3f} | {r130['proxy_precision']:.3f} |",
        f"| 116_occupancy_entropy_gap_activation | {r116['bonn_acc_cm']:.3f} | {r116['bonn_comp_r_5cm']:.2f} | {r116['bonn_rear_ghost_sum']:.0f} | - | - |",
        "",
        f"- `130` 相比 `125` 的 Acc 差值：`{r130['bonn_acc_cm'] - r125['bonn_acc_cm']:+.3f} cm`",
        f"- `130` 相比 `125` 的 Comp-R 差值：`{r130['bonn_comp_r_5cm'] - r125['bonn_comp_r_5cm']:+.2f}`",
        f"- `130` 相比 `125` 的 Ghost 差值：`{r130['bonn_rear_ghost_sum'] - r125['bonn_rear_ghost_sum']:+.0f}`",
        "",
        "解释：",
        "- `130` 不再直接把高 deficit 候选整体激活，而是先做 cluster-level support pruning；",
        "- 这等价于把旧 RPS 的 cluster support 思想迁移到当前 `125` 架构，但删除了对历史中间产物的依赖；",
        "- 若 `130` 的 recall 保持而 ghost 下降，说明 cluster support 已成功变成独立闭环内的有效模块。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Independent GT-free clustered activation.")
    ap.add_argument("--cluster_min_points", type=int, default=CLUSTER_MIN_SIZE)
    ap.add_argument("--activation_threshold", type=float, default=P_OCC_MIN)
    ap.add_argument("--cluster_keep_top_frac", type=float, default=CLUSTER_KEEP_TOP_FRAC)
    ap.add_argument("--variant_name", type=str, default="130_clustered_gtfree_activation")
    args = ap.parse_args()

    ensure_current_chain()
    compare_rows = [
        normalize_row(load_csv(OUT_DIR / "S2_NATIVE_BASELINE_111.csv")[0]),
        normalize_row(pick_variant(OUT_DIR / "S2_OCCUPANCY_ENTROPY_COMPARE.csv", "116_occupancy_entropy_gap_activation")),
        normalize_row(pick_variant(OUT_DIR / "S2_VISIBILITY_DEFICIT_COMPARE.csv", "122_evidential_visibility_deficit")),
        normalize_row(pick_variant(OUT_DIR / "S2_VISIBILITY_DEFICIT_COMPARE.csv", "123_ray_deficit_accumulation")),
        normalize_row(pick_variant(OUT_DIR / "S2_HYBRID_INTEGRATION_COMPARE.csv", "125_hybrid_papg_constrained")),
    ]
    merged = evaluate_config(
        cluster_min_points=int(args.cluster_min_points),
        activation_threshold=float(args.activation_threshold),
        cluster_keep_top_frac=float(args.cluster_keep_top_frac),
        variant_name=str(args.variant_name),
    )
    rows = compare_rows + [merged]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    write_report(REPORT_PATH, rows)
    write_analysis(ANALYSIS_PATH, rows)
    print("[done]", CSV_PATH)


if __name__ == "__main__":
    main()
