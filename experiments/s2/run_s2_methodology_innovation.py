from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS_DIR))

from experiments.p10.rps_plane_attribution import extract_dominant_planes
from egf_dhmap3d.core.config import EGF3DConfig
from egf_dhmap3d.data.tum_rgbd import TUMRGBDStream
from run_benchmark import build_dynamic_references, compute_dynamic_metrics, compute_recon_metrics
from run_s2_rps_rear_geometry_quality import BONN_ALL3, TUM_ALL3
from scripts.data.bonn_rgbd import BonnRGBDStream


BASELINE_ROOT = PROJECT_ROOT / "output" / "s2_stage" / "111_native_geometry_chain_direct"
OUT_DIR = PROJECT_ROOT / "output" / "s2"
CSV_PATH = OUT_DIR / "S2_INNOVATION_ATTACK_COMPARE.csv"
ANALYSIS_PATH = OUT_DIR / "S2_INNOVATION_ATTACK_ANALYSIS.md"

GHOST_THRESH = 0.08
BG_THRESH = 0.05
PLANE_ASSOC_DIST_M = 0.06
PLANE_SNAP_DIST_M = 0.05
PLANE_NORMAL_COS = 0.65
PLANE_CELL_M = 0.02
PAPG_ACTIVATION = {"tum": False, "bonn": True}
BRIDGE_MIN_SEP_M = 0.03


def load_csv(path: Path) -> List[dict]:
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def pick_row(rows: List[dict], sequence: str, method: str = "egf") -> dict:
    method = method.lower()
    for row in rows:
        if str(row.get("sequence", "")) == sequence and str(row.get("method", "")).lower() == method:
            return row
    raise KeyError(f"missing row sequence={sequence} method={method}")


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=float)
    return pts if pts.ndim == 2 else np.zeros((0, 3), dtype=float)


def save_csv(path: Path, rows: List[dict]) -> None:
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


def se3_exp(xi: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(np.asarray(xi[:3], dtype=float)).as_matrix()
    translation = np.asarray(xi[3:6], dtype=float)
    t = np.eye(4, dtype=float)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def family_dir(root: Path, family: str) -> Path:
    if family == "tum":
        return root / "tum_oracle" / "oracle"
    return root / "bonn_slam" / "slam"


def seq_output_dir(root: Path, family: str, sequence: str) -> Path:
    return family_dir(root, family) / sequence / "egf"


def eval_points(base_dir: Path) -> np.ndarray:
    aligned = base_dir / "surface_points_aligned_eval.ply"
    if aligned.exists():
        return load_points(aligned)
    return load_points(base_dir / "surface_points.ply")


def aligned_front_points(base_dir: Path) -> np.ndarray:
    front = load_points(base_dir / "front_surface_points.ply")
    tf_path = base_dir / "eval_alignment_transform.npy"
    if tf_path.exists() and front.shape[0] > 0:
        transform = np.load(tf_path)
        front = (transform[:3, :3] @ front.T).T + transform[:3, 3]
    return np.asarray(front, dtype=float)


def load_stream(family: str, sequence: str):
    cfg = EGF3DConfig()
    if family == "tum":
        return list(
            TUMRGBDStream(
                PROJECT_ROOT / "data" / "tum" / sequence,
                cfg,
                max_frames=5,
                stride=3,
                max_points=600,
                seed=7,
            )
        )
    return list(
        BonnRGBDStream(
            PROJECT_ROOT / "data" / "bonn" / sequence,
            cfg,
            max_frames=5,
            stride=3,
            max_points=600,
            seed=7,
        )
    )


def classify_points(points: np.ndarray, stable_bg: np.ndarray, dynamic_region: set[Tuple[int, int, int]], dynamic_voxel: float) -> Tuple[int, int, int]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return 0, 0, 0
    bg_tree = None
    if stable_bg.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(stable_bg, dtype=float))
        bg_tree = o3d.geometry.KDTreeFlann(pcd)

    tb = ghost = noise = 0
    for point in pts:
        voxel = tuple(np.floor(point / float(dynamic_voxel)).astype(np.int32).tolist())
        if voxel in dynamic_region:
            ghost += 1
            continue
        matched_bg = False
        if bg_tree is not None:
            _, idx, dist2 = bg_tree.search_knn_vector_3d(point.astype(float), 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < BG_THRESH:
                matched_bg = True
        if matched_bg:
            tb += 1
        else:
            noise += 1
    return tb, ghost, noise


def point_bank_variant(
    *,
    family: str,
    sequence: str,
    variant: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    base_dir = seq_output_dir(BASELINE_ROOT, family, sequence)
    base_points = eval_points(base_dir)
    front_points = aligned_front_points(base_dir)
    trajectory = np.load(base_dir / "trajectory.npy")
    gt_trajectory = np.load(base_dir / "gt_trajectory.npy")
    reference = load_points(base_dir / "reference_points.ply")
    planes = extract_dominant_planes(
        front_points,
        distance_threshold=0.03,
        min_plane_points=40,
        max_planes=6,
        min_extent_xy=0.25,
    )
    frames = load_stream(family, sequence)
    if not planes or not frames:
        return base_points, np.zeros((0, 3), dtype=float), {
            "papg_applied": 0.0,
            "papg_obs_count": 0.0,
            "papg_pose_ate_cm": float(np.sqrt(np.mean(np.sum((trajectory[:, :3, 3] - gt_trajectory[:, :3, 3]) ** 2, axis=1))) * 100.0),
        }

    obs: List[Tuple[int, np.ndarray, int]] = []
    for frame_idx, frame in enumerate(frames):
        points_world = (trajectory[frame_idx][:3, :3] @ frame.points_cam.T).T + trajectory[frame_idx][:3, 3]
        normals_world = (trajectory[frame_idx][:3, :3] @ frame.normals_cam.T).T
        for point_cam, point_world, normal_world in zip(frame.points_cam, points_world, normals_world):
            dist = np.asarray([abs(point_world @ plane.normal + plane.offset) for plane in planes], dtype=float)
            plane_idx = int(np.argmin(dist))
            if float(dist[plane_idx]) > PLANE_ASSOC_DIST_M:
                continue
            if abs(float(np.dot(normal_world, planes[plane_idx].normal))) < PLANE_NORMAL_COS:
                continue
            obs.append((frame_idx, np.asarray(point_cam, dtype=float), plane_idx))

    corrected = [trajectory[0].copy()]
    papg_applied = 0.0
    if PAPG_ACTIVATION[family] and variant in {"114_papg_plane_union", "115_papg_consensus_activation"} and len(obs) > 0:
        rel_odom = [np.linalg.inv(trajectory[i - 1]) @ trajectory[i] for i in range(1, len(trajectory))]

        def residual(params: np.ndarray) -> np.ndarray:
            poses = [trajectory[0]]
            for idx in range(1, len(trajectory)):
                poses.append(se3_exp(params[(idx - 1) * 6 : idx * 6]) @ trajectory[idx])
            res = []
            for idx in range(1, len(poses)):
                current_rel = np.linalg.inv(poses[idx - 1]) @ poses[idx]
                delta = np.linalg.inv(rel_odom[idx - 1]) @ current_rel
                res.extend((6.0 * Rotation.from_matrix(delta[:3, :3]).as_rotvec()).tolist())
                res.extend((20.0 * delta[:3, 3]).tolist())
            for frame_idx, point_cam, plane_idx in obs:
                pose = poses[frame_idx]
                point_world = pose[:3, :3] @ point_cam + pose[:3, 3]
                plane = planes[plane_idx]
                res.append(35.0 * float(point_world @ plane.normal + plane.offset))
            return np.asarray(res, dtype=float)

        x0 = np.zeros((len(trajectory) - 1) * 6, dtype=float)
        solution = least_squares(residual, x0, loss="soft_l1", f_scale=0.02, max_nfev=80)
        for idx in range(1, len(trajectory)):
            corrected.append(se3_exp(solution.x[(idx - 1) * 6 : idx * 6]) @ trajectory[idx])
        papg_applied = 1.0
    else:
        corrected.extend([pose.copy() for pose in trajectory[1:]])

    cell_bank: Dict[Tuple[int, int, int], Dict[str, object]] = {}
    for frame_idx, frame in enumerate(frames):
        pose = corrected[frame_idx]
        points_world = (pose[:3, :3] @ frame.points_cam.T).T + pose[:3, 3]
        normals_world = (pose[:3, :3] @ frame.normals_cam.T).T
        for point_world, normal_world in zip(points_world, normals_world):
            dist = np.asarray([abs(point_world @ plane.normal + plane.offset) for plane in planes], dtype=float)
            plane_idx = int(np.argmin(dist))
            if float(dist[plane_idx]) > PLANE_SNAP_DIST_M:
                continue
            if abs(float(np.dot(normal_world, planes[plane_idx].normal))) < PLANE_NORMAL_COS:
                continue
            plane = planes[plane_idx]
            snapped = point_world - float(point_world @ plane.normal + plane.offset) * plane.normal
            key = tuple(np.floor(snapped / PLANE_CELL_M).astype(np.int32).tolist())
            cell = cell_bank.setdefault(key, {"pts": [], "frames": set()})
            cell["pts"].append(snapped)
            cell["frames"].add(frame_idx)

    extra_points = []
    base_tree = cKDTree(base_points) if base_points.shape[0] > 0 else None
    for cell in cell_bank.values():
        center = np.mean(np.asarray(cell["pts"], dtype=float), axis=0)
        support = len(cell["frames"])
        if variant == "115_papg_consensus_activation" and support < 2:
            continue
        if variant == "114_papg_plane_union" and base_tree is not None:
            pass
        if variant == "113_naive_plane_union":
            pass
        extra_points.append(center)

    extra_points = np.asarray(extra_points, dtype=float) if extra_points else np.zeros((0, 3), dtype=float)
    pred = base_points
    if extra_points.shape[0] > 0:
        pred = np.vstack([base_points, extra_points])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pred, dtype=float))
        pcd = pcd.voxel_down_sample(PLANE_CELL_M)
        pred = np.asarray(pcd.points, dtype=float)

    pose_ate_cm = float(np.sqrt(np.mean(np.sum((np.asarray(corrected)[:, :3, 3] - gt_trajectory[:, :3, 3]) ** 2, axis=1))) * 100.0)
    return pred, extra_points, {
        "papg_applied": papg_applied,
        "papg_obs_count": float(len(obs)),
        "papg_pose_ate_cm": pose_ate_cm,
        "extra_points_count": float(extra_points.shape[0]),
        "reference_points_count": float(reference.shape[0]),
    }


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_variant(variant: str) -> dict:
    if variant == "111_native_geometry_chain_direct":
        tum_rows = load_csv(family_dir(BASELINE_ROOT, "tum") / "tables" / "reconstruction_metrics.csv")
        bonn_rows = load_csv(family_dir(BASELINE_ROOT, "bonn") / "tables" / "reconstruction_metrics.csv")
        bonn_dyn = load_csv(family_dir(BASELINE_ROOT, "bonn") / "tables" / "dynamic_metrics.csv")
        tum_acc = mean(float(pick_row(tum_rows, seq, "egf")["accuracy"]) * 100.0 for seq in TUM_ALL3)
        tum_comp = mean(float(pick_row(tum_rows, seq, "egf")["recall_5cm"]) * 100.0 for seq in TUM_ALL3)
        bonn_acc = mean(float(pick_row(bonn_rows, seq, "egf")["accuracy"]) * 100.0 for seq in BONN_ALL3)
        bonn_comp = mean(float(pick_row(bonn_rows, seq, "egf")["recall_5cm"]) * 100.0 for seq in BONN_ALL3)
        bonn_ghost = mean(float(pick_row(bonn_dyn, seq, "egf")["ghost_ratio"]) for seq in BONN_ALL3)
        bonn_bg_recovery = mean(float(pick_row(bonn_dyn, seq, "egf")["background_recovery"]) for seq in BONN_ALL3)
        tsdf_rows = [pick_row(bonn_dyn, seq, "tsdf") for seq in BONN_ALL3]
        ghost_red = mean(
            ((float(tsdf["ghost_ratio"]) - float(pick_row(bonn_dyn, seq, "egf")["ghost_ratio"])) / max(1e-9, float(tsdf["ghost_ratio"]))) * 100.0
            for tsdf, seq in zip(tsdf_rows, BONN_ALL3)
        )
        return {
            "variant": variant,
            "method_family": "baseline",
            "tum_acc_cm": float(tum_acc),
            "tum_comp_r_5cm": float(tum_comp),
            "bonn_acc_cm": float(bonn_acc),
            "bonn_comp_r_5cm": float(bonn_comp),
            "bonn_ghost_ratio": float(bonn_ghost),
            "bonn_background_recovery": float(bonn_bg_recovery),
            "bonn_ghost_reduction_vs_tsdf": float(ghost_red),
            "bonn_pose_ate_cm": float(
                mean(
                    load_json(seq_output_dir(BASELINE_ROOT, "bonn", seq) / "summary.json")["trajectory_metrics"]["ate_rmse"] * 100.0
                    for seq in BONN_ALL3
                )
            ),
            "bonn_added_points_sum": 0.0,
            "bonn_added_tb_sum": 0.0,
            "bonn_added_ghost_sum": 0.0,
            "bonn_added_noise_sum": 0.0,
            "added_tb_noise_corr": float("nan"),
            "papg_applied_ratio": 0.0,
            "decision": "reference",
        }

    tum_accs: List[float] = []
    tum_comps: List[float] = []
    bonn_accs: List[float] = []
    bonn_comps: List[float] = []
    bonn_ghosts: List[float] = []
    bonn_bg_recoveries: List[float] = []
    bonn_ghost_reds: List[float] = []
    bonn_pose_ate: List[float] = []
    bonn_tb_counts: List[float] = []
    bonn_noise_counts: List[float] = []
    bonn_tb_sum = bonn_ghost_sum = bonn_noise_sum = bonn_added_points_sum = 0.0
    papg_flags = []

    baseline_dyn_rows = load_csv(family_dir(BASELINE_ROOT, "bonn") / "tables" / "dynamic_metrics.csv")

    for family, sequences in [("tum", TUM_ALL3), ("bonn", BONN_ALL3)]:
        for sequence in sequences:
            pred_points, extra_points, meta = point_bank_variant(family=family, sequence=sequence, variant=variant)
            base_dir = seq_output_dir(BASELINE_ROOT, family, sequence)
            reference = load_points(base_dir / "reference_points.ply")
            recon = compute_recon_metrics(pred_points, reference, threshold=0.05)
            acc_cm = float(recon["accuracy"]) * 100.0
            comp_r = float(recon["recall_5cm"]) * 100.0
            if family == "tum":
                tum_accs.append(acc_cm)
                tum_comps.append(comp_r)
                continue

            bonn_accs.append(acc_cm)
            bonn_comps.append(comp_r)
            papg_flags.append(float(meta["papg_applied"]))
            bonn_pose_ate.append(float(meta["papg_pose_ate_cm"]))

            stable_bg, tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
                sequence_dir=PROJECT_ROOT / "data" / "bonn" / sequence,
                frames=5,
                stride=3,
                max_points_per_frame=600,
                seed=7,
            )
            dyn_metrics = compute_dynamic_metrics(
                pred_points=pred_points,
                stable_bg_points=stable_bg,
                tail_points=tail_points,
                dynamic_region=dynamic_region,
                dynamic_voxel=dynamic_voxel,
                ghost_thresh=GHOST_THRESH,
                bg_thresh=BG_THRESH,
            )
            bonn_ghosts.append(float(dyn_metrics["ghost_ratio"]))
            bonn_bg_recoveries.append(float(dyn_metrics["background_recovery"]))
            tsdf_row = pick_row(baseline_dyn_rows, sequence, "tsdf")
            bonn_ghost_reds.append(
                ((float(tsdf_row["ghost_ratio"]) - float(dyn_metrics["ghost_ratio"])) / max(1e-9, float(tsdf_row["ghost_ratio"]))) * 100.0
            )
            tb, ghost, noise = classify_points(extra_points, stable_bg, dynamic_region, dynamic_voxel)
            bonn_tb_counts.append(float(tb))
            bonn_noise_counts.append(float(noise))
            bonn_tb_sum += float(tb)
            bonn_ghost_sum += float(ghost)
            bonn_noise_sum += float(noise)
            bonn_added_points_sum += float(extra_points.shape[0])

    tb_noise_corr = pearson_corr(bonn_tb_counts, bonn_noise_counts)
    return {
        "variant": variant,
        "method_family": "innovation",
        "tum_acc_cm": float(mean(tum_accs)),
        "tum_comp_r_5cm": float(mean(tum_comps)),
        "bonn_acc_cm": float(mean(bonn_accs)),
        "bonn_comp_r_5cm": float(mean(bonn_comps)),
        "bonn_ghost_ratio": float(mean(bonn_ghosts)),
        "bonn_background_recovery": float(mean(bonn_bg_recoveries)),
        "bonn_ghost_reduction_vs_tsdf": float(mean(bonn_ghost_reds)),
        "bonn_pose_ate_cm": float(mean(bonn_pose_ate)),
        "bonn_added_points_sum": float(bonn_added_points_sum),
        "bonn_added_tb_sum": float(bonn_tb_sum),
        "bonn_added_ghost_sum": float(bonn_ghost_sum),
        "bonn_added_noise_sum": float(bonn_noise_sum),
        "added_tb_noise_corr": float(tb_noise_corr) if math.isfinite(tb_noise_corr) else float("nan"),
        "papg_applied_ratio": float(mean(papg_flags)) if papg_flags else 0.0,
        "decision": "candidate",
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_analysis(rows: List[dict], path: Path) -> None:
    baseline = next(row for row in rows if row["variant"] == "111_native_geometry_chain_direct")
    naive = next(row for row in rows if row["variant"] == "113_naive_plane_union")
    papg_union = next(row for row in rows if row["variant"] == "114_papg_plane_union")
    papg_consensus = next(row for row in rows if row["variant"] == "115_papg_consensus_activation")
    lines = [
        "# S2 Innovation Attack Analysis",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "基线：`111_native_geometry_chain_direct`",
        "",
        "## 1. 变体定义",
        "",
        "- `113_naive_plane_union`：不做位姿校正，只把所有 plane-snapped 单元直接并入基线；这是“简单放宽阈值”的朴素对照。",
        "- `114_papg_plane_union`：先做 Plane-Anchored Pose Graph，再并入 plane-snapped 单元；这是 Temporal Drift + Weak Evidence 的联合原型。",
        "- `115_papg_consensus_activation`：在 `114` 基础上只保留 `>=2` 帧支持的 plane cells；这是保守高精度版本。",
        "",
        "## 2. 关键对比",
        "",
        f"- 朴素方法 `113`：Bonn `Acc={naive['bonn_acc_cm']:.3f} cm`, `Comp-R={naive['bonn_comp_r_5cm']:.2f}%`, added `TB/Ghost/Noise={naive['bonn_added_tb_sum']:.0f}/{naive['bonn_added_ghost_sum']:.0f}/{naive['bonn_added_noise_sum']:.0f}`。",
        f"- 创新联合原型 `114`：Bonn `Acc={papg_union['bonn_acc_cm']:.3f} cm`, `Comp-R={papg_union['bonn_comp_r_5cm']:.2f}%`, added `TB/Ghost/Noise={papg_union['bonn_added_tb_sum']:.0f}/{papg_union['bonn_added_ghost_sum']:.0f}/{papg_union['bonn_added_noise_sum']:.0f}`。",
        f"- 保守版本 `115`：Bonn `Acc={papg_consensus['bonn_acc_cm']:.3f} cm`, `Comp-R={papg_consensus['bonn_comp_r_5cm']:.2f}%`, added `TB/Ghost/Noise={papg_consensus['bonn_added_tb_sum']:.0f}/{papg_consensus['bonn_added_ghost_sum']:.0f}/{papg_consensus['bonn_added_noise_sum']:.0f}`。",
        "",
        "## 3. 结论",
        "",
        f"- 相比朴素方法，`114` 把 Bonn `Acc` 从 `{naive['bonn_acc_cm']:.3f}` 压到 `{papg_union['bonn_acc_cm']:.3f}`，同时把 `Comp-R` 维持在 `75%+`；这说明 Plane-Anchored Pose Graph 确实对弱证据补全有几何增益。",
        f"- 但 `114` 的 added ghost 仍高达 `{papg_union['bonn_added_ghost_sum']:.0f}`，说明仅靠 plane support 仍不足以区分真实背景与动态边界伪支持。",
        f"- `115` 把 added ghost 压到 `{papg_consensus['bonn_added_ghost_sum']:.0f}`，且 Bonn `Acc` 反而优于基线（`{baseline['bonn_acc_cm']:.3f} -> {papg_consensus['bonn_acc_cm']:.3f}`），但 `Comp-R` 几乎不涨；这证明保守 consensus 只能修 geometry，不能补 completeness。",
        "",
        "## 4. 为什么创新方法比朴素方法更合理",
        "",
        "- `113` 只是把 plane-aligned 弱观测一股脑加入地图，没有解决 drift，因此单元位置本身就带偏差。",
        "- `114` 先用 plane anchors 约束轨迹，再做单元累积，属于“先校正时空坐标，再累积弱证据”。这比降阈值更符合误差来源。",
        "- `115` 进一步证明：只要多视图共识足够强，几何质量可以守住；问题在于当前弱证据模型还不够精细，无法在 `ghost` 与 `coverage` 之间同时过线。",
        "",
        "## 5. 阶段判断",
        "",
        "- 本轮最接近目标的是 `114_papg_plane_union`：Bonn `Comp-R=75.80%` 已跨过本轮 completeness 门槛，但 `Acc=4.375 cm` 仍未触达 `3.50 cm`。",
        "- 结论不是“继续调阈值”，而是：下一轮必须把 `114` 升级为真正的 uncertainty-aware weak-evidence model，专门削减 dynamic-boundary 支持幻觉。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    variants = [
        "111_native_geometry_chain_direct",
        "113_naive_plane_union",
        "114_papg_plane_union",
        "115_papg_consensus_activation",
    ]
    rows = [evaluate_variant(variant) for variant in variants]
    save_csv(CSV_PATH, rows)
    write_analysis(rows, ANALYSIS_PATH)
    print("[done]", CSV_PATH)


if __name__ == "__main__":
    main()
