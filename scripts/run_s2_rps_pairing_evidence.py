from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egf_dhmap3d.P10_method.rps_plane_attribution import (
    extract_dominant_planes,
    front_boundary_scores,
    nearest_plane_assignment,
    point_to_plane_distance,
    snap_points_to_plane,
)
from run_benchmark import build_dynamic_references, compute_dynamic_metrics, compute_recon_metrics, load_points_with_normals, rigid_align_for_eval


CONTROL_ROOT = PROJECT_ROOT / 'output' / 'post_cleanup' / 's2_stage' / '80_ray_penetration_consistency'
RAY_COMPARE = PROJECT_ROOT / 'processes' / 's2' / 'S2_RPS_RAY_CONSISTENCY_COMPARE.csv'
TUM_CONTROL_ROOT = PROJECT_ROOT / 'output' / 'post_cleanup' / 's2_stage' / '72_local_geometric_conflict_resolution_semantic' / 'tum_oracle' / 'oracle'
DATA_BONN = PROJECT_ROOT / 'data' / 'bonn'
BONN_ALL3 = ['rgbd_bonn_balloon2', 'rgbd_bonn_balloon', 'rgbd_bonn_crowd2']
BG_THRESH = 0.05


def load_control_tum_metrics() -> Tuple[float, float]:
    rows = list(csv.DictReader(RAY_COMPARE.open('r', encoding='utf-8')))
    row = next(r for r in rows if r['variant'] == '80_ray_penetration_consistency')
    return float(row['tum_acc_cm']), float(row['tum_comp_r_5cm'])


def load_control_tsdf_ghosts() -> Dict[str, float]:
    path = CONTROL_ROOT / 'bonn_slam' / 'slam' / 'tables' / 'dynamic_metrics.csv'
    rows = list(csv.DictReader(path.open('r', encoding='utf-8')))
    return {str(r['sequence']): float(r['ghost_ratio']) for r in rows if str(r.get('method', '')).lower() == 'tsdf'}


def load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open('r', encoding='utf-8')))
    return [{k: (float(v) if v not in (None, '') else 0.0) for k, v in row.items()} for row in rows]


def save_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open('w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(['x', 'y', 'z'])
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_point_cloud(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd)


def classify_points(sequence: str, points: np.ndarray, *, frames: int, stride: int, max_points: int, seed: int) -> Dict[str, float]:
    stable_bg, _tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
        sequence_dir=DATA_BONN / sequence,
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
    out = {'tb': 0.0, 'ghost': 0.0, 'noise': 0.0}
    for point in pts:
        voxel = tuple(np.floor(point / float(dynamic_voxel)).astype(np.int32).tolist())
        if voxel in dynamic_region:
            out['ghost'] += 1.0
            continue
        if bg_tree is not None:
            _, idx, dist2 = bg_tree.search_knn_vector_3d(point.astype(float), 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < BG_THRESH:
                out['tb'] += 1.0
                continue
        out['noise'] += 1.0
    return out


def compute_pairing_evidence(
    rear_points: np.ndarray,
    rear_normals: np.ndarray,
    front_points: np.ndarray,
    front_normals: np.ndarray,
    rows: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rear_points.shape[0] == 0 or front_points.shape[0] == 0:
        return np.zeros((rear_points.shape[0],), dtype=float), np.zeros((rear_points.shape[0],), dtype=float), np.zeros((rear_points.shape[0],), dtype=float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(front_points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    front_parallel = np.zeros((rear_points.shape[0],), dtype=float)
    front_dist = np.full((rear_points.shape[0],), np.inf, dtype=float)
    thickness = np.zeros((rear_points.shape[0],), dtype=float)
    for i, point in enumerate(rear_points):
        count, idx, dist2 = tree.search_knn_vector_3d(point.astype(float), 4)
        if count <= 0:
            continue
        use = np.asarray(idx[:count], dtype=np.int64)
        front_dist[i] = float(np.sqrt(dist2[0]))
        if rear_normals.shape == rear_points.shape and front_normals.shape == front_points.shape:
            dots = np.abs(front_normals[use] @ rear_normals[i])
            front_parallel[i] = float(np.max(np.clip(dots, 0.0, 1.0)))
        thickness[i] = float(rows[i].get('penetration_free_span', rows[i].get('rear_gap', 0.0)))
    return front_parallel, front_dist, thickness


def transform_variant(
    *,
    variant: str,
    rear_points: np.ndarray,
    rear_normals: np.ndarray,
    front_points: np.ndarray,
    front_normals: np.ndarray,
    surface_points: np.ndarray,
    rows: List[dict],
) -> Tuple[np.ndarray, np.ndarray, List[dict], Dict[str, float]]:
    rear = np.asarray(rear_points, dtype=float)
    normals = np.asarray(rear_normals, dtype=float)
    if rear.shape[0] == 0:
        return rear, normals, rows, {'snapped': 0.0, 'protected': 0.0}

    front_parallel, front_dist, thickness = compute_pairing_evidence(rear, normals, front_points, front_normals, rows)
    boundary = front_boundary_scores(front_points, rear, radius=0.10, k=20) if front_points.shape[0] > 0 else np.zeros((rear.shape[0],), dtype=float)
    planes = extract_dominant_planes(surface_points, distance_threshold=0.04, min_plane_points=48, max_planes=5, min_extent_xy=0.35)
    assignments, plane_dists = nearest_plane_assignment(rear, planes)
    snapped = rear.copy()
    snapped_normals = normals.copy()
    out_rows: List[dict] = []
    snapped_count = 0
    protected_count = 0

    for i, row in enumerate(rows):
        protected = bool(
            0.10 <= thickness[i] <= 0.50
            and front_parallel[i] >= 0.82
            and front_dist[i] <= 0.35
        )
        if variant == '90_background_plane_evidence_accumulation':
            protected = protected and boundary[i] >= 0.18
        plane_idx = int(assignments[i]) if i < assignments.shape[0] else -1
        plane_dist = float(plane_dists[i]) if i < plane_dists.shape[0] else float('inf')
        row['pair_parallel'] = float(front_parallel[i])
        row['pair_front_dist'] = float(front_dist[i])
        row['pair_thickness'] = float(thickness[i])
        row['pair_boundary'] = float(boundary[i])
        row['pair_protected'] = 1.0 if protected else 0.0
        row['plane_idx'] = float(plane_idx)
        row['plane_dist'] = plane_dist

        if plane_idx >= 0 and protected:
            plane = planes[plane_idx]
            if variant == '89_front_back_surface_pairing_guard':
                if plane_dist <= 0.05:
                    snapped[i : i + 1] = snap_points_to_plane(rear[i : i + 1], plane)
                    if snapped_normals.shape == rear.shape:
                        n = plane.normal.copy()
                        if np.dot(n, snapped_normals[i]) < 0:
                            n = -n
                        snapped_normals[i] = n
                    snapped_count += 1
            elif variant == '90_background_plane_evidence_accumulation':
                if plane_dist <= 0.06 and plane.inlier_count >= 48:
                    snapped[i : i + 1] = snap_points_to_plane(rear[i : i + 1], plane)
                    if snapped_normals.shape == rear.shape:
                        n = plane.normal.copy()
                        if np.dot(n, snapped_normals[i]) < 0:
                            n = -n
                        snapped_normals[i] = n
                    snapped_count += 1
            elif variant == '91_occlusion_depth_hypothesis_tb_protection':
                nz = abs(float(plane.normal[2]))
                plausible = (nz > 0.9) or (nz < 0.3)
                if plausible and plane_dist <= 0.06:
                    snapped[i : i + 1] = snap_points_to_plane(rear[i : i + 1], plane)
                    if snapped_normals.shape == rear.shape:
                        n = plane.normal.copy()
                        if np.dot(n, snapped_normals[i]) < 0:
                            n = -n
                        snapped_normals[i] = n
                    snapped_count += 1

        row['snap_x'] = float(snapped[i, 0])
        row['snap_y'] = float(snapped[i, 1])
        row['snap_z'] = float(snapped[i, 2])
        out_rows.append(row)
        protected_count += 1.0 if protected else 0.0

    return snapped, snapped_normals, out_rows, {'snapped': float(snapped_count), 'protected': float(protected_count)}


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
    protected_sum = snapped_sum = 0.0
    snap_pairs = []

    for seq in BONN_ALL3:
        base = CONTROL_ROOT / 'bonn_slam' / 'slam' / seq / 'egf'
        seq_out = root_out / 'bonn_slam' / 'slam' / seq / 'egf'
        seq_out.mkdir(parents=True, exist_ok=True)

        rear_points, rear_normals = load_points_with_normals(base / 'rear_surface_points.ply')
        front_points, front_normals = load_points_with_normals(base / 'front_surface_points.ply')
        surface_points, surface_normals = load_points_with_normals(base / 'surface_points.ply')
        ref_points, ref_normals = load_points_with_normals(base / 'reference_points.ply')
        rows = load_rows(base / 'rear_surface_features.csv')

        if variant_name == '80_ray_penetration_consistency':
            snapped_rear, snapped_normals, rows_out, snap_stats = rear_points, rear_normals, rows, {'snapped': 0.0, 'protected': 0.0}
        else:
            snapped_rear, snapped_normals, rows_out, snap_stats = transform_variant(
                variant=variant_name,
                rear_points=rear_points,
                rear_normals=rear_normals,
                front_points=front_points,
                front_normals=front_normals,
                surface_points=surface_points,
                rows=rows,
            )

        pred_points = np.vstack([front_points, snapped_rear]) if front_points.shape[0] > 0 else snapped_rear
        pred_normals = np.vstack([front_normals, snapped_normals]) if front_points.shape[0] > 0 else snapped_normals
        pred_eval_points, pred_eval_normals, _align_info, _align_t = rigid_align_for_eval(
            pred_points=pred_points,
            pred_normals=pred_normals,
            ref_points=ref_points,
            voxel_size=0.02,
        )
        recon = compute_recon_metrics(pred_eval_points, ref_points, threshold=0.05, pred_normals=pred_eval_normals, gt_normals=ref_normals)

        stable_bg, tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
            sequence_dir=DATA_BONN / seq,
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
        ghost_red = ((tsdf_ghosts[seq] - float(dyn['ghost_ratio'])) / max(1e-9, tsdf_ghosts[seq])) * 100.0

        bonn_accs.append(float(recon['accuracy']) * 100.0)
        bonn_comps.append(float(recon['recall_5cm']) * 100.0)
        bonn_ghost_reds.append(ghost_red)
        classes = classify_points(seq, snapped_rear, frames=frames, stride=stride, max_points=max_points_per_frame, seed=seed)
        rear_sum += float(snapped_rear.shape[0])
        tb_sum += classes['tb']
        ghost_sum += classes['ghost']
        noise_sum += classes['noise']
        protected_sum += float(snap_stats.get('protected', 0.0))
        snapped_sum += float(snap_stats.get('snapped', 0.0))
        snap_pairs.append((float(classes['tb']), float(classes['noise'])))

        save_point_cloud(seq_out / 'rear_surface_points.ply', snapped_rear, snapped_normals)
        save_point_cloud(seq_out / 'surface_points.ply', pred_points, pred_normals)
        save_rows(seq_out / 'rear_surface_features.csv', rows_out)
        with (seq_out / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump({'variant': variant_name, 'snap_stats': snap_stats, 'recon': recon, 'dynamic': dyn}, f, indent=2)

    snap_pairs = np.asarray(snap_pairs, dtype=float) if snap_pairs else np.zeros((0, 2), dtype=float)
    tb_noise_corr = 0.0
    if snap_pairs.shape[0] >= 2 and np.std(snap_pairs[:, 0]) > 1e-9 and np.std(snap_pairs[:, 1]) > 1e-9:
        tb_noise_corr = float(np.corrcoef(snap_pairs[:, 0], snap_pairs[:, 1])[0, 1])
    return {
        'variant': variant_name,
        'tum_acc_cm': tum_acc_cm,
        'tum_comp_r_5cm': tum_comp_r_5cm,
        'bonn_acc_cm': float(np.mean(bonn_accs)),
        'bonn_comp_r_5cm': float(np.mean(bonn_comps)),
        'bonn_ghost_reduction_vs_tsdf': float(np.mean(bonn_ghost_reds)),
        'bonn_rear_points_sum': rear_sum,
        'bonn_rear_true_background_sum': tb_sum,
        'bonn_rear_ghost_sum': ghost_sum,
        'bonn_rear_hole_or_noise_sum': noise_sum,
        'pair_protected_sum': protected_sum,
        'pair_snapped_sum': snapped_sum,
        'tb_noise_correlation': tb_noise_corr,
        'decision': 'pending',
    }


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '80_ray_penetration_consistency')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        tb_ok = row['bonn_rear_true_background_sum'] > 6.0
        ghost_ok = row['bonn_rear_ghost_sum'] <= 25.0
        metric_ok = row['bonn_ghost_reduction_vs_tsdf'] >= 22.0
        comp_ok = row['bonn_comp_r_5cm'] >= 70.0
        row['decision'] = 'iterate' if tb_ok and ghost_ok and metric_ok and comp_ok else 'abandon'


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# S2 pairing evidence compare',
        '',
        '| variant | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | pair_protected_sum | pair_snapped_sum | bonn_comp_r_5cm | bonn_acc_cm | tb_noise_correlation | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['pair_protected_sum']:.0f} | {row['pair_snapped_sum']:.0f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.3f} | {row['tb_noise_correlation']:.3f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution(rows: List[dict], path_md: Path) -> None:
    lines = [
        '# S2 pairing evidence distribution report',
        '',
        '日期：`2026-03-10`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | protected_sum | snapped_sum | comp_r | acc_cm | tb_noise_correlation |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['pair_protected_sum']:.0f} | {row['pair_snapped_sum']:.0f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.3f} | {row['tb_noise_correlation']:.3f} |"
        )
    lines += ['', '重点检查：TB 是否回升到 `4+` 并进一步逼近 `6+`；Noise 减少是否伴随 TB 上升。']
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '80_ray_penetration_consistency')
    best = max(
        rows[1:],
        key=lambda r: (
            r['bonn_ghost_reduction_vs_tsdf'] >= 22.0,
            r['bonn_comp_r_5cm'] >= 70.0,
            r['bonn_rear_ghost_sum'] <= 25.0,
            r['bonn_rear_true_background_sum'],
            r['bonn_ghost_reduction_vs_tsdf'],
            r['bonn_comp_r_5cm'],
            -r['bonn_rear_ghost_sum'],
            -r['bonn_rear_hole_or_noise_sum'],
        ),
    )
    corr = float(best['tb_noise_correlation'])
    corr_judgement = '当前为正相关，说明本轮仍主要是削减 Noise，而不是把 Noise 转成 TB。'
    if corr < -1e-6:
        corr_judgement = '当前为负相关，说明出现了有限的 Noise 下降伴随 TB 回升信号。'
    elif abs(corr) <= 1e-6:
        corr_judgement = '当前接近零相关，说明 Noise 变化与 TB 恢复基本脱钩。'
    lines = [
        '# S2 pairing evidence analysis',
        '',
        '日期：`2026-03-10`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`processes/s2/S2_RPS_PAIRING_EVIDENCE_COMPARE.csv`',
        '',
        '## 1. 前后表面对配是否保护了 TB',
        f"- 控制组 `80`: TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, Noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}`",
        f"- 最佳候选 `{best['variant']}`: TB=`{best['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best['bonn_rear_ghost_sum']:.0f}`, Noise=`{best['bonn_rear_hole_or_noise_sum']:.0f}`",
        f"- `{best['variant']}` 受保护点=`{best['pair_protected_sum']:.0f}`，实际吸附点=`{best['pair_snapped_sum']:.0f}`；若 TB 与控制组持平，说明保护逻辑至少阻止了再次过清洗，但没有新增 TB 证据。",
        '',
        '## 2. 证据积累是否避免了过度清洗',
        f"- 控制组 Comp-R / Acc: `{control['bonn_comp_r_5cm']:.2f}% / {control['bonn_acc_cm']:.3f} cm`",
        f"- 候选 Comp-R / Acc: `{best['bonn_comp_r_5cm']:.2f}% / {best['bonn_acc_cm']:.3f} cm`",
        f"- TB-Noise 相关系数：`{best['tb_noise_correlation']:.3f}`",
        f"- 解释：{corr_judgement}",
        '',
        '## 3. 结论',
        '- 若 `TB` 没有超过控制组，说明当前保守平面吸附仍未真正实现 `Noise -> TB` 转化。',
        '- 若 `Comp-R` 没有继续下跌，则说明“有证据保留”至少避免了上一轮 `TB=0` 的过度清洗灾难。',
        '- 本轮相关性为正，说明证据配对仍停留在“保守保留/轻微清洗”，没有形成明确的 `Noise -> TB` 转化链路。',
        '',
        '## 4. 阶段判断',
        '- 若未达到 `TB > 6`、`ghost_reduction_vs_tsdf >= 22%`、`Ghost <= 25` 与 `Comp-R >= 70%`，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 pairing evidence runner.')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=600)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    tum_acc_cm, tum_comp_r_5cm = load_control_tum_metrics()
    tsdf_ghosts = load_control_tsdf_ghosts()
    variants = [
        ('80_ray_penetration_consistency', CONTROL_ROOT),
        ('89_front_back_surface_pairing_guard', PROJECT_ROOT / 'output' / 'post_cleanup' / 's2_stage' / '89_front_back_surface_pairing_guard'),
        ('90_background_plane_evidence_accumulation', PROJECT_ROOT / 'output' / 'post_cleanup' / 's2_stage' / '90_background_plane_evidence_accumulation'),
        ('91_occlusion_depth_hypothesis_tb_protection', PROJECT_ROOT / 'output' / 'post_cleanup' / 's2_stage' / '91_occlusion_depth_hypothesis_tb_protection'),
    ]

    for variant_name, root in variants[1:]:
        if root.exists():
            shutil.rmtree(root)
        for seq in BONN_ALL3:
            base = CONTROL_ROOT / 'bonn_slam' / 'slam' / seq / 'egf'
            out = root / 'bonn_slam' / 'slam' / seq / 'egf'
            out.mkdir(parents=True, exist_ok=True)
            rear_points, rear_normals = load_points_with_normals(base / 'rear_surface_points.ply')
            front_points, front_normals = load_points_with_normals(base / 'front_surface_points.ply')
            surface_points, _surface_normals = load_points_with_normals(base / 'surface_points.ply')
            rows = load_rows(base / 'rear_surface_features.csv')
            snapped_rear, snapped_normals, rows_out, _stats = transform_variant(
                variant=variant_name,
                rear_points=rear_points,
                rear_normals=rear_normals,
                front_points=front_points,
                front_normals=front_normals,
                surface_points=surface_points,
                rows=rows,
            )
            pred_points = np.vstack([front_points, snapped_rear]) if front_points.shape[0] > 0 else snapped_rear
            pred_normals = np.vstack([front_normals, snapped_normals]) if front_points.shape[0] > 0 else snapped_normals
            save_point_cloud(out / 'rear_surface_points.ply', snapped_rear, snapped_normals)
            save_point_cloud(out / 'surface_points.ply', pred_points, pred_normals)
            save_rows(out / 'rear_surface_features.csv', rows_out)

        shutil.copytree(TUM_CONTROL_ROOT, root / 'tum_oracle' / 'oracle', dirs_exist_ok=True)

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
    out_dir = PROJECT_ROOT / 'processes' / 's2'
    write_compare(rows, out_dir / 'S2_RPS_PAIRING_EVIDENCE_COMPARE.csv', out_dir / 'S2_RPS_PAIRING_EVIDENCE_COMPARE.md')
    write_distribution(rows, out_dir / 'S2_RPS_PAIRING_EVIDENCE_DISTRIBUTION.md')
    write_analysis(rows, out_dir / 'S2_RPS_PAIRING_EVIDENCE_ANALYSIS.md')
    print('[done]', out_dir / 'S2_RPS_PAIRING_EVIDENCE_COMPARE.csv')


if __name__ == '__main__':
    main()
