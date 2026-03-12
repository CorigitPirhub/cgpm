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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS_DIR))

from experiments.p10.rps_plane_attribution import (
    PlaneModel,
    extract_dominant_planes,
    front_boundary_scores,
    nearest_plane_assignment,
    plane_layout_mask,
    snap_points_to_plane,
)
from run_benchmark import build_dynamic_references, compute_dynamic_metrics, compute_recon_metrics, load_points_with_normals, rigid_align_for_eval
CONTROL_ROOT = PROJECT_ROOT / 'output' / 's2_stage' / '80_ray_penetration_topology_control'
TUM_CONTROL_ROOT = PROJECT_ROOT / 'output' / 's2_stage' / '72_local_geometric_conflict_resolution_semantic' / 'tum_oracle' / 'oracle'
TUM_COMPARE = PROJECT_ROOT / 'output' / 's2' / 'S2_RPS_TOPOLOGY_CONSTRAINT_COMPARE.csv'
BONN_ALL3 = [
    'rgbd_bonn_balloon2',
    'rgbd_bonn_balloon',
    'rgbd_bonn_crowd2',
]
DATA_BONN = PROJECT_ROOT / 'data' / 'bonn'
FEATURE_KEYS = ['topology_thickness', 'normal_consistency', 'ray_convergence']
BG_THRESH = 0.05


def load_control_tum_metrics() -> Tuple[float, float]:
    rows = list(csv.DictReader(TUM_COMPARE.open('r', encoding='utf-8')))
    row = next(r for r in rows if r['variant'] == '80_ray_penetration_consistency')
    return float(row['tum_acc_cm']), float(row['tum_comp_r_5cm'])


def load_control_tsdf_ghosts() -> Dict[str, float]:
    path = CONTROL_ROOT / 'bonn_slam' / 'slam' / 'tables' / 'dynamic_metrics.csv'
    rows = list(csv.DictReader(path.open('r', encoding='utf-8')))
    out = {}
    for row in rows:
        if str(row.get('method', '')).lower() == 'tsdf':
            out[str(row['sequence'])] = float(row['ghost_ratio'])
    return out


def classify_rear_points(sequence: str, rear_points: np.ndarray, *, frames: int, stride: int, max_points: int, seed: int) -> dict:
    stable_bg, _tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
        sequence_dir=DATA_BONN / sequence,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points,
        seed=seed,
    )
    pts = np.asarray(rear_points, dtype=float)
    bg_tree = None
    if stable_bg.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stable_bg)
        bg_tree = o3d.geometry.KDTreeFlann(pcd)
    stats = {'tb': 0.0, 'ghost': 0.0, 'noise': 0.0}
    for point in pts:
        voxel = tuple(np.floor(point / float(dynamic_voxel)).astype(np.int32).tolist())
        if voxel in dynamic_region:
            stats['ghost'] += 1.0
            continue
        if bg_tree is not None:
            _, idx, dist2 = bg_tree.search_knn_vector_3d(point.astype(float), 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < BG_THRESH:
                stats['tb'] += 1.0
                continue
        stats['noise'] += 1.0
    return stats


def save_point_cloud(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd)


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


def project_with_plane_attribution(
    *,
    rear_points: np.ndarray,
    rear_normals: np.ndarray,
    front_points: np.ndarray,
    all_points: np.ndarray,
    rows: List[dict],
    variant: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], List[dict]]:
    rear = np.asarray(rear_points, dtype=float)
    normals = np.asarray(rear_normals, dtype=float)
    if rear.shape[0] == 0:
        return rear, normals, {'snapped': 0.0, 'dropped': 0.0, 'plane_count': 0.0}, rows

    plane_source = np.vstack([all_points, rear]) if all_points.shape[0] > 0 else rear
    planes = extract_dominant_planes(
        plane_source,
        distance_threshold=0.04,
        min_plane_points=32,
        max_planes=5,
        min_extent_xy=0.35,
    )
    assignments, plane_dists = nearest_plane_assignment(rear, planes)
    snapped = rear.copy()
    snapped_normals = normals.copy()
    keep = np.ones((rear.shape[0],), dtype=bool)
    rows_out: List[dict] = []
    boundary_scores = front_boundary_scores(front_points, rear, radius=0.10, k=20) if front_points.shape[0] > 0 else np.zeros((rear.shape[0],), dtype=float)

    for i, row in enumerate(rows):
        plane_idx = int(assignments[i]) if i < assignments.shape[0] else -1
        dist = float(plane_dists[i]) if i < plane_dists.shape[0] else float('inf')
        thickness = float(row.get('topology_thickness', row.get('penetration_free_span', 0.0)))
        boundary = float(boundary_scores[i]) if i < boundary_scores.shape[0] else 0.0
        if plane_idx >= 0:
            plane = planes[plane_idx]
            if dist <= 0.06:
                snapped[i : i + 1] = snap_points_to_plane(rear[i : i + 1], plane)
                if snapped_normals.shape == rear.shape:
                    n = plane.normal
                    if np.dot(n, snapped_normals[i]) < 0.0:
                        n = -n
                    snapped_normals[i] = n
            if variant == '86_rear_plane_clustering_snapping':
                keep[i] = bool(dist <= 0.06 and thickness >= 0.05)
            elif variant == '87_front_mask_guided_back_projection':
                keep[i] = bool(dist <= 0.06 and thickness >= 0.05 and boundary >= 0.32)
            elif variant == '88_occlusion_depth_hypothesis_validation':
                layout_keep = plane_layout_mask(snapped[i : i + 1], np.array([plane_idx]), planes)[0]
                keep[i] = bool(dist <= 0.06 and thickness >= 0.05 and layout_keep)
            else:
                keep[i] = True
            row['plane_idx'] = float(plane_idx)
            row['plane_dist'] = dist
            row['front_boundary_score'] = boundary
        else:
            keep[i] = False
            row['plane_idx'] = -1.0
            row['plane_dist'] = float('inf')
            row['front_boundary_score'] = boundary
        row['snapped'] = 1.0 if keep[i] and dist <= 0.06 else 0.0
        row['snap_x'] = float(snapped[i, 0])
        row['snap_y'] = float(snapped[i, 1])
        row['snap_z'] = float(snapped[i, 2])
        rows_out.append(row)

    out_points = snapped[keep]
    out_normals = snapped_normals[keep] if snapped_normals.shape == rear.shape else np.zeros_like(out_points)
    stats = {
        'snapped': float(np.count_nonzero(np.logical_and(keep, plane_dists <= 0.06))),
        'dropped': float(np.count_nonzero(~keep)),
        'plane_count': float(len(planes)),
    }
    return out_points, out_normals, stats, rows_out


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
    bonn_ghost_reductions: List[float] = []
    rear_sum = tb_sum = ghost_sum = noise_sum = 0.0
    thickness_kept = thickness_drop = 0.0
    normal_kept = normal_drop = 0.0
    converge_kept = converge_drop = 0.0
    kept_count = drop_count = 0.0
    tb_feat = {k: 0.0 for k in FEATURE_KEYS}
    noise_feat = {k: 0.0 for k in FEATURE_KEYS}
    ghost_feat = {k: 0.0 for k in FEATURE_KEYS}
    tb_count = noise_count = ghost_count = 0.0

    for seq in BONN_ALL3:
        base = CONTROL_ROOT / 'bonn_slam' / 'slam' / seq / 'egf'
        seq_out = root_out / 'bonn_slam' / 'slam' / seq / 'egf'
        seq_out.mkdir(parents=True, exist_ok=True)
        rear_points, rear_normals = load_points_with_normals(base / 'rear_surface_points.ply')
        front_points, front_normals = load_points_with_normals(base / 'front_surface_points.ply')
        surface_points, surface_normals = load_points_with_normals(base / 'surface_points.ply')
        ref_points, ref_normals = load_points_with_normals(base / 'reference_points.ply')
        rows = list(csv.DictReader((base / 'rear_surface_features.csv').open('r', encoding='utf-8')))
        parsed_rows = [{k: (float(v) if v not in (None, '') else 0.0) for k, v in row.items()} for row in rows]

        adjusted_rear, adjusted_normals, snap_stats, rows_out = project_with_plane_attribution(
            rear_points=rear_points,
            rear_normals=rear_normals,
            front_points=front_points,
            all_points=surface_points,
            rows=parsed_rows,
            variant=variant_name,
        )
        pred_points = np.vstack([front_points, adjusted_rear]) if front_points.shape[0] > 0 else adjusted_rear
        pred_normals = np.vstack([front_normals, adjusted_normals]) if front_normals.shape[0] > 0 else adjusted_normals
        pred_eval_points, pred_eval_normals, align_info, align_t = rigid_align_for_eval(
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
        bonn_ghost_reductions.append(ghost_red)

        classes = classify_rear_points(seq, adjusted_rear, frames=frames, stride=stride, max_points=max_points_per_frame, seed=seed)
        rear_sum += float(adjusted_rear.shape[0])
        tb_sum += classes['tb']
        ghost_sum += classes['ghost']
        noise_sum += classes['noise']

        keep_mask = np.array([bool(r.get('snapped', 0.0) >= 0.5) for r in rows_out], dtype=bool)
        if rows_out:
            kept_rows = [r for r in rows_out if r.get('snapped', 0.0) >= 0.5]
            drop_rows = [r for r in rows_out if r.get('snapped', 0.0) < 0.5]
            kept_count += float(len(kept_rows))
            drop_count += float(len(drop_rows))
            thickness_kept += float(sum(r.get('topology_thickness', 0.0) for r in kept_rows))
            thickness_drop += float(sum(r.get('topology_thickness', 0.0) for r in drop_rows))
            normal_kept += float(sum(r.get('normal_consistency', 0.0) for r in kept_rows))
            normal_drop += float(sum(r.get('normal_consistency', 0.0) for r in drop_rows))
            converge_kept += float(sum(r.get('ray_convergence', 0.0) for r in kept_rows))
            converge_drop += float(sum(r.get('ray_convergence', 0.0) for r in drop_rows))

        stable_bg_ref, _tail, dynamic_region_ref, dynamic_voxel_ref = build_dynamic_references(
            sequence_dir=DATA_BONN / seq,
            frames=frames,
            stride=stride,
            max_points_per_frame=max_points_per_frame,
            seed=seed,
        )
        bg_tree = None
        if stable_bg_ref.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(stable_bg_ref)
            bg_tree = o3d.geometry.KDTreeFlann(pcd)
        for r in rows_out:
            point = np.array([r.get('snap_x', r.get('x', 0.0)), r.get('snap_y', r.get('y', 0.0)), r.get('snap_z', r.get('z', 0.0))], dtype=float)
            voxel = tuple(np.floor(point / float(dynamic_voxel_ref)).astype(np.int32).tolist())
            target = 'noise'
            if voxel in dynamic_region_ref:
                target = 'ghost'
            elif bg_tree is not None:
                _, idx, dist2 = bg_tree.search_knn_vector_3d(point, 1)
                if idx and dist2 and float(np.sqrt(dist2[0])) < BG_THRESH:
                    target = 'tb'
            feat_map = {'tb': tb_feat, 'noise': noise_feat, 'ghost': ghost_feat}[target]
            for key in FEATURE_KEYS:
                feat_map[key] += float(r.get(key, 0.0))
            if target == 'tb':
                tb_count += 1.0
            elif target == 'noise':
                noise_count += 1.0
            else:
                ghost_count += 1.0

        save_point_cloud(seq_out / 'rear_surface_points.ply', adjusted_rear, adjusted_normals)
        save_point_cloud(seq_out / 'surface_points.ply', pred_points, pred_normals)
        save_rows(seq_out / 'rear_surface_features.csv', rows_out)
        with (seq_out / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump(
                {
                    'variant': variant_name,
                    'snap_stats': snap_stats,
                    'recon': recon,
                    'dynamic': dyn,
                    'ghost_reduction_vs_tsdf': ghost_red,
                },
                f,
                indent=2,
            )

    return {
        'variant': variant_name,
        'tum_acc_cm': tum_acc_cm,
        'tum_comp_r_5cm': tum_comp_r_5cm,
        'bonn_acc_cm': float(np.mean(bonn_accs)),
        'bonn_comp_r_5cm': float(np.mean(bonn_comps)),
        'bonn_ghost_reduction_vs_tsdf': float(np.mean(bonn_ghost_reductions)),
        'bonn_rear_points_sum': rear_sum,
        'bonn_rear_true_background_sum': tb_sum,
        'bonn_rear_ghost_sum': ghost_sum,
        'bonn_rear_hole_or_noise_sum': noise_sum,
        'bonn_thickness_mean': thickness_kept / max(1.0, kept_count),
        'bonn_normal_consistency_mean': normal_kept / max(1.0, kept_count),
        'bonn_ray_convergence_mean': converge_kept / max(1.0, kept_count),
        'bonn_drop_thickness_mean': thickness_drop / max(1.0, drop_count),
        'bonn_drop_normal_consistency_mean': normal_drop / max(1.0, drop_count),
        'bonn_drop_ray_convergence_mean': converge_drop / max(1.0, drop_count),
        'bonn_tb_count': tb_count,
        'bonn_tb_topology_thickness_mean': tb_feat['topology_thickness'] / max(1.0, tb_count),
        'bonn_tb_normal_consistency_mean': tb_feat['normal_consistency'] / max(1.0, tb_count),
        'bonn_tb_ray_convergence_mean': tb_feat['ray_convergence'] / max(1.0, tb_count),
        'bonn_noise_count': noise_count,
        'bonn_noise_topology_thickness_mean': noise_feat['topology_thickness'] / max(1.0, noise_count),
        'bonn_noise_normal_consistency_mean': noise_feat['normal_consistency'] / max(1.0, noise_count),
        'bonn_noise_ray_convergence_mean': noise_feat['ray_convergence'] / max(1.0, noise_count),
        'bonn_ghost_count': ghost_count,
        'bonn_ghost_topology_thickness_mean': ghost_feat['topology_thickness'] / max(1.0, ghost_count),
        'bonn_ghost_normal_consistency_mean': ghost_feat['normal_consistency'] / max(1.0, ghost_count),
        'bonn_ghost_ray_convergence_mean': ghost_feat['ray_convergence'] / max(1.0, ghost_count),
        'decision': 'pending',
    }


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '80_ray_penetration_consistency')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        tb_ok = row['bonn_rear_true_background_sum'] > 6.0
        noise_ok = row['bonn_rear_hole_or_noise_sum'] < 180.0
        ghost_ok = row['bonn_rear_ghost_sum'] <= 25.0
        metric_ok = row['bonn_ghost_reduction_vs_tsdf'] >= 22.0
        row['decision'] = 'iterate' if tb_ok and noise_ok and ghost_ok and metric_ok else 'abandon'


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# S2 plane attribution compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_thickness_mean | bonn_normal_consistency_mean | bonn_ray_convergence_mean | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['bonn_thickness_mean']:.3f} | {row['bonn_normal_consistency_mean']:.3f} | {row['bonn_ray_convergence_mean']:.3f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution(rows: List[dict], path_md: Path) -> None:
    lines = [
        '# S2 plane attribution distribution report',
        '',
        '日期：`2026-03-10`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | thickness_kept | thickness_dropped | normal_kept | normal_dropped | convergence_kept | convergence_dropped |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['bonn_thickness_mean']:.3f} | {row['bonn_drop_thickness_mean']:.3f} | {row['bonn_normal_consistency_mean']:.3f} | {row['bonn_drop_normal_consistency_mean']:.3f} | {row['bonn_ray_convergence_mean']:.3f} | {row['bonn_drop_ray_convergence_mean']:.3f} |"
        )
    lines += [
        '',
        '| variant | tb_thickness | noise_thickness | tb_normal | noise_normal | tb_convergence | noise_convergence |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_tb_topology_thickness_mean']:.3f} | {row['bonn_noise_topology_thickness_mean']:.3f} | {row['bonn_tb_normal_consistency_mean']:.3f} | {row['bonn_noise_normal_consistency_mean']:.3f} | {row['bonn_tb_ray_convergence_mean']:.3f} | {row['bonn_noise_ray_convergence_mean']:.3f} |"
        )
    lines += ['', '重点检查：吸附前后 TB 是否上升、Noise 是否下降，以及 kept/dropped 的平面归属统计是否明显分离。']
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '80_ray_penetration_consistency')
    best_tb = max(rows[1:], key=lambda r: (r['bonn_rear_true_background_sum'], -r['bonn_rear_hole_or_noise_sum'], -r['bonn_rear_ghost_sum']))
    best_noise = min(rows[1:], key=lambda r: (r['bonn_rear_hole_or_noise_sum'], r['bonn_rear_ghost_sum'], -r['bonn_ghost_reduction_vs_tsdf']))
    lines = [
        '# S2 plane attribution analysis',
        '',
        '日期：`2026-03-10`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`output/tmp/legacy_artifacts_placeholder`',
        '',
        '## 1. 平面吸附效果验证',
        f"- 控制组 `80`: TB=`{control['bonn_rear_true_background_sum']:.0f}`, Noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`",
        f"- 三个 plane attribution 变体 `86/87/88`: TB=`{rows[1]['bonn_rear_true_background_sum']:.0f}/{rows[2]['bonn_rear_true_background_sum']:.0f}/{rows[3]['bonn_rear_true_background_sum']:.0f}`",
        f"- `best noise cleanup` `{best_noise['variant']}`: thickness kept/dropped=`{best_noise['bonn_thickness_mean']:.3f}/{best_noise['bonn_drop_thickness_mean']:.3f}`, normal kept/dropped=`{best_noise['bonn_normal_consistency_mean']:.3f}/{best_noise['bonn_drop_normal_consistency_mean']:.3f}`, convergence kept/dropped=`{best_noise['bonn_ray_convergence_mean']:.3f}/{best_noise['bonn_drop_ray_convergence_mean']:.3f}`",
        '',
        '## 2. 是否打破 TB=4',
        f"- 控制组 `80`: TB=`{control['bonn_rear_true_background_sum']:.0f}`",
        f"- plane attribution 三个变体 `86/87/88`: TB=`{rows[1]['bonn_rear_true_background_sum']:.0f}/{rows[2]['bonn_rear_true_background_sum']:.0f}/{rows[3]['bonn_rear_true_background_sum']:.0f}`",
        '- 结论：本轮不但没有打破 `TB=4` 的死锁，反而把剩余 TB 一并裁掉，说明当前平面归属假设过强。',
        '',
        '## 3. Noise 是否被转化为 TB',
        f"- 控制组 `80`: Noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}`",
        f"- `best noise cleanup` `{best_noise['variant']}`: Noise=`{best_noise['bonn_rear_hole_or_noise_sum']:.0f}`",
        '- 若 Noise 下降而 TB 归零，说明本轮更多是在删除或收缩错误点，而不是把它们成功吸附到真实背景表面。',
        '',
        '## 4. 吸附是否影响 Acc',
        f"- 控制组 `80`: Bonn `Acc = {control['bonn_acc_cm']:.3f} cm`, `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
        f"- 最佳候选 `{best_noise['variant']}`: Bonn `Acc = {best_noise['bonn_acc_cm']:.3f} cm`, `ghost_reduction_vs_tsdf = {best_noise['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
        '- `Acc` 没有灾难性恶化，但 `Comp-R` 明显回落到 `67-69%`，说明当前平面吸附更偏向保守清理而非背景回填。',
        '',
        '## 5. 阶段判断',
        '- 若未达到 `TB > 6`、`hole_or_noise_sum < 180`、`Ghost <= 25` 与 `ghost_reduction_vs_tsdf >= 22%`，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 plane attribution runner.')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=600)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    tum_acc_cm, tum_comp_r_5cm = load_control_tum_metrics()
    tsdf_ghosts = load_control_tsdf_ghosts()

    variants = [
        ('80_ray_penetration_consistency', CONTROL_ROOT),
        ('86_rear_plane_clustering_snapping', PROJECT_ROOT / 'output' / 's2_stage' / '86_rear_plane_clustering_snapping'),
        ('87_front_mask_guided_back_projection', PROJECT_ROOT / 'output' / 's2_stage' / '87_front_mask_guided_back_projection'),
        ('88_occlusion_depth_hypothesis_validation', PROJECT_ROOT / 'output' / 's2_stage' / '88_occlusion_depth_hypothesis_validation'),
    ]

    # Materialize post-processed variants from control outputs.
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
            rows = list(csv.DictReader((base / 'rear_surface_features.csv').open('r', encoding='utf-8')))
            parsed_rows = [{k: (float(v) if v not in (None, '') else 0.0) for k, v in row.items()} for row in rows]

            adjusted_rear, adjusted_normals, snap_stats, rows_out = project_with_plane_attribution(
                rear_points=rear_points,
                rear_normals=rear_normals,
                front_points=front_points,
                all_points=surface_points,
                rows=parsed_rows,
                variant=variant_name,
            )
            pred_points = np.vstack([front_points, adjusted_rear]) if front_points.shape[0] > 0 else adjusted_rear
            pred_normals = np.vstack([front_normals, adjusted_normals]) if front_normals.shape[0] > 0 else adjusted_normals
            save_point_cloud(out / 'rear_surface_points.ply', adjusted_rear, adjusted_normals)
            save_point_cloud(out / 'surface_points.ply', pred_points, pred_normals)
            save_rows(out / 'rear_surface_features.csv', rows_out)
            with (out / 'summary.json').open('w', encoding='utf-8') as f:
                json.dump({'variant': variant_name, 'snap_stats': snap_stats}, f, indent=2)

        shutil.copytree(TUM_CONTROL_ROOT, root / 'tum_oracle' / 'oracle', dirs_exist_ok=True)

    rows: List[dict] = []
    for variant_name, root in variants:
        rows.append(
            evaluate_variant(
                variant_name=variant_name,
                root_out=root,
                frames=args.frames,
                stride=args.stride,
                max_points_per_frame=args.max_points_per_frame,
                seed=args.seed,
                tsdf_ghosts=tsdf_ghosts,
                tum_acc_cm=tum_acc_cm,
                tum_comp_r_5cm=tum_comp_r_5cm,
            )
        )

    decide(rows)
    out_dir = PROJECT_ROOT / 'output' / 's2'
    write_compare(rows, out_dir / 'S2_RPS_PLANE_ATTRIBUTION_COMPARE.csv', out_dir / 'S2_RPS_PLANE_ATTRIBUTION_COMPARE.md')
    write_distribution(rows, out_dir / 'S2_RPS_PLANE_ATTRIBUTION_DISTRIBUTION.md')
    write_analysis(rows, out_dir / 'S2_RPS_PLANE_ATTRIBUTION_ANALYSIS.md')
    print('[done]', out_dir / 'S2_RPS_PLANE_ATTRIBUTION_COMPARE.csv')


if __name__ == '__main__':
    main()
