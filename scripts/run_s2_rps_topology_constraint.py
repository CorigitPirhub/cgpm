from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d

from run_s2_write_time_synthesis import PROJECT_ROOT, dryrun_base_cmds, ensure_flag, remove_flag, run, set_arg
from run_s2_rps_rear_geometry_quality import BONN_ALL3, build_commands, summarize_variant
from run_benchmark import build_dynamic_references


DATA_BONN = PROJECT_ROOT / 'data' / 'bonn'
TUM_CONTROL_ROOT = PROJECT_ROOT / 'output' / 'post_cleanup' / 's2_stage' / '72_local_geometric_conflict_resolution_semantic' / 'tum_oracle' / 'oracle'
BG_THRESH = 0.05
FEATURE_KEYS = ['topology_thickness', 'normal_consistency', 'ray_convergence']
TOPO_ARG_PAIRS = [
    ('bonn_rps_rear_selectivity_thickness_weight', '--egf_rps_rear_selectivity_thickness_weight'),
    ('bonn_rps_rear_selectivity_thickness_floor', '--egf_rps_rear_selectivity_thickness_floor'),
    ('bonn_rps_rear_selectivity_thickness_risk_weight', '--egf_rps_rear_selectivity_thickness_risk_weight'),
    ('bonn_rps_rear_selectivity_thickness_ref', '--egf_rps_rear_selectivity_thickness_ref'),
    ('bonn_rps_rear_selectivity_normal_consistency_weight', '--egf_rps_rear_selectivity_normal_consistency_weight'),
    ('bonn_rps_rear_selectivity_normal_consistency_floor', '--egf_rps_rear_selectivity_normal_consistency_floor'),
    ('bonn_rps_rear_selectivity_normal_consistency_relief_weight', '--egf_rps_rear_selectivity_normal_consistency_relief_weight'),
    ('bonn_rps_rear_selectivity_normal_consistency_radius_cells', '--egf_rps_rear_selectivity_normal_consistency_radius_cells'),
    ('bonn_rps_rear_selectivity_normal_consistency_dyn_max', '--egf_rps_rear_selectivity_normal_consistency_dyn_max'),
    ('bonn_rps_rear_selectivity_ray_convergence_weight', '--egf_rps_rear_selectivity_ray_convergence_weight'),
    ('bonn_rps_rear_selectivity_ray_convergence_floor', '--egf_rps_rear_selectivity_ray_convergence_floor'),
    ('bonn_rps_rear_selectivity_ray_convergence_relief_weight', '--egf_rps_rear_selectivity_ray_convergence_relief_weight'),
    ('bonn_rps_rear_selectivity_ray_convergence_radius_cells', '--egf_rps_rear_selectivity_ray_convergence_radius_cells'),
    ('bonn_rps_rear_selectivity_ray_convergence_gap_ref', '--egf_rps_rear_selectivity_ray_convergence_gap_ref'),
    ('bonn_rps_rear_selectivity_ray_convergence_normal_cos', '--egf_rps_rear_selectivity_ray_convergence_normal_cos'),
    ('bonn_rps_rear_selectivity_ray_convergence_thickness_ref', '--egf_rps_rear_selectivity_ray_convergence_thickness_ref'),
    ('bonn_rps_rear_selectivity_ray_convergence_ref', '--egf_rps_rear_selectivity_ray_convergence_ref'),
]
TOPO_BOOL_PAIRS: list[tuple[str, str]] = []


def load_summary(root: Path, family: str, sequence: str) -> dict:
    sub = 'tum_oracle/oracle' if family == 'tum' else 'bonn_slam/slam'
    return json.loads((root / sub / sequence / 'egf' / 'summary.json').read_text(encoding='utf-8'))


def load_feature_rows(root: Path, sequence: str) -> List[dict]:
    path = root / 'bonn_slam' / 'slam' / sequence / 'egf' / 'rear_surface_features.csv'
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open('r', encoding='utf-8')))
    parsed: List[dict] = []
    for row in rows:
        parsed.append({k: (float(v) if v not in (None, '') else 0.0) for k, v in row.items()})
    return parsed


def classify_feature_rows(sequence: str, rows: List[dict], *, frames: int, stride: int, max_points: int, seed: int) -> dict:
    stable_bg, _tail_points, dynamic_region, dynamic_voxel = build_dynamic_references(
        sequence_dir=DATA_BONN / sequence,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points,
        seed=seed,
    )
    bg_tree = None
    if stable_bg.shape[0] > 0:
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(stable_bg)
        bg_tree = o3d.geometry.KDTreeFlann(bg_pcd)
    buckets = {
        'true_background': {'count': 0.0, **{f'{k}_sum': 0.0 for k in FEATURE_KEYS}},
        'ghost_region': {'count': 0.0, **{f'{k}_sum': 0.0 for k in FEATURE_KEYS}},
        'hole_or_noise': {'count': 0.0, **{f'{k}_sum': 0.0 for k in FEATURE_KEYS}},
    }
    for row in rows:
        point = np.array([row.get('x', 0.0), row.get('y', 0.0), row.get('z', 0.0)], dtype=float)
        voxel = tuple(np.floor(point / float(dynamic_voxel)).astype(np.int32).tolist())
        label = 'hole_or_noise'
        if voxel in dynamic_region:
            label = 'ghost_region'
        elif bg_tree is not None:
            _, idx, dist2 = bg_tree.search_knn_vector_3d(point, 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < BG_THRESH:
                label = 'true_background'
        buckets[label]['count'] += 1.0
        for key in FEATURE_KEYS:
            buckets[label][f'{key}_sum'] += float(row.get(key, 0.0))
    for bucket in buckets.values():
        denom = max(1.0, bucket['count'])
        for key in FEATURE_KEYS:
            bucket[f'{key}_mean'] = bucket[f'{key}_sum'] / denom
    return buckets


def apply_topology_args(cmd: List[str], spec: dict) -> List[str]:
    out = list(cmd)
    for key, flag in TOPO_ARG_PAIRS:
        if key in spec:
            out = set_arg(out, flag, str(spec[key]))
    for key, flag in TOPO_BOOL_PAIRS:
        if bool(spec.get(key, False)):
            out = ensure_flag(out, flag)
        else:
            out = remove_flag(out, flag)
    return out


def enrich_row(root: Path, row: dict, *, frames: int, stride: int, max_points: int, seed: int) -> dict:
    row = dict(row)
    sum_fields = [
        'bonn_extract_rear_selected_sum',
        'bonn_extract_score_ready_sum',
        'bonn_extract_support_protected_sum',
        'bonn_extract_fail_score_sum',
        'bonn_rear_selectivity_pre_sum',
        'bonn_rear_selectivity_kept_sum',
        'bonn_rear_selectivity_drop_sum',
        'bonn_rear_selectivity_topk_drop_sum',
        'bonn_rear_selectivity_topology_thickness_sum',
        'bonn_rear_selectivity_normal_consistency_sum',
        'bonn_rear_selectivity_ray_convergence_sum',
        'bonn_rear_selectivity_pre_topology_thickness_sum',
        'bonn_rear_selectivity_pre_normal_consistency_sum',
        'bonn_rear_selectivity_pre_ray_convergence_sum',
    ]
    for key in sum_fields:
        row[key] = 0.0

    classes = {
        'true_background': {'count': 0.0, **{f'{k}_sum': 0.0 for k in FEATURE_KEYS}},
        'ghost_region': {'count': 0.0, **{f'{k}_sum': 0.0 for k in FEATURE_KEYS}},
        'hole_or_noise': {'count': 0.0, **{f'{k}_sum': 0.0 for k in FEATURE_KEYS}},
    }
    for seq in BONN_ALL3:
        summary = load_summary(root, 'bonn', seq)
        admission_diag = summary.get('rps_admission_diag', {})
        competition_diag = summary.get('rps_competition_diag', {})
        row['bonn_extract_rear_selected_sum'] += float(competition_diag.get('extract_rear_selected', 0.0))
        row['bonn_extract_score_ready_sum'] += float(admission_diag.get('extract_score_ready', 0.0))
        row['bonn_extract_support_protected_sum'] += float(admission_diag.get('extract_support_protected', 0.0))
        row['bonn_extract_fail_score_sum'] += float(admission_diag.get('extract_fail_score', 0.0))
        row['bonn_rear_selectivity_pre_sum'] += float(summary.get('rear_selectivity_pre_count', 0.0))
        row['bonn_rear_selectivity_kept_sum'] += float(summary.get('rear_selectivity_kept_count', 0.0))
        row['bonn_rear_selectivity_drop_sum'] += float(summary.get('rear_selectivity_drop_count', 0.0))
        row['bonn_rear_selectivity_topk_drop_sum'] += float(summary.get('rear_selectivity_topk_drop_count', 0.0))
        row['bonn_rear_selectivity_topology_thickness_sum'] += float(summary.get('rear_selectivity_topology_thickness_sum', 0.0))
        row['bonn_rear_selectivity_normal_consistency_sum'] += float(summary.get('rear_selectivity_normal_consistency_sum', 0.0))
        row['bonn_rear_selectivity_ray_convergence_sum'] += float(summary.get('rear_selectivity_ray_convergence_sum', 0.0))
        row['bonn_rear_selectivity_pre_topology_thickness_sum'] += float(summary.get('rear_selectivity_pre_topology_thickness_sum', 0.0))
        row['bonn_rear_selectivity_pre_normal_consistency_sum'] += float(summary.get('rear_selectivity_pre_normal_consistency_sum', 0.0))
        row['bonn_rear_selectivity_pre_ray_convergence_sum'] += float(summary.get('rear_selectivity_pre_ray_convergence_sum', 0.0))

        cls = classify_feature_rows(seq, load_feature_rows(root, seq), frames=frames, stride=stride, max_points=max_points, seed=seed)
        for label in classes:
            classes[label]['count'] += cls[label]['count']
            for key in FEATURE_KEYS:
                classes[label][f'{key}_sum'] += cls[label][f'{key}_sum']

    pre = max(1.0, row['bonn_rear_selectivity_pre_sum'])
    kept = max(1.0, row['bonn_rear_selectivity_kept_sum'])
    dropped = max(1.0, row['bonn_rear_selectivity_pre_sum'] - row['bonn_rear_selectivity_kept_sum'])
    row['bonn_thickness_mean'] = row['bonn_rear_selectivity_topology_thickness_sum'] / kept
    row['bonn_normal_consistency_mean'] = row['bonn_rear_selectivity_normal_consistency_sum'] / kept
    row['bonn_ray_convergence_mean'] = row['bonn_rear_selectivity_ray_convergence_sum'] / kept
    row['bonn_pre_thickness_mean'] = row['bonn_rear_selectivity_pre_topology_thickness_sum'] / pre
    row['bonn_pre_normal_consistency_mean'] = row['bonn_rear_selectivity_pre_normal_consistency_sum'] / pre
    row['bonn_pre_ray_convergence_mean'] = row['bonn_rear_selectivity_pre_ray_convergence_sum'] / pre
    row['bonn_drop_thickness_mean'] = max(0.0, row['bonn_rear_selectivity_pre_topology_thickness_sum'] - row['bonn_rear_selectivity_topology_thickness_sum']) / dropped
    row['bonn_drop_normal_consistency_mean'] = max(0.0, row['bonn_rear_selectivity_pre_normal_consistency_sum'] - row['bonn_rear_selectivity_normal_consistency_sum']) / dropped
    row['bonn_drop_ray_convergence_mean'] = max(0.0, row['bonn_rear_selectivity_pre_ray_convergence_sum'] - row['bonn_rear_selectivity_ray_convergence_sum']) / dropped
    row['bonn_noise_ratio'] = row['bonn_rear_hole_or_noise_sum'] / max(1.0, row['bonn_rear_points_sum'])

    for label, prefix in [('true_background', 'tb'), ('hole_or_noise', 'noise'), ('ghost_region', 'ghost')]:
        count = max(1.0, classes[label]['count'])
        row[f'bonn_{prefix}_count'] = classes[label]['count']
        for key in FEATURE_KEYS:
            row[f'bonn_{prefix}_{key}_mean'] = classes[label][f'{key}_sum'] / count
    return row


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
        '# S2 topology constraint compare',
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
        '# S2 topology constraint distribution report',
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
    lines += ['', '重点检查：被剔除点与被保留点的 `thickness / normal / convergence` 是否明显分离；TB 是否突破 `4`。']
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '80_ray_penetration_consistency')
    best_tb = max(rows[1:], key=lambda r: (r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum'], -r['bonn_rear_hole_or_noise_sum']))
    best_noise = min(rows[1:], key=lambda r: (r['bonn_rear_hole_or_noise_sum'], r['bonn_rear_ghost_sum'], -r['bonn_ghost_reduction_vs_tsdf']))
    lines = [
        '# S2 topology constraint analysis',
        '',
        '日期：`2026-03-10`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`processes/s2/S2_RPS_TOPOLOGY_CONSTRAINT_COMPARE.csv`',
        '',
        '## 1. 拓扑特征有效性',
        f"- 控制组 `80`: thickness=`{control['bonn_thickness_mean']:.3f}`, normal=`{control['bonn_normal_consistency_mean']:.3f}`, convergence=`{control['bonn_ray_convergence_mean']:.3f}`",
        f"- `best TB` `{best_tb['variant']}`: thickness=`{best_tb['bonn_thickness_mean']:.3f}`, normal=`{best_tb['bonn_normal_consistency_mean']:.3f}`, convergence=`{best_tb['bonn_ray_convergence_mean']:.3f}`",
        f"- `best noise cleanup` `{best_noise['variant']}`: thickness kept/dropped=`{best_noise['bonn_thickness_mean']:.3f}/{best_noise['bonn_drop_thickness_mean']:.3f}`, normal kept/dropped=`{best_noise['bonn_normal_consistency_mean']:.3f}/{best_noise['bonn_drop_normal_consistency_mean']:.3f}`, convergence kept/dropped=`{best_noise['bonn_ray_convergence_mean']:.3f}/{best_noise['bonn_drop_ray_convergence_mean']:.3f}`",
        '',
        '## 2. 是否清理了 Noise',
        f"- 控制组 `80`: TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, Noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}`, `ghost_reduction_vs_tsdf={control['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
        f"- `best noise cleanup` `{best_noise['variant']}`: TB=`{best_noise['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best_noise['bonn_rear_ghost_sum']:.0f}`, Noise=`{best_noise['bonn_rear_hole_or_noise_sum']:.0f}`, `ghost_reduction_vs_tsdf={best_noise['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
        '',
        '## 3. TB 是否突破',
        f"- 控制组 `80`: TB=`{control['bonn_rear_true_background_sum']:.0f}`",
        f"- 最佳候选 `{best_tb['variant']}`: TB=`{best_tb['bonn_rear_true_background_sum']:.0f}`",
        '- 若 `TB` 仍停在 `4`，说明拓扑约束只能净化已有候选，尚不能把隐藏在 `Noise` 中的点重新定位到真实背景表面。',
        '',
        '## 4. 为什么拓扑约束比标量特征更有效',
        '- `thickness` 直接编码前后表面间的最小物理间隔，能优先打掉贴着遮挡物后方的薄壁伪影。',
        '- `front-back normal consistency` 检查后方点是否能与静态背景法向量构成连续表面，避免孤立浮点混入。',
        '- `ray convergence` 要求多个相邻 rear 候选在厚度与法向上达成一致，比单点分数更接近真实遮挡拓扑。',
        '',
        '## 5. 阶段判断',
        '- 若未达到 `TB > 6`、`hole_or_noise_sum < 180`、`Ghost <= 25` 与 `ghost_reduction_vs_tsdf >= 22%`，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 topology constraint runner.')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=600)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    base_tum, base_bonn = dryrun_base_cmds(args.frames, args.stride, args.max_points_per_frame)
    base_spec = dict(
        frames=args.frames,
        stride=args.stride,
        max_points=args.max_points_per_frame,
        rps_candidate_rescue_enable=True,
        rps_candidate_support_gain=0.36,
        rps_candidate_bg_gain=0.30,
        rps_candidate_rho_gain=0.22,
        rps_candidate_front_relax=0.38,
        joint_bg_state_enable=True,
        joint_bg_state_on=0.15,
        joint_bg_state_gain=0.76,
        joint_bg_state_rho_gain=0.30,
        joint_bg_state_front_penalty=0.10,
        rps_commit_activation_enable=True,
        rps_commit_threshold=0.34,
        rps_commit_release=0.22,
        rps_commit_age_threshold=1.0,
        rps_commit_rho_ref=0.035,
        rps_commit_weight_ref=0.35,
        rps_commit_min_cand_rho=0.006,
        rps_commit_min_cand_w=0.025,
        rps_commit_evidence_weight=0.38,
        rps_commit_geometry_weight=0.32,
        rps_commit_bg_weight=0.20,
        rps_commit_static_weight=0.10,
        rps_commit_front_penalty=0.12,
        rps_surface_bank_enable=True,
        rps_bank_margin=0.02,
        rps_bank_separation_ref=0.02,
        rps_bank_rear_min_score=0.18,
        bonn_rps_admission_support_enable=True,
        bonn_rps_admission_support_on=0.36,
        bonn_rps_admission_support_gain=0.60,
        bonn_rps_admission_score_relax=0.06,
        bonn_rps_admission_active_floor=0.28,
        bonn_rps_admission_rho_ref=0.07,
        bonn_rps_admission_weight_ref=0.30,
        bonn_rps_rear_state_protect_enable=True,
        bonn_rps_rear_state_decay_relax=0.65,
        bonn_rps_rear_state_active_floor=0.34,
        bonn_rps_bg_manifold_state_enable=True,
        bonn_rps_bg_dense_state_enable=True,
        bonn_rps_bg_surface_constrained_enable=True,
        bonn_rps_bg_bridge_enable=True,
        bonn_rps_bg_bridge_min_visible=0.20,
        bonn_rps_bg_bridge_min_obstruction=0.22,
        bonn_rps_bg_bridge_min_step=2,
        bonn_rps_bg_bridge_max_step=5,
        bonn_rps_bg_bridge_gain=0.86,
        bonn_rps_bg_bridge_target_dyn_max=0.28,
        bonn_rps_bg_bridge_target_surface_max=0.30,
        bonn_rps_bg_bridge_ghost_suppress_enable=True,
        bonn_rps_bg_bridge_ghost_suppress_weight=0.18,
        bonn_rps_bg_bridge_relaxed_dyn_max=0.44,
        bonn_rps_bg_bridge_keep_multi_hits=True,
        bonn_rps_bg_bridge_max_hits_per_source=18,
        bonn_rps_bg_bridge_patch_radius_cells=1,
        bonn_rps_bg_bridge_patch_gain_scale=0.58,
        bonn_rps_bg_bridge_depth_hypothesis_count=1,
        bonn_rps_bg_bridge_depth_step_scale=0.30,
        bonn_rps_bg_bridge_cone_enable=True,
        bonn_rps_bg_bridge_cone_radius_cells=1,
        bonn_rps_bg_bridge_cone_gain_scale=0.52,
        bonn_rps_bg_bridge_rear_synth_enable=True,
        bonn_rps_bg_bridge_rear_support_gain=0.18,
        bonn_rps_bg_bridge_rear_rho_gain=0.09,
        bonn_rps_bg_bridge_rear_phi_blend=0.86,
        bonn_rps_bg_bridge_rear_score_floor=0.23,
        bonn_rps_bg_bridge_rear_active_floor=0.53,
        bonn_rps_bg_bridge_rear_age_floor=1.0,
        bonn_rps_bg_manifold_history_weight=0.00,
        bonn_rps_bg_manifold_obstruction_weight=0.00,
        bonn_rps_bg_manifold_visible_lo=0.00,
        bonn_rps_bg_manifold_visible_hi=0.00,
        bonn_rps_rear_selectivity_enable=True,
        bonn_rps_rear_selectivity_support_weight=0.10,
        bonn_rps_rear_selectivity_history_weight=0.18,
        bonn_rps_rear_selectivity_static_weight=0.12,
        bonn_rps_rear_selectivity_geom_weight=0.18,
        bonn_rps_rear_selectivity_bridge_weight=0.08,
        bonn_rps_rear_selectivity_density_weight=0.08,
        bonn_rps_rear_selectivity_rear_score_weight=0.18,
        bonn_rps_rear_selectivity_front_score_weight=0.22,
        bonn_rps_rear_selectivity_competition_weight=0.34,
        bonn_rps_rear_selectivity_competition_alpha=0.90,
        bonn_rps_rear_selectivity_gap_weight=0.08,
        bonn_rps_rear_selectivity_sep_weight=0.08,
        bonn_rps_rear_selectivity_dyn_weight=0.12,
        bonn_rps_rear_selectivity_ghost_weight=0.12,
        bonn_rps_rear_selectivity_front_weight=0.10,
        bonn_rps_rear_selectivity_geom_risk_weight=0.18,
        bonn_rps_rear_selectivity_history_risk_weight=0.10,
        bonn_rps_rear_selectivity_density_risk_weight=0.08,
        bonn_rps_rear_selectivity_bridge_relief_weight=0.08,
        bonn_rps_rear_selectivity_static_relief_weight=0.06,
        bonn_rps_rear_selectivity_gap_risk_weight=0.06,
        bonn_rps_rear_selectivity_score_min=0.40,
        bonn_rps_rear_selectivity_risk_max=0.60,
        bonn_rps_rear_selectivity_geom_floor=0.36,
        bonn_rps_rear_selectivity_history_floor=0.30,
        bonn_rps_rear_selectivity_bridge_floor=0.12,
        bonn_rps_rear_selectivity_competition_floor=0.02,
        bonn_rps_rear_selectivity_front_score_max=0.72,
        bonn_rps_rear_selectivity_gap_valid_min=0.10,
        bonn_rps_rear_selectivity_occlusion_order_weight=0.20,
        bonn_rps_rear_selectivity_local_conflict_weight=0.60,
        bonn_rps_rear_selectivity_local_conflict_max=0.55,
        bonn_rps_rear_selectivity_conflict_radius_cells=2,
        bonn_rps_rear_selectivity_conflict_front_score_min=0.24,
        bonn_rps_rear_selectivity_conflict_static_score_min=0.42,
        bonn_rps_rear_selectivity_conflict_dist_scale=1.0,
        bonn_rps_rear_selectivity_conflict_gap_ref=0.05,
        bonn_rps_rear_selectivity_conflict_ref=1.2,
        bonn_rps_rear_selectivity_front_residual_weight=0.12,
    )
    variants = [
        dict(
            base_spec,
            name='80_ray_penetration_consistency',
            output_name='80_ray_penetration_topology_control',
            bonn_rps_rear_selectivity_penetration_weight=0.55,
            bonn_rps_rear_selectivity_penetration_floor=0.18,
            bonn_rps_rear_selectivity_penetration_risk_weight=0.20,
            bonn_rps_rear_selectivity_penetration_free_ref=0.07,
            bonn_rps_rear_selectivity_penetration_max_steps=12,
            bonn_rps_rear_selectivity_topk=92,
        ),
        dict(
            base_spec,
            name='83_minimum_thickness_topology_filter',
            output_name='83_minimum_thickness_topology_filter',
            bonn_rps_rear_selectivity_penetration_weight=0.35,
            bonn_rps_rear_selectivity_penetration_floor=0.10,
            bonn_rps_rear_selectivity_penetration_risk_weight=0.10,
            bonn_rps_rear_selectivity_thickness_weight=0.80,
            bonn_rps_rear_selectivity_thickness_floor=0.07,
            bonn_rps_rear_selectivity_thickness_risk_weight=0.30,
            bonn_rps_rear_selectivity_thickness_ref=0.09,
            bonn_rps_rear_selectivity_topk=100,
        ),
        dict(
            base_spec,
            name='84_front_back_normal_consistency',
            output_name='84_front_back_normal_consistency',
            bonn_rps_rear_selectivity_penetration_weight=0.18,
            bonn_rps_rear_selectivity_penetration_floor=0.08,
            bonn_rps_rear_selectivity_thickness_weight=0.45,
            bonn_rps_rear_selectivity_thickness_floor=0.06,
            bonn_rps_rear_selectivity_normal_consistency_weight=0.95,
            bonn_rps_rear_selectivity_normal_consistency_floor=0.82,
            bonn_rps_rear_selectivity_normal_consistency_relief_weight=0.25,
            bonn_rps_rear_selectivity_normal_consistency_radius_cells=2,
            bonn_rps_rear_selectivity_topk=105,
        ),
        dict(
            base_spec,
            name='85_occlusion_ray_convergence_constraint',
            output_name='85_occlusion_ray_convergence_constraint',
            bonn_rps_rear_selectivity_penetration_weight=0.20,
            bonn_rps_rear_selectivity_penetration_floor=0.06,
            bonn_rps_rear_selectivity_thickness_weight=0.25,
            bonn_rps_rear_selectivity_thickness_floor=0.05,
            bonn_rps_rear_selectivity_normal_consistency_weight=0.25,
            bonn_rps_rear_selectivity_normal_consistency_floor=0.72,
            bonn_rps_rear_selectivity_ray_convergence_weight=1.00,
            bonn_rps_rear_selectivity_ray_convergence_floor=0.45,
            bonn_rps_rear_selectivity_ray_convergence_relief_weight=0.30,
            bonn_rps_rear_selectivity_ray_convergence_radius_cells=2,
            bonn_rps_rear_selectivity_ray_convergence_gap_ref=0.05,
            bonn_rps_rear_selectivity_ray_convergence_normal_cos=0.82,
            bonn_rps_rear_selectivity_ray_convergence_thickness_ref=0.08,
            bonn_rps_rear_selectivity_ray_convergence_ref=1.8,
            bonn_rps_rear_selectivity_topk=120,
        ),
    ]

    rows: List[dict] = []
    for spec in variants:
        _tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec)
        bonn_cmd = apply_topology_args(bonn_cmd, spec)
        if root.exists():
            shutil.rmtree(root)
        run(bonn_cmd)
        shutil.copytree(TUM_CONTROL_ROOT, root / 'tum_oracle' / 'oracle', dirs_exist_ok=True)
        row, _tum_details, _bonn_payload = summarize_variant(
            root,
            str(spec['name']),
            frames=args.frames,
            stride=args.stride,
            max_points=args.max_points_per_frame,
            seed=args.seed,
        )
        rows.append(enrich_row(root, row, frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame, seed=args.seed))

    decide(rows)
    out_dir = PROJECT_ROOT / 'processes' / 's2'
    write_compare(rows, out_dir / 'S2_RPS_TOPOLOGY_CONSTRAINT_COMPARE.csv', out_dir / 'S2_RPS_TOPOLOGY_CONSTRAINT_COMPARE.md')
    write_distribution(rows, out_dir / 'S2_RPS_TOPOLOGY_CONSTRAINT_DISTRIBUTION.md')
    write_analysis(rows, out_dir / 'S2_RPS_TOPOLOGY_CONSTRAINT_ANALYSIS.md')
    print('[done]', out_dir / 'S2_RPS_TOPOLOGY_CONSTRAINT_COMPARE.csv')


if __name__ == '__main__':
    main()
