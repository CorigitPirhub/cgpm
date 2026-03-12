from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

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

from run_s2_write_time_synthesis import PROJECT_ROOT, dryrun_base_cmds, ensure_flag, remove_flag, run, set_arg
from run_s2_rps_rear_geometry_quality import BONN_ALL3, build_commands, summarize_variant
from run_benchmark import build_dynamic_references


FEATURE_KEYS = [
    'penetration_score',
    'penetration_free_span',
    'observation_count',
    'observation_support',
    'static_coherence',
]
RAY_ARG_PAIRS = [
    ('bonn_rps_rear_selectivity_penetration_weight', '--egf_rps_rear_selectivity_penetration_weight'),
    ('bonn_rps_rear_selectivity_penetration_floor', '--egf_rps_rear_selectivity_penetration_floor'),
    ('bonn_rps_rear_selectivity_penetration_risk_weight', '--egf_rps_rear_selectivity_penetration_risk_weight'),
    ('bonn_rps_rear_selectivity_penetration_free_ref', '--egf_rps_rear_selectivity_penetration_free_ref'),
    ('bonn_rps_rear_selectivity_penetration_max_steps', '--egf_rps_rear_selectivity_penetration_max_steps'),
    ('bonn_rps_rear_selectivity_observation_weight', '--egf_rps_rear_selectivity_observation_weight'),
    ('bonn_rps_rear_selectivity_observation_floor', '--egf_rps_rear_selectivity_observation_floor'),
    ('bonn_rps_rear_selectivity_observation_risk_weight', '--egf_rps_rear_selectivity_observation_risk_weight'),
    ('bonn_rps_rear_selectivity_observation_count_ref', '--egf_rps_rear_selectivity_observation_count_ref'),
    ('bonn_rps_rear_selectivity_observation_min_count', '--egf_rps_rear_selectivity_observation_min_count'),
    ('bonn_rps_rear_selectivity_static_coherence_weight', '--egf_rps_rear_selectivity_static_coherence_weight'),
    ('bonn_rps_rear_selectivity_static_coherence_floor', '--egf_rps_rear_selectivity_static_coherence_floor'),
    ('bonn_rps_rear_selectivity_static_coherence_relief_weight', '--egf_rps_rear_selectivity_static_coherence_relief_weight'),
    ('bonn_rps_rear_selectivity_static_coherence_radius_cells', '--egf_rps_rear_selectivity_static_coherence_radius_cells'),
    ('bonn_rps_rear_selectivity_static_coherence_ref', '--egf_rps_rear_selectivity_static_coherence_ref'),
    ('bonn_rps_rear_selectivity_static_neighbor_min_weight', '--egf_rps_rear_selectivity_static_neighbor_min_weight'),
    ('bonn_rps_rear_selectivity_static_neighbor_dyn_max', '--egf_rps_rear_selectivity_static_neighbor_dyn_max'),
]
RAY_BOOL_PAIRS = [
    ('bonn_rps_rear_selectivity_unobserved_veto_enable', '--egf_rps_rear_selectivity_unobserved_veto_enable'),
]
DATA_BONN = PROJECT_ROOT / 'data' / 'bonn'
TUM_CONTROL_ROOT = PROJECT_ROOT / 'output' / 's2_stage' / '72_local_geometric_conflict_resolution_semantic' / 'tum_oracle' / 'oracle'
GHOST_THRESH = 0.08
BG_THRESH = 0.05


def load_summary(root: Path, family: str, sequence: str) -> dict:
    sub = 'tum_oracle/oracle' if family == 'tum' else 'bonn_slam/slam'
    return json.loads((root / sub / sequence / 'egf' / 'summary.json').read_text(encoding='utf-8'))


def load_feature_rows(root: Path, sequence: str) -> List[dict]:
    path = root / 'bonn_slam' / 'slam' / sequence / 'egf' / 'rear_surface_features.csv'
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open('r', encoding='utf-8')))
    out: List[dict] = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            if value is None or value == '':
                parsed[key] = 0.0
            else:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
        out.append(parsed)
    return out


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

    stats = {
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
            _, idx, dist2 = bg_tree.search_knn_vector_3d(point.astype(float), 1)
            if idx and dist2 and float(np.sqrt(dist2[0])) < BG_THRESH:
                label = 'true_background'
        stats[label]['count'] += 1.0
        for key in FEATURE_KEYS:
            stats[label][f'{key}_sum'] += float(row.get(key, 0.0))
    for bucket in stats.values():
        count = max(1.0, bucket['count'])
        for key in FEATURE_KEYS:
            bucket[f'{key}_mean'] = float(bucket[f'{key}_sum'] / count)
    return stats


def apply_ray_args(cmd: List[str], spec: dict) -> List[str]:
    out = list(cmd)
    for key, flag in RAY_ARG_PAIRS:
        if key in spec:
            out = set_arg(out, flag, str(spec[key]))
    for key, flag in RAY_BOOL_PAIRS:
        if bool(spec.get(key, False)):
            out = ensure_flag(out, flag)
        else:
            out = remove_flag(out, flag)
    return out


def enrich_row(root: Path, row: dict, *, frames: int, stride: int, max_points: int, seed: int) -> tuple[dict, dict]:
    row = dict(row)
    fields = [
        'bonn_extract_rear_selected_sum',
        'bonn_extract_score_ready_sum',
        'bonn_extract_support_protected_sum',
        'bonn_extract_fail_score_sum',
        'bonn_rear_selectivity_pre_sum',
        'bonn_rear_selectivity_kept_sum',
        'bonn_rear_selectivity_drop_sum',
        'bonn_rear_selectivity_topk_drop_sum',
        'bonn_rear_selectivity_penetration_sum',
        'bonn_rear_selectivity_penetration_free_span_sum',
        'bonn_rear_selectivity_observation_count_sum',
        'bonn_rear_selectivity_observation_support_sum',
        'bonn_rear_selectivity_static_coherence_sum',
        'bonn_rear_selectivity_pre_penetration_sum',
        'bonn_rear_selectivity_pre_penetration_free_span_sum',
        'bonn_rear_selectivity_pre_observation_count_sum',
        'bonn_rear_selectivity_pre_observation_support_sum',
        'bonn_rear_selectivity_pre_static_coherence_sum',
    ]
    for key in fields:
        row[key] = 0.0

    class_stats = {
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
        row['bonn_rear_selectivity_penetration_sum'] += float(summary.get('rear_selectivity_penetration_sum', 0.0))
        row['bonn_rear_selectivity_penetration_free_span_sum'] += float(summary.get('rear_selectivity_penetration_free_span_sum', 0.0))
        row['bonn_rear_selectivity_observation_count_sum'] += float(summary.get('rear_selectivity_observation_count_sum', 0.0))
        row['bonn_rear_selectivity_observation_support_sum'] += float(summary.get('rear_selectivity_observation_support_sum', 0.0))
        row['bonn_rear_selectivity_static_coherence_sum'] += float(summary.get('rear_selectivity_static_coherence_sum', 0.0))
        row['bonn_rear_selectivity_pre_penetration_sum'] += float(summary.get('rear_selectivity_pre_penetration_sum', 0.0))
        row['bonn_rear_selectivity_pre_penetration_free_span_sum'] += float(summary.get('rear_selectivity_pre_penetration_free_span_sum', 0.0))
        row['bonn_rear_selectivity_pre_observation_count_sum'] += float(summary.get('rear_selectivity_pre_observation_count_sum', 0.0))
        row['bonn_rear_selectivity_pre_observation_support_sum'] += float(summary.get('rear_selectivity_pre_observation_support_sum', 0.0))
        row['bonn_rear_selectivity_pre_static_coherence_sum'] += float(summary.get('rear_selectivity_pre_static_coherence_sum', 0.0))

        feature_rows = load_feature_rows(root, seq)
        seq_class = classify_feature_rows(seq, feature_rows, frames=frames, stride=stride, max_points=max_points, seed=seed)
        for label in class_stats:
            class_stats[label]['count'] += seq_class[label]['count']
            for key in FEATURE_KEYS:
                class_stats[label][f'{key}_sum'] += seq_class[label][f'{key}_sum']

    kept = max(1.0, row['bonn_rear_selectivity_kept_sum'])
    pre = max(1.0, row['bonn_rear_selectivity_pre_sum'])
    row['bonn_penetration_mean'] = row['bonn_rear_selectivity_penetration_sum'] / kept
    row['bonn_penetration_free_span_mean'] = row['bonn_rear_selectivity_penetration_free_span_sum'] / kept
    row['bonn_observation_count_mean'] = row['bonn_rear_selectivity_observation_count_sum'] / kept
    row['bonn_observation_support_mean'] = row['bonn_rear_selectivity_observation_support_sum'] / kept
    row['bonn_static_coherence_mean'] = row['bonn_rear_selectivity_static_coherence_sum'] / kept
    row['bonn_pre_penetration_mean'] = row['bonn_rear_selectivity_pre_penetration_sum'] / pre
    row['bonn_pre_penetration_free_span_mean'] = row['bonn_rear_selectivity_pre_penetration_free_span_sum'] / pre
    row['bonn_pre_observation_count_mean'] = row['bonn_rear_selectivity_pre_observation_count_sum'] / pre
    row['bonn_pre_observation_support_mean'] = row['bonn_rear_selectivity_pre_observation_support_sum'] / pre
    row['bonn_pre_static_coherence_mean'] = row['bonn_rear_selectivity_pre_static_coherence_sum'] / pre
    row['bonn_noise_ratio'] = row['bonn_rear_hole_or_noise_sum'] / max(1.0, row['bonn_rear_points_sum'])

    for label, prefix in [('true_background', 'tb'), ('hole_or_noise', 'noise'), ('ghost_region', 'ghost')]:
        count = max(1.0, class_stats[label]['count'])
        row[f'bonn_{prefix}_count'] = class_stats[label]['count']
        for key in FEATURE_KEYS:
            row[f'bonn_{prefix}_{key}_mean'] = class_stats[label][f'{key}_sum'] / count
    return row, class_stats


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '72_local_geometric_conflict_resolution')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        tb_ok = row['bonn_rear_true_background_sum'] >= 6.0
        ghost_ok = row['bonn_rear_ghost_sum'] <= 25.0
        metric_ok = row['bonn_ghost_reduction_vs_tsdf'] >= 22.0
        noise_ok = row['bonn_noise_ratio'] <= 0.75
        row['decision'] = 'iterate' if tb_ok and ghost_ok and metric_ok and noise_ok else 'abandon'
    candidates = [r for r in rows if r['decision'] == 'iterate']
    if candidates:
        best = max(
            candidates,
            key=lambda r: (
                r['bonn_rear_true_background_sum'],
                -r['bonn_rear_hole_or_noise_sum'],
                -r['bonn_rear_ghost_sum'],
                r['bonn_ghost_reduction_vs_tsdf'],
            ),
        )
        for row in candidates:
            if row is not best:
                row['decision'] = 'abandon'


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# S2 ray consistency compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_noise_ratio | bonn_penetration_mean | bonn_observation_support_mean | bonn_static_coherence_mean | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['bonn_noise_ratio']:.3f} | {row['bonn_penetration_mean']:.3f} | {row['bonn_observation_support_mean']:.3f} | {row['bonn_static_coherence_mean']:.3f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution(rows: List[dict], payloads: Dict[str, dict], path_md: Path) -> None:
    lines = [
        '# S2 ray consistency distribution report',
        '',
        '日期：`2026-03-10`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | noise_ratio | penetration_mean | thickness_proxy_mean | observation_support_mean | static_coherence_mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['bonn_noise_ratio']:.3f} | {row['bonn_penetration_mean']:.3f} | {row['bonn_penetration_free_span_mean']:.3f} | {row['bonn_observation_support_mean']:.3f} | {row['bonn_static_coherence_mean']:.3f} |"
        )
    lines += [
        '',
        '| variant | tb_penetration | noise_penetration | tb_observation | noise_observation | tb_static_coherence | noise_static_coherence |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_tb_penetration_score_mean']:.3f} | {row['bonn_noise_penetration_score_mean']:.3f} | {row['bonn_tb_observation_support_mean']:.3f} | {row['bonn_noise_observation_support_mean']:.3f} | {row['bonn_tb_static_coherence_mean']:.3f} | {row['bonn_noise_static_coherence_mean']:.3f} |"
        )
    lines += ['', '重点检查：TB 是否达到 `>=6`；Noise 占比是否降到 `< 0.75`；`ghost_reduction_vs_tsdf` 是否保持 `>=22%`。', '', '注：`thickness_proxy_mean` 使用 `penetration_free_span_mean`，表示 rear 点前方可追溯空洞/穿透长度的平均值。']
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '72_local_geometric_conflict_resolution')
    best_tb = max(rows[1:], key=lambda r: (r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum'], -r['bonn_rear_hole_or_noise_sum']))
    best_ghost = min(rows[1:], key=lambda r: (r['bonn_rear_ghost_sum'], -r['bonn_ghost_reduction_vs_tsdf'], r['bonn_rear_hole_or_noise_sum']))
    best_metric = max(rows[1:], key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], -r['bonn_rear_ghost_sum'], -r['bonn_rear_hole_or_noise_sum']))
    lines = [
        '# S2 ray consistency analysis',
        '',
        '日期：`2026-03-10`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`output/tmp/legacy_artifacts_placeholder`',
        '',
        '## 1. 新特征是否摆脱饱和',
        f"- 控制组 `72`: penetration=`{control['bonn_penetration_mean']:.3f}`, observation=`{control['bonn_observation_support_mean']:.3f}`, static_coherence=`{control['bonn_static_coherence_mean']:.3f}`",
        f"- `best TB` `{best_tb['variant']}`: penetration=`{best_tb['bonn_penetration_mean']:.3f}`, observation=`{best_tb['bonn_observation_support_mean']:.3f}`, static_coherence=`{best_tb['bonn_static_coherence_mean']:.3f}`",
        f"- 控制组 TB vs Noise: penetration=`{control['bonn_tb_penetration_score_mean']:.3f}` vs `{control['bonn_noise_penetration_score_mean']:.3f}`, observation=`{control['bonn_tb_observation_support_mean']:.3f}` vs `{control['bonn_noise_observation_support_mean']:.3f}`, coherence=`{control['bonn_tb_static_coherence_mean']:.3f}` vs `{control['bonn_noise_static_coherence_mean']:.3f}`",
        f"- 厚度/空洞跨度代理 `penetration_free_span_mean`: 控制组 `72 = {control['bonn_penetration_free_span_mean']:.3f} m`，`best TB = {best_tb['bonn_penetration_free_span_mean']:.3f} m`，`best metric = {best_metric['bonn_penetration_free_span_mean']:.3f} m`",
    ]
    if best_tb['bonn_tb_penetration_score_mean'] > best_tb['bonn_noise_penetration_score_mean'] or best_tb['bonn_tb_observation_support_mean'] > best_tb['bonn_noise_observation_support_mean']:
        lines.append('- 新特征已经不再是全零/全饱和，并对 TB 与 Noise 给出了可见分离。')
    else:
        lines.append('- 新特征虽然非零，但对 TB 与 Noise 的区分仍然很弱。')
    lines += [
        '',
        '## 2. Noise 与 Ghost 是否被削减',
        f"- 控制组 `72`: rear=`{control['bonn_rear_points_sum']:.0f}`, TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, Noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}` (`noise_ratio={control['bonn_noise_ratio']:.3f}`), `ghost_reduction_vs_tsdf={control['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
        f"- `best ghost suppression` `{best_ghost['variant']}`: rear=`{best_ghost['bonn_rear_points_sum']:.0f}`, TB=`{best_ghost['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best_ghost['bonn_rear_ghost_sum']:.0f}`, Noise=`{best_ghost['bonn_rear_hole_or_noise_sum']:.0f}` (`noise_ratio={best_ghost['bonn_noise_ratio']:.3f}`), `ghost_reduction_vs_tsdf={best_ghost['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
        f"- `best metric` `{best_metric['variant']}`: rear=`{best_metric['bonn_rear_points_sum']:.0f}`, TB=`{best_metric['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best_metric['bonn_rear_ghost_sum']:.0f}`, Noise=`{best_metric['bonn_rear_hole_or_noise_sum']:.0f}` (`noise_ratio={best_metric['bonn_noise_ratio']:.3f}`), `ghost_reduction_vs_tsdf={best_metric['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
    ]
    lines += [
        '',
        '## 3. 诊断结论',
        '- 若 `penetration_score(TB)` 高于 `penetration_score(Noise)` 但 TB 总量仍不上升，说明当前候选生成本身仍缺 TB 密度，筛选只能做有限净化。',
        '- 若 `observation_support` 成功拉开 TB 与 Noise，但 ghost 仍居高，说明 synthetic-only 噪声被清掉了，剩余误差主要来自已观测动态残留。',
        '- 若 `static_coherence` 对 TB 与 Noise 可分，但 `noise_ratio` 仍高，说明当前孤立噪声并不完全缺邻域结构，后续还需要更强的 front/back 几何冲突特征。',
        '',
        '## 4. 阶段判断',
        '- 若未达到 `TB >= 6`、`noise_ratio < 0.75`、`Ghost <= 25` 与 `ghost_reduction_vs_tsdf >= 22%`，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 ray consistency runner.')
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
        bonn_rps_rear_selectivity_front_score_weight=0.26,
        bonn_rps_rear_selectivity_competition_weight=0.38,
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
        bonn_rps_rear_selectivity_topk=85,
        bonn_rps_rear_selectivity_rank_risk_weight=0.45,
        bonn_rps_rear_selectivity_penetration_free_ref=0.06,
        bonn_rps_rear_selectivity_penetration_max_steps=10,
        bonn_rps_rear_selectivity_observation_count_ref=6.0,
        bonn_rps_rear_selectivity_static_coherence_radius_cells=1,
        bonn_rps_rear_selectivity_static_coherence_ref=0.35,
        bonn_rps_rear_selectivity_static_neighbor_min_weight=0.20,
        bonn_rps_rear_selectivity_static_neighbor_dyn_max=0.35,
    )

    control_spec = dict(
        base_spec,
        name='72_local_geometric_conflict_resolution',
        output_name='72_local_geometric_conflict_resolution_ray_control',
        bonn_rps_rear_selectivity_rear_score_weight=0.18,
        bonn_rps_rear_selectivity_front_score_weight=0.26,
        bonn_rps_rear_selectivity_competition_weight=0.38,
        bonn_rps_rear_selectivity_competition_alpha=0.90,
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
        bonn_rps_rear_selectivity_topk=85,
    )

    variants = [
        control_spec,
        dict(
            control_spec,
            name='80_ray_penetration_consistency',
            bonn_rps_rear_selectivity_penetration_weight=0.55,
            bonn_rps_rear_selectivity_penetration_floor=0.18,
            bonn_rps_rear_selectivity_penetration_risk_weight=0.20,
            bonn_rps_rear_selectivity_penetration_free_ref=0.07,
            bonn_rps_rear_selectivity_penetration_max_steps=12,
            bonn_rps_rear_selectivity_front_score_weight=0.22,
            bonn_rps_rear_selectivity_competition_weight=0.34,
            bonn_rps_rear_selectivity_topk=92,
        ),
        dict(
            control_spec,
            name='81_unobserved_space_veto',
            bonn_rps_rear_selectivity_observation_weight=0.65,
            bonn_rps_rear_selectivity_observation_floor=0.28,
            bonn_rps_rear_selectivity_observation_risk_weight=0.36,
            bonn_rps_rear_selectivity_observation_count_ref=5.0,
            bonn_rps_rear_selectivity_observation_min_count=1.0,
            bonn_rps_rear_selectivity_unobserved_veto_enable=True,
            bonn_rps_rear_selectivity_topk=88,
        ),
        dict(
            control_spec,
            name='82_static_neighborhood_coherence',
            bonn_rps_rear_selectivity_penetration_weight=0.28,
            bonn_rps_rear_selectivity_penetration_floor=0.10,
            bonn_rps_rear_selectivity_observation_weight=0.35,
            bonn_rps_rear_selectivity_observation_floor=0.18,
            bonn_rps_rear_selectivity_observation_min_count=1.0,
            bonn_rps_rear_selectivity_unobserved_veto_enable=True,
            bonn_rps_rear_selectivity_static_coherence_weight=0.72,
            bonn_rps_rear_selectivity_static_coherence_floor=0.24,
            bonn_rps_rear_selectivity_static_coherence_relief_weight=0.18,
            bonn_rps_rear_selectivity_static_coherence_radius_cells=1,
            bonn_rps_rear_selectivity_static_coherence_ref=0.30,
            bonn_rps_rear_selectivity_topk=95,
        ),
    ]

    rows: List[dict] = []
    payloads: Dict[str, dict] = {}
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec)
        tum_cmd = apply_ray_args(tum_cmd, spec)
        bonn_cmd = apply_ray_args(bonn_cmd, spec)
        if root.exists():
            shutil.rmtree(root)
        run(bonn_cmd)
        shutil.copytree(TUM_CONTROL_ROOT, root / 'tum_oracle' / 'oracle', dirs_exist_ok=True)
        row, tum_details, bonn_payload = summarize_variant(
            root,
            str(spec['name']),
            frames=args.frames,
            stride=args.stride,
            max_points=args.max_points_per_frame,
            seed=args.seed,
        )
        row, class_stats = enrich_row(root, row, frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame, seed=args.seed)
        rows.append(row)
        payloads[str(spec['name'])] = {'tum': tum_details, 'bonn': bonn_payload, 'class_stats': class_stats}

    decide(rows)
    out_dir = PROJECT_ROOT / 'output' / 's2'
    write_compare(rows, out_dir / 'S2_RPS_RAY_CONSISTENCY_COMPARE.csv', out_dir / 'S2_RPS_RAY_CONSISTENCY_COMPARE.md')
    write_distribution(rows, payloads, out_dir / 'S2_RPS_RAY_CONSISTENCY_DISTRIBUTION.md')
    write_analysis(rows, out_dir / 'S2_RPS_RAY_CONSISTENCY_ANALYSIS.md')
    print('[done]', out_dir / 'S2_RPS_RAY_CONSISTENCY_COMPARE.csv')


if __name__ == '__main__':
    main()
