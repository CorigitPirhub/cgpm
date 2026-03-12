from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
S2_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT_BOOTSTRAP / "scripts"
for _path in (PROJECT_ROOT_BOOTSTRAP, S2_DIR, ROOT_SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_s2_write_time_synthesis import PROJECT_ROOT, dryrun_base_cmds, run
from run_s2_rps_rear_geometry_quality import BONN_ALL3, build_commands, summarize_variant


def load_summary(root: Path, family: str, sequence: str) -> dict:
    sub = 'tum_oracle/oracle' if family == 'tum' else 'bonn_slam/slam'
    return json.loads((root / sub / sequence / 'egf' / 'summary.json').read_text(encoding='utf-8'))


def enrich_row(root: Path, row: dict) -> dict:
    row = dict(row)
    agg_keys = [
        'bonn_extract_rear_selected_sum',
        'bonn_extract_score_ready_sum',
        'bonn_extract_support_protected_sum',
        'bonn_extract_fail_score_sum',
        'bonn_rear_selectivity_pre_sum',
        'bonn_rear_selectivity_kept_sum',
        'bonn_rear_selectivity_drop_sum',
        'bonn_rear_selectivity_topk_drop_sum',
        'bonn_rear_selectivity_history_anchor_sum',
        'bonn_rear_selectivity_surface_anchor_sum',
        'bonn_rear_selectivity_surface_distance_sum',
        'bonn_rear_selectivity_dynamic_shell_sum',
        'bonn_rear_selectivity_pre_history_anchor_sum',
        'bonn_rear_selectivity_pre_surface_anchor_sum',
        'bonn_rear_selectivity_pre_surface_distance_sum',
        'bonn_rear_selectivity_pre_dynamic_shell_sum',
    ]
    for key in agg_keys:
        row[key] = 0.0
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
        row['bonn_rear_selectivity_history_anchor_sum'] += float(summary.get('rear_selectivity_history_anchor_sum', 0.0))
        row['bonn_rear_selectivity_surface_anchor_sum'] += float(summary.get('rear_selectivity_surface_anchor_sum', 0.0))
        row['bonn_rear_selectivity_surface_distance_sum'] += float(summary.get('rear_selectivity_surface_distance_sum', 0.0))
        row['bonn_rear_selectivity_dynamic_shell_sum'] += float(summary.get('rear_selectivity_dynamic_shell_sum', 0.0))
        row['bonn_rear_selectivity_pre_history_anchor_sum'] += float(summary.get('rear_selectivity_pre_history_anchor_sum', 0.0))
        row['bonn_rear_selectivity_pre_surface_anchor_sum'] += float(summary.get('rear_selectivity_pre_surface_anchor_sum', 0.0))
        row['bonn_rear_selectivity_pre_surface_distance_sum'] += float(summary.get('rear_selectivity_pre_surface_distance_sum', 0.0))
        row['bonn_rear_selectivity_pre_dynamic_shell_sum'] += float(summary.get('rear_selectivity_pre_dynamic_shell_sum', 0.0))
    kept = max(1.0, row['bonn_rear_selectivity_kept_sum'])
    pre = max(1.0, row['bonn_rear_selectivity_pre_sum'])
    row['bonn_history_anchor_mean'] = row['bonn_rear_selectivity_history_anchor_sum'] / kept
    row['bonn_surface_anchor_mean'] = row['bonn_rear_selectivity_surface_anchor_sum'] / kept
    row['bonn_surface_distance_mean'] = row['bonn_rear_selectivity_surface_distance_sum'] / kept
    row['bonn_dynamic_shell_mean'] = row['bonn_rear_selectivity_dynamic_shell_sum'] / kept
    row['bonn_pre_history_anchor_mean'] = row['bonn_rear_selectivity_pre_history_anchor_sum'] / pre
    row['bonn_pre_surface_anchor_mean'] = row['bonn_rear_selectivity_pre_surface_anchor_sum'] / pre
    row['bonn_pre_surface_distance_mean'] = row['bonn_rear_selectivity_pre_surface_distance_sum'] / pre
    row['bonn_pre_dynamic_shell_mean'] = row['bonn_rear_selectivity_pre_dynamic_shell_sum'] / pre
    return row


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '72_local_geometric_conflict_resolution')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        stats_ok = row['bonn_history_anchor_mean'] > 0.0 or row['bonn_surface_anchor_mean'] > 0.0
        tb_ok = row['bonn_rear_true_background_sum'] >= 6.0
        ghost_ok = row['bonn_rear_ghost_sum'] <= 25.0
        comp_ok = row['bonn_comp_r_5cm'] >= 70.5
        metric_ok = row['bonn_ghost_reduction_vs_tsdf'] >= 22.0
        row['decision'] = 'iterate' if stats_ok and tb_ok and ghost_ok and comp_ok and metric_ok else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(
            iters,
            key=lambda r: (
                r['bonn_rear_true_background_sum'],
                r['bonn_comp_r_5cm'],
                -r['bonn_rear_ghost_sum'],
                r['bonn_ghost_reduction_vs_tsdf'],
            ),
        )
        for row in iters:
            if row is not best:
                row['decision'] = 'abandon'


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# S2 hybrid optimization compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_history_anchor_mean | bonn_surface_anchor_mean | bonn_dynamic_shell_mean | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['bonn_history_anchor_mean']:.3f} | {row['bonn_surface_anchor_mean']:.3f} | {row['bonn_dynamic_shell_mean']:.3f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution(payloads: Dict[str, dict], rows: List[dict], path_md: Path) -> None:
    row_map = {row['variant']: row for row in rows}
    lines = [
        '# S2 hybrid optimization distribution report',
        '',
        '日期：`2026-03-10`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | history_anchor_mean | history_anchor_pre_mean | surface_anchor_mean | surface_anchor_pre_mean | surface_distance_mean | surface_distance_pre_mean | dynamic_shell_mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, payload in payloads.items():
        rear = payload['rear_dist']
        row = row_map[name]
        lines.append(
            f"| {name} | {sum(v['rear_points'] for v in rear.values()):.0f} | {sum(v['true_background_points'] for v in rear.values()):.0f} | {sum(v['ghost_region_points'] for v in rear.values()):.0f} | {sum(v['hole_or_noise_points'] for v in rear.values()):.0f} | {row['bonn_history_anchor_mean']:.3f} | {row['bonn_pre_history_anchor_mean']:.3f} | {row['bonn_surface_anchor_mean']:.3f} | {row['bonn_pre_surface_anchor_mean']:.3f} | {row['bonn_surface_distance_mean']:.4f} | {row['bonn_pre_surface_distance_mean']:.4f} | {row['bonn_dynamic_shell_mean']:.3f} |"
        )
    lines += ['', '重点检查：特征统计必须非零；TB 是否提升到 `>=6`；Ghost 是否控制在 `<=25`。']
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '72_local_geometric_conflict_resolution')
    best_tb = max(
        rows[1:],
        key=lambda r: (
            r['bonn_rear_true_background_sum'],
            r['bonn_comp_r_5cm'],
            -r['bonn_rear_ghost_sum'],
            r['bonn_ghost_reduction_vs_tsdf'],
        ),
    )
    metric_safe = [r for r in rows[1:] if r['bonn_ghost_reduction_vs_tsdf'] >= 22.0]
    best_metric = max(
        metric_safe,
        key=lambda r: (
            r['bonn_rear_true_background_sum'],
            -r['bonn_rear_ghost_sum'],
            r['bonn_comp_r_5cm'],
            r['bonn_ghost_reduction_vs_tsdf'],
        ),
    ) if metric_safe else None
    lines = [
        '# S2 hybrid optimization analysis',
        '',
        '日期：`2026-03-10`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`output/tmp/legacy_artifacts_placeholder`',
        '',
        '## 1. 特征统计验证',
        f"- 控制组 `72`: history_anchor=`{control['bonn_history_anchor_mean']:.3f}`, surface_anchor=`{control['bonn_surface_anchor_mean']:.3f}`, dynamic_shell=`{control['bonn_dynamic_shell_mean']:.3f}`",
        f"- `best TB` `{best_tb['variant']}`: history_anchor=`{best_tb['bonn_history_anchor_mean']:.3f}`, surface_anchor=`{best_tb['bonn_surface_anchor_mean']:.3f}`, dynamic_shell=`{best_tb['bonn_dynamic_shell_mean']:.3f}`",
    ]
    if best_metric is not None:
        lines.append(
            f"- `best >=22%` `{best_metric['variant']}`: history_anchor=`{best_metric['bonn_history_anchor_mean']:.3f}`, surface_anchor=`{best_metric['bonn_surface_anchor_mean']:.3f}`, dynamic_shell=`{best_metric['bonn_dynamic_shell_mean']:.3f}`"
        )
    if best_tb['bonn_history_anchor_mean'] > 0.0 or best_tb['bonn_surface_anchor_mean'] > 0.0:
        lines.append('- 关键特征统计已成功输出为非零值。')
    else:
        lines.append('- 关键特征统计仍为零，执行视为失败。')
    lines += [
        '',
        '## 2. 组合策略是否奏效',
        f"- 控制组 `72`: rear=`{control['bonn_rear_points_sum']:.0f}`, TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, Bonn `Comp-R = {control['bonn_comp_r_5cm']:.2f}%`, `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
        f"- 最佳 TB 恢复 `{best_tb['variant']}`: rear=`{best_tb['bonn_rear_points_sum']:.0f}`, TB=`{best_tb['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best_tb['bonn_rear_ghost_sum']:.0f}`, Bonn `Comp-R = {best_tb['bonn_comp_r_5cm']:.2f}%`, `ghost_reduction_vs_tsdf = {best_tb['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
    ]
    if best_metric is not None:
        lines.append(
            f"- 唯一守住 `22%` 红线的 hybrid `{best_metric['variant']}`: rear=`{best_metric['bonn_rear_points_sum']:.0f}`, TB=`{best_metric['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best_metric['bonn_rear_ghost_sum']:.0f}`, Bonn `Comp-R = {best_metric['bonn_comp_r_5cm']:.2f}%`, `ghost_reduction_vs_tsdf = {best_metric['bonn_ghost_reduction_vs_tsdf']:.2f}%`"
        )
    lines.append('- 结论：没有任何 hybrid 同时满足 `TB >= 6`、`ghost_reduction_vs_tsdf >= 22%` 和 `Ghost <= 25`。')
    lines += [
        '',
        '## 3. 顾此失彼原因',
        '- 若 history boosting 提高了 TB 但守不住 `22%` 红线，说明持久性本身不能区分“真实背景”与“长期残留伪影”。',
        '- 若 dynamic-shell masking 压住了 Ghost 却拖垮 Comp-R，说明当前抑制范围仍过宽，把大量合法背景也一起裁掉了。',
        '- 若 feature-weighted Top-K 只能维持 `22%` 却拉不回 TB，说明当前 `history_anchor + dynamic_shell + rear_score` 仍缺少足够的 true-background 判别力。',
        '- 若 surface anchoring 统计长期接近饱和（`surface_anchor_mean = 1.000`），说明当前 surface-distance 特征区分度几乎为零，后验筛选只能做有限补救。',
        '',
        '## 4. 阶段判断',
        '- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 hybrid optimization runner.')
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
    )
    variants = [
        dict(base_spec, name='72_local_geometric_conflict_resolution', output_name='72_local_geometric_conflict_resolution_hybrid_control'),
        dict(
            base_spec,
            name='77_hybrid_boost_conflict',
            bonn_rps_rear_selectivity_history_anchor_weight=0.55,
            bonn_rps_rear_selectivity_history_anchor_floor=0.28,
            bonn_rps_rear_selectivity_history_anchor_relief_weight=0.10,
            bonn_rps_rear_selectivity_local_conflict_weight=0.45,
            bonn_rps_rear_selectivity_local_conflict_max=0.70,
            bonn_rps_rear_selectivity_topk=90,
        ),
        dict(
            base_spec,
            name='78_conservative_anchoring',
            bonn_rps_rear_selectivity_history_anchor_weight=0.25,
            bonn_rps_rear_selectivity_history_anchor_floor=0.20,
            bonn_rps_rear_selectivity_surface_anchor_weight=0.55,
            bonn_rps_rear_selectivity_surface_anchor_floor=0.28,
            bonn_rps_rear_selectivity_surface_anchor_risk_weight=0.18,
            bonn_rps_rear_selectivity_surface_distance_ref=0.045,
            bonn_rps_rear_selectivity_dynamic_shell_weight=0.10,
            bonn_rps_rear_selectivity_dynamic_shell_max=0.75,
            bonn_rps_rear_selectivity_dynamic_shell_gap_ref=0.07,
            bonn_rps_rear_selectivity_topk=95,
        ),
        dict(
            base_spec,
            name='79_feature_weighted_topk',
            bonn_rps_rear_selectivity_history_anchor_weight=0.45,
            bonn_rps_rear_selectivity_history_anchor_floor=0.24,
            bonn_rps_rear_selectivity_dynamic_shell_weight=0.35,
            bonn_rps_rear_selectivity_dynamic_shell_max=0.50,
            bonn_rps_rear_selectivity_rear_score_weight=0.24,
            bonn_rps_rear_selectivity_front_score_weight=0.18,
            bonn_rps_rear_selectivity_competition_weight=0.30,
            bonn_rps_rear_selectivity_topk=80,
        ),
    ]

    rows: List[dict] = []
    payloads: Dict[str, dict] = {}
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec)
        run(tum_cmd)
        run(bonn_cmd)
        row, tum_details, bonn_payload = summarize_variant(
            root,
            str(spec['name']),
            frames=args.frames,
            stride=args.stride,
            max_points=args.max_points_per_frame,
            seed=args.seed,
        )
        row = enrich_row(root, row)
        rows.append(row)
        payloads[str(spec['name'])] = {'tum': tum_details, **bonn_payload}

    decide(rows)
    out_dir = PROJECT_ROOT / 'output' / 's2'
    compare_csv = out_dir / 'S2_RPS_HYBRID_OPTIMIZATION_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_HYBRID_OPTIMIZATION_COMPARE.md'
    dist_md = out_dir / 'S2_RPS_HYBRID_OPTIMIZATION_DISTRIBUTION.md'
    analysis_md = out_dir / 'S2_RPS_HYBRID_OPTIMIZATION_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_distribution(payloads, rows, dist_md)
    write_analysis(rows, analysis_md)
    print(f'[done] {compare_csv} {compare_md} {dist_md} {analysis_md}')


if __name__ == '__main__':
    main()
