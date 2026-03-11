from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from run_s2_write_time_synthesis import PROJECT_ROOT, dryrun_base_cmds, run
from run_s2_rps_rear_geometry_quality import BONN_ALL3, build_commands, summarize_variant


def load_summary(root: Path, family: str, sequence: str) -> dict:
    sub = 'tum_oracle/oracle' if family == 'tum' else 'bonn_slam/slam'
    return json.loads((root / sub / sequence / 'egf' / 'summary.json').read_text(encoding='utf-8'))


def enrich_row(root: Path, row: dict) -> dict:
    row = dict(row)
    keys = [
        'bonn_extract_rear_selected_sum',
        'bonn_extract_score_ready_sum',
        'bonn_extract_support_protected_sum',
        'bonn_extract_fail_score_sum',
        'bonn_rear_selectivity_pre_sum',
        'bonn_rear_selectivity_kept_sum',
        'bonn_rear_selectivity_drop_sum',
        'bonn_rear_selectivity_topk_drop_sum',
        'bonn_rear_selectivity_score_sum',
        'bonn_rear_selectivity_risk_sum',
        'bonn_rear_selectivity_front_score_sum',
        'bonn_rear_selectivity_rear_score_sum',
        'bonn_rear_selectivity_gap_sum',
        'bonn_rear_selectivity_competition_sum',
    ]
    for key in keys:
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
        row['bonn_rear_selectivity_score_sum'] += float(summary.get('rear_selectivity_score_sum', 0.0))
        row['bonn_rear_selectivity_risk_sum'] += float(summary.get('rear_selectivity_risk_sum', 0.0))
        row['bonn_rear_selectivity_front_score_sum'] += float(summary.get('rear_selectivity_front_score_sum', 0.0))
        row['bonn_rear_selectivity_rear_score_sum'] += float(summary.get('rear_selectivity_rear_score_sum', 0.0))
        row['bonn_rear_selectivity_gap_sum'] += float(summary.get('rear_selectivity_gap_sum', 0.0))
        row['bonn_rear_selectivity_competition_sum'] += float(summary.get('rear_selectivity_competition_sum', 0.0))
    denom = max(1.0, row['bonn_rear_selectivity_kept_sum'])
    row['bonn_front_score_mean'] = row['bonn_rear_selectivity_front_score_sum'] / denom
    row['bonn_rear_score_mean'] = row['bonn_rear_selectivity_rear_score_sum'] / denom
    row['bonn_rear_gap_mean'] = row['bonn_rear_selectivity_gap_sum'] / denom
    row['bonn_competition_mean'] = row['bonn_rear_selectivity_competition_sum'] / denom
    return row


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '67_topk_selective_generation')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        ghost_ok = row['bonn_rear_ghost_sum'] <= 15.0
        tb_ok = row['bonn_rear_true_background_sum'] >= 8.0
        metric_ok = row['bonn_ghost_reduction_vs_tsdf'] >= 22.0
        no_regress = row['tum_comp_r_5cm'] >= control['tum_comp_r_5cm'] - 0.50
        row['decision'] = 'iterate' if ghost_ok and tb_ok and metric_ok and no_regress else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(
            iters,
            key=lambda r: (
                r['bonn_ghost_reduction_vs_tsdf'],
                -r['bonn_rear_ghost_sum'],
                r['bonn_rear_true_background_sum'],
                r['bonn_competition_mean'],
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
        '# S2 discriminative fusion compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_front_score_mean | bonn_rear_score_mean | bonn_rear_gap_mean | bonn_competition_mean | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_front_score_mean']:.3f} | {row['bonn_rear_score_mean']:.3f} | {row['bonn_rear_gap_mean']:.4f} | {row['bonn_competition_mean']:.3f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution(payloads: Dict[str, dict], rows: List[dict], path_md: Path) -> None:
    row_map = {row['variant']: row for row in rows}
    lines = [
        '# S2 discriminative fusion distribution report',
        '',
        '日期：`2026-03-10`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | front_score_mean | rear_score_mean | rear_gap_mean | competition_mean | selectivity_topk_drop_sum |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, payload in payloads.items():
        rear = payload['rear_dist']
        row = row_map[name]
        lines.append(
            f"| {name} | {sum(v['rear_points'] for v in rear.values()):.0f} | {sum(v['true_background_points'] for v in rear.values()):.0f} | {sum(v['ghost_region_points'] for v in rear.values()):.0f} | {sum(v['hole_or_noise_points'] for v in rear.values()):.0f} | {row['bonn_front_score_mean']:.3f} | {row['bonn_rear_score_mean']:.3f} | {row['bonn_rear_gap_mean']:.4f} | {row['bonn_competition_mean']:.3f} | {row['bonn_rear_selectivity_topk_drop_sum']:.0f} |"
        )
    lines += ['', '重点检查：Ghost 是否压回 `<=15`，TB 是否恢复到 `>=8`，以及 `rear_score/front_score` 是否出现可分离均值。']
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '67_topk_selective_generation')
    valid_rows = [r for r in rows[1:] if r['bonn_rear_points_sum'] > 0.0]
    best_pool = valid_rows if valid_rows else rows[1:]
    best = max(
        best_pool,
        key=lambda r: (
            r['bonn_ghost_reduction_vs_tsdf'],
            -r['bonn_rear_ghost_sum'],
            r['bonn_rear_true_background_sum'],
            r['bonn_competition_mean'],
        ),
    )
    lines = [
        '# S2 discriminative fusion analysis',
        '',
        '日期：`2026-03-10`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`processes/s2/S2_RPS_DISCRIMINATIVE_FUSION_COMPARE.csv`',
        '',
        '## 1. 判别特征是否分离了 TB/Ghost',
        f"- 基组 `67`: rear=`{control['bonn_rear_points_sum']:.0f}`, TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, front=`{control['bonn_front_score_mean']:.3f}`, rear=`{control['bonn_rear_score_mean']:.3f}`, gap=`{control['bonn_rear_gap_mean']:.4f}`, comp=`{control['bonn_competition_mean']:.3f}`",
        f"- 本轮最佳候选 `{best['variant']}`: rear=`{best['bonn_rear_points_sum']:.0f}`, TB=`{best['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best['bonn_rear_ghost_sum']:.0f}`, front=`{best['bonn_front_score_mean']:.3f}`, rear=`{best['bonn_rear_score_mean']:.3f}`, gap=`{best['bonn_rear_gap_mean']:.4f}`, comp=`{best['bonn_competition_mean']:.3f}`",
    ]
    zero_rows = [r for r in rows[1:] if r['bonn_rear_points_sum'] <= 0.0]
    if zero_rows:
        lines.append(f"- 退化候选：`{', '.join(r['variant'] for r in zero_rows)}` 直接把 rear 裁到 `0`，其 ghost 指标不具备可用性。")
    if best['bonn_competition_mean'] > control['bonn_competition_mean'] + 1e-6:
        lines.append('- 判别融合提高了后前竞争均值。')
    else:
        lines.append('- 判别融合没有提高后前竞争均值。')
    if best['bonn_front_score_mean'] < control['bonn_front_score_mean'] - 1e-6:
        lines.append('- 平均 front_score 被压低。')
    else:
        lines.append('- 平均 front_score 没有被有效压低。')
    lines += [
        '',
        '## 2. 指标是否站稳 22%',
        f"- 基组 `67`: Bonn `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {control['bonn_comp_r_5cm']:.2f}%`",
        f"- 候选 `{best['variant']}`: Bonn `ghost_reduction_vs_tsdf = {best['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {best['bonn_comp_r_5cm']:.2f}%`",
    ]
    if best['bonn_ghost_reduction_vs_tsdf'] >= 22.0:
        lines.append('- ghost_reduction_vs_tsdf 站稳了 `22%`。')
    else:
        lines.append('- ghost_reduction_vs_tsdf 没有站稳 `22%`。')
    if best['bonn_rear_ghost_sum'] <= 15.0 and best['bonn_rear_true_background_sum'] >= 8.0:
        lines.append('- 同时满足了 `Ghost <= 15` 和 `TB >= 8`。')
    else:
        lines.append('- 仍未同时满足 `Ghost <= 15` 和 `TB >= 8`。')
    lines += [
        '',
        '## 3. 诊断结论',
        '- 若 `competition_mean` 提升而 Ghost 仍高，说明单纯 `rear-front` 竞争还不能覆盖动态残留的复杂模式。',
        '- 若 `rear_gap_mean` 收窄但 TB 下降，说明 gap 约束更像是在裁覆盖，而不是识别正确背景。',
        '- 若 `ghost_reduction_vs_tsdf` 继续保持在 `22%+`，但 Ghost Count 仍下不来，说明当前指标改善主要来自总量裁剪，而非真正的 TB/Ghost 分类正确率提升。',
        '',
        '## 4. 阶段判断',
        '- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 discriminative fusion runner.')
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
    )
    variants = [
        dict(
            base_spec,
            name='67_topk_selective_generation',
            output_name='67_topk_selective_generation_df_control',
            bonn_rps_rear_selectivity_enable=True,
            bonn_rps_rear_selectivity_support_weight=0.12,
            bonn_rps_rear_selectivity_history_weight=0.26,
            bonn_rps_rear_selectivity_static_weight=0.16,
            bonn_rps_rear_selectivity_geom_weight=0.26,
            bonn_rps_rear_selectivity_bridge_weight=0.10,
            bonn_rps_rear_selectivity_density_weight=0.10,
            bonn_rps_rear_selectivity_rear_score_weight=0.0,
            bonn_rps_rear_selectivity_front_score_weight=0.0,
            bonn_rps_rear_selectivity_competition_weight=0.0,
            bonn_rps_rear_selectivity_gap_weight=0.0,
            bonn_rps_rear_selectivity_sep_weight=0.0,
            bonn_rps_rear_selectivity_dyn_weight=0.28,
            bonn_rps_rear_selectivity_ghost_weight=0.20,
            bonn_rps_rear_selectivity_front_weight=0.18,
            bonn_rps_rear_selectivity_geom_risk_weight=0.26,
            bonn_rps_rear_selectivity_history_risk_weight=0.16,
            bonn_rps_rear_selectivity_density_risk_weight=0.12,
            bonn_rps_rear_selectivity_bridge_relief_weight=0.10,
            bonn_rps_rear_selectivity_static_relief_weight=0.08,
            bonn_rps_rear_selectivity_gap_risk_weight=0.0,
            bonn_rps_rear_selectivity_score_min=0.48,
            bonn_rps_rear_selectivity_risk_max=0.44,
            bonn_rps_rear_selectivity_geom_floor=0.54,
            bonn_rps_rear_selectivity_history_floor=0.38,
            bonn_rps_rear_selectivity_bridge_floor=0.14,
            bonn_rps_rear_selectivity_competition_floor=-1.0,
            bonn_rps_rear_selectivity_front_score_max=1.0,
            bonn_rps_rear_selectivity_gap_valid_min=0.0,
            bonn_rps_rear_selectivity_topk=80,
            bonn_rps_rear_selectivity_rank_risk_weight=0.70,
        ),
        dict(
            base_spec,
            name='68_rear_front_score_competition',
            bonn_rps_rear_selectivity_enable=True,
            bonn_rps_rear_selectivity_support_weight=0.10,
            bonn_rps_rear_selectivity_history_weight=0.18,
            bonn_rps_rear_selectivity_static_weight=0.12,
            bonn_rps_rear_selectivity_geom_weight=0.18,
            bonn_rps_rear_selectivity_bridge_weight=0.08,
            bonn_rps_rear_selectivity_density_weight=0.08,
            bonn_rps_rear_selectivity_rear_score_weight=0.22,
            bonn_rps_rear_selectivity_front_score_weight=0.34,
            bonn_rps_rear_selectivity_competition_weight=0.54,
            bonn_rps_rear_selectivity_competition_alpha=0.95,
            bonn_rps_rear_selectivity_gap_weight=0.06,
            bonn_rps_rear_selectivity_sep_weight=0.06,
            bonn_rps_rear_selectivity_dyn_weight=0.18,
            bonn_rps_rear_selectivity_ghost_weight=0.16,
            bonn_rps_rear_selectivity_front_weight=0.12,
            bonn_rps_rear_selectivity_geom_risk_weight=0.18,
            bonn_rps_rear_selectivity_history_risk_weight=0.12,
            bonn_rps_rear_selectivity_density_risk_weight=0.08,
            bonn_rps_rear_selectivity_bridge_relief_weight=0.08,
            bonn_rps_rear_selectivity_static_relief_weight=0.06,
            bonn_rps_rear_selectivity_gap_risk_weight=0.06,
            bonn_rps_rear_selectivity_score_min=0.42,
            bonn_rps_rear_selectivity_risk_max=0.55,
            bonn_rps_rear_selectivity_geom_floor=0.40,
            bonn_rps_rear_selectivity_history_floor=0.30,
            bonn_rps_rear_selectivity_bridge_floor=0.12,
            bonn_rps_rear_selectivity_competition_floor=0.04,
            bonn_rps_rear_selectivity_front_score_max=0.62,
            bonn_rps_rear_selectivity_gap_valid_min=0.10,
            bonn_rps_rear_selectivity_topk=70,
            bonn_rps_rear_selectivity_rank_risk_weight=0.50,
        ),
        dict(
            base_spec,
            name='69_depth_gap_validation',
            bonn_rps_rear_selectivity_enable=True,
            bonn_rps_rear_selectivity_support_weight=0.10,
            bonn_rps_rear_selectivity_history_weight=0.18,
            bonn_rps_rear_selectivity_static_weight=0.14,
            bonn_rps_rear_selectivity_geom_weight=0.22,
            bonn_rps_rear_selectivity_bridge_weight=0.08,
            bonn_rps_rear_selectivity_density_weight=0.08,
            bonn_rps_rear_selectivity_rear_score_weight=0.14,
            bonn_rps_rear_selectivity_front_score_weight=0.18,
            bonn_rps_rear_selectivity_competition_weight=0.18,
            bonn_rps_rear_selectivity_competition_alpha=0.80,
            bonn_rps_rear_selectivity_gap_weight=0.42,
            bonn_rps_rear_selectivity_sep_weight=0.12,
            bonn_rps_rear_selectivity_dyn_weight=0.14,
            bonn_rps_rear_selectivity_ghost_weight=0.12,
            bonn_rps_rear_selectivity_front_weight=0.10,
            bonn_rps_rear_selectivity_geom_risk_weight=0.20,
            bonn_rps_rear_selectivity_history_risk_weight=0.12,
            bonn_rps_rear_selectivity_density_risk_weight=0.08,
            bonn_rps_rear_selectivity_bridge_relief_weight=0.08,
            bonn_rps_rear_selectivity_static_relief_weight=0.06,
            bonn_rps_rear_selectivity_gap_risk_weight=0.36,
            bonn_rps_rear_selectivity_score_min=0.40,
            bonn_rps_rear_selectivity_risk_max=0.58,
            bonn_rps_rear_selectivity_geom_floor=0.36,
            bonn_rps_rear_selectivity_history_floor=0.28,
            bonn_rps_rear_selectivity_bridge_floor=0.10,
            bonn_rps_rear_selectivity_competition_floor=-0.04,
            bonn_rps_rear_selectivity_front_score_max=0.82,
            bonn_rps_rear_selectivity_gap_min=0.024,
            bonn_rps_rear_selectivity_gap_max=0.070,
            bonn_rps_rear_selectivity_gap_valid_min=0.58,
            bonn_rps_rear_selectivity_topk=70,
            bonn_rps_rear_selectivity_rank_risk_weight=0.55,
        ),
        dict(
            base_spec,
            name='70_fused_discriminator_topk',
            bonn_rps_rear_selectivity_enable=True,
            bonn_rps_rear_selectivity_support_weight=0.08,
            bonn_rps_rear_selectivity_history_weight=0.16,
            bonn_rps_rear_selectivity_static_weight=0.12,
            bonn_rps_rear_selectivity_geom_weight=0.16,
            bonn_rps_rear_selectivity_bridge_weight=0.08,
            bonn_rps_rear_selectivity_density_weight=0.06,
            bonn_rps_rear_selectivity_rear_score_weight=0.24,
            bonn_rps_rear_selectivity_front_score_weight=0.30,
            bonn_rps_rear_selectivity_competition_weight=0.48,
            bonn_rps_rear_selectivity_competition_alpha=0.92,
            bonn_rps_rear_selectivity_gap_weight=0.28,
            bonn_rps_rear_selectivity_sep_weight=0.08,
            bonn_rps_rear_selectivity_dyn_weight=0.14,
            bonn_rps_rear_selectivity_ghost_weight=0.12,
            bonn_rps_rear_selectivity_front_weight=0.10,
            bonn_rps_rear_selectivity_geom_risk_weight=0.16,
            bonn_rps_rear_selectivity_history_risk_weight=0.10,
            bonn_rps_rear_selectivity_density_risk_weight=0.08,
            bonn_rps_rear_selectivity_bridge_relief_weight=0.10,
            bonn_rps_rear_selectivity_static_relief_weight=0.08,
            bonn_rps_rear_selectivity_gap_risk_weight=0.28,
            bonn_rps_rear_selectivity_score_min=0.38,
            bonn_rps_rear_selectivity_risk_max=0.60,
            bonn_rps_rear_selectivity_geom_floor=0.34,
            bonn_rps_rear_selectivity_history_floor=0.26,
            bonn_rps_rear_selectivity_bridge_floor=0.10,
            bonn_rps_rear_selectivity_competition_floor=0.02,
            bonn_rps_rear_selectivity_front_score_max=0.72,
            bonn_rps_rear_selectivity_gap_min=0.020,
            bonn_rps_rear_selectivity_gap_max=0.078,
            bonn_rps_rear_selectivity_gap_valid_min=0.40,
            bonn_rps_rear_selectivity_topk=55,
            bonn_rps_rear_selectivity_rank_risk_weight=0.42,
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
        rows.append(enrich_row(root, row))
        payloads[str(spec['name'])] = {'tum': tum_details, **bonn_payload}

    decide(rows)
    out_dir = PROJECT_ROOT / 'processes' / 's2'
    compare_csv = out_dir / 'S2_RPS_DISCRIMINATIVE_FUSION_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_DISCRIMINATIVE_FUSION_COMPARE.md'
    dist_md = out_dir / 'S2_RPS_DISCRIMINATIVE_FUSION_DISTRIBUTION.md'
    analysis_md = out_dir / 'S2_RPS_DISCRIMINATIVE_FUSION_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_distribution(payloads, rows, dist_md)
    write_analysis(rows, analysis_md)
    print(f'[done] {compare_csv} {compare_md} {dist_md} {analysis_md}')


if __name__ == '__main__':
    main()
