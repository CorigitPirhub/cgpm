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
    fields = [
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
        'bonn_rear_selectivity_occlusion_order_sum',
        'bonn_rear_selectivity_local_conflict_sum',
        'bonn_rear_selectivity_front_residual_sum',
        'bonn_rear_selectivity_pre_front_residual_sum',
        'bonn_rear_selectivity_pre_dyn_risk_sum',
        'bonn_rear_selectivity_dyn_risk_sum',
    ]
    for key in fields:
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
        row['bonn_rear_selectivity_occlusion_order_sum'] += float(summary.get('rear_selectivity_occlusion_order_sum', 0.0))
        row['bonn_rear_selectivity_local_conflict_sum'] += float(summary.get('rear_selectivity_local_conflict_sum', 0.0))
        row['bonn_rear_selectivity_front_residual_sum'] += float(summary.get('rear_selectivity_front_residual_sum', 0.0))
        row['bonn_rear_selectivity_pre_front_residual_sum'] += float(summary.get('rear_selectivity_pre_front_residual_sum', 0.0))
        row['bonn_rear_selectivity_pre_dyn_risk_sum'] += float(summary.get('rear_selectivity_pre_dyn_risk_sum', 0.0))
        row['bonn_rear_selectivity_dyn_risk_sum'] += float(summary.get('rear_selectivity_dyn_risk_sum', 0.0))
    denom = max(1.0, row['bonn_rear_selectivity_kept_sum'])
    drop_count = max(1.0, row['bonn_rear_selectivity_pre_sum'] - row['bonn_rear_selectivity_kept_sum'])
    row['bonn_front_score_mean'] = row['bonn_rear_selectivity_front_score_sum'] / denom
    row['bonn_rear_score_mean'] = row['bonn_rear_selectivity_rear_score_sum'] / denom
    row['bonn_rear_gap_mean'] = row['bonn_rear_selectivity_gap_sum'] / denom
    row['bonn_competition_mean'] = row['bonn_rear_selectivity_competition_sum'] / denom
    row['bonn_occlusion_order_mean'] = row['bonn_rear_selectivity_occlusion_order_sum'] / denom
    row['bonn_local_conflict_mean'] = row['bonn_rear_selectivity_local_conflict_sum'] / denom
    row['bonn_front_residual_mean'] = row['bonn_rear_selectivity_front_residual_sum'] / denom
    row['bonn_front_residual_drop_mean'] = (row['bonn_rear_selectivity_pre_front_residual_sum'] - row['bonn_rear_selectivity_front_residual_sum']) / drop_count
    row['bonn_dyn_risk_mean'] = row['bonn_rear_selectivity_dyn_risk_sum'] / denom
    row['bonn_dyn_risk_drop_mean'] = (row['bonn_rear_selectivity_pre_dyn_risk_sum'] - row['bonn_rear_selectivity_dyn_risk_sum']) / drop_count
    return row


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '68_rear_front_score_competition')
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
                r['bonn_occlusion_order_mean'],
                -r['bonn_local_conflict_mean'],
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
        '# S2 semantic classification compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_occlusion_order_mean | bonn_local_conflict_mean | bonn_front_residual_mean | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_occlusion_order_mean']:.3f} | {row['bonn_local_conflict_mean']:.3f} | {row['bonn_front_residual_mean']:.3f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution(payloads: Dict[str, dict], rows: List[dict], path_md: Path) -> None:
    row_map = {row['variant']: row for row in rows}
    lines = [
        '# S2 semantic classification distribution report',
        '',
        '日期：`2026-03-10`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | occlusion_order_mean | local_conflict_mean | front_residual_mean | front_residual_drop_mean | dyn_risk_mean | dyn_risk_drop_mean | competition_mean | selectivity_topk_drop_sum |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, payload in payloads.items():
        rear = payload['rear_dist']
        row = row_map[name]
        lines.append(
            f"| {name} | {sum(v['rear_points'] for v in rear.values()):.0f} | {sum(v['true_background_points'] for v in rear.values()):.0f} | {sum(v['ghost_region_points'] for v in rear.values()):.0f} | {sum(v['hole_or_noise_points'] for v in rear.values()):.0f} | {row['bonn_occlusion_order_mean']:.3f} | {row['bonn_local_conflict_mean']:.3f} | {row['bonn_front_residual_mean']:.3f} | {row['bonn_front_residual_drop_mean']:.3f} | {row['bonn_dyn_risk_mean']:.3f} | {row['bonn_dyn_risk_drop_mean']:.3f} | {row['bonn_competition_mean']:.3f} | {row['bonn_rear_selectivity_topk_drop_sum']:.0f} |"
        )
    lines += ['', '重点检查：Ghost 是否压回 `<=15`，TB 是否恢复到 `>=8`，以及新时空特征均值是否拉开。']
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '68_rear_front_score_competition')
    valid_rows = [r for r in rows[1:] if r['bonn_rear_points_sum'] > 0.0]
    best_pool = valid_rows if valid_rows else rows[1:]
    best = max(
        best_pool,
        key=lambda r: (
            r['bonn_ghost_reduction_vs_tsdf'],
            -r['bonn_rear_ghost_sum'],
            r['bonn_rear_true_background_sum'],
            r['bonn_occlusion_order_mean'],
            -r['bonn_local_conflict_mean'],
        ),
    )
    lines = [
        '# S2 semantic classification analysis',
        '',
        '日期：`2026-03-10`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`output/tmp/legacy_artifacts_placeholder`',
        '',
        '## 1. 新特征是否区分了 TB 和 Ghost',
        f"- 基组 `68`: rear=`{control['bonn_rear_points_sum']:.0f}`, TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, order=`{control['bonn_occlusion_order_mean']:.3f}`, conflict=`{control['bonn_local_conflict_mean']:.3f}`, residual=`{control['bonn_front_residual_mean']:.3f}`",
        f"- 本轮最佳候选 `{best['variant']}`: rear=`{best['bonn_rear_points_sum']:.0f}`, TB=`{best['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best['bonn_rear_ghost_sum']:.0f}`, order=`{best['bonn_occlusion_order_mean']:.3f}`, conflict=`{best['bonn_local_conflict_mean']:.3f}`, residual=`{best['bonn_front_residual_mean']:.3f}`",
        f"- 基组 kept/drop front_residual=`{control['bonn_front_residual_mean']:.3f}/{control['bonn_front_residual_drop_mean']:.3f}`, dyn_risk=`{control['bonn_dyn_risk_mean']:.3f}/{control['bonn_dyn_risk_drop_mean']:.3f}`",
        f"- 候选 kept/drop front_residual=`{best['bonn_front_residual_mean']:.3f}/{best['bonn_front_residual_drop_mean']:.3f}`, dyn_risk=`{best['bonn_dyn_risk_mean']:.3f}/{best['bonn_dyn_risk_drop_mean']:.3f}`",
    ]
    zero_rows = [r for r in rows[1:] if r['bonn_rear_points_sum'] <= 0.0]
    if zero_rows:
        lines.append(f"- 退化候选：`{', '.join(r['variant'] for r in zero_rows)}` 直接把 rear 裁到 `0`，说明门控仍过硬。")
    if best['bonn_occlusion_order_mean'] > control['bonn_occlusion_order_mean'] + 1e-6:
        lines.append('- `occlusion_order` 均值提升。')
    else:
        lines.append('- `occlusion_order` 没有形成更强区分。')
    if best['bonn_local_conflict_mean'] < control['bonn_local_conflict_mean'] - 1e-6:
        lines.append('- `local_conflict` 均值下降，局部冲突压制更强。')
    else:
        lines.append('- `local_conflict` 没有明显下降。')
    lines += [
        '',
        '## 2. 是否解决“指标虚高但分布恶化”',
        f"- 基组 `68`: Bonn `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {control['bonn_comp_r_5cm']:.2f}%`",
        f"- 候选 `{best['variant']}`: Bonn `ghost_reduction_vs_tsdf = {best['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {best['bonn_comp_r_5cm']:.2f}%`",
    ]
    if best['bonn_ghost_reduction_vs_tsdf'] >= 22.0:
        lines.append('- ghost_reduction_vs_tsdf 保持在 `22%+`。')
    else:
        lines.append('- ghost_reduction_vs_tsdf 未能保持在 `22%+`。')
    if best['bonn_rear_ghost_sum'] <= 15.0 and best['bonn_rear_true_background_sum'] >= 8.0:
        lines.append('- 同时达到了 `Ghost <= 15` 与 `TB >= 8`。')
    else:
        lines.append('- 仍未同时达到 `Ghost <= 15` 与 `TB >= 8`。')
    lines += [
        '',
        '## 3. 诊断结论',
        '- 若 `occlusion_order` 上升但 TB 不升，说明当前时序顺序特征仍偏向“保守裁剪”，没有真正识别被遮挡背景。',
        '- 若 `local_conflict` 降低但 Ghost 仍高，说明冲突检测只裁掉了少量浅层残留，未覆盖主要 Ghost 来源。',
        '- 若保留/剔除点在 `front_residual` 或 `dyn_risk` 上均值差异仍然很小，说明当前动态上下文特征没有形成有效分类边界。',
        '- 若 `front_residual` 压制后仍主要降低 TB，说明当前前景残留信号与 TB 深度区高度耦合，仍缺少可分离特征。',
        '',
        '## 4. 阶段判断',
        '- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 occlusion conflict runner.')
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
    )
    variants = [
        dict(base_spec, name='68_rear_front_score_competition', output_name='68_rear_front_score_competition_semantic_control'),
        dict(
            base_spec,
            name='71_occlusion_order_consistency',
            output_name='71_occlusion_order_consistency_semantic',
            bonn_rps_rear_selectivity_rear_score_weight=0.18,
            bonn_rps_rear_selectivity_front_score_weight=0.24,
            bonn_rps_rear_selectivity_competition_weight=0.42,
            bonn_rps_rear_selectivity_competition_alpha=0.90,
            bonn_rps_rear_selectivity_occlusion_order_weight=0.55,
            bonn_rps_rear_selectivity_occlusion_order_floor=0.08,
            bonn_rps_rear_selectivity_occlusion_order_risk_weight=0.18,
            bonn_rps_rear_selectivity_local_conflict_weight=0.10,
            bonn_rps_rear_selectivity_front_residual_weight=0.12,
            bonn_rps_rear_selectivity_topk=90,
        ),
        dict(
            base_spec,
            name='72_local_geometric_conflict_resolution',
            output_name='72_local_geometric_conflict_resolution_semantic',
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
        ),
        dict(
            base_spec,
            name='73_front_residual_aware_suppression',
            output_name='73_front_residual_aware_suppression_semantic',
            bonn_rps_rear_selectivity_rear_score_weight=0.16,
            bonn_rps_rear_selectivity_front_score_weight=0.24,
            bonn_rps_rear_selectivity_competition_weight=0.36,
            bonn_rps_rear_selectivity_competition_alpha=0.88,
            bonn_rps_rear_selectivity_occlusion_order_weight=0.18,
            bonn_rps_rear_selectivity_local_conflict_weight=0.18,
            bonn_rps_rear_selectivity_front_residual_weight=0.72,
            bonn_rps_rear_selectivity_front_residual_max=0.42,
            bonn_rps_rear_selectivity_topk=100,
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
    out_dir = PROJECT_ROOT / 'output' / 's2'
    compare_csv = out_dir / 'S2_RPS_SEMANTIC_CLASSIFICATION_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_SEMANTIC_CLASSIFICATION_COMPARE.md'
    dist_md = out_dir / 'S2_RPS_SEMANTIC_CLASSIFICATION_DISTRIBUTION.md'
    analysis_md = out_dir / 'S2_RPS_SEMANTIC_CLASSIFICATION_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_distribution(payloads, rows, dist_md)
    write_analysis(rows, analysis_md)
    print(f'[done] {compare_csv} {compare_md} {dist_md} {analysis_md}')


if __name__ == '__main__':
    main()
