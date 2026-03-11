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
    extract_rear_selected_sum = 0.0
    extract_score_ready_sum = 0.0
    extract_support_protected_sum = 0.0
    extract_fail_score_sum = 0.0
    bridge_rear_synth_updates_sum = 0.0
    bridge_rear_synth_w_sum = 0.0
    for seq in BONN_ALL3:
        summary = load_summary(root, 'bonn', seq)
        ptdsf_export_diag = summary.get('ptdsf_export_diag', {})
        admission_diag = summary.get('rps_admission_diag', {})
        bg_manifold_diag = summary.get('bg_manifold_diag', {})
        extract_rear_selected_sum += float(ptdsf_export_diag.get('rear_selected', 0.0))
        extract_score_ready_sum += float(admission_diag.get('extract_score_ready', 0.0))
        extract_support_protected_sum += float(admission_diag.get('extract_support_protected', 0.0))
        extract_fail_score_sum += float(admission_diag.get('extract_fail_score', 0.0))
        bridge_rear_synth_updates_sum += float(bg_manifold_diag.get('bridge_rear_synth_updates', 0.0))
        bridge_rear_synth_w_sum += float(bg_manifold_diag.get('bridge_rear_synth_w_sum', 0.0))
    row = dict(row)
    row['bonn_extract_rear_selected_sum'] = extract_rear_selected_sum
    row['bonn_extract_score_ready_sum'] = extract_score_ready_sum
    row['bonn_extract_support_protected_sum'] = extract_support_protected_sum
    row['bonn_extract_fail_score_sum'] = extract_fail_score_sum
    row['bonn_bridge_rear_synth_updates_sum'] = bridge_rear_synth_updates_sum
    row['bonn_bridge_rear_synth_w_sum'] = bridge_rear_synth_w_sum
    return row


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        coverage_ok = row['bonn_rear_points_sum'] >= 100.0
        bg_ok = row['bonn_rear_true_background_sum'] >= 10.0
        ghost_ok = row['bonn_rear_ghost_sum'] <= 12.0
        metric_ok = row['bonn_ghost_reduction_vs_tsdf'] >= 15.5
        no_regress = row['tum_comp_r_5cm'] >= control['tum_comp_r_5cm'] - 0.50
        row['decision'] = 'iterate' if coverage_ok and bg_ok and ghost_ok and metric_ok and no_regress else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(
            iters,
            key=lambda r: (
                r['bonn_ghost_reduction_vs_tsdf'],
                r['bonn_rear_true_background_sum'],
                -r['bonn_rear_ghost_sum'],
                r['bonn_rear_points_sum'],
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
        '# S2 multi-candidate generation compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | bonn_extract_rear_selected_sum | bonn_bridge_rear_synth_updates_sum | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['bonn_extract_rear_selected_sum']:.0f} | {row['bonn_bridge_rear_synth_updates_sum']:.0f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution(payloads: Dict[str, dict], rows: List[dict], path_md: Path) -> None:
    row_map = {row['variant']: row for row in rows}
    lines = [
        '# S2 multi-candidate generation distribution report',
        '',
        '日期：`2026-03-09`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum | extract_rear_selected_sum | extract_score_ready_sum | extract_support_protected_sum | extract_fail_score_sum | bridge_rear_synth_updates_sum | bridge_rear_synth_w_sum |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, payload in payloads.items():
        rear = payload['rear_dist']
        row = row_map[name]
        lines.append(
            f"| {name} | {sum(v['rear_points'] for v in rear.values()):.0f} | {sum(v['true_background_points'] for v in rear.values()):.0f} | {sum(v['ghost_region_points'] for v in rear.values()):.0f} | {sum(v['hole_or_noise_points'] for v in rear.values()):.0f} | {row['bonn_extract_rear_selected_sum']:.0f} | {row['bonn_extract_score_ready_sum']:.0f} | {row['bonn_extract_support_protected_sum']:.0f} | {row['bonn_extract_fail_score_sum']:.0f} | {row['bonn_bridge_rear_synth_updates_sum']:.0f} | {row['bonn_bridge_rear_synth_w_sum']:.2f} |"
        )
    lines += ['', '重点检查：rear 是否恢复到 `>=100`，TB 是否突破 `>=10`，Ghost 是否仍 `<=12`。']
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    best = max(
        rows[1:],
        key=lambda r: (
            r['bonn_ghost_reduction_vs_tsdf'],
            r['bonn_rear_true_background_sum'],
            -r['bonn_rear_ghost_sum'],
            r['bonn_rear_points_sum'],
        ),
    )
    lines = [
        '# S2 multi-candidate generation analysis',
        '',
        '日期：`2026-03-09`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`processes/s2/S2_RPS_MULTI_CANDIDATE_GENERATION_COMPARE.csv`',
        '',
        '## 1. 覆盖率是否提升',
        f"- 控制组 `38`: rear=`{control['bonn_rear_points_sum']:.0f}`, TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, extract_selected=`{control['bonn_extract_rear_selected_sum']:.0f}`",
        f"- 本轮最佳候选 `{best['variant']}`: rear=`{best['bonn_rear_points_sum']:.0f}`, TB=`{best['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best['bonn_rear_ghost_sum']:.0f}`, extract_selected=`{best['bonn_extract_rear_selected_sum']:.0f}`",
        f"- bridge 合成更新数：`{best['bonn_bridge_rear_synth_updates_sum']:.0f}`，合成权重和：`{best['bonn_bridge_rear_synth_w_sum']:.2f}`",
    ]
    if best['bonn_rear_points_sum'] >= 100.0:
        lines.append('- rear 覆盖量达到了本轮目标。')
    else:
        lines.append('- rear 覆盖量仍未恢复到目标区间。')
    if best['bonn_rear_true_background_sum'] >= 10.0:
        lines.append('- True Background 达到了倍数级增长目标。')
    else:
        lines.append('- True Background 仍未突破 10。')
    if best['bonn_rear_ghost_sum'] <= 12.0:
        lines.append('- Ghost 仍处于可控范围。')
    else:
        lines.append('- Ghost 超出了本轮允许上界。')
    lines += [
        '',
        '## 2. 指标是否改善',
        f"- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {control['bonn_comp_r_5cm']:.2f}%`",
        f"- 本轮最佳候选 `{best['variant']}`: Bonn `ghost_reduction_vs_tsdf = {best['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {best['bonn_comp_r_5cm']:.2f}%`",
        f"- extract `support_protected`: `{control['bonn_extract_support_protected_sum']:.0f} -> {best['bonn_extract_support_protected_sum']:.0f}`",
        f"- extract `score_ready`: `{control['bonn_extract_score_ready_sum']:.0f} -> {best['bonn_extract_score_ready_sum']:.0f}`",
        f"- extract `fail_score`: `{control['bonn_extract_fail_score_sum']:.0f} -> {best['bonn_extract_fail_score_sum']:.0f}`",
    ]
    if best['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6:
        lines.append('- ghost 指标有净提升。')
    else:
        lines.append('- ghost 指标没有超过控制组。')
    if best['bonn_comp_r_5cm'] > control['bonn_comp_r_5cm'] + 1e-6:
        lines.append('- Comp-R 有提升。')
    else:
        lines.append('- Comp-R 仍无明显提升。')
    lines += [
        '',
        '## 3. 诊断结论',
        '- 若 `bridge_rear_synth_updates_sum` 明显增加但 `extract_score_ready_sum` 没有同步恢复，说明新增 rear state 仍在 admission/score_gate 被拦截。',
        '- 若 `rear_points_sum` 增加但 `true_background_sum` 不增，说明多点生成主要放大了 hole/noise，而非命中真实背景。',
        '- 若 `true_background_sum` 上升且 `ghost_sum` 仍受控，则说明多点生成开始把 bridge 方向转化为有效背景覆盖。',
        '',
        '## 4. 阶段判断',
        '- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 multi-candidate rear generation runner.')
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
    )
    variants = [
        dict(
            base_spec,
            name='38_bonn_state_protect',
            output_name='38_bonn_state_protect_high_coverage_bridge_control',
        ),
        dict(
            base_spec,
            name='62_dense_patch_projection',
            bonn_rps_bg_manifold_state_enable=True,
            bonn_rps_bg_dense_state_enable=True,
            bonn_rps_bg_surface_constrained_enable=True,
            bonn_rps_bg_bridge_enable=True,
            bonn_rps_bg_bridge_min_visible=0.22,
            bonn_rps_bg_bridge_min_obstruction=0.24,
            bonn_rps_bg_bridge_min_step=2,
            bonn_rps_bg_bridge_max_step=4,
            bonn_rps_bg_bridge_gain=0.84,
            bonn_rps_bg_bridge_target_dyn_max=0.30,
            bonn_rps_bg_bridge_target_surface_max=0.32,
            bonn_rps_bg_bridge_ghost_suppress_enable=True,
            bonn_rps_bg_bridge_ghost_suppress_weight=0.16,
            bonn_rps_bg_bridge_relaxed_dyn_max=0.48,
            bonn_rps_bg_bridge_keep_multi_hits=True,
            bonn_rps_bg_bridge_max_hits_per_source=12,
            bonn_rps_bg_bridge_patch_radius_cells=1,
            bonn_rps_bg_bridge_patch_gain_scale=0.62,
            bonn_rps_bg_bridge_rear_synth_enable=True,
            bonn_rps_bg_bridge_rear_support_gain=0.20,
            bonn_rps_bg_bridge_rear_rho_gain=0.10,
            bonn_rps_bg_bridge_rear_phi_blend=0.84,
            bonn_rps_bg_bridge_rear_score_floor=0.24,
            bonn_rps_bg_bridge_rear_active_floor=0.54,
            bonn_rps_bg_bridge_rear_age_floor=1.0,
            bonn_rps_bg_manifold_history_weight=0.00,
            bonn_rps_bg_manifold_obstruction_weight=0.00,
            bonn_rps_bg_manifold_visible_lo=0.00,
            bonn_rps_bg_manifold_visible_hi=0.00,
        ),
        dict(
            base_spec,
            name='63_multi_hypothesis_depth_sampling',
            bonn_rps_bg_manifold_state_enable=True,
            bonn_rps_bg_dense_state_enable=True,
            bonn_rps_bg_surface_constrained_enable=True,
            bonn_rps_bg_bridge_enable=True,
            bonn_rps_bg_bridge_min_visible=0.22,
            bonn_rps_bg_bridge_min_obstruction=0.24,
            bonn_rps_bg_bridge_min_step=2,
            bonn_rps_bg_bridge_max_step=5,
            bonn_rps_bg_bridge_gain=0.82,
            bonn_rps_bg_bridge_target_dyn_max=0.28,
            bonn_rps_bg_bridge_target_surface_max=0.32,
            bonn_rps_bg_bridge_ghost_suppress_enable=True,
            bonn_rps_bg_bridge_ghost_suppress_weight=0.18,
            bonn_rps_bg_bridge_relaxed_dyn_max=0.46,
            bonn_rps_bg_bridge_keep_multi_hits=True,
            bonn_rps_bg_bridge_max_hits_per_source=14,
            bonn_rps_bg_bridge_depth_hypothesis_count=2,
            bonn_rps_bg_bridge_depth_step_scale=0.35,
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
        ),
        dict(
            base_spec,
            name='64_patch_depth_hybrid_generation',
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
    out_dir = PROJECT_ROOT / 'processes' / 's2'
    compare_csv = out_dir / 'S2_RPS_MULTI_CANDIDATE_GENERATION_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_MULTI_CANDIDATE_GENERATION_COMPARE.md'
    dist_md = out_dir / 'S2_RPS_MULTI_CANDIDATE_GENERATION_DISTRIBUTION.md'
    analysis_md = out_dir / 'S2_RPS_MULTI_CANDIDATE_GENERATION_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_distribution(payloads, rows, dist_md)
    write_analysis(rows, analysis_md)
    print(f'[done] {compare_csv} {compare_md} {dist_md} {analysis_md}')


if __name__ == '__main__':
    main()
