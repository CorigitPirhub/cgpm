from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from run_s2_write_time_synthesis import PROJECT_ROOT, dryrun_base_cmds, run
from run_s2_rps_rear_geometry_quality import build_commands, summarize_variant


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        bg_goal = row['bonn_rear_true_background_sum'] >= 10.0
        ghost_ok = row['bonn_rear_ghost_sum'] <= 12.0
        metric_ok = row['bonn_ghost_reduction_vs_tsdf'] >= 16.0
        no_regress = row['tum_comp_r_5cm'] >= control['tum_comp_r_5cm'] - 0.50
        row['decision'] = 'iterate' if bg_goal and ghost_ok and metric_ok and no_regress else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(iters, key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum']))
        for r in iters:
            if r is not best:
                r['decision'] = 'abandon'


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        '# S2 surface constrained compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_points_sum | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_points_sum']:.0f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution(payloads: Dict[str, dict], path_md: Path) -> None:
    lines = [
        '# S2 surface constrained distribution report',
        '',
        '日期：`2026-03-09`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, payload in payloads.items():
        rear = payload['rear_dist']
        lines.append(
            f"| {name} | {sum(v['rear_points'] for v in rear.values()):.0f} | {sum(v['true_background_points'] for v in rear.values()):.0f} | {sum(v['ghost_region_points'] for v in rear.values()):.0f} | {sum(v['hole_or_noise_points'] for v in rear.values()):.0f} |"
        )
    lines += ['', '重点检查：True Background 是否有倍数级提升，以及 Ghost 是否仍控制在 `<=12`。']
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    best = max(rows[1:], key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum']))
    lines = [
        '# S2 surface constrained analysis',
        '',
        '日期：`2026-03-09`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`processes/s2/S2_RPS_SURFACE_CONSTRAINED_COMPARE.csv`',
        '',
        '## 1. 表面约束是否成功',
        f"- 控制组 `38`: TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, Hole/Noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}`",
        f"- 本轮最佳候选 `{best['variant']}`: TB=`{best['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best['bonn_rear_ghost_sum']:.0f}`, Hole/Noise=`{best['bonn_rear_hole_or_noise_sum']:.0f}`",
    ]
    if best['bonn_rear_true_background_sum'] > control['bonn_rear_true_background_sum']:
        lines.append('- 表面约束至少提升了 True Background 覆盖。')
    else:
        lines.append('- 表面约束未能提升 True Background 覆盖。')
    if best['bonn_rear_ghost_sum'] <= 12.0:
        lines.append('- Ghost 仍被控制在阈值以内。')
    else:
        lines.append('- Ghost 仍高于本轮目标。')
    lines += [
        '',
        '## 2. 指标是否达到新高度',
        f"- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {control['bonn_comp_r_5cm']:.2f}%`",
        f"- 本轮最佳候选 `{best['variant']}`: Bonn `ghost_reduction_vs_tsdf = {best['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {best['bonn_comp_r_5cm']:.2f}%`",
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
        '## 3. 若未达标，原因判断',
        '- 若 TB 仍然很低，说明表面约束虽然限制了扩散方向，但支持仍未跨越遮挡缝隙。',
        '- 若 rear 总量显著下降，则说明约束过硬，传播覆盖面不足。',
        '- 若 Ghost 仍升高，说明表面掩码仍不足以区分动态表面与真实背景表面。',
        '',
        '## 4. 阶段判断',
        '- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 surface constrained runner.')
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
        dict(base_spec, name='38_bonn_state_protect', output_name='38_bonn_state_protect_surface_constrained_control'),
        dict(
            base_spec,
            name='53_surface_adjacent_propagation',
            bonn_rps_bg_manifold_state_enable=True,
            bonn_rps_bg_manifold_alpha_up=0.10,
            bonn_rps_bg_manifold_alpha_down=0.01,
            bonn_rps_bg_manifold_rho_alpha=0.12,
            bonn_rps_bg_manifold_weight_gain=0.74,
            bonn_rps_bg_manifold_rho_ref=0.07,
            bonn_rps_bg_manifold_weight_ref=0.28,
            bonn_rps_bg_manifold_history_weight=0.46,
            bonn_rps_bg_manifold_obstruction_weight=0.10,
            bonn_rps_bg_manifold_visible_lo=0.12,
            bonn_rps_bg_manifold_visible_hi=0.36,
            bonn_rps_bg_dense_state_enable=True,
            bonn_rps_bg_dense_neighbor_radius=1,
            bonn_rps_bg_dense_neighbor_weight=0.92,
            bonn_rps_bg_dense_geometry_weight=0.40,
            bonn_rps_bg_dense_support_floor=0.16,
            bonn_rps_bg_dense_decay=0.997,
            bonn_rps_bg_surface_constrained_enable=True,
            bonn_rps_bg_surface_min_conf=0.18,
            bonn_rps_bg_surface_agree_weight=0.55,
        ),
        dict(
            base_spec,
            name='54_normal_guided_manifold_extension',
            bonn_rps_bg_manifold_state_enable=True,
            bonn_rps_bg_manifold_alpha_up=0.10,
            bonn_rps_bg_manifold_alpha_down=0.01,
            bonn_rps_bg_manifold_rho_alpha=0.12,
            bonn_rps_bg_manifold_weight_gain=0.74,
            bonn_rps_bg_manifold_rho_ref=0.07,
            bonn_rps_bg_manifold_weight_ref=0.28,
            bonn_rps_bg_manifold_history_weight=0.46,
            bonn_rps_bg_manifold_obstruction_weight=0.08,
            bonn_rps_bg_manifold_visible_lo=0.10,
            bonn_rps_bg_manifold_visible_hi=0.34,
            bonn_rps_bg_dense_state_enable=True,
            bonn_rps_bg_dense_neighbor_radius=1,
            bonn_rps_bg_dense_neighbor_weight=0.88,
            bonn_rps_bg_dense_geometry_weight=0.42,
            bonn_rps_bg_dense_support_floor=0.14,
            bonn_rps_bg_dense_decay=0.997,
            bonn_rps_bg_surface_constrained_enable=True,
            bonn_rps_bg_surface_min_conf=0.16,
            bonn_rps_bg_surface_agree_weight=0.52,
            bonn_rps_bg_surface_tangent_enable=True,
            bonn_rps_bg_surface_tangent_weight=0.82,
            bonn_rps_bg_surface_tangent_floor=0.10,
        ),
        dict(
            base_spec,
            name='55_surface_constrained_ray_projection',
            bonn_rps_bg_manifold_state_enable=True,
            bonn_rps_bg_manifold_alpha_up=0.10,
            bonn_rps_bg_manifold_alpha_down=0.01,
            bonn_rps_bg_manifold_rho_alpha=0.12,
            bonn_rps_bg_manifold_weight_gain=0.76,
            bonn_rps_bg_manifold_rho_ref=0.07,
            bonn_rps_bg_manifold_weight_ref=0.28,
            bonn_rps_bg_manifold_history_weight=0.48,
            bonn_rps_bg_manifold_obstruction_weight=0.10,
            bonn_rps_bg_manifold_visible_lo=0.10,
            bonn_rps_bg_manifold_visible_hi=0.34,
            bonn_rps_bg_dense_state_enable=True,
            bonn_rps_bg_dense_neighbor_radius=1,
            bonn_rps_bg_dense_neighbor_weight=0.90,
            bonn_rps_bg_dense_geometry_weight=0.45,
            bonn_rps_bg_dense_support_floor=0.14,
            bonn_rps_bg_dense_decay=0.998,
            bonn_rps_bg_surface_constrained_enable=True,
            bonn_rps_bg_surface_min_conf=0.16,
            bonn_rps_bg_surface_agree_weight=0.60,
            bonn_rps_bg_surface_tangent_enable=True,
            bonn_rps_bg_surface_tangent_weight=0.78,
            bonn_rps_bg_surface_tangent_floor=0.12,
            bonn_rps_history_manifold_enable=True,
            bonn_rps_history_manifold_visible_min=0.22,
            bonn_rps_history_manifold_obstruction_min=0.14,
            bonn_rps_history_manifold_bg_weight=0.92,
            bonn_rps_history_manifold_geo_weight=0.06,
            bonn_rps_history_manifold_static_weight=0.02,
            bonn_rps_history_manifold_blend=0.96,
            bonn_rps_history_manifold_max_offset=0.008,
        ),
    ]

    rows: List[dict] = []
    payloads: Dict[str, dict] = {}
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec)
        run(tum_cmd)
        run(bonn_cmd)
        row, tum_details, bonn_payload = summarize_variant(root, str(spec['name']), frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame, seed=args.seed)
        rows.append(row)
        payloads[str(spec['name'])] = {'tum': tum_details, **bonn_payload}

    decide(rows)
    out_dir = PROJECT_ROOT / 'processes' / 's2'
    compare_csv = out_dir / 'S2_RPS_SURFACE_CONSTRAINED_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_SURFACE_CONSTRAINED_COMPARE.md'
    dist_md = out_dir / 'S2_RPS_SURFACE_CONSTRAINED_DISTRIBUTION.md'
    analysis_md = out_dir / 'S2_RPS_SURFACE_CONSTRAINED_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_distribution(payloads, dist_md)
    write_analysis(rows, analysis_md)
    print(f'[done] {compare_csv} {compare_md} {dist_md} {analysis_md}')


if __name__ == '__main__':
    main()
