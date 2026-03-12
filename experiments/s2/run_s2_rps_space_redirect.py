from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
S2_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT_BOOTSTRAP / "scripts"
for _path in (PROJECT_ROOT_BOOTSTRAP, S2_DIR, ROOT_SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_s2_rps_rear_geometry_quality import (
    PROJECT_ROOT,
    TUM_ALL3,
    BONN_ALL3,
    build_commands,
    dryrun_base_cmds,
    run,
    summarize_variant,
)


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        bg_gain = row['bonn_rear_true_background_sum'] > control['bonn_rear_true_background_sum']
        ghost_ok = row['bonn_rear_ghost_sum'] <= control['bonn_rear_ghost_sum'] + 2.0
        better = row['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6
        no_regress = row['tum_comp_r_5cm'] >= control['tum_comp_r_5cm'] - 0.50 and row['bonn_acc_cm'] <= control['bonn_acc_cm'] * 1.10
        row['decision'] = 'iterate' if bg_gain and ghost_ok and better and no_regress else 'abandon'
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
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# S2 space redirect compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | bonn_rear_ghost_sum | bonn_rear_hole_or_noise_sum | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_true_background_sum']:.0f} | {row['bonn_rear_ghost_sum']:.0f} | {row['bonn_rear_hole_or_noise_sum']:.0f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_distribution_report(payloads: Dict[str, dict], path_md: Path) -> None:
    lines = [
        '# S2 rear-point space redirect distribution report',
        '',
        '日期：`2026-03-09`',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | rear_points_sum | true_background_sum | ghost_sum | hole_or_noise_sum |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, payload in payloads.items():
        rear_dist = payload['rear_dist']
        lines.append(
            f"| {name} | {sum(v['rear_points'] for v in rear_dist.values()):.0f} | {sum(v['true_background_points'] for v in rear_dist.values()):.0f} | {sum(v['ghost_region_points'] for v in rear_dist.values()):.0f} | {sum(v['hole_or_noise_points'] for v in rear_dist.values()):.0f} |"
        )
    lines += [
        '',
        '空间重定向是否有效，重点看两点：',
        '- `true_background_sum` 是否显著高于 `38`；',
        '- 同时 `ghost_sum` 是否没有明显恶化。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    best_bg = max(rows[1:], key=lambda r: (r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum']))
    best_metric = max(rows[1:], key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_comp_r_5cm']))
    lines = [
        '# S2 space redirect analysis',
        '',
        '日期：`2026-03-09`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`output/tmp/legacy_artifacts_placeholder`',
        '',
        '## 1. 空间分布是否得到重定向',
        f"- 控制组 `38`: true background=`{control['bonn_rear_true_background_sum']:.0f}`, ghost=`{control['bonn_rear_ghost_sum']:.0f}`, hole/noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}`",
        f"- True-background 最强候选 `{best_bg['variant']}`: true background=`{best_bg['bonn_rear_true_background_sum']:.0f}`, ghost=`{best_bg['bonn_rear_ghost_sum']:.0f}`, hole/noise=`{best_bg['bonn_rear_hole_or_noise_sum']:.0f}`",
    ]
    if best_bg['bonn_rear_true_background_sum'] > control['bonn_rear_true_background_sum'] and best_bg['bonn_rear_ghost_sum'] <= control['bonn_rear_ghost_sum']:
        lines.append('- 结论：空间重定向有效，True Background 覆盖提升且 Ghost 误入未恶化。')
    else:
        lines.append('- 结论：空间重定向未真正成功；虽然部分候选提高了 True Background 点数，但 Ghost 或 Hole/Noise 代价同步上升。')
    lines += [
        '',
        '## 2. 指标是否得到实质性推动',
        f"- 指标最强候选 `{best_metric['variant']}`: Bonn `ghost_reduction_vs_tsdf = {best_metric['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {best_metric['bonn_comp_r_5cm']:.2f}%`",
        f"- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {control['bonn_comp_r_5cm']:.2f}%`",
    ]
    if best_metric['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6:
        lines.append('- 结论：本轮对 Ghost 指标有净提升。')
    else:
        lines.append('- 结论：本轮没有实质性推动 Ghost 指标。')
    if best_metric['bonn_comp_r_5cm'] > control['bonn_comp_r_5cm'] + 1e-6:
        lines.append('- `Comp-R` 也获得了提升。')
    else:
        lines.append('- `Comp-R` 仍然没有被拉起来。')
    lines += [
        '',
        '## 3. 若未达标，主要原因',
        '- 若 `true_background` 上升但 `ghost` 同时上升，说明重定向信号仍与动态区域耦合过强。',
        '- 若 `hole/noise` 仍占主导，说明历史背景锚定仍不足，Rear Points 还在填补无效空洞。',
        '',
        '## 4. 阶段判断',
        '- 若未同时达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 rear-point space redirect runner.')
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
        dict(base_spec, name='38_bonn_state_protect', output_name='38_bonn_state_protect_space_redirect_control'),
        dict(
            base_spec,
            name='43_history_guided_background_location',
            bonn_rps_space_redirect_history_enable=True,
            bonn_rps_space_redirect_history_weight=0.42,
            bonn_rps_space_redirect_history_bg_weight=0.72,
            bonn_rps_space_redirect_history_static_weight=0.28,
            bonn_rps_space_redirect_history_floor=0.42,
            bonn_rps_space_redirect_visual_anchor_enable=True,
            bonn_rps_space_redirect_visual_anchor_weight=0.16,
            bonn_rps_space_redirect_visual_anchor_min=0.44,
        ),
        dict(
            base_spec,
            name='44_history_plus_ghost_suppress',
            bonn_rps_space_redirect_history_enable=True,
            bonn_rps_space_redirect_history_weight=0.38,
            bonn_rps_space_redirect_history_bg_weight=0.70,
            bonn_rps_space_redirect_history_static_weight=0.30,
            bonn_rps_space_redirect_history_floor=0.40,
            bonn_rps_space_redirect_ghost_suppress_enable=True,
            bonn_rps_space_redirect_ghost_suppress_weight=0.18,
            bonn_rps_space_redirect_visual_anchor_enable=True,
            bonn_rps_space_redirect_visual_anchor_weight=0.14,
            bonn_rps_space_redirect_visual_anchor_min=0.40,
        ),
        dict(
            base_spec,
            name='45_visual_evidence_anchor_strict',
            bonn_rps_space_redirect_history_enable=True,
            bonn_rps_space_redirect_history_weight=0.34,
            bonn_rps_space_redirect_history_bg_weight=0.65,
            bonn_rps_space_redirect_history_static_weight=0.35,
            bonn_rps_space_redirect_history_floor=0.46,
            bonn_rps_space_redirect_ghost_suppress_enable=True,
            bonn_rps_space_redirect_ghost_suppress_weight=0.22,
            bonn_rps_space_redirect_visual_anchor_enable=True,
            bonn_rps_space_redirect_visual_anchor_weight=0.20,
            bonn_rps_space_redirect_visual_anchor_min=0.52,
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
    out_dir = PROJECT_ROOT / 'output' / 's2'
    compare_csv = out_dir / 'S2_RPS_SPACE_REDIRECT_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_SPACE_REDIRECT_COMPARE.md'
    dist_md = out_dir / 'S2_RPS_SPACE_REDIRECT_DISTRIBUTION.md'
    analysis_md = out_dir / 'S2_RPS_SPACE_REDIRECT_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_distribution_report(payloads, dist_md)
    write_analysis(rows, analysis_md)
    print(f'[done] {compare_csv} {compare_md} {dist_md} {analysis_md}')


if __name__ == '__main__':
    main()
