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

from run_s2_write_time_synthesis import PROJECT_ROOT, dryrun_base_cmds, run
from run_s2_rps_rear_geometry_quality import build_commands, summarize_variant


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        bg_goal = row['bonn_rear_true_background_sum'] >= 20.0
        ghost_ok = row['bonn_rear_ghost_sum'] <= 15.0
        better = row['bonn_ghost_reduction_vs_tsdf'] >= 16.5 and row['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6
        comp_ok = row['bonn_comp_r_5cm'] >= control['bonn_comp_r_5cm']
        row['decision'] = 'iterate' if bg_goal and ghost_ok and better and comp_ok else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(iters, key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_comp_r_5cm'], r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum']))
        for r in iters:
            if r is not best:
                r['decision'] = 'abandon'


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        '# S2 history-visible obstructed compare',
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


def write_distribution(payloads: Dict[str, dict], path_md: Path) -> None:
    lines = [
        '# S2 history-visible obstructed distribution report',
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
    lines += [
        '',
        '目标检查：`true_background >= 20` 且 `ghost <= 15` 才算空间分布真正接近本轮目标。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '38_bonn_state_protect')
    best_bg = max(rows[1:], key=lambda r: (r['bonn_rear_true_background_sum'], -r['bonn_rear_ghost_sum']))
    best_metric = max(rows[1:], key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_comp_r_5cm']))
    lines = [
        '# S2 history-visible obstructed analysis',
        '',
        '日期：`2026-03-09`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`output/tmp/legacy_artifacts_placeholder`',
        '',
        '## 1. True Background / Ghost 变化',
        f"- 控制组 `38`: TB=`{control['bonn_rear_true_background_sum']:.0f}`, Ghost=`{control['bonn_rear_ghost_sum']:.0f}`, Hole/Noise=`{control['bonn_rear_hole_or_noise_sum']:.0f}`",
        f"- True-background 最强候选 `{best_bg['variant']}`: TB=`{best_bg['bonn_rear_true_background_sum']:.0f}`, Ghost=`{best_bg['bonn_rear_ghost_sum']:.0f}`, Hole/Noise=`{best_bg['bonn_rear_hole_or_noise_sum']:.0f}`",
    ]
    if best_bg['bonn_rear_true_background_sum'] >= 20.0 and best_bg['bonn_rear_ghost_sum'] <= 15.0:
        lines.append('- 结论：新的约束成功把 Rear Points 引导到 True Background，并控制了 Ghost。')
    else:
        lines.append('- 结论：新的约束没有达成目标分布；True Background 仍远未到 20，或 Ghost 已明显超出允许范围。')
    lines += [
        '',
        '## 2. 指标是否得到推动',
        f"- 控制组 `38`: Bonn `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {control['bonn_comp_r_5cm']:.2f}%`",
        f"- 指标最强候选 `{best_metric['variant']}`: Bonn `ghost_reduction_vs_tsdf = {best_metric['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `Comp-R = {best_metric['bonn_comp_r_5cm']:.2f}%`",
    ]
    if best_metric['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6:
        lines.append('- Ghost 指标获得了净提升。')
    else:
        lines.append('- Ghost 指标没有超过控制组。')
    if best_metric['bonn_comp_r_5cm'] > control['bonn_comp_r_5cm'] + 1e-6:
        lines.append('- `Comp-R` 也获得了提升。')
    else:
        lines.append('- `Comp-R` 依然没有提升。')
    lines += [
        '',
        '## 3. 若未达标，原因判断',
        '- 若 TB 上升但 Ghost 同步大幅上升，说明“历史可见 + 当前遮挡”约束仍与动态区域耦合。',
        '- 若 TB 仍很低，说明当前 rear generation 仍在历史 manifold 之外生成。',
        '- 若指标没有提升，则说明该约束尚未实质性改善 Rear Points 的空间有效性。',
        '',
        '## 4. 阶段判断',
        '- 若未达到 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。',
    ]
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 history-visible obstructed runner.')
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
        dict(base_spec, name='38_bonn_state_protect', output_name='38_bonn_state_protect_hvo_control'),
        dict(
            base_spec,
            name='46_history_background_only_admission',
            bonn_rps_history_obstructed_gate_enable=True,
            bonn_rps_history_visible_min=0.60,
            bonn_rps_obstruction_min=0.38,
            bonn_rps_non_hole_min=0.42,
        ),
        dict(
            base_spec,
            name='47_history_visible_obstructed_manifold',
            bonn_rps_history_obstructed_gate_enable=True,
            bonn_rps_history_visible_min=0.58,
            bonn_rps_obstruction_min=0.36,
            bonn_rps_non_hole_min=0.40,
            bonn_rps_history_manifold_enable=True,
            bonn_rps_history_manifold_visible_min=0.58,
            bonn_rps_history_manifold_obstruction_min=0.36,
            bonn_rps_history_manifold_bg_weight=0.70,
            bonn_rps_history_manifold_geo_weight=0.20,
            bonn_rps_history_manifold_static_weight=0.10,
            bonn_rps_history_manifold_blend=0.88,
            bonn_rps_history_manifold_max_offset=0.015,
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
    compare_csv = out_dir / 'S2_RPS_HISTORY_VISIBLE_OBSTRUCTED_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_HISTORY_VISIBLE_OBSTRUCTED_COMPARE.md'
    dist_md = out_dir / 'S2_RPS_HISTORY_VISIBLE_OBSTRUCTED_DISTRIBUTION.md'
    analysis_md = out_dir / 'S2_RPS_HISTORY_VISIBLE_OBSTRUCTED_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_distribution(payloads, dist_md)
    write_analysis(rows, analysis_md)
    print(f'[done] {compare_csv} {compare_md} {dist_md} {analysis_md}')


if __name__ == '__main__':
    main()
