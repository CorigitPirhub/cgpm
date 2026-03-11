from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from run_s2_write_time_synthesis import PROJECT_ROOT, dryrun_base_cmds, ensure_flag, run, set_arg
from run_s2_rear_bg_state_recovery import build_variant_cmds as build_rearbg


TUM_ALL3 = [
    'rgbd_dataset_freiburg3_walking_xyz',
    'rgbd_dataset_freiburg3_walking_static',
    'rgbd_dataset_freiburg3_walking_halfsphere',
]
BONN_ALL3 = [
    'rgbd_bonn_balloon2',
    'rgbd_bonn_balloon',
    'rgbd_bonn_crowd2',
]


def load_csv(path: Path) -> List[dict]:
    return list(csv.DictReader(path.open('r', encoding='utf-8')))


def pick_row(rows: List[dict], sequence: str, method: str) -> dict:
    method = method.lower()
    for row in rows:
        if str(row.get('sequence', '')) == sequence and str(row.get('method', '')).lower() == method:
            return row
    raise KeyError(f'missing row sequence={sequence} method={method}')


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def seq_summary(root: Path, family: str, sequence: str) -> dict:
    if family == 'tum':
        return load_summary(root / 'tum_oracle' / 'oracle' / sequence / 'egf' / 'summary.json')
    return load_summary(root / 'bonn_slam' / 'slam' / sequence / 'egf' / 'summary.json')


def family_metrics(root: Path, family: str, sequences: List[str]) -> Tuple[Dict[str, float], Dict[str, dict]]:
    sub = root / ('tum_oracle/oracle' if family == 'tum' else 'bonn_slam/slam')
    rec = load_csv(sub / 'tables' / 'reconstruction_metrics.csv')
    dyn = load_csv(sub / 'tables' / 'dynamic_metrics.csv')
    accs: List[float] = []
    comps: List[float] = []
    ghost_reds: List[float] = []
    details: Dict[str, dict] = {}
    for seq in sequences:
        regf = pick_row(rec, seq, 'egf')
        rdyn = pick_row(dyn, seq, 'egf')
        rtsdf = pick_row(dyn, seq, 'tsdf')
        acc_cm = float(regf['accuracy']) * 100.0
        comp_r = float(regf['recall_5cm']) * 100.0
        ghost_ratio = float(rdyn['ghost_ratio'])
        tsdf_ghost = float(rtsdf['ghost_ratio'])
        ghost_red = ((tsdf_ghost - ghost_ratio) / max(1e-9, tsdf_ghost)) * 100.0
        accs.append(acc_cm)
        comps.append(comp_r)
        ghost_reds.append(ghost_red)
        details[seq] = {
            'acc_cm': acc_cm,
            'comp_r_5cm': comp_r,
            'ghost_reduction_vs_tsdf': ghost_red,
        }
    return {
        'acc_cm_mean': float(mean(accs)),
        'comp_r_5cm_mean': float(mean(comps)),
        'ghost_reduction_vs_tsdf_mean': float(mean(ghost_reds)),
    }, details


def chain_diag(summary: dict) -> dict:
    state = summary.get('rear_bg_state_diag', {})
    admission = summary.get('rps_admission_diag', {})
    competition = summary.get('rps_competition_diag', {})
    return {
        'committed_cells': float(state.get('rear_w_nonzero', 0.0)),
        'active_cells': float(state.get('rps_active_nonzero', 0.0)),
        'score_ge_on': float(state.get('rps_commit_score_ge_on', 0.0)),
        'score_ge_release': float(state.get('rps_commit_score_ge_release', 0.0)),
        'sync_wr_present': float(admission.get('sync_wr_present', 0.0)),
        'sync_hard_commit_on': float(admission.get('sync_hard_commit_on', 0.0)),
        'sync_rear_enabled': float(admission.get('sync_rear_enabled', 0.0)),
        'sync_support_protected': float(admission.get('sync_support_protected', 0.0)),
        'extract_wr_present': float(admission.get('extract_wr_present', 0.0)),
        'extract_hard_commit_on': float(admission.get('extract_hard_commit_on', 0.0)),
        'extract_rear_enabled': float(admission.get('extract_rear_enabled', 0.0)),
        'extract_support_protected': float(admission.get('extract_support_protected', 0.0)),
        'extract_fail_active': float(admission.get('extract_fail_active', 0.0)),
        'extract_fail_score': float(admission.get('extract_fail_score', 0.0)),
        'extract_support_mean': float(admission.get('extract_support_sum', 0.0) / max(1.0, admission.get('extract_calls', 0.0))),
        'extract_active_like_mean': float(admission.get('extract_active_like_sum', 0.0) / max(1.0, admission.get('extract_calls', 0.0))),
        'extract_rear_selected': float(competition.get('extract_rear_selected', 0.0) + competition.get('extract_rear_soft_selected', 0.0)),
    }


def summarize_variant(root: Path, name: str) -> Tuple[dict, Dict[str, dict], Dict[str, dict]]:
    tum_family, tum_details = family_metrics(root, 'tum', TUM_ALL3)
    bonn_family, bonn_details = family_metrics(root, 'bonn', BONN_ALL3)
    bonn_chain = {seq: chain_diag(seq_summary(root, 'bonn', seq)) for seq in BONN_ALL3}
    row = {
        'variant': name,
        'tum_acc_cm': tum_family['acc_cm_mean'],
        'tum_comp_r_5cm': tum_family['comp_r_5cm_mean'],
        'bonn_acc_cm': bonn_family['acc_cm_mean'],
        'bonn_comp_r_5cm': bonn_family['comp_r_5cm_mean'],
        'bonn_ghost_reduction_vs_tsdf': bonn_family['ghost_reduction_vs_tsdf_mean'],
        'bonn_committed_cells_sum': sum(v['committed_cells'] for v in bonn_chain.values()),
        'bonn_active_cells_sum': sum(v['active_cells'] for v in bonn_chain.values()),
        'bonn_extract_wr_present_sum': sum(v['extract_wr_present'] for v in bonn_chain.values()),
        'bonn_extract_hard_commit_on_sum': sum(v['extract_hard_commit_on'] for v in bonn_chain.values()),
        'bonn_extract_rear_enabled_sum': sum(v['extract_rear_enabled'] for v in bonn_chain.values()),
        'bonn_extract_rear_selected_sum': sum(v['extract_rear_selected'] for v in bonn_chain.values()),
        'bonn_extract_fail_active_sum': sum(v['extract_fail_active'] for v in bonn_chain.values()),
        'bonn_extract_fail_score_sum': sum(v['extract_fail_score'] for v in bonn_chain.values()),
        'hit_2of4_partial': float(
            (tum_family['acc_cm_mean'] <= 2.55)
            + (bonn_family['acc_cm_mean'] <= 3.10)
            + (bonn_family['ghost_reduction_vs_tsdf_mean'] >= 22.0)
            + ((tum_family['comp_r_5cm_mean'] >= 98.0) or (bonn_family['comp_r_5cm_mean'] >= 98.0))
        ),
        'pass_comp': bool(tum_family['comp_r_5cm_mean'] >= 98.0 and bonn_family['comp_r_5cm_mean'] >= 98.0),
        'decision': 'pending',
    }
    return row, tum_details, {'metrics': bonn_details, 'chain': bonn_chain}


def build_commands(base_tum: List[str], base_bonn: List[str], spec: dict) -> Tuple[List[str], List[str], Path]:
    spec_for_build = dict(spec)
    if 'output_name' in spec_for_build:
        spec_for_build['name'] = str(spec_for_build['output_name'])
    tum_cmd, bonn_cmd, root = build_rearbg(base_tum, base_bonn, spec_for_build)
    tum_cmd = set_arg(tum_cmd, '--dynamic_sequences', ','.join(TUM_ALL3))
    bonn_cmd = set_arg(bonn_cmd, '--bonn_dynamic_preset', 'all3')
    bonn_cmd = set_arg(bonn_cmd, '--dynamic_sequences', ','.join(BONN_ALL3))

    for cmd_name in ('tum', 'bonn'):
        cmd = tum_cmd if cmd_name == 'tum' else bonn_cmd
        if bool(spec.get('rps_commit_activation_enable', False)):
            cmd = ensure_flag(cmd, '--egf_rps_commit_activation_enable')
        if bool(spec.get('rps_surface_bank_enable', False)):
            cmd = ensure_flag(cmd, '--egf_rps_surface_bank_enable')
        if cmd_name == 'bonn' and bool(spec.get('bonn_rps_admission_support_enable', False)):
            cmd = ensure_flag(cmd, '--egf_rps_admission_support_enable')
        if cmd_name == 'bonn' and bool(spec.get('bonn_rps_rear_state_protect_enable', False)):
            cmd = ensure_flag(cmd, '--egf_rps_rear_state_protect_enable')
        if cmd_name == 'tum':
            tum_cmd = cmd
        else:
            bonn_cmd = cmd

    common_pairs = [
        ('rps_bank_margin', '--egf_rps_bank_margin'),
        ('rps_bank_separation_ref', '--egf_rps_bank_separation_ref'),
        ('rps_bank_rear_min_score', '--egf_rps_bank_rear_min_score'),
        ('rps_commit_threshold', '--egf_rps_commit_threshold'),
        ('rps_commit_release', '--egf_rps_commit_release'),
        ('rps_commit_age_threshold', '--egf_rps_commit_age_threshold'),
        ('rps_commit_rho_ref', '--egf_rps_commit_rho_ref'),
        ('rps_commit_weight_ref', '--egf_rps_commit_weight_ref'),
        ('rps_commit_min_cand_rho', '--egf_rps_commit_min_cand_rho'),
        ('rps_commit_min_cand_w', '--egf_rps_commit_min_cand_w'),
        ('rps_commit_evidence_weight', '--egf_rps_commit_evidence_weight'),
        ('rps_commit_geometry_weight', '--egf_rps_commit_geometry_weight'),
        ('rps_commit_bg_weight', '--egf_rps_commit_bg_weight'),
        ('rps_commit_static_weight', '--egf_rps_commit_static_weight'),
        ('rps_commit_front_penalty', '--egf_rps_commit_front_penalty'),
    ]
    bonn_only_pairs = [
        ('bonn_rps_admission_support_on', '--egf_rps_admission_support_on'),
        ('bonn_rps_admission_support_gain', '--egf_rps_admission_support_gain'),
        ('bonn_rps_admission_score_relax', '--egf_rps_admission_score_relax'),
        ('bonn_rps_admission_active_floor', '--egf_rps_admission_active_floor'),
        ('bonn_rps_admission_rho_ref', '--egf_rps_admission_rho_ref'),
        ('bonn_rps_admission_weight_ref', '--egf_rps_admission_weight_ref'),
        ('bonn_rps_rear_state_decay_relax', '--egf_rps_rear_state_decay_relax'),
        ('bonn_rps_rear_state_active_floor', '--egf_rps_rear_state_active_floor'),
    ]
    for key, flag in common_pairs:
        if key in spec:
            tum_cmd = set_arg(tum_cmd, flag, str(spec[key]))
            bonn_cmd = set_arg(bonn_cmd, flag, str(spec[key]))
    for key, flag in bonn_only_pairs:
        if key in spec:
            bonn_cmd = set_arg(bonn_cmd, flag, str(spec[key]))
    return tum_cmd, bonn_cmd, root


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '30_rps_commit_geom_bg_soft_bank')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        active_gain = (
            row['bonn_extract_rear_selected_sum'] > control['bonn_extract_rear_selected_sum']
            or row['bonn_extract_rear_enabled_sum'] > control['bonn_extract_rear_enabled_sum']
            or row['bonn_extract_hard_commit_on_sum'] > control['bonn_extract_hard_commit_on_sum']
        )
        better = (
            row['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6
            or row['bonn_comp_r_5cm'] > control['bonn_comp_r_5cm'] + 0.05
        )
        no_regress = (
            row['tum_comp_r_5cm'] >= control['tum_comp_r_5cm'] - 0.50
            and row['tum_acc_cm'] <= control['tum_acc_cm'] * 1.10
            and row['bonn_acc_cm'] <= control['bonn_acc_cm'] * 1.10
        )
        row['decision'] = 'iterate' if active_gain and better and no_regress else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(iters, key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_extract_rear_selected_sum'], r['bonn_extract_rear_enabled_sum']))
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
        '# S2 committed rear-bank admission compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_extract_hard_commit_on_sum | bonn_extract_rear_enabled_sum | bonn_extract_rear_selected_sum | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_extract_hard_commit_on_sum']:.0f} | {row['bonn_extract_rear_enabled_sum']:.0f} | {row['bonn_extract_rear_selected_sum']:.0f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_diag_report(control_payload: Dict[str, dict], path_md: Path) -> None:
    chain = control_payload['chain']
    lines = [
        '# S2 committed rear-bank admission diagnosis',
        '',
        '日期：`2026-03-09`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '控制组：`30_rps_commit_geom_bg_soft_bank`',
        '',
        '## 1. 诊断目标',
        '- 追踪 committed rear states 在 `commit -> active -> sync -> extract` 各阶段的数量流失。',
        '- 定位最主要的丢失环节，判断当前瓶颈是否位于 active 维护、hard-commit admission，还是 extract 前的准入筛选。',
        '',
        '## 2. Bonn all3 链路统计',
    ]
    for seq in BONN_ALL3:
        d = chain[seq]
        lines.extend([
            f"### `{seq}`",
            f"- committed cells: `{d['committed_cells']:.0f}`",
            f"- active cells: `{d['active_cells']:.0f}`",
            f"- sync hard-commit on: `{d['sync_hard_commit_on']:.0f}` / sync rear enabled: `{d['sync_rear_enabled']:.0f}`",
            f"- extract wr present: `{d['extract_wr_present']:.0f}`",
            f"- extract hard-commit on: `{d['extract_hard_commit_on']:.0f}`",
            f"- extract rear enabled: `{d['extract_rear_enabled']:.0f}`",
            f"- extract rear selected: `{d['extract_rear_selected']:.0f}`",
            f"- extract fail_active: `{d['extract_fail_active']:.0f}` / fail_score: `{d['extract_fail_score']:.0f}`",
            f"- extract support_mean: `{d['extract_support_mean']:.4f}` / active_like_mean: `{d['extract_active_like_mean']:.4f}`",
            '',
        ])
    lines.extend([
        '## 3. 控制组瓶颈判断',
        '- 若 `committed_cells >> extract_hard_commit_on`，则说明 committed rear state 在 extract admission 之前已被 hard-commit gate 大量拦截。',
        '- 若 `extract_hard_commit_on ≈ extract_rear_selected`，则说明一旦通过 admission，后续 competition 并不是主要问题。',
        '- 因此本轮主线应优先提升 `extract_hard_commit_on` 与 `extract_rear_enabled`，而不是继续调 competition。',
    ])
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], control_payload: Dict[str, dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '30_rps_commit_geom_bg_soft_bank')
    best = max(rows[1:], key=lambda r: (r['bonn_extract_rear_selected_sum'], r['bonn_extract_rear_enabled_sum'], r['bonn_ghost_reduction_vs_tsdf']))
    lines = [
        '# S2 committed rear-bank admission analysis',
        '',
        '日期：`2026-03-09`',
        '协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '对比表：`processes/s2/S2_RPS_COMMITTED_BANK_ADMISSION_COMPARE.csv`',
        '',
        '## 1. 结果概览',
    ]
    for row in rows:
        lines.append(
            f"- `{row['variant']}`: Bonn `extract_hard_commit_on_sum = {row['bonn_extract_hard_commit_on_sum']:.0f}`, `extract_rear_enabled_sum = {row['bonn_extract_rear_enabled_sum']:.0f}`, `extract_rear_selected_sum = {row['bonn_extract_rear_selected_sum']:.0f}`, `ghost_reduction_vs_tsdf = {row['bonn_ghost_reduction_vs_tsdf']:.2f}%`, decision=`{row['decision']}`"
        )
    lines.extend([
        '',
        '## 2. 哪个变体提升了 `bonn_extract_rear_selected_sum`',
        f"- 最佳变体：`{best['variant']}`",
        f"- 控制组 `bonn_extract_rear_selected_sum = {control['bonn_extract_rear_selected_sum']:.0f}`，最佳变体为 `{best['bonn_extract_rear_selected_sum']:.0f}`。",
        f"- 同时其 `bonn_extract_hard_commit_on_sum` 从 `{control['bonn_extract_hard_commit_on_sum']:.0f}` 变化到 `{best['bonn_extract_hard_commit_on_sum']:.0f}`；`bonn_extract_rear_enabled_sum` 从 `{control['bonn_extract_rear_enabled_sum']:.0f}` 变化到 `{best['bonn_extract_rear_enabled_sum']:.0f}`。",
        '',
        '## 3. 是否转化为指标改善',
        f"- 控制组 Bonn `ghost_reduction_vs_tsdf = {control['bonn_ghost_reduction_vs_tsdf']:.2f}%`，最佳变体为 `{best['bonn_ghost_reduction_vs_tsdf']:.2f}%`。",
        f"- 控制组 Bonn `Comp-R = {control['bonn_comp_r_5cm']:.2f}%`，最佳变体为 `{best['bonn_comp_r_5cm']:.2f}%`。",
    ])
    if best['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6 or best['bonn_comp_r_5cm'] > control['bonn_comp_r_5cm'] + 1e-6:
        lines.append('- 结论：rear candidate admission 的提升已经开始转化为 Bonn 指标改善。')
    else:
        lines.append('- 结论：即便 admission 链有所改善，也尚未转化为 Bonn ghost / Comp-R 的可见提升。')
    lines.extend([
        '',
        '## 4. 是否达到 S2 门槛',
        f"- TUM mean(best): `Acc = {best['tum_acc_cm']:.4f} cm`, `Comp-R = {best['tum_comp_r_5cm']:.2f}%`",
        f"- Bonn mean(best): `Acc = {best['bonn_acc_cm']:.4f} cm`, `Comp-R = {best['bonn_comp_r_5cm']:.2f}%`, `ghost_reduction_vs_tsdf = {best['bonn_ghost_reduction_vs_tsdf']:.2f}%`",
        '- 结论：本轮若未同时触达 `Acc / Comp-R / ghost` 门槛，则 `S2` 仍未通过，绝对不能进入 `S3`。',
        '',
        '## 5. 失败归因与下一轮最合理方向',
    ])
    if best['bonn_extract_rear_selected_sum'] <= control['bonn_extract_rear_selected_sum']:
        lines.append('- 当前 admission 设计仍未把更多 committed rear states 推进到 extract 阶段，说明 admission 规则本身仍偏保守或作用位置不够靠前。')
    else:
        lines.append('- 当前 admission 设计已增加 extract 阶段 rear 候选，但 downstream geometry 质量或 extract filter 仍在限制最终指标转化。')
    lines.append('- 综合判断：若未过线，更接近 admission 设计仍不够对症，而不是简单的参数未收敛。')
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 committed rear-bank admission runner.')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=600)
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
    )
    variants = [
        dict(base_spec, name='30_rps_commit_geom_bg_soft_bank', output_name='30_rps_commit_geom_bg_soft_bank_admission_all3'),
        dict(
            base_spec,
            name='37_bonn_admission_gate_relax',
            bonn_rps_admission_support_enable=True,
            bonn_rps_admission_support_on=0.34,
            bonn_rps_admission_support_gain=0.70,
            bonn_rps_admission_score_relax=0.10,
            bonn_rps_admission_active_floor=0.30,
            bonn_rps_admission_rho_ref=0.07,
            bonn_rps_admission_weight_ref=0.30,
        ),
        dict(
            base_spec,
            name='38_bonn_state_protect',
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
        ),
        dict(
            base_spec,
            name='39_bonn_admission_gate_plus_protect',
            bonn_rps_admission_support_enable=True,
            bonn_rps_admission_support_on=0.30,
            bonn_rps_admission_support_gain=0.80,
            bonn_rps_admission_score_relax=0.14,
            bonn_rps_admission_active_floor=0.34,
            bonn_rps_admission_rho_ref=0.06,
            bonn_rps_admission_weight_ref=0.28,
            bonn_rps_rear_state_protect_enable=True,
            bonn_rps_rear_state_decay_relax=0.75,
            bonn_rps_rear_state_active_floor=0.38,
        ),
    ]

    rows: List[dict] = []
    payload: Dict[str, dict] = {}
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec)
        run(tum_cmd)
        run(bonn_cmd)
        row, tum_details, bonn_payload = summarize_variant(root, str(spec['name']))
        rows.append(row)
        payload[str(spec['name'])] = {'tum': tum_details, **bonn_payload}

    decide(rows)
    out_dir = PROJECT_ROOT / 'processes' / 's2'
    compare_csv = out_dir / 'S2_RPS_COMMITTED_BANK_ADMISSION_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_COMMITTED_BANK_ADMISSION_COMPARE.md'
    diag_md = out_dir / 'S2_RPS_COMMITTED_BANK_ADMISSION_DIAGNOSIS.md'
    analysis_md = out_dir / 'S2_RPS_COMMITTED_BANK_ADMISSION_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_diag_report(payload['30_rps_commit_geom_bg_soft_bank'], diag_md)
    write_analysis(rows, payload['30_rps_commit_geom_bg_soft_bank'], analysis_md)
    print(f'[done] {compare_csv} {compare_md} {diag_md} {analysis_md}')


if __name__ == '__main__':
    main()
