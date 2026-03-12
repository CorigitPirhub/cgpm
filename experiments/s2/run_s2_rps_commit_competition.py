from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
S2_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT_BOOTSTRAP / "scripts"
for _path in (PROJECT_ROOT_BOOTSTRAP, S2_DIR, ROOT_SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_s2_write_time_synthesis import (
    PROJECT_ROOT,
    dryrun_base_cmds,
    ensure_flag,
    run,
    set_arg,
)
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
            'ghost_ratio': ghost_ratio,
            'tsdf_ghost_ratio': tsdf_ghost,
        }
    return {
        'acc_cm_mean': float(mean(accs)),
        'comp_r_5cm_mean': float(mean(comps)),
        'ghost_reduction_vs_tsdf_mean': float(mean(ghost_reds)),
    }, details


def extract_diag(summary: dict) -> dict:
    export_diag = summary.get('ptdsf_export_diag', {})
    comp_diag = summary.get('rps_competition_diag', {})
    return {
        'rear_selected_sync_export': float(export_diag.get('rear_selected', 0.0)),
        'extract_considered': float(comp_diag.get('extract_considered', 0.0)),
        'extract_rear_selected': float(comp_diag.get('extract_rear_selected', 0.0)),
        'extract_rear_soft_selected': float(comp_diag.get('extract_rear_soft_selected', 0.0)),
        'extract_front_kept': float(comp_diag.get('extract_front_kept', 0.0)),
        'extract_hard_ready': float(comp_diag.get('extract_hard_ready', 0.0)),
        'extract_soft_ready': float(comp_diag.get('extract_soft_ready', 0.0)),
        'extract_fail_sep': float(comp_diag.get('extract_fail_sep', 0.0)),
        'extract_fail_min': float(comp_diag.get('extract_fail_min', 0.0)),
        'extract_fail_margin': float(comp_diag.get('extract_fail_margin', 0.0)),
        'extract_fail_soft_support': float(comp_diag.get('extract_fail_soft_support', 0.0)),
        'extract_close_gap': float(comp_diag.get('extract_close_gap', 0.0)),
        'front_score_mean': float(comp_diag.get('extract_front_score_sum', 0.0) / max(1.0, comp_diag.get('extract_considered', 0.0))),
        'rear_score_mean': float(comp_diag.get('extract_rear_score_sum', 0.0) / max(1.0, comp_diag.get('extract_considered', 0.0))),
        'rear_score_eff_mean': float(comp_diag.get('extract_rear_score_eff_sum', 0.0) / max(1.0, comp_diag.get('extract_considered', 0.0))),
        'rear_gap_mean': float(comp_diag.get('extract_gap_sum', 0.0) / max(1.0, comp_diag.get('extract_considered', 0.0))),
        'rear_sep_mean': float(comp_diag.get('extract_sep_sum', 0.0) / max(1.0, comp_diag.get('extract_considered', 0.0))),
        'rear_bg_support_mean': float(comp_diag.get('extract_bg_support_sum', 0.0) / max(1.0, comp_diag.get('extract_considered', 0.0))),
        'rear_local_support_mean': float(comp_diag.get('extract_local_support_sum', 0.0) / max(1.0, comp_diag.get('extract_considered', 0.0))),
        'gap_hist': {k: float(v) for k, v in comp_diag.items() if str(k).startswith('extract_gap_')},
    }


def summarize_variant(root: Path, name: str) -> Tuple[dict, Dict[str, dict], Dict[str, dict]]:
    tum_family, tum_details = family_metrics(root, 'tum', TUM_ALL3)
    bonn_family, bonn_details = family_metrics(root, 'bonn', BONN_ALL3)
    seq_diags: Dict[str, dict] = {}
    for seq in BONN_ALL3:
        seq_diags[seq] = extract_diag(seq_summary(root, 'bonn', seq))
    rear_sync = sum(diag['rear_selected_sync_export'] for diag in seq_diags.values())
    rear_extract = sum(diag['extract_rear_selected'] + diag['extract_rear_soft_selected'] for diag in seq_diags.values())
    row = {
        'variant': name,
        'tum_acc_cm': tum_family['acc_cm_mean'],
        'tum_comp_r_5cm': tum_family['comp_r_5cm_mean'],
        'bonn_acc_cm': bonn_family['acc_cm_mean'],
        'bonn_comp_r_5cm': bonn_family['comp_r_5cm_mean'],
        'bonn_ghost_reduction_vs_tsdf': bonn_family['ghost_reduction_vs_tsdf_mean'],
        'bonn_rear_selected_sync_sum': rear_sync,
        'bonn_extract_rear_selected_sum': rear_extract,
        'bonn_extract_considered_sum': sum(diag['extract_considered'] for diag in seq_diags.values()),
        'bonn_extract_close_gap_sum': sum(diag['extract_close_gap'] for diag in seq_diags.values()),
        'hit_2of4_partial': float(
            (tum_family['acc_cm_mean'] <= 2.55)
            + (bonn_family['acc_cm_mean'] <= 3.10)
            + (bonn_family['ghost_reduction_vs_tsdf_mean'] >= 22.0)
            + (tum_family['comp_r_5cm_mean'] >= 98.0 or bonn_family['comp_r_5cm_mean'] >= 98.0)
        ),
        'pass_comp': bool(tum_family['comp_r_5cm_mean'] >= 98.0 and bonn_family['comp_r_5cm_mean'] >= 98.0),
        'decision': 'pending',
    }
    return row, tum_details, {'metrics': bonn_details, 'diag': seq_diags}


def set_bonn_only_arg(tum_cmd: List[str], bonn_cmd: List[str], flag: str, value: str | None) -> Tuple[List[str], List[str]]:
    return tum_cmd, set_arg(bonn_cmd, flag, value)


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
        if cmd_name == 'bonn' and bool(spec.get('bonn_rps_bank_soft_competition_enable', False)):
            cmd = ensure_flag(cmd, '--egf_rps_bank_soft_competition_enable')
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
        ('bonn_rps_bank_margin', '--egf_rps_bank_margin'),
        ('bonn_rps_bank_rear_min_score', '--egf_rps_bank_rear_min_score'),
        ('bonn_rps_bank_sep_gate', '--egf_rps_bank_sep_gate'),
        ('bonn_rps_bank_bg_support_gain', '--egf_rps_bank_bg_support_gain'),
        ('bonn_rps_bank_front_dyn_penalty_gain', '--egf_rps_bank_front_dyn_penalty_gain'),
        ('bonn_rps_bank_rear_score_bias', '--egf_rps_bank_rear_score_bias'),
        ('bonn_rps_bank_soft_competition_gap', '--egf_rps_bank_soft_competition_gap'),
        ('bonn_rps_bank_soft_sep_relax', '--egf_rps_bank_soft_sep_relax'),
        ('bonn_rps_bank_soft_rear_min_relax', '--egf_rps_bank_soft_rear_min_relax'),
        ('bonn_rps_bank_soft_support_min', '--egf_rps_bank_soft_support_min'),
        ('bonn_rps_bank_soft_local_support_gain', '--egf_rps_bank_soft_local_support_gain'),
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
            or row['bonn_rear_selected_sync_sum'] > control['bonn_rear_selected_sync_sum']
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
        best = max(iters, key=lambda r: (r['bonn_ghost_reduction_vs_tsdf'], r['bonn_extract_rear_selected_sum'], r['bonn_comp_r_5cm']))
        for row in iters:
            if row is not best:
                row['decision'] = 'abandon'


def write_compare(rows: List[dict], path_csv: Path, path_md: Path) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# S2 committed rear-bank competition compare',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_selected_sync_sum | bonn_extract_rear_selected_sum | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_selected_sync_sum']:.0f} | {row['bonn_extract_rear_selected_sum']:.0f} | {row['decision']} |"
        )
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_diag_report(diag_payload: Dict[str, dict], path_md: Path) -> None:
    lines = [
        '# S2 Bonn committed rear-bank competition diagnosis',
        '',
        '协议：`Bonn all3 / slam / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '## 1. 现象',
        '- 当前 `30_rps_commit_geom_bg_soft_bank` 已能形成 committed rear state，但 Bonn 最终导出中的 rear 胜出仍偏少。',
        '- 本页只分析 extract 阶段的 rear-vs-front competition，不再混入 sync 期调用。',
        '',
    ]
    control = diag_payload['30_rps_commit_geom_bg_soft_bank']
    lines.extend([
        '## 2. 控制组根因诊断',
        f"- 控制组 Bonn extract considered 总数：`{sum(v['extract_considered'] for v in control['diag'].values()):.0f}`",
        f"- 控制组 Bonn extract rear 胜出总数：`{sum(v['extract_rear_selected'] + v['extract_rear_soft_selected'] for v in control['diag'].values()):.0f}`",
        f"- 控制组 Bonn sync/export 旧口径 rear_selected 总数：`{sum(v['rear_selected_sync_export'] for v in control['diag'].values()):.0f}`",
        '',
    ])
    for seq in BONN_ALL3:
        d = control['diag'][seq]
        lines.extend([
            f"### `{seq}`",
            f"- `front_score_mean = {d['front_score_mean']:.4f}`，`rear_score_mean = {d['rear_score_mean']:.4f}`，`rear_score_eff_mean = {d['rear_score_eff_mean']:.4f}`",
            f"- `rear_gap_mean = {d['rear_gap_mean']:.4f}`，`rear_sep_mean = {d['rear_sep_mean']:.4f}`",
            f"- `rear_bg_support_mean = {d['rear_bg_support_mean']:.4f}`，`rear_local_support_mean = {d['rear_local_support_mean']:.4f}`",
            f"- `extract_considered = {d['extract_considered']:.0f}`，`extract_rear_selected = {d['extract_rear_selected']:.0f}`，`extract_rear_soft_selected = {d['extract_rear_soft_selected']:.0f}`",
            f"- fail counts: `sep={d['extract_fail_sep']:.0f}` / `margin={d['extract_fail_margin']:.0f}` / `min={d['extract_fail_min']:.0f}` / `soft_support={d['extract_fail_soft_support']:.0f}`",
            f"- gap hist: `{d['gap_hist']}`",
            '',
        ])
    lines.extend([
        '## 3. 根因总结',
        '- 控制组的核心问题若表现为 `extract_fail_margin` 高于 `extract_rear_selected`，则说明 rear state 已形成，但最终输在 front-vs-rear score margin。',
        '- 若 `extract_fail_sep` 占比高，则说明不是证据不足，而是 rear/front separation gate 过硬。',
        '- 若 `extract_close_gap` 较多，则说明适合尝试 soft competition / near-tie rescue。',
        '',
    ])
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_analysis(rows: List[dict], diag_payload: Dict[str, dict], path_md: Path) -> None:
    control = next(r for r in rows if r['variant'] == '30_rps_commit_geom_bg_soft_bank')
    best = max(rows[1:], key=lambda r: (r['bonn_extract_rear_selected_sum'], r['bonn_rear_selected_sync_sum'], r['bonn_ghost_reduction_vs_tsdf']))
    lines = [
        '# S2 committed rear-bank competition analysis',
        '',
        '协议：`TUM walking all3 / Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        f"对比表：`{path_md.parent / 'S2_RPS_COMMITTED_BANK_COMPETITION_COMPARE.csv'}`",
        '',
        '## 1. 结果概览',
    ]
    for row in rows:
        lines.append(
            f"- `{row['variant']}`: Bonn `ghost_reduction_vs_tsdf = {row['bonn_ghost_reduction_vs_tsdf']:.2f}%`, `rear_selected_sync_sum = {row['bonn_rear_selected_sync_sum']:.0f}`, `extract_rear_selected_sum = {row['bonn_extract_rear_selected_sum']:.0f}`, decision=`{row['decision']}`"
        )
    lines.extend([
        '',
        '## 2. 哪个变体提升了 rear_selected',
        f"- 相对控制组，rear 胜出提升最大的变体是 `{best['variant']}`：`extract_rear_selected_sum` 从 `{control['bonn_extract_rear_selected_sum']:.0f}` 提到 `{best['bonn_extract_rear_selected_sum']:.0f}`。",
        f"- 同时其旧口径 `rear_selected_sync_sum` 从 `{control['bonn_rear_selected_sync_sum']:.0f}` 变化到 `{best['bonn_rear_selected_sync_sum']:.0f}`。",
        '',
        '## 3. 是否转化为 ghost 改善',
        f"- 最佳变体 `{best['variant']}` 的 Bonn `ghost_reduction_vs_tsdf` 为 `{best['bonn_ghost_reduction_vs_tsdf']:.2f}%`；控制组为 `{control['bonn_ghost_reduction_vs_tsdf']:.2f}%`。",
    ])
    if best['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1e-6:
        lines.append(f"- 结论：rear 胜出增量已转化为 ghost 改善，改善幅度为 `{best['bonn_ghost_reduction_vs_tsdf'] - control['bonn_ghost_reduction_vs_tsdf']:.2f}` 个百分点。")
    else:
        lines.append('- 结论：rear 胜出即使有变化，也尚未转化为可见的 Bonn ghost 改善。')
    lines.extend([
        '',
        '## 4. 是否达到 S2 门槛',
        f"- TUM mean: `Acc = {control['tum_acc_cm']:.4f} cm`, `Comp-R = {control['tum_comp_r_5cm']:.2f}%`（门槛：`<=2.55 / >=98`）",
        f"- Bonn mean(best): `Acc = {best['bonn_acc_cm']:.4f} cm`, `Comp-R = {best['bonn_comp_r_5cm']:.2f}%`, `ghost_reduction_vs_tsdf = {best['bonn_ghost_reduction_vs_tsdf']:.2f}%`（门槛：`<=3.10 / >=98 / >=22`）",
        '- 结论：本轮仍未达到 S2 门槛，绝对不能进入 S3。',
        '',
        '## 5. 失败归因',
    ])
    if best['bonn_extract_rear_selected_sum'] <= control['bonn_extract_rear_selected_sum']:
        lines.append('- 当前变体甚至未能明显扩大 extract 阶段 rear 胜出规模，说明竞争公式本身仍不够有效。')
    else:
        lines.append('- 当前变体已扩大 extract 阶段 rear 胜出规模，但仍未推动 ghost 指标，说明仅靠竞争阈值软化还不够，rear 胜出区域的空间组织或后验过滤仍在吞噬收益。')
    lines.append('- 综合判断：当前更接近“竞争机制仍偏硬或分数设计仍错配”，而不是简单的参数未收敛。')
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 committed rear-bank competition runner.')
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
        dict(base_spec, name='30_rps_commit_geom_bg_soft_bank', output_name='30_rps_commit_geom_bg_soft_bank_compete_all3'),
        dict(
            base_spec,
            name='34_bonn_compete_bgscore',
            bonn_rps_bank_margin=0.015,
            bonn_rps_bank_rear_min_score=0.16,
            bonn_rps_bank_sep_gate=0.18,
            bonn_rps_bank_bg_support_gain=0.14,
            bonn_rps_bank_front_dyn_penalty_gain=0.06,
            bonn_rps_bank_rear_score_bias=0.03,
        ),
        dict(
            base_spec,
            name='35_bonn_compete_softgap',
            bonn_rps_bank_margin=0.012,
            bonn_rps_bank_rear_min_score=0.15,
            bonn_rps_bank_sep_gate=0.17,
            bonn_rps_bank_bg_support_gain=0.16,
            bonn_rps_bank_front_dyn_penalty_gain=0.08,
            bonn_rps_bank_rear_score_bias=0.04,
            bonn_rps_bank_soft_competition_enable=True,
            bonn_rps_bank_soft_competition_gap=0.08,
            bonn_rps_bank_soft_sep_relax=0.05,
            bonn_rps_bank_soft_rear_min_relax=0.04,
            bonn_rps_bank_soft_support_min=0.40,
            bonn_rps_bank_soft_local_support_gain=1.05,
        ),
        dict(
            base_spec,
            name='36_bonn_compete_softgap_support',
            bonn_rps_bank_margin=0.010,
            bonn_rps_bank_rear_min_score=0.14,
            bonn_rps_bank_sep_gate=0.15,
            bonn_rps_bank_bg_support_gain=0.20,
            bonn_rps_bank_front_dyn_penalty_gain=0.10,
            bonn_rps_bank_rear_score_bias=0.05,
            bonn_rps_bank_soft_competition_enable=True,
            bonn_rps_bank_soft_competition_gap=0.12,
            bonn_rps_bank_soft_sep_relax=0.08,
            bonn_rps_bank_soft_rear_min_relax=0.06,
            bonn_rps_bank_soft_support_min=0.34,
            bonn_rps_bank_soft_local_support_gain=1.20,
        ),
    ]

    rows: List[dict] = []
    diag_payload: Dict[str, dict] = {}
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec)
        run(tum_cmd)
        run(bonn_cmd)
        row, tum_details, bonn_payload = summarize_variant(root, str(spec['name']))
        rows.append(row)
        diag_payload[str(spec['name'])] = {'tum': tum_details, **bonn_payload}

    decide(rows)
    out_dir = PROJECT_ROOT / 'output' / 's2'
    compare_csv = out_dir / 'S2_RPS_COMMITTED_BANK_COMPETITION_COMPARE.csv'
    compare_md = out_dir / 'S2_RPS_COMMITTED_BANK_COMPETITION_COMPARE.md'
    diag_md = out_dir / 'S2_RPS_COMMITTED_BANK_COMPETITION_DIAGNOSIS.md'
    analysis_md = out_dir / 'S2_RPS_COMMITTED_BANK_COMPETITION_ANALYSIS.md'
    write_compare(rows, compare_csv, compare_md)
    write_diag_report(diag_payload, diag_md)
    write_analysis(rows, diag_payload, analysis_md)
    print(f'[done] {compare_csv} {compare_md} {diag_md} {analysis_md}')


if __name__ == '__main__':
    main()
