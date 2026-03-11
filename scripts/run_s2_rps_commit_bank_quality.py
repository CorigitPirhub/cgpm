from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

from run_s2_write_time_synthesis import (
    DEV_BONN,
    DEV_TUM,
    PROJECT_ROOT,
    dryrun_base_cmds,
    ensure_flag,
    eval_variant,
    run,
    set_arg,
)
from run_s2_rear_bg_state_recovery import build_variant_cmds as build_rearbg


def load_summary(root: Path, ds: str, seq: str) -> dict:
    return json.loads((root / ds / seq / 'egf' / 'summary.json').read_text(encoding='utf-8'))


def enrich_row(root: Path, metrics: dict, name: str) -> dict:
    tum_summary = load_summary(root, 'tum_oracle/oracle', DEV_TUM)
    bonn_summary = load_summary(root, 'bonn_slam/slam', DEV_BONN)
    tum_state = tum_summary.get('rear_bg_state_diag', {})
    bonn_state = bonn_summary.get('rear_bg_state_diag', {})
    tum_export = tum_summary.get('ptdsf_export_diag', {})
    bonn_export = bonn_summary.get('ptdsf_export_diag', {})
    tum_commit = tum_summary.get('rps_commit_diag', {})
    bonn_commit = bonn_summary.get('rps_commit_diag', {})
    return {
        'variant': name,
        **metrics,
        'tum_rear_w_nz': float(tum_state.get('rear_w_nonzero', 0.0)),
        'tum_rps_active_nz': float(tum_state.get('rps_active_nonzero', 0.0)),
        'tum_score_ge_on': float(tum_state.get('rps_commit_score_ge_on', 0.0)),
        'tum_rear_selected': float(tum_export.get('rear_selected', 0.0)),
        'tum_commit_ready': float(tum_commit.get('commit_ready', 0.0)),
        'tum_commit_w_sum': float(tum_commit.get('commit_w_sum', 0.0)),
        'bonn_rear_w_nz': float(bonn_state.get('rear_w_nonzero', 0.0)),
        'bonn_rps_active_nz': float(bonn_state.get('rps_active_nonzero', 0.0)),
        'bonn_score_ge_on': float(bonn_state.get('rps_commit_score_ge_on', 0.0)),
        'bonn_rear_selected': float(bonn_export.get('rear_selected', 0.0)),
        'bonn_commit_ready': float(bonn_commit.get('commit_ready', 0.0)),
        'bonn_commit_w_sum': float(bonn_commit.get('commit_w_sum', 0.0)),
        'bonn_commit_rho_sum': float(bonn_commit.get('commit_rho_sum', 0.0)),
        'bonn_commit_shift_sum': float(bonn_commit.get('commit_shift_sum', 0.0)),
        'decision': 'pending',
    }


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '30_rps_commit_geom_bg_soft_bank')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        active_gain = (
            row['bonn_rear_w_nz'] > control['bonn_rear_w_nz']
            or row['bonn_rear_selected'] > control['bonn_rear_selected']
            or row['bonn_commit_w_sum'] > control['bonn_commit_w_sum']
            or row['bonn_commit_ready'] > control['bonn_commit_ready']
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
        best = max(
            iters,
            key=lambda r: (
                r['bonn_ghost_reduction_vs_tsdf'],
                r['bonn_rear_selected'],
                r['bonn_commit_w_sum'],
                r['tum_comp_r_5cm'],
                -r['bonn_acc_cm'],
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
        '# S2 committed rear-bank quality compare',
        '',
        '协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_w_nz | bonn_rear_selected | bonn_commit_w_sum | bonn_commit_rho_sum | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_w_nz']:.0f} | {row['bonn_rear_selected']:.0f} | {row['bonn_commit_w_sum']:.3f} | {row['bonn_commit_rho_sum']:.3f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 committed rear-bank quality runner.')
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
        dict(base_spec, name='30_rps_commit_geom_bg_soft_bank'),
        dict(
            base_spec,
            name='31_rps_commit_quality_bank_mid',
            rps_commit_quality_enable=True,
            rps_commit_quality_transfer_gain=0.20,
            rps_commit_quality_rho_gain=0.60,
            rps_commit_quality_geom_blend=0.35,
            rps_commit_quality_sep_scale=0.75,
            rps_commit_min_cand_rho=0.0055,
            rps_commit_min_cand_w=0.022,
            rps_bank_margin=0.018,
            rps_bank_rear_min_score=0.17,
        ),
        dict(
            base_spec,
            name='32_rps_commit_quality_bank_geom',
            rps_commit_quality_enable=True,
            rps_commit_quality_transfer_gain=0.28,
            rps_commit_quality_rho_gain=0.75,
            rps_commit_quality_geom_blend=0.48,
            rps_commit_quality_sep_scale=0.90,
            rps_commit_threshold=0.33,
            rps_commit_release=0.20,
            rps_commit_min_cand_rho=0.0050,
            rps_commit_min_cand_w=0.020,
            rps_bank_margin=0.016,
            rps_bank_rear_min_score=0.16,
        ),
        dict(
            base_spec,
            name='33_rps_commit_quality_bank_push',
            rps_commit_quality_enable=True,
            rps_commit_quality_transfer_gain=0.34,
            rps_commit_quality_rho_gain=0.90,
            rps_commit_quality_geom_blend=0.58,
            rps_commit_quality_sep_scale=1.05,
            rps_commit_threshold=0.31,
            rps_commit_release=0.19,
            rps_commit_min_cand_rho=0.0045,
            rps_commit_min_cand_w=0.018,
            rps_bank_margin=0.014,
            rps_bank_rear_min_score=0.15,
        ),
    ]

    rows: List[dict] = []
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_rearbg(base_tum, base_bonn, spec)
        for cmd_name in ('tum', 'bonn'):
            cmd = tum_cmd if cmd_name == 'tum' else bonn_cmd
            if bool(spec.get('rps_commit_activation_enable', False)):
                cmd = ensure_flag(cmd, '--egf_rps_commit_activation_enable')
            if bool(spec.get('rps_surface_bank_enable', False)):
                cmd = ensure_flag(cmd, '--egf_rps_surface_bank_enable')
            if bool(spec.get('rps_commit_quality_enable', False)):
                cmd = ensure_flag(cmd, '--egf_rps_commit_quality_enable')
            for key, flag in [
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
                ('rps_commit_quality_transfer_gain', '--egf_rps_commit_quality_transfer_gain'),
                ('rps_commit_quality_rho_gain', '--egf_rps_commit_quality_rho_gain'),
                ('rps_commit_quality_geom_blend', '--egf_rps_commit_quality_geom_blend'),
                ('rps_commit_quality_sep_scale', '--egf_rps_commit_quality_sep_scale'),
            ]:
                if key in spec:
                    cmd = set_arg(cmd, flag, str(spec[key]))
            if cmd_name == 'tum':
                tum_cmd = cmd
            else:
                bonn_cmd = cmd
        run(tum_cmd)
        run(bonn_cmd)
        metrics = eval_variant(root, DEV_TUM, DEV_BONN)
        rows.append(enrich_row(root, metrics, str(spec['name'])))

    decide(rows)
    csv_path = PROJECT_ROOT / 'processes' / 's2' / 'S2_RPS_COMMITTED_BANK_QUALITY_COMPARE.csv'
    md_path = PROJECT_ROOT / 'processes' / 's2' / 'S2_RPS_COMMITTED_BANK_QUALITY_COMPARE.md'
    write_compare(rows, csv_path, md_path)
    print(f'[done] {csv_path} {md_path}')


if __name__ == '__main__':
    main()
