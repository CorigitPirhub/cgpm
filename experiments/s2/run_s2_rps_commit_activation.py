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
    return {
        'variant': name,
        **metrics,
        'tum_rear_w_nz': float(tum_state.get('rear_w_nonzero', 0.0)),
        'tum_rear_cand_nz': float(tum_state.get('rear_cand_w_nonzero', 0.0)),
        'tum_rps_active_nz': float(tum_state.get('rps_active_nonzero', 0.0)),
        'bonn_rear_w_nz': float(bonn_state.get('rear_w_nonzero', 0.0)),
        'bonn_rear_cand_nz': float(bonn_state.get('rear_cand_w_nonzero', 0.0)),
        'bonn_rps_active_nz': float(bonn_state.get('rps_active_nonzero', 0.0)),
        'decision': 'pending',
    }


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '25_rps_commit_control')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        active = row['tum_rear_w_nz'] > control['tum_rear_w_nz'] or row['bonn_rear_w_nz'] > control['bonn_rear_w_nz'] or row['tum_rps_active_nz'] > control['tum_rps_active_nz'] or row['bonn_rps_active_nz'] > control['bonn_rps_active_nz']
        better = row['tum_comp_r_5cm'] >= control['tum_comp_r_5cm'] and row['bonn_ghost_reduction_vs_tsdf'] >= control['bonn_ghost_reduction_vs_tsdf']
        no_regress = row['tum_acc_cm'] <= control['tum_acc_cm'] * 1.10 and row['bonn_acc_cm'] <= control['bonn_acc_cm'] * 1.10
        row['decision'] = 'iterate' if active and better and no_regress else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(iters, key=lambda r: (r['tum_comp_r_5cm'] + 0.02 * r['bonn_ghost_reduction_vs_tsdf'] + 0.001 * (r['tum_rear_w_nz'] + r['bonn_rear_w_nz'])))
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
        '# S2 RPS commit activation compare',
        '',
        '协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | tum_rear_w_nz | tum_rps_active_nz | bonn_rear_w_nz | bonn_rps_active_nz | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['tum_rear_w_nz']:.0f} | {row['tum_rps_active_nz']:.0f} | {row['bonn_rear_w_nz']:.0f} | {row['bonn_rps_active_nz']:.0f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 RPS commit activation runner.')
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
    )
    variants = [
        dict(name='25_rps_commit_control', **base_spec),
        dict(
            name='28_rps_commit_geom_bg_soft',
            **base_spec,
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
        ),
        dict(
            name='29_rps_commit_geom_bg_mid',
            **base_spec,
            rps_commit_activation_enable=True,
            rps_commit_threshold=0.42,
            rps_commit_release=0.28,
            rps_commit_age_threshold=1.0,
            rps_commit_rho_ref=0.045,
            rps_commit_weight_ref=0.45,
            rps_commit_min_cand_rho=0.008,
            rps_commit_min_cand_w=0.04,
            rps_commit_evidence_weight=0.34,
            rps_commit_geometry_weight=0.36,
            rps_commit_bg_weight=0.18,
            rps_commit_static_weight=0.12,
            rps_commit_front_penalty=0.16,
        ),
        dict(
            name='30_rps_commit_geom_bg_soft_bank',
            **base_spec,
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
    csv_path = PROJECT_ROOT / 'output' / 's2' / 'S2_RPS_COMMIT_ACTIVATION_COMPARE.csv'
    md_path = PROJECT_ROOT / 'output' / 's2' / 'S2_RPS_COMMIT_ACTIVATION_COMPARE.md'
    write_compare(rows, csv_path, md_path)
    print(f'[done] {csv_path} {md_path}')


if __name__ == '__main__':
    main()
