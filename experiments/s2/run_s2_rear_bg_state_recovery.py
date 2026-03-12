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
    remove_flag,
    run,
    set_arg,
)
from run_s2_bonn_localclip_refine import apply_bonn_calibration, apply_frozen_14, apply_tum_surface_defaults


def load_summary(root: Path, ds: str, seq: str) -> dict:
    return json.loads((root / ds / seq / 'egf' / 'summary.json').read_text(encoding='utf-8'))


def build_variant_cmds(base_tum: List[str], base_bonn: List[str], spec: Dict[str, float | str | bool]) -> tuple[List[str], List[str], Path]:
    name = str(spec['name'])
    root = PROJECT_ROOT / 'output' / 's2_stage' / name
    tum = apply_frozen_14(base_tum, include_tsdf=True)
    tum = apply_tum_surface_defaults(tum)
    bonn = apply_frozen_14(base_bonn, include_tsdf=True)
    bonn = apply_bonn_calibration(
        bonn,
        phi_min=float(spec.get('bonn_phi_min', 0.64)),
        phi_max=float(spec.get('bonn_phi_max', 1.58)),
        min_weight=float(spec.get('bonn_min_weight', 0.01)),
        free_ratio=float(spec.get('bonn_free_ratio', 0.03)),
    )
    for cmd_name in ('tum', 'bonn'):
        cmd = tum if cmd_name == 'tum' else bonn
        cmd = set_arg(cmd, '--frames', str(int(spec['frames'])))
        cmd = set_arg(cmd, '--stride', str(int(spec['stride'])))
        cmd = set_arg(cmd, '--seed', '7')
        cmd = set_arg(cmd, '--max_points_per_frame', str(int(spec['max_points'])))
        cmd = set_arg(cmd, '--out_root', str(root / ('tum_oracle' if cmd_name == 'tum' else 'bonn_slam')))
        cmd = set_arg(cmd, '--static_sequences', '')
        if cmd_name == 'tum':
            cmd = set_arg(cmd, '--dynamic_sequences', DEV_TUM)
        else:
            cmd = set_arg(cmd, '--bonn_dynamic_preset', 'balloon2')
        for flag_key, flag in [
            ('rps_candidate_rescue_enable', '--egf_rps_candidate_rescue_enable'),
            ('joint_bg_state_enable', '--egf_joint_bg_state_enable'),
        ]:
            if bool(spec.get(flag_key, False)):
                cmd = ensure_flag(cmd, flag)
            else:
                cmd = remove_flag(cmd, flag)
        for key, flag in [
            ('rps_candidate_support_gain', '--egf_rps_candidate_support_gain'),
            ('rps_candidate_bg_gain', '--egf_rps_candidate_bg_gain'),
            ('rps_candidate_rho_gain', '--egf_rps_candidate_rho_gain'),
            ('rps_candidate_front_relax', '--egf_rps_candidate_front_relax'),
            ('joint_bg_state_on', '--egf_joint_bg_state_on'),
            ('joint_bg_state_gain', '--egf_joint_bg_state_gain'),
            ('joint_bg_state_rho_gain', '--egf_joint_bg_state_rho_gain'),
            ('joint_bg_state_front_penalty', '--egf_joint_bg_state_front_penalty'),
        ]:
            if key in spec:
                cmd = set_arg(cmd, flag, str(spec[key]))
        if cmd_name == 'tum':
            tum = cmd
        else:
            bonn = cmd
    return tum, bonn, root


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
        'tum_bg_w_nz': float(tum_state.get('bg_w_nonzero', 0.0)),
        'tum_bg_cand_nz': float(tum_state.get('bg_cand_w_nonzero', 0.0)),
        'bonn_rear_w_nz': float(bonn_state.get('rear_w_nonzero', 0.0)),
        'bonn_rear_cand_nz': float(bonn_state.get('rear_cand_w_nonzero', 0.0)),
        'bonn_bg_w_nz': float(bonn_state.get('bg_w_nonzero', 0.0)),
        'bonn_bg_cand_nz': float(bonn_state.get('bg_cand_w_nonzero', 0.0)),
        'decision': 'pending',
    }


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '14_rearbg_control')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        active = (
            row['tum_rear_cand_nz'] > control['tum_rear_cand_nz']
            or row['bonn_rear_cand_nz'] > control['bonn_rear_cand_nz']
            or row['tum_bg_w_nz'] > control['tum_bg_w_nz']
            or row['bonn_bg_w_nz'] > control['bonn_bg_w_nz']
            or row['tum_bg_cand_nz'] > control['tum_bg_cand_nz']
            or row['bonn_bg_cand_nz'] > control['bonn_bg_cand_nz']
        )
        better = (
            row['tum_comp_r_5cm'] > control['tum_comp_r_5cm'] + 0.10
            or row['bonn_comp_r_5cm'] > control['bonn_comp_r_5cm'] + 0.10
            or row['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 0.50
        )
        no_regress = row['tum_acc_cm'] <= control['tum_acc_cm'] + 0.12 and row['bonn_acc_cm'] <= control['bonn_acc_cm'] + 0.12
        row['decision'] = 'iterate' if active and better and no_regress else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(iters, key=lambda r: (r['tum_comp_r_5cm'] + 0.5 * r['bonn_comp_r_5cm'] + 0.02 * r['bonn_ghost_reduction_vs_tsdf']))
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
        '# S2 rear/bg state formation compare',
        '',
        '协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | tum_rear_cand_nz | tum_bg_w_nz | tum_bg_cand_nz | bonn_rear_cand_nz | bonn_bg_w_nz | bonn_bg_cand_nz | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['tum_rear_cand_nz']:.0f} | {row['tum_bg_w_nz']:.0f} | {row['tum_bg_cand_nz']:.0f} | {row['bonn_rear_cand_nz']:.0f} | {row['bonn_bg_w_nz']:.0f} | {row['bonn_bg_cand_nz']:.0f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 rear/bg state formation runner.')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=600)
    args = ap.parse_args()

    base_tum, base_bonn = dryrun_base_cmds(args.frames, args.stride, args.max_points_per_frame)
    variants: List[Dict[str, float | str | bool]] = [
        dict(name='14_rearbg_control', frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame),
        dict(
            name='23_rear_candidate_support_rescue',
            frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame,
            rps_candidate_rescue_enable=True,
            rps_candidate_support_gain=0.34,
            rps_candidate_bg_gain=0.24,
            rps_candidate_rho_gain=0.20,
            rps_candidate_front_relax=0.32,
        ),
        dict(
            name='24_joint_bg_state_coformation',
            frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame,
            joint_bg_state_enable=True,
            joint_bg_state_on=0.16,
            joint_bg_state_gain=0.72,
            joint_bg_state_rho_gain=0.28,
            joint_bg_state_front_penalty=0.12,
        ),
        dict(
            name='25_rear_bg_coupled_formation',
            frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame,
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
        ),
    ]

    rows: List[dict] = []
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_variant_cmds(base_tum, base_bonn, spec)
        run(tum_cmd)
        run(bonn_cmd)
        metrics = eval_variant(root, DEV_TUM, DEV_BONN)
        rows.append(enrich_row(root, metrics, str(spec['name'])))

    decide(rows)
    csv_path = PROJECT_ROOT / 'output' / 's2' / 'S2_REAR_BG_STATE_FORMATION_COMPARE.csv'
    md_path = PROJECT_ROOT / 'output' / 's2' / 'S2_REAR_BG_STATE_FORMATION_COMPARE.md'
    write_compare(rows, csv_path, md_path)
    print(f'[done] {csv_path} {md_path}')


if __name__ == '__main__':
    main()
