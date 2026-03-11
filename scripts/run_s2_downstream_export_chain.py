from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

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
    root = PROJECT_ROOT / 'output' / 'post_cleanup' / 's2_stage' / name

    tum = apply_frozen_14(base_tum, include_tsdf=True)
    tum = apply_tum_surface_defaults(tum)
    tum = set_arg(tum, '--frames', str(int(spec['frames'])))
    tum = set_arg(tum, '--stride', str(int(spec['stride'])))
    tum = set_arg(tum, '--seed', '7')
    tum = set_arg(tum, '--max_points_per_frame', str(int(spec['max_points'])))
    tum = set_arg(tum, '--out_root', str(root / 'tum_oracle'))
    tum = set_arg(tum, '--static_sequences', '')
    tum = set_arg(tum, '--dynamic_sequences', DEV_TUM)

    bonn = apply_frozen_14(base_bonn, include_tsdf=True)
    bonn = set_arg(bonn, '--frames', str(int(spec['frames'])))
    bonn = set_arg(bonn, '--stride', str(int(spec['stride'])))
    bonn = set_arg(bonn, '--seed', '7')
    bonn = set_arg(bonn, '--max_points_per_frame', str(int(spec['max_points'])))
    bonn = set_arg(bonn, '--out_root', str(root / 'bonn_slam'))
    bonn = set_arg(bonn, '--static_sequences', '')
    bonn = set_arg(bonn, '--bonn_dynamic_preset', 'balloon2')
    bonn = apply_bonn_calibration(
        bonn,
        phi_min=float(spec.get('bonn_phi_min', 0.64)),
        phi_max=float(spec.get('bonn_phi_max', 1.58)),
        min_weight=float(spec.get('bonn_min_weight', 0.01)),
        free_ratio=float(spec.get('bonn_free_ratio', 0.03)),
    )

    for cmd in (tum, bonn):
        if bool(spec.get('rps_surface_bank_enable', False)):
            ensure_flag(cmd, '--egf_rps_surface_bank_enable')
        else:
            remove_flag(cmd, '--egf_rps_surface_bank_enable')

    for cmd_name in ('tum', 'bonn'):
        cmd = tum if cmd_name == 'tum' else bonn
        for key, flag in [
            ('rps_bank_margin', '--egf_rps_bank_margin'),
            ('rps_bank_separation_ref', '--egf_rps_bank_separation_ref'),
            ('rps_bank_rear_min_score', '--egf_rps_bank_rear_min_score'),
            ('surface_dual_layer_compete_margin', '--egf_surface_dual_layer_compete_margin'),
            ('surface_dual_layer_compete_geo_weight', '--egf_surface_dual_layer_compete_geo_weight'),
            ('surface_dual_layer_compete_dyn_mix_weight', '--egf_surface_dual_layer_compete_dyn_mix_weight'),
            ('surface_dual_layer_compete_dyn_conf_weight', '--egf_surface_dual_layer_compete_dyn_conf_weight'),
            ('surface_dual_p_static_min', '--egf_surface_dual_p_static_min'),
        ]:
            if key in spec:
                cmd = set_arg(cmd, flag, str(spec[key]))
        if bool(spec.get('surface_dual_layer_compete_enable', False)):
            cmd = ensure_flag(cmd, '--egf_surface_dual_layer_compete_enable')
        else:
            cmd = remove_flag(cmd, '--egf_surface_dual_layer_compete_enable')
        if bool(spec.get('disable_ptdsf_persistent_only', False)):
            cmd = remove_flag(cmd, '--egf_surface_ptdsf_persistent_only_enable')
        if cmd_name == 'tum':
            tum = cmd
        else:
            bonn = cmd
    return tum, bonn, root


def enrich_row(root: Path, metrics: dict, name: str) -> dict:
    tum_summary = load_summary(root, 'tum_oracle/oracle', DEV_TUM)
    bonn_summary = load_summary(root, 'bonn_slam/slam', DEV_BONN)
    tum_diag = tum_summary.get('ptdsf_export_diag', {})
    bonn_diag = bonn_summary.get('ptdsf_export_diag', {})
    return {
        'variant': name,
        **metrics,
        'tum_bank_front': float(tum_diag.get('front_selected', 0.0)),
        'tum_bank_bg': float(tum_diag.get('bg_selected', 0.0)),
        'tum_bank_rear': float(tum_diag.get('rear_selected', 0.0)),
        'bonn_bank_front': float(bonn_diag.get('front_selected', 0.0)),
        'bonn_bank_bg': float(bonn_diag.get('bg_selected', 0.0)),
        'bonn_bank_rear': float(bonn_diag.get('rear_selected', 0.0)),
        'decision': 'pending',
    }


def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '14_downstream_control')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        better = (
            row['tum_comp_r_5cm'] > control['tum_comp_r_5cm'] + 0.10
            or row['bonn_comp_r_5cm'] > control['bonn_comp_r_5cm'] + 0.10
            or row['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 0.50
        )
        no_regress = row['tum_acc_cm'] <= control['tum_acc_cm'] + 0.10 and row['bonn_acc_cm'] <= control['bonn_acc_cm'] + 0.10
        row['decision'] = 'iterate' if better and no_regress else 'abandon'
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
        '# S2 downstream sync/export chain compare',
        '',
        '协议：`TUM/Bonn dev quick / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | tum_bank_front | tum_bank_bg | tum_bank_rear | bonn_bank_front | bonn_bank_bg | bonn_bank_rear | decision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['tum_bank_front']:.0f} | {row['tum_bank_bg']:.0f} | {row['tum_bank_rear']:.0f} | {row['bonn_bank_front']:.0f} | {row['bonn_bank_bg']:.0f} | {row['bonn_bank_rear']:.0f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='S2 downstream sync/export chain runner.')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=600)
    args = ap.parse_args()

    base_tum, base_bonn = dryrun_base_cmds(args.frames, args.stride, args.max_points_per_frame)
    variants: List[Dict[str, float | str | bool]] = [
        dict(name='14_downstream_control', frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame),
        dict(
            name='17_banked_export_compete',
            frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame,
            rps_surface_bank_enable=True,
            rps_bank_margin=0.04,
            rps_bank_separation_ref=0.03,
            rps_bank_rear_min_score=0.42,
        ),
        dict(
            name='18_banked_dual_compete_export',
            frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame,
            rps_surface_bank_enable=True,
            rps_bank_margin=0.04,
            rps_bank_separation_ref=0.03,
            rps_bank_rear_min_score=0.42,
            surface_dual_layer_compete_enable=True,
            surface_dual_layer_compete_margin=0.04,
            surface_dual_layer_compete_geo_weight=0.68,
            surface_dual_layer_compete_dyn_mix_weight=0.62,
            surface_dual_layer_compete_dyn_conf_weight=0.30,
        ),
        dict(
            name='19_banked_nonpersistent_export',
            frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame,
            rps_surface_bank_enable=True,
            rps_bank_margin=0.03,
            rps_bank_separation_ref=0.03,
            rps_bank_rear_min_score=0.40,
            disable_ptdsf_persistent_only=True,
            surface_dual_p_static_min=0.08,
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
    csv_path = PROJECT_ROOT / 'processes' / 's2' / 'S2_DOWNSTREAM_EXPORT_CHAIN_COMPARE.csv'
    md_path = PROJECT_ROOT / 'processes' / 's2' / 'S2_DOWNSTREAM_EXPORT_CHAIN_COMPARE.md'
    write_compare(rows, csv_path, md_path)
    print(f'[done] {csv_path} {md_path}')


if __name__ == '__main__':
    main()
