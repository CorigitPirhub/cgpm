from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS_DIR))

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


def apply_frozen_14(cmd: List[str], *, include_tsdf: bool) -> List[str]:
    out = list(cmd)
    out = ensure_flag(out, '--egf_rps_enable')
    out = ensure_flag(out, '--egf_rps_hard_commit_enable')
    out = ensure_flag(out, '--egf_wdsg_enable')
    out = remove_flag(out, '--egf_wdsg_route_enable')
    out = remove_flag(out, '--egf_spg_enable')
    out = set_arg(out, '--methods', 'egf,tsdf' if include_tsdf else 'egf')
    out = set_arg(out, '--egf_wdsg_synth_mode', 'anchor')
    out = set_arg(out, '--egf_wdsg_synth_anchor_gain', '0.18')
    out = set_arg(out, '--egf_wdsg_synth_geo_gain', '0.11')
    out = set_arg(out, '--egf_wdsg_synth_bg_gain', '0.07')
    out = set_arg(out, '--egf_wdsg_synth_counterfactual_gain', '0.06')
    out = set_arg(out, '--egf_wdsg_synth_front_repel_gain', '0.012')
    out = set_arg(out, '--egf_wdsg_synth_energy_temp', '0.22')
    out = set_arg(out, '--egf_wdsg_synth_clip_vox', '0.9')
    out = remove_flag(out, '--egf_wdsg_local_clip_enable')
    out = set_arg(out, '--egf_wdsg_local_clip_min_scale', '0.70')
    out = set_arg(out, '--egf_wdsg_local_clip_max_scale', '1.18')
    out = set_arg(out, '--egf_wdsg_local_clip_risk_gain', '0.52')
    out = set_arg(out, '--egf_wdsg_local_clip_expand_gain', '0.22')
    out = set_arg(out, '--egf_wdsg_local_clip_front_gate', '0.48')
    out = set_arg(out, '--egf_wdsg_local_clip_support_gate', '0.52')
    out = set_arg(out, '--egf_wdsg_local_clip_ambiguity_gate', '0.12')
    out = set_arg(out, '--egf_wdsg_local_clip_pfv_gain', '0.20')
    out = set_arg(out, '--egf_wdsg_local_clip_bg_gain', '0.18')
    return out


def apply_tum_surface_defaults(cmd: List[str]) -> List[str]:
    out = list(cmd)
    out = ensure_flag(out, '--egf_surface_adaptive_enable')
    out = set_arg(out, '--egf_surface_adaptive_rho_ref', '2.0')
    out = set_arg(out, '--egf_surface_adaptive_phi_min_scale', '0.55')
    out = set_arg(out, '--egf_surface_adaptive_phi_max_scale', '1.15')
    out = set_arg(out, '--egf_surface_adaptive_min_weight_gain', '0.8')
    out = set_arg(out, '--egf_surface_adaptive_free_ratio_gain', '0.5')
    return out


def apply_bonn_calibration(cmd: List[str], *, phi_min: float, phi_max: float, min_weight: float, free_ratio: float) -> List[str]:
    out = list(cmd)
    out = ensure_flag(out, '--egf_bonn_surface_adaptive_enable')
    out = set_arg(out, '--egf_bonn_surface_adaptive_rho_ref', '1.4')
    out = set_arg(out, '--egf_bonn_surface_adaptive_phi_min_scale', str(phi_min))
    out = set_arg(out, '--egf_bonn_surface_adaptive_phi_max_scale', str(phi_max))
    out = set_arg(out, '--egf_bonn_surface_adaptive_min_weight_gain', str(min_weight))
    out = set_arg(out, '--egf_bonn_surface_adaptive_free_ratio_gain', str(free_ratio))
    return out


def build_variant_cmds(base_tum: List[str], base_bonn: List[str], spec: Dict[str, float | str | bool]) -> tuple[List[str], List[str], Path]:
    name = str(spec['name'])
    root = PROJECT_ROOT / 'output' / 's2_stage' / name

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
    bonn = set_arg(bonn, '--egf_wdsg_synth_geo_gain', str(spec.get('bonn_geo_gain', 0.11)))
    bonn = set_arg(bonn, '--egf_wdsg_synth_bg_gain', str(spec.get('bonn_bg_gain', 0.07)))
    bonn = set_arg(bonn, '--egf_wdsg_synth_front_repel_gain', str(spec.get('bonn_front_repel', 0.012)))
    bonn = set_arg(bonn, '--egf_wdsg_synth_clip_vox', str(spec.get('bonn_clip_vox', 0.9)))
    bonn = apply_bonn_calibration(
        bonn,
        phi_min=float(spec.get('bonn_phi_min', 0.64)),
        phi_max=float(spec.get('bonn_phi_max', 1.58)),
        min_weight=float(spec.get('bonn_min_weight', 0.01)),
        free_ratio=float(spec.get('bonn_free_ratio', 0.03)),
    )
    if bool(spec.get('local_clip_enable', False)):
        bonn = ensure_flag(bonn, '--egf_wdsg_local_clip_enable')
    else:
        bonn = remove_flag(bonn, '--egf_wdsg_local_clip_enable')
    for key, flag in [
        ('local_clip_min_scale', '--egf_wdsg_local_clip_min_scale'),
        ('local_clip_max_scale', '--egf_wdsg_local_clip_max_scale'),
        ('local_clip_risk_gain', '--egf_wdsg_local_clip_risk_gain'),
        ('local_clip_expand_gain', '--egf_wdsg_local_clip_expand_gain'),
        ('local_clip_front_gate', '--egf_wdsg_local_clip_front_gate'),
        ('local_clip_support_gate', '--egf_wdsg_local_clip_support_gate'),
        ('local_clip_ambiguity_gate', '--egf_wdsg_local_clip_ambiguity_gate'),
        ('local_clip_pfv_gain', '--egf_wdsg_local_clip_pfv_gain'),
        ('local_clip_bg_gain', '--egf_wdsg_local_clip_bg_gain'),
    ]:
        if key in spec:
            bonn = set_arg(bonn, flag, str(spec[key]))
    return tum, bonn, root


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# S2 Bonn local clipping / calibration refinement compare',
        '',
        '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | decision |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['decision']} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')



def decide(rows: List[dict]) -> None:
    control = next(r for r in rows if r['variant'] == '14_bonn_localclip_drive_recheck')
    for row in rows:
        if row is control:
            row['decision'] = 'control'
            continue
        keep_tum = row['tum_comp_r_5cm'] >= 99.0
        bonn_better = (
            row['bonn_comp_r_5cm'] > control['bonn_comp_r_5cm'] + 0.10
            or row['bonn_acc_cm'] < control['bonn_acc_cm'] - 0.03
            or row['bonn_ghost_reduction_vs_tsdf'] > control['bonn_ghost_reduction_vs_tsdf'] + 1.00
        )
        row['decision'] = 'iterate' if keep_tum and bonn_better else 'abandon'
    iters = [r for r in rows if r['decision'] == 'iterate']
    if iters:
        best = max(
            iters,
            key=lambda r: (
                r['bonn_comp_r_5cm'],
                -r['bonn_acc_cm'],
                r['bonn_ghost_reduction_vs_tsdf'],
                -abs(r['tum_comp_r_5cm'] - control['tum_comp_r_5cm']),
            ),
        )
        for row in iters:
            if row is not best:
                row['decision'] = 'abandon'



def main() -> None:
    ap = argparse.ArgumentParser(description='S2 Bonn local clipping / calibration refinement runner.')
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=600)
    args = ap.parse_args()

    base_tum, base_bonn = dryrun_base_cmds(args.frames, args.stride, args.max_points_per_frame)
    variants: List[Dict[str, float | str | bool]] = [
        dict(
            name='14_bonn_localclip_drive_recheck',
            frames=args.frames,
            stride=args.stride,
            max_points=args.max_points_per_frame,
            bonn_phi_min=0.64,
            bonn_phi_max=1.58,
            bonn_min_weight=0.01,
            bonn_free_ratio=0.03,
            bonn_geo_gain=0.11,
            bonn_bg_gain=0.07,
            bonn_front_repel=0.012,
            bonn_clip_vox=0.90,
            local_clip_enable=False,
        ),
        dict(
            name='15_bonn_localclip_band_relax',
            frames=args.frames,
            stride=args.stride,
            max_points=args.max_points_per_frame,
            bonn_phi_min=0.62,
            bonn_phi_max=1.62,
            bonn_min_weight=0.01,
            bonn_free_ratio=0.02,
            bonn_geo_gain=0.115,
            bonn_bg_gain=0.075,
            bonn_front_repel=0.010,
            bonn_clip_vox=0.92,
            local_clip_enable=True,
            local_clip_min_scale=0.80,
            local_clip_max_scale=1.26,
            local_clip_risk_gain=0.44,
            local_clip_expand_gain=0.24,
            local_clip_front_gate=0.50,
            local_clip_support_gate=0.56,
            local_clip_ambiguity_gate=0.10,
            local_clip_pfv_gain=0.10,
            local_clip_bg_gain=0.24,
        ),
        dict(
            name='16_bonn_localclip_pfv_rearexpand',
            frames=args.frames,
            stride=args.stride,
            max_points=args.max_points_per_frame,
            bonn_phi_min=0.61,
            bonn_phi_max=1.66,
            bonn_min_weight=0.005,
            bonn_free_ratio=0.02,
            bonn_geo_gain=0.12,
            bonn_bg_gain=0.08,
            bonn_front_repel=0.010,
            bonn_clip_vox=0.95,
            local_clip_enable=True,
            local_clip_min_scale=0.74,
            local_clip_max_scale=1.34,
            local_clip_risk_gain=0.56,
            local_clip_expand_gain=0.28,
            local_clip_front_gate=0.46,
            local_clip_support_gate=0.54,
            local_clip_ambiguity_gate=0.08,
            local_clip_pfv_gain=0.28,
            local_clip_bg_gain=0.26,
        ),
    ]

    rows: List[dict] = []
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_variant_cmds(base_tum, base_bonn, spec)
        run(tum_cmd)
        run(bonn_cmd)
        metrics = eval_variant(root, DEV_TUM, DEV_BONN)
        rows.append({'variant': spec['name'], **metrics, 'decision': 'pending'})

    decide(rows)
    csv_path = PROJECT_ROOT / 'output' / 's2' / 'S2_BONN_LOCALCLIP_REFINEMENT_COMPARE.csv'
    md_path = PROJECT_ROOT / 'output' / 's2' / 'S2_BONN_LOCALCLIP_REFINEMENT_COMPARE.md'
    write_compare(rows, csv_path, md_path)
    print(f'[done] {csv_path} {md_path}')


if __name__ == '__main__':
    main()
