from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

DEV_TUM = 'rgbd_dataset_freiburg3_walking_xyz'
DEV_BONN = 'rgbd_bonn_balloon2'
LOCKBOX_TUM = 'rgbd_dataset_freiburg3_walking_static'
LOCKBOX_BONN = 'rgbd_bonn_balloon'


def run(cmd: List[str], capture: bool = False) -> str:
    print('[cmd]', ' '.join(shlex.quote(str(x)) for x in cmd))
    if capture:
        cp = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, text=True, capture_output=True)
        return cp.stdout
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return ''


def set_arg(cmd: List[str], flag: str, value: str | None) -> List[str]:
    out = []
    i = 0
    replaced = False
    while i < len(cmd):
        if cmd[i] == flag:
            replaced = True
            if value is None:
                i += 1
                if i < len(cmd) and not str(cmd[i]).startswith('--'):
                    i += 1
                continue
            out.extend([flag, value])
            i += 1
            if i < len(cmd) and not str(cmd[i]).startswith('--'):
                i += 1
            continue
        out.append(cmd[i])
        i += 1
    if not replaced and value is not None:
        out.extend([flag, value])
    return out


def remove_flag(cmd: List[str], flag: str) -> List[str]:
    out = []
    i = 0
    while i < len(cmd):
        if cmd[i] == flag:
            i += 1
            if i < len(cmd) and not str(cmd[i]).startswith('--'):
                i += 1
            continue
        out.append(cmd[i])
        i += 1
    return out


def ensure_flag(cmd: List[str], flag: str) -> List[str]:
    if flag not in cmd:
        cmd.append(flag)
    return cmd


def dryrun_base_cmds(frames: int, stride: int, max_points: int) -> Tuple[List[str], List[str]]:
    dry = run([
        PY,
        'experiments/p10/run_p10_precision_profile.py',
        '--profiles', 'p10_ptdsf_zcbf_dccm_wdsgr_spg_a',
        '--frames', str(frames),
        '--stride', str(stride),
        '--max_points_per_frame', str(max_points),
        '--out_root', 'output/tmp/_s2_cmd_seed',
        '--summary_csv', 'output/tmp/_s2_cmd_seed.csv',
        '--figure', 'assets/_s2_cmd_seed.png',
        '--dry_run',
    ], capture=True)
    lines = [line for line in dry.splitlines() if line.startswith('[cmd] ')]
    if len(lines) < 2:
        raise RuntimeError('Failed to derive base commands from dry_run.')
    tum_cmd = shlex.split(lines[0][6:])
    bonn_cmd = shlex.split(lines[1][6:])
    return tum_cmd, bonn_cmd


def load_tables(root: Path) -> Tuple[List[dict], List[dict]]:
    rec = list(csv.DictReader((root / 'tables' / 'reconstruction_metrics.csv').open('r', encoding='utf-8')))
    dyn = list(csv.DictReader((root / 'tables' / 'dynamic_metrics.csv').open('r', encoding='utf-8')))
    return rec, dyn


def pick_row(rows: List[dict], sequence: str, method: str) -> dict:
    method = method.lower()
    for row in rows:
        if str(row.get('sequence', '')) == sequence and str(row.get('method', '')).lower() == method:
            return row
    raise KeyError(f'missing row sequence={sequence} method={method}')


def to_float(v: object) -> float:
    return float(v)


def eval_variant(root: Path, tum_seq: str, bonn_seq: str) -> Dict[str, float | str | bool]:
    tum_rec, tum_dyn = load_tables(root / 'tum_oracle' / 'oracle')
    bonn_rec, bonn_dyn = load_tables(root / 'bonn_slam' / 'slam')
    tr_egf = pick_row(tum_rec, tum_seq, 'egf')
    tr_tsdf = pick_row(tum_rec, tum_seq, 'tsdf') if any(str(r.get('method','')).lower() == 'tsdf' for r in tum_rec) else None
    td_egf = pick_row(tum_dyn, tum_seq, 'egf')
    td_tsdf = pick_row(tum_dyn, tum_seq, 'tsdf') if any(str(r.get('method','')).lower() == 'tsdf' for r in tum_dyn) else None
    br_egf = pick_row(bonn_rec, bonn_seq, 'egf')
    br_tsdf = pick_row(bonn_rec, bonn_seq, 'tsdf') if any(str(r.get('method','')).lower() == 'tsdf' for r in bonn_rec) else None
    bd_egf = pick_row(bonn_dyn, bonn_seq, 'egf')
    bd_tsdf = pick_row(bonn_dyn, bonn_seq, 'tsdf') if any(str(r.get('method','')).lower() == 'tsdf' for r in bonn_dyn) else None

    tum_acc = to_float(tr_egf['accuracy']) * 100.0
    tum_comp = to_float(tr_egf['recall_5cm']) * 100.0
    bonn_acc = to_float(br_egf['accuracy']) * 100.0
    bonn_comp = to_float(br_egf['recall_5cm']) * 100.0
    tum_ghost = to_float(td_egf['ghost_ratio'])
    bonn_ghost = to_float(bd_egf['ghost_ratio'])
    tum_red = ((to_float(td_tsdf['ghost_ratio']) - tum_ghost) / max(1e-9, to_float(td_tsdf['ghost_ratio']))) if td_tsdf else float('nan')
    bonn_red = ((to_float(bd_tsdf['ghost_ratio']) - bonn_ghost) / max(1e-9, to_float(bd_tsdf['ghost_ratio']))) if bd_tsdf else float('nan')
    hit_count = 0
    hit_count += 1 if tum_acc <= 2.55 else 0
    hit_count += 1 if bonn_acc <= 3.10 else 0
    hit_count += 1 if bonn_red * 100.0 >= 22.0 else 0
    return {
        'tum_acc_cm': tum_acc,
        'tum_comp_r_5cm': tum_comp,
        'bonn_acc_cm': bonn_acc,
        'bonn_comp_r_5cm': bonn_comp,
        'tum_ghost_reduction_vs_tsdf': tum_red * 100.0,
        'bonn_ghost_reduction_vs_tsdf': bonn_red * 100.0,
        'hit_2of4_partial': hit_count,
        'pass_comp': bool(tum_comp >= 98.0 and bonn_comp >= 98.0),
    }


def build_variant_cmds(base_tum: List[str], base_bonn: List[str], name: str, mode: str,
                       frames: int, stride: int, max_points: int, include_tsdf: bool,
                       no_synth: bool = False,
                       anchor: float = 0.55, geo: float = 0.35, bg: float = 0.20,
                       counter: float = 0.45, front_repel: float = 0.35, temp: float = 0.18, clip_vox: float = 2.4) -> Tuple[List[str], List[str], Path]:
    root = PROJECT_ROOT / 'output' / 's2_stage' / name
    tum = list(base_tum)
    bonn = list(base_bonn)
    tum = set_arg(tum, '--frames', str(frames))
    tum = set_arg(tum, '--stride', str(stride))
    tum = set_arg(tum, '--seed', '7')
    tum = set_arg(tum, '--max_points_per_frame', str(max_points))
    tum = set_arg(tum, '--out_root', str(root / 'tum_oracle'))
    tum = set_arg(tum, '--static_sequences', '')
    tum = set_arg(tum, '--dynamic_sequences', DEV_TUM)
    tum = set_arg(tum, '--methods', 'egf,tsdf' if include_tsdf else 'egf')
    bonn = set_arg(bonn, '--frames', str(frames))
    bonn = set_arg(bonn, '--stride', str(stride))
    bonn = set_arg(bonn, '--seed', '7')
    bonn = set_arg(bonn, '--max_points_per_frame', str(max_points))
    bonn = set_arg(bonn, '--out_root', str(root / 'bonn_slam'))
    bonn = set_arg(bonn, '--static_sequences', '')
    bonn = set_arg(bonn, '--bonn_dynamic_preset', 'balloon2')
    bonn = set_arg(bonn, '--methods', 'egf,tsdf' if include_tsdf else 'egf')
    for cmd_name in ('tum', 'bonn'):
        cmd = tum if cmd_name == 'tum' else bonn
        if no_synth:
            for flag in ('--egf_wdsg_enable', '--egf_wdsg_route_enable', '--egf_spg_enable'):
                cmd = remove_flag(cmd, flag)
        else:
            for flag in ('--egf_wdsg_enable', '--egf_wdsg_route_enable', '--egf_spg_enable'):
                cmd = ensure_flag(cmd, flag)
        cmd = set_arg(cmd, '--egf_wdsg_synth_mode', mode)
        cmd = set_arg(cmd, '--egf_wdsg_synth_anchor_gain', str(anchor))
        cmd = set_arg(cmd, '--egf_wdsg_synth_geo_gain', str(geo))
        cmd = set_arg(cmd, '--egf_wdsg_synth_bg_gain', str(bg))
        cmd = set_arg(cmd, '--egf_wdsg_synth_counterfactual_gain', str(counter))
        cmd = set_arg(cmd, '--egf_wdsg_synth_front_repel_gain', str(front_repel))
        cmd = set_arg(cmd, '--egf_wdsg_synth_energy_temp', str(temp))
        cmd = set_arg(cmd, '--egf_wdsg_synth_clip_vox', str(clip_vox))
        if cmd_name == 'tum':
            tum = cmd
        else:
            bonn = cmd
    return tum, bonn, root


def write_csv_md(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# S2 write-time target synthesis 对比表', '', '| variant | tum_acc_cm | tum_comp_r_5cm | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | hit_2of4_partial | pass_comp | decision |', '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        lines.append(f"| {row['variant']} | {row['tum_acc_cm']:.4f} | {row['tum_comp_r_5cm']:.2f} | {row['bonn_acc_cm']:.4f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['hit_2of4_partial']} | {row['pass_comp']} | {row['decision']} |")
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='Focused S2 runner for write-time target synthesis candidates.')
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--max_points_per_frame', type=int, default=600)
    args = ap.parse_args()

    base_tum, base_bonn = dryrun_base_cmds(args.frames, args.stride, args.max_points_per_frame)
    variants = [
        dict(name='00_no_synthesis_rps', mode='legacy', no_synth=True, include_tsdf=True),
        dict(name='01_weak_legacy_wdsg', mode='legacy', no_synth=False, include_tsdf=False),
        dict(name='02_anchor_synthesis', mode='anchor', no_synth=False, include_tsdf=False, anchor=0.62, geo=0.38, bg=0.08, counter=0.18, front_repel=0.22, temp=0.18, clip_vox=2.2),
        dict(name='03_counterfactual_synthesis', mode='counterfactual', no_synth=False, include_tsdf=False, anchor=0.66, geo=0.42, bg=0.10, counter=0.62, front_repel=0.38, temp=0.16, clip_vox=2.6),
        dict(name='04_energy_synthesis', mode='energy', no_synth=False, include_tsdf=False, anchor=0.70, geo=0.48, bg=0.16, counter=0.40, front_repel=0.30, temp=0.12, clip_vox=2.8),
    ]
    rows: List[dict] = []
    for spec in variants:
        tum_cmd, bonn_cmd, root = build_variant_cmds(base_tum, base_bonn, frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame, **spec)
        run(tum_cmd)
        run(bonn_cmd)
        metrics = eval_variant(root, DEV_TUM, DEV_BONN)
        row = {'variant': spec['name'], **metrics, 'decision': 'iterate'}
        rows.append(row)

    baseline = next(r for r in rows if r['variant'] == '01_weak_legacy_wdsg')
    for row in rows:
        if row['variant'] == '00_no_synthesis_rps':
            row['decision'] = 'baseline'
            continue
        if row['variant'] == '01_weak_legacy_wdsg':
            row['decision'] = 'weak'
            continue
        better = (row['tum_acc_cm'] < baseline['tum_acc_cm'] - 1e-6) or (row['bonn_acc_cm'] < baseline['bonn_acc_cm'] - 1e-6) or (row['bonn_ghost_reduction_vs_tsdf'] > baseline['bonn_ghost_reduction_vs_tsdf'] + 1e-6)
        row['decision'] = 'accept' if better and row['hit_2of4_partial'] >= 2 and row['pass_comp'] else 'abandon'

    # keep the best accepted candidate, if any
    accepted = [r for r in rows if r['decision'] == 'accept']
    if accepted:
        best = min(accepted, key=lambda r: (0 if r['pass_comp'] else 1, -(r['hit_2of4_partial']), r['tum_acc_cm'] + r['bonn_acc_cm'] - 0.05 * r['bonn_ghost_reduction_vs_tsdf']))
        for r in accepted:
            if r is not best:
                r['decision'] = 'abandon'
        best['decision'] = 'accept'

    csv_path = PROJECT_ROOT / 'output' / 's2' / 'S2_CONTROLLED_ABLATION_COMPARE.csv'
    md_path = PROJECT_ROOT / 'output' / 's2' / 'S2_CONTROLLED_ABLATION_COMPARE.md'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_md(rows, csv_path, md_path)
    print(f'[done] {csv_path} {md_path}')


if __name__ == '__main__':
    main()
