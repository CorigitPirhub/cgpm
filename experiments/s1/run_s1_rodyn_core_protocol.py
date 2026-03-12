from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
RUNNER = PROJECT_ROOT / 'scripts' / 'external' / 'run_rodyn_slam_runner.py'
ADAPTER = PROJECT_ROOT / 'scripts' / 'adapters' / 'run_rodyn_slam_adapter.py'

SEQUENCES = [
    {'sequence': 'rgbd_dataset_freiburg3_walking_xyz', 'dataset_kind': 'tum', 'dataset_root': 'data/tum', 'tag': 'walking_xyz_f5', 'frames': 5, 'stride': 3, 'first_iters': 20, 'iters': 3, 'track_iters': 5, 'edge_iters': 0, 'mesh_resolution': 96, 'mesh_voxel': 0.10, 'map_every': 999999, 'keyframe_every': 999999},
    {'sequence': 'rgbd_dataset_freiburg3_walking_static', 'dataset_kind': 'tum', 'dataset_root': 'data/tum', 'tag': 'walking_static_f5', 'frames': 5, 'stride': 3, 'first_iters': 20, 'iters': 3, 'track_iters': 5, 'edge_iters': 0, 'mesh_resolution': 96, 'mesh_voxel': 0.10, 'map_every': 999999, 'keyframe_every': 999999},
    {'sequence': 'rgbd_dataset_freiburg3_walking_halfsphere', 'dataset_kind': 'tum', 'dataset_root': 'data/tum', 'tag': 'walking_halfsphere_f10', 'frames': 10, 'stride': 2, 'first_iters': 30, 'iters': 5, 'track_iters': 5, 'edge_iters': 0, 'mesh_resolution': 128, 'mesh_voxel': 0.08, 'map_every': 999999, 'keyframe_every': 999999},
    {'sequence': 'rgbd_dataset_freiburg1_xyz', 'dataset_kind': 'tum', 'dataset_root': 'data/tum', 'tag': 'freiburg1_xyz_f10_noba', 'frames': 10, 'stride': 2, 'first_iters': 30, 'iters': 5, 'track_iters': 5, 'edge_iters': 0, 'mesh_resolution': 128, 'mesh_voxel': 0.08, 'map_every': 999999, 'keyframe_every': 999999},
    {'sequence': 'rgbd_bonn_balloon', 'dataset_kind': 'bonn', 'dataset_root': 'data/bonn', 'tag': 'bonn_balloon_f5', 'frames': 5, 'stride': 3, 'first_iters': 20, 'iters': 3, 'track_iters': 5, 'edge_iters': 0, 'mesh_resolution': 96, 'mesh_voxel': 0.10, 'map_every': 999999, 'keyframe_every': 999999},
    {'sequence': 'rgbd_bonn_balloon2', 'dataset_kind': 'bonn', 'dataset_root': 'data/bonn', 'tag': 'bonn_balloon2_f5', 'frames': 5, 'stride': 3, 'first_iters': 20, 'iters': 3, 'track_iters': 5, 'edge_iters': 0, 'mesh_resolution': 96, 'mesh_voxel': 0.10, 'map_every': 999999, 'keyframe_every': 999999},
    {'sequence': 'rgbd_bonn_crowd2', 'dataset_kind': 'bonn', 'dataset_root': 'data/bonn', 'tag': 'bonn_crowd2_f5', 'frames': 5, 'stride': 3, 'first_iters': 20, 'iters': 3, 'track_iters': 5, 'edge_iters': 0, 'mesh_resolution': 96, 'mesh_voxel': 0.10, 'map_every': 999999, 'keyframe_every': 999999},
]


def run(cmd: List[str]) -> None:
    print('[cmd]', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description='Run/evaluate the S1 RoDyn-SLAM core-dataset protocol matrix.')
    ap.add_argument('--run_missing', action='store_true')
    ap.add_argument('--eval_only', action='store_true')
    args = ap.parse_args()

    smoke_root = PROJECT_ROOT / 'output' / 'post_cleanup' / 'rodyn_smoke'
    eval_root = PROJECT_ROOT / 'output' / 'post_cleanup' / 's1_rodyn_core_eval'
    eval_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for spec in SEQUENCES:
        smoke_dir = smoke_root / spec['tag']
        mesh_path = smoke_dir / 'mesh.ply'
        meta_path = smoke_dir / 'meta.json'
        if args.run_missing and not mesh_path.exists():
            cmd = [
                PYTHON, str(RUNNER),
                '--dataset_root', spec['dataset_root'],
                '--sequence', spec['sequence'],
                '--dataset_kind', spec['dataset_kind'],
                '--out_mesh', str(mesh_path),
                '--out_meta', str(meta_path),
                '--frames', str(spec['frames']),
                '--stride', str(spec['stride']),
                '--first_iters', str(spec['first_iters']),
                '--iters', str(spec['iters']),
                '--track_iters', str(spec['track_iters']),
                '--edge_iters', str(spec['edge_iters']),
                '--mesh_resolution', str(spec['mesh_resolution']),
                '--mesh_voxel', str(spec['mesh_voxel']),
                '--map_every', str(spec['map_every']),
                '--keyframe_every', str(spec['keyframe_every']),
            ]
            run(cmd)
        if not mesh_path.exists():
            rows.append({
                'sequence': spec['sequence'], 'dataset_kind': spec['dataset_kind'], 'status': 'missing_mesh', 'frames': spec['frames'], 'stride': spec['stride'],
                'chamfer': '', 'fscore': '', 'comp_r_5cm': '', 'mesh_path': '', 'summary_path': ''
            })
            continue

        out_dir = eval_root / spec['sequence'] / 'rodyn_slam'
        if not args.eval_only or not (out_dir / 'summary.json').exists():
            cmd = [
                PYTHON, str(ADAPTER),
                '--dataset_root', spec['dataset_root'],
                '--sequence', spec['sequence'],
                '--out', str(out_dir),
                '--dataset_kind', spec['dataset_kind'],
                '--frames', str(spec['frames']),
                '--stride', str(spec['stride']),
                '--surface_eval_thresh', '0.05',
                '--voxel_size', '0.02',
                '--pred_mesh', str(mesh_path),
            ]
            run(cmd)
        summary_path = out_dir / 'summary.json'
        data = json.loads(summary_path.read_text(encoding='utf-8'))
        metrics = data.get('metrics', {})
        rows.append({
            'sequence': spec['sequence'],
            'dataset_kind': spec['dataset_kind'],
            'status': data.get('status', 'ok'),
            'frames': spec['frames'],
            'stride': spec['stride'],
            'chamfer': metrics.get('chamfer', ''),
            'fscore': metrics.get('fscore', ''),
            'comp_r_5cm': metrics.get('recall_5cm', ''),
            'mesh_path': str(mesh_path),
            'summary_path': str(summary_path),
        })

    proc_dir = PROJECT_ROOT / 'processes' / 's1'
    proc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = proc_dir / 'S1_RODYN_CORE_PROTOCOL_CHECK.csv'
    md_path = proc_dir / 'S1_RODYN_CORE_PROTOCOL_CHECK.md'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence', 'dataset_kind', 'status', 'frames', 'stride', 'chamfer', 'fscore', 'comp_r_5cm', 'mesh_path', 'summary_path'])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# S1 RoDyn-SLAM 核心数据集口径确认表',
        '',
        '| sequence | dataset_kind | status | frames | stride | chamfer | fscore | comp_r_5cm |',
        '|---|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f"| {row['sequence']} | {row['dataset_kind']} | {row['status']} | {row['frames']} | {row['stride']} | {row['chamfer']} | {row['fscore']} | {row['comp_r_5cm']} |")
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[done] {csv_path} {md_path}')


if __name__ == '__main__':
    main()
