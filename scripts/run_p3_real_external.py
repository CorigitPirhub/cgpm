#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None)


def ensure_subset(src_seq: Path, subset_dir: Path, frames: int) -> None:
    subset_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rgb", "depth", "groundtruth.txt"):
        src = src_seq / name
        dst = subset_dir / name
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        dst.symlink_to(src)

    for txt in ("rgb.txt", "depth.txt"):
        src_txt = src_seq / txt
        dst_txt = subset_dir / txt
        kept: List[str] = []
        with src_txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                kept.append(line)
                if len(kept) >= frames:
                    break
        with dst_txt.open("w", encoding="utf-8") as f:
            f.write("# generated subset\n")
            for line in kept:
                f.write(line + "\n")


def write_niceslam_fast_config(
    config_path: Path,
    input_folder: Path,
    output_folder: Path,
    map_first_iters: int,
    map_iters: int,
    tracking_iters: int,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""inherit_from: configs/TUM_RGBD/tum.yaml
mapping:
  bound: [[-6.0,6.0],[-6.0,6.0],[-3.0,9.0]]
  marching_cubes_bound: [[-3.0,3.0],[-3.0,3.0],[0.2,6.5]]
  every_frame: 1
  mesh_freq: 5
  vis_freq: 100000
  vis_inside_freq: 100000
  ckpt_freq: 1000
  keyframe_every: 5
  mapping_window_size: 3
  pixels: 400
  iters_first: {int(map_first_iters)}
  iters: {int(map_iters)}
tracking:
  vis_freq: 100000
  vis_inside_freq: 100000
  pixels: 600
  iters: {int(tracking_iters)}
  gt_camera: False
data:
  input_folder: {input_folder}
  output: {output_folder}
cam:
  H: 480
  W: 640
  fx: 535.4
  fy: 539.2
  cx: 320.1
  cy: 247.6
  crop_edge: 8
  crop_size: [384,512]
"""
    config_path.write_text(text, encoding="utf-8")


def find_latest_mesh(mesh_dir: Path) -> Path:
    if not mesh_dir.exists():
        raise FileNotFoundError(f"mesh directory not found: {mesh_dir}")
    candidates = sorted(mesh_dir.glob("*_mesh.ply"))
    if not candidates:
        candidates = sorted(mesh_dir.glob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"no mesh ply found in {mesh_dir}")
    return candidates[-1]


def run_niceslam_until_mesh(cmd: List[str], cwd: Path, mesh_dir: Path, timeout_sec: int) -> Path:
    print("[cmd]", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd), start_new_session=True)
    start_t = time.time()
    while True:
        rc = proc.poll()
        if rc is not None:
            if rc != 0:
                if mesh_dir.exists():
                    try:
                        return find_latest_mesh(mesh_dir)
                    except FileNotFoundError:
                        raise subprocess.CalledProcessError(rc, cmd)
                raise subprocess.CalledProcessError(rc, cmd)
            return find_latest_mesh(mesh_dir)

        if mesh_dir.exists():
            try:
                mesh = find_latest_mesh(mesh_dir)
                if mesh.stat().st_size > 0:
                    # NICE-SLAM may keep child processes alive after producing mesh.
                    # For our benchmark chain, a valid mesh is sufficient.
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    return mesh
            except FileNotFoundError:
                pass

        if (time.time() - start_t) > float(timeout_sec):
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            if mesh_dir.exists():
                try:
                    return find_latest_mesh(mesh_dir)
                except FileNotFoundError:
                    pass
            raise TimeoutError(f"NICE-SLAM timeout: {timeout_sec}s, no mesh generated at {mesh_dir}")
        time.sleep(3.0)


def load_method_rows(csv_path: Path) -> Dict[str, Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for r in reader:
            rows[str(r.get("method", ""))] = dict(r)
    return rows


def copy_summary_tables(run_tables_dir: Path, summary_root: Path) -> Dict[str, str]:
    summary_root.mkdir(parents=True, exist_ok=True)
    recon_src = run_tables_dir / "reconstruction_metrics.csv"
    dyn_src = run_tables_dir / "dynamic_metrics.csv"
    recon_dst = summary_root / "p3_real_external_reconstruction.csv"
    dyn_dst = summary_root / "p3_real_external_dynamic.csv"
    shutil.copy2(recon_src, recon_dst)
    shutil.copy2(dyn_src, dyn_dst)
    return {"reconstruction": str(recon_dst), "dynamic": str(dyn_dst)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P3 with real external NICE-SLAM output and unified evaluation.")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--dataset_root", type=str, default="data/tum")
    parser.add_argument("--sequence", type=str, default="rgbd_dataset_freiburg3_walking_xyz")
    parser.add_argument("--subset_frames", type=int, default=10)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_points_per_frame", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--ghost_thresh", type=float, default=0.08)
    parser.add_argument("--bg_thresh", type=float, default=0.05)
    parser.add_argument("--protocol", type=str, default="slam", choices=["oracle", "slam"])
    parser.add_argument("--out_root", type=str, default="output/post_cleanup/p3_real_external_niceslam")
    parser.add_argument("--summary_root", type=str, default="output/summary_tables")
    parser.add_argument("--niceslam_repo", type=str, default="third_party/nice-slam")
    parser.add_argument("--niceslam_output", type=str, default="output/external/nice_slam/rgbd_dataset_freiburg3_walking_xyz_f010_fast")
    parser.add_argument("--niceslam_config", type=str, default="configs/TUM_RGBD/freiburg3_walking_xyz_f010_fast.yaml")
    parser.add_argument("--niceslam_map_first_iters", type=int, default=80)
    parser.add_argument("--niceslam_map_iters", type=int, default=8)
    parser.add_argument("--niceslam_tracking_iters", type=int, default=15)
    parser.add_argument("--niceslam_timeout_sec", type=int, default=900)
    parser.add_argument("--neural_mesh", type=str, default="")
    parser.add_argument("--skip_niceslam", action="store_true")
    parser.add_argument("--ensure_rtree", action="store_true")
    parser.add_argument("--force_benchmark", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = (project_root / args.dataset_root).resolve()
    sequence_dir = dataset_root / args.sequence
    if not sequence_dir.exists():
        raise FileNotFoundError(f"sequence not found: {sequence_dir}")

    niceslam_repo = (project_root / args.niceslam_repo).resolve()
    if not niceslam_repo.exists():
        raise FileNotFoundError(f"nice-slam repo not found: {niceslam_repo}")

    niceslam_output = (project_root / args.niceslam_output).resolve()
    niceslam_config = (niceslam_repo / args.niceslam_config).resolve()

    subset_name = f"{args.sequence}_f{int(args.subset_frames):03d}"
    subset_dir = (project_root / "data" / "external" / "tum_niceslam_subsets" / subset_name).resolve()
    ensure_subset(sequence_dir, subset_dir, int(args.subset_frames))

    if args.ensure_rtree:
        try:
            run_cmd([args.python_bin, "-c", "import rtree"])
        except subprocess.CalledProcessError:
            run_cmd([args.python_bin, "-m", "pip", "install", "rtree"])

    neural_mesh_path = Path(args.neural_mesh).resolve() if args.neural_mesh.strip() else None
    mesh_dir = niceslam_output / "mesh"
    mesh_already_exists = mesh_dir.exists() and any(mesh_dir.glob("*.ply"))

    if not args.skip_niceslam and neural_mesh_path is None and not mesh_already_exists:
        write_niceslam_fast_config(
            config_path=niceslam_config,
            input_folder=subset_dir,
            output_folder=niceslam_output,
            map_first_iters=int(args.niceslam_map_first_iters),
            map_iters=int(args.niceslam_map_iters),
            tracking_iters=int(args.niceslam_tracking_iters),
        )
        neural_mesh = run_niceslam_until_mesh(
            [
                args.python_bin,
                "-W",
                "ignore",
                "run.py",
                str(niceslam_config.relative_to(niceslam_repo)),
                "--output",
                str(niceslam_output),
            ],
            cwd=niceslam_repo,
            mesh_dir=mesh_dir,
            timeout_sec=int(args.niceslam_timeout_sec),
        )
    else:
        if neural_mesh_path is not None:
            neural_mesh = neural_mesh_path
        else:
            neural_mesh = find_latest_mesh(mesh_dir)
    if not neural_mesh.exists():
        raise FileNotFoundError(f"neural mesh not found: {neural_mesh}")

    benchmark_cmd = [
        args.python_bin,
        str((project_root / "scripts" / "run_benchmark.py").resolve()),
        "--dataset_kind",
        "tum",
        "--dataset_root",
        str(dataset_root),
        "--out_root",
        str((project_root / args.out_root).resolve()),
        "--protocol",
        str(args.protocol),
        "--static_sequences",
        "",
        "--dynamic_sequences",
        args.sequence,
        "--frames",
        str(int(args.frames)),
        "--stride",
        str(int(args.stride)),
        "--max_points_per_frame",
        str(int(args.max_points_per_frame)),
        "--voxel_size",
        str(float(args.voxel_size)),
        "--eval_thresh",
        str(float(args.eval_thresh)),
        "--ghost_thresh",
        str(float(args.ghost_thresh)),
        "--bg_thresh",
        str(float(args.bg_thresh)),
        "--seed",
        str(int(args.seed)),
        "--methods",
        "egf,tsdf,simple_removal,neural_implicit",
        "--external_require_real",
        "--neural_pred_mesh_template",
        str(neural_mesh),
    ]
    if args.force_benchmark:
        benchmark_cmd.append("--force")
    run_cmd(benchmark_cmd, cwd=project_root)

    run_root = (project_root / args.out_root / args.protocol).resolve()
    run_tables_dir = run_root / "tables"
    copied = copy_summary_tables(run_tables_dir, (project_root / args.summary_root).resolve())
    recon_rows = load_method_rows(run_tables_dir / "reconstruction_metrics.csv")
    dyn_rows = load_method_rows(run_tables_dir / "dynamic_metrics.csv")

    report = {
        "status": "ok",
        "sequence": args.sequence,
        "protocol": args.protocol,
        "seed": int(args.seed),
        "niceslam_subset": str(subset_dir),
        "niceslam_output": str(niceslam_output),
        "neural_mesh": str(neural_mesh),
        "run_tables_dir": str(run_tables_dir),
        "summary_tables": copied,
        "methods": {
            "egf": {
                "fscore": float(recon_rows.get("egf", {}).get("fscore", 0.0)),
                "ghost_tail_ratio": float(dyn_rows.get("egf", {}).get("ghost_tail_ratio", 0.0)),
            },
            "tsdf": {
                "fscore": float(recon_rows.get("tsdf", {}).get("fscore", 0.0)),
                "ghost_tail_ratio": float(dyn_rows.get("tsdf", {}).get("ghost_tail_ratio", 0.0)),
            },
            "simple_removal": {
                "fscore": float(recon_rows.get("simple_removal", {}).get("fscore", 0.0)),
                "ghost_tail_ratio": float(dyn_rows.get("simple_removal", {}).get("ghost_tail_ratio", 0.0)),
            },
            "neural_implicit_real": {
                "fscore": float(recon_rows.get("neural_implicit", {}).get("fscore", 0.0)),
                "ghost_tail_ratio": float(dyn_rows.get("neural_implicit", {}).get("ghost_tail_ratio", 0.0)),
                "source_path": str(
                    json.loads(
                        (run_root / args.sequence / "neural_implicit" / "summary.json").read_text(encoding="utf-8")
                    ).get("source_path", "")
                ),
            },
        },
    }
    report_json = run_root / "p3_real_external_report.json"
    report_md = run_root / "P3_REAL_EXTERNAL_REPORT.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(
        "\n".join(
            [
                "# P3 Real External Baseline Report",
                "",
                f"- Sequence: `{args.sequence}`",
                f"- Protocol: `{args.protocol}`",
                f"- Neural external source: `{report['methods']['neural_implicit_real']['source_path']}`",
                f"- Tables: `{run_tables_dir}`",
                "",
                "## Metrics",
                "",
                "| Method | F-score | ghost_tail_ratio |",
                "|---|---:|---:|",
                f"| EGF | {report['methods']['egf']['fscore']:.6f} | {report['methods']['egf']['ghost_tail_ratio']:.6f} |",
                f"| TSDF | {report['methods']['tsdf']['fscore']:.6f} | {report['methods']['tsdf']['ghost_tail_ratio']:.6f} |",
                f"| Simple Removal | {report['methods']['simple_removal']['fscore']:.6f} | {report['methods']['simple_removal']['ghost_tail_ratio']:.6f} |",
                f"| Neural Implicit (real external) | {report['methods']['neural_implicit_real']['fscore']:.6f} | {report['methods']['neural_implicit_real']['ghost_tail_ratio']:.6f} |",
                "",
                "## Copied Summary CSV",
                "",
                f"- `{copied['reconstruction']}`",
                f"- `{copied['dynamic']}`",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[done] report: {report_json}")
    print(f"[done] markdown: {report_md}")


if __name__ == "__main__":
    main()
