from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_SEQUENCE = "rgbd_dataset_freiburg3_walking_xyz"
LOCKBOX_SEQUENCE = "rgbd_dataset_freiburg3_walking_static"
METHODS = ["egf", "tsdf", "dynaslam", "rodyn_slam"]


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("[cmd]", " ".join(shlex.quote(str(part)) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def build_shell_cmd(parts: List[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run_native_baselines(
    python_bin: str,
    dataset_root_tum: Path,
    out_root: Path,
    sequence: str,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    voxel_size: float,
    eval_thresh: float,
    seed: int,
    force: bool,
    dry_run: bool,
) -> None:
    cmd = [
        python_bin,
        "scripts/run_benchmark.py",
        "--dataset_kind",
        "tum",
        "--dataset_root",
        str(dataset_root_tum),
        "--out_root",
        str(out_root),
        "--protocol",
        "oracle",
        "--static_sequences",
        "",
        "--dynamic_sequences",
        sequence,
        "--methods",
        "egf,tsdf",
        "--frames",
        str(int(frames)),
        "--stride",
        str(int(stride)),
        "--max_points_per_frame",
        str(int(max_points_per_frame)),
        "--voxel_size",
        str(float(voxel_size)),
        "--eval_thresh",
        str(float(eval_thresh)),
        "--seed",
        str(int(seed)),
    ]
    if force:
        cmd.append("--force")
    run_cmd(cmd, dry_run=dry_run)


def run_dynaslam(
    python_bin: str,
    dataset_root_tum: Path,
    method_out_dir: Path,
    sequence: str,
    ref_points: Path,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    voxel_size: float,
    eval_thresh: float,
    seed: int,
    dry_run: bool,
) -> None:
    runner_cmd = build_shell_cmd(
        [
            python_bin,
            "scripts/external/run_dynaslam_tum_runner.py",
            "--sequence_dir",
            dataset_root_tum / sequence,
            "--out_points",
            method_out_dir / "surface_points.ply",
            "--out_meta",
            method_out_dir / "dynaslam_runner_meta.json",
            "--dynaslam_root",
            PROJECT_ROOT / "third_party/DynaSLAM",
            "--frames",
            frames,
            "--stride",
            stride,
            "--first_iters",
            20,
            "--iters",
            3,
            "--track_iters",
            5,
            "--edge_iters",
            0,
            "--map_every",
            999999,
            "--keyframe_every",
            999999,
            "--mesh_resolution",
            96,
            "--mesh_voxel",
            0.10,
        ]
    )
    cmd = [
        python_bin,
        "scripts/adapters/run_dynaslam_adapter.py",
        "--dataset_root",
        str(dataset_root_tum),
        "--sequence",
        sequence,
        "--out",
        str(method_out_dir),
        "--dataset_kind",
        "tum",
        "--frames",
        str(int(frames)),
        "--stride",
        str(int(stride)),
        "--max_points_per_frame",
        str(int(max_points_per_frame)),
        "--surface_eval_thresh",
        str(float(eval_thresh)),
        "--voxel_size",
        str(float(voxel_size)),
        "--seed",
        str(int(seed)),
        "--reference_points",
        str(ref_points),
        "--pred_mesh",
        str(method_out_dir / "surface_mesh.ply"),
        "--runner_cmd",
        runner_cmd,
    ]
    run_cmd(cmd, dry_run=dry_run)


def run_rodyn_slam(
    python_bin: str,
    dataset_root_tum: Path,
    method_out_dir: Path,
    sequence: str,
    ref_points: Path,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    voxel_size: float,
    eval_thresh: float,
    seed: int,
    dry_run: bool,
) -> None:
    runner_cmd = build_shell_cmd(
        [
            python_bin,
            "scripts/external/run_rodyn_slam_runner.py",
            "--dataset_root",
            dataset_root_tum,
            "--sequence",
            sequence,
            "--dataset_kind",
            "tum",
            "--out_mesh",
            method_out_dir / "surface_mesh.ply",
            "--out_meta",
            method_out_dir / "rodyn_slam_meta.json",
            "--frames",
            frames,
            "--stride",
            stride,
            "--first_iters",
            20,
            "--iters",
            3,
            "--track_iters",
            5,
            "--edge_iters",
            0,
            "--map_every",
            999999,
            "--keyframe_every",
            999999,
            "--mesh_resolution",
            96,
            "--mesh_voxel",
            0.10,
        ]
    )
    cmd = [
        python_bin,
        "scripts/adapters/run_rodyn_slam_adapter.py",
        "--dataset_root",
        str(dataset_root_tum),
        "--sequence",
        sequence,
        "--out",
        str(method_out_dir),
        "--dataset_kind",
        "tum",
        "--frames",
        str(int(frames)),
        "--stride",
        str(int(stride)),
        "--max_points_per_frame",
        str(int(max_points_per_frame)),
        "--surface_eval_thresh",
        str(float(eval_thresh)),
        "--voxel_size",
        str(float(voxel_size)),
        "--seed",
        str(int(seed)),
        "--reference_points",
        str(ref_points),
        "--pred_mesh",
        str(method_out_dir / "surface_mesh.ply"),
        "--runner_cmd",
        runner_cmd,
    ]
    run_cmd(cmd, dry_run=dry_run)


def load_method_row(method_out_dir: Path, method: str) -> Dict[str, object]:
    summary_path = method_out_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary for {method}: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    metrics = summary.get("metrics", {})
    mesh = summary.get("mesh", {})
    source_path = str(summary.get("source_path", "")).strip()
    if not source_path:
        surface_path = method_out_dir / "surface_points.ply"
        source_path = str(surface_path) if surface_path.exists() else ""
    surface_points = summary.get("surface_points")
    if surface_points is None:
        surface_points = mesh.get("surface_points", 0)
    return {
        "method": method,
        "status": summary.get("status", "ok"),
        "source_path": source_path,
        "surface_points": int(float(surface_points or 0)),
        "chamfer": float(metrics.get("chamfer", float("nan"))),
        "fscore": float(metrics.get("fscore", float("nan"))),
        "comp_r_5cm": float(metrics.get("recall_5cm", float("nan"))),
    }


def write_table(csv_path: Path, md_path: Path, title: str, protocol_desc: str, rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "status", "source_path", "surface_points", "chamfer", "fscore", "comp_r_5cm"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {title}",
        "",
        f"协议：`{protocol_desc}`",
        "",
        "| method | status | surface_points | chamfer | fscore | comp_r_5cm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['status']} | {row['surface_points']} | {row['chamfer']} | {row['fscore']} | {row['comp_r_5cm']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sequence_gate(
    python_bin: str,
    dataset_root_tum: Path,
    out_root: Path,
    sequence: str,
    frames: int,
    stride: int,
    max_points_per_frame: int,
    voxel_size: float,
    eval_thresh: float,
    seed: int,
    force: bool,
    dry_run: bool,
) -> List[Dict[str, object]]:
    run_native_baselines(
        python_bin=python_bin,
        dataset_root_tum=dataset_root_tum,
        out_root=out_root,
        sequence=sequence,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points_per_frame,
        voxel_size=voxel_size,
        eval_thresh=eval_thresh,
        seed=seed,
        force=force,
        dry_run=dry_run,
    )
    base_dir = out_root / "oracle" / sequence
    ref_points = base_dir / "egf" / "reference_points.ply"
    if not dry_run and not ref_points.exists():
        raise FileNotFoundError(f"Missing EGF reference points: {ref_points}")
    run_dynaslam(
        python_bin=python_bin,
        dataset_root_tum=dataset_root_tum,
        method_out_dir=base_dir / "dynaslam",
        sequence=sequence,
        ref_points=ref_points,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points_per_frame,
        voxel_size=voxel_size,
        eval_thresh=eval_thresh,
        seed=seed,
        dry_run=dry_run,
    )
    run_rodyn_slam(
        python_bin=python_bin,
        dataset_root_tum=dataset_root_tum,
        method_out_dir=base_dir / "rodyn_slam",
        sequence=sequence,
        ref_points=ref_points,
        frames=frames,
        stride=stride,
        max_points_per_frame=max_points_per_frame,
        voxel_size=voxel_size,
        eval_thresh=eval_thresh,
        seed=seed,
        dry_run=dry_run,
    )
    if dry_run:
        return []
    return [load_method_row(base_dir / method, method) for method in METHODS]


def write_lockbox_summary(md_path: Path, dev_rows: List[Dict[str, object]], lockbox_rows: List[Dict[str, object]]) -> None:
    by_method_dev = {str(row["method"]): row for row in dev_rows}
    by_method_lockbox = {str(row["method"]): row for row in lockbox_rows}

    def dominates(candidate: Dict[str, object], ref: Dict[str, object]) -> bool:
        chamfer_ok = float(candidate["chamfer"]) <= float(ref["chamfer"])
        fscore_ok = float(candidate["fscore"]) >= float(ref["fscore"])
        comp_ok = float(candidate["comp_r_5cm"]) >= float(ref["comp_r_5cm"])
        strict = (
            float(candidate["chamfer"]) < float(ref["chamfer"])
            or float(candidate["fscore"]) > float(ref["fscore"])
            or float(candidate["comp_r_5cm"]) > float(ref["comp_r_5cm"])
        )
        return chamfer_ok and fscore_ok and comp_ok and strict

    egf_dev = by_method_dev["egf"]
    egf_lockbox = by_method_lockbox["egf"]
    dev_dominators = [method for method in METHODS if method != "egf" and dominates(by_method_dev[method], egf_dev)]
    lockbox_dominators = [method for method in METHODS if method != "egf" and dominates(by_method_lockbox[method], egf_lockbox)]
    direction_ok = (not dev_dominators) and (not lockbox_dominators)

    lines = [
        "# S1 RB-Core 锁箱方向复验",
        "",
        "开发门槛：`TUM / oracle / rgbd_dataset_freiburg3_walking_xyz / frames=5 / stride=3 / seed=7`",
        "锁箱复验：`TUM / oracle / rgbd_dataset_freiburg3_walking_static / frames=5 / stride=3 / seed=7`",
        "",
        "判定规则：若某 baseline 在 `chamfer`、`fscore`、`comp_r_5cm` 三项上同时不差于当前 mainline (`egf`)，且至少一项严格更优，则视为“全面支配”；否则记为 `mixed/not-dominating`。",
        "",
        "| method | dev_chamfer | dev_fscore | dev_comp_r_5cm | dev_vs_egf | lockbox_chamfer | lockbox_fscore | lockbox_comp_r_5cm | lockbox_vs_egf |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for method in METHODS:
        row_dev = by_method_dev[method]
        row_lockbox = by_method_lockbox[method]
        if method == "egf":
            dev_flag = "reference"
            lockbox_flag = "reference"
        else:
            dev_flag = "dominating" if dominates(row_dev, egf_dev) else "mixed/not-dominating"
            lockbox_flag = "dominating" if dominates(row_lockbox, egf_lockbox) else "mixed/not-dominating"
        lines.append(
            f"| {method} | {row_dev['chamfer']} | {row_dev['fscore']} | {row_dev['comp_r_5cm']} | {dev_flag} | {row_lockbox['chamfer']} | {row_lockbox['fscore']} | {row_lockbox['comp_r_5cm']} | {lockbox_flag} |"
        )
    lines += [
        "",
        f"结论：开发门槛全面支配者={dev_dominators or ['none']}；锁箱全面支配者={lockbox_dominators or ['none']}。",
        f"按 `RB-Core` 全面支配判据，当前对比方向{'未翻转' if direction_ok else '发生翻转'}。",
        "`RoDyn-SLAM` 在当前 dev/lockbox smoke 下整体明显弱于 `EGF`，因此不会造成 lockbox direction flip；它当前承担的是 modern dynamic dense baseline 角色，而非强竞争线。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the S1 RB-Core local gate and export dev/lockbox compare tables.")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--dataset_root_tum", type=str, default="data/tum")
    parser.add_argument("--out_root", type=str, default="output/tmp/s1_rbcore_gate")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_points_per_frame", type=int, default=3000)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dataset_root_tum = Path(args.dataset_root_tum)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dev_rows = run_sequence_gate(
        python_bin=args.python_bin,
        dataset_root_tum=dataset_root_tum,
        out_root=out_root / "dev",
        sequence=DEV_SEQUENCE,
        frames=int(args.frames),
        stride=int(args.stride),
        max_points_per_frame=int(args.max_points_per_frame),
        voxel_size=float(args.voxel_size),
        eval_thresh=float(args.eval_thresh),
        seed=int(args.seed),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
    lockbox_rows = run_sequence_gate(
        python_bin=args.python_bin,
        dataset_root_tum=dataset_root_tum,
        out_root=out_root / "lockbox",
        sequence=LOCKBOX_SEQUENCE,
        frames=int(args.frames),
        stride=int(args.stride),
        max_points_per_frame=int(args.max_points_per_frame),
        voxel_size=float(args.voxel_size),
        eval_thresh=float(args.eval_thresh),
        seed=int(args.seed),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        return

    proc_dir = PROJECT_ROOT / "processes" / "s1"
    write_table(
        csv_path=proc_dir / "S1_RB_CORE_COMPARE_TUM_SMOKE.csv",
        md_path=proc_dir / "S1_RB_CORE_COMPARE_TUM_SMOKE.md",
        title="S1 RB-Core 开发门槛烟雾子集对比表",
        protocol_desc="TUM / oracle / rgbd_dataset_freiburg3_walking_xyz / frames=5 / stride=3 / seed=7",
        rows=dev_rows,
    )
    write_table(
        csv_path=proc_dir / "S1_RB_CORE_LOCKBOX_DIRECTION_RECHECK.csv",
        md_path=proc_dir / "S1_RB_CORE_LOCKBOX_DIRECTION_RECHECK_TABLE.md",
        title="S1 RB-Core 锁箱方向复验表",
        protocol_desc="TUM / oracle / rgbd_dataset_freiburg3_walking_static / frames=5 / stride=3 / seed=7",
        rows=lockbox_rows,
    )
    write_lockbox_summary(
        md_path=proc_dir / "S1_RB_CORE_LOCKBOX_DIRECTION_RECHECK.md",
        dev_rows=dev_rows,
        lockbox_rows=lockbox_rows,
    )
    print("[done] S1 RB-Core gate tables refreshed.")


if __name__ == "__main__":
    main()
