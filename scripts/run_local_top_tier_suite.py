#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _split_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _run(cmd: List[str], dry_run: bool = False) -> None:
    print("[cmd]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


@dataclass
class SuiteConfig:
    python_bin: str
    dataset_root_tum: str
    dataset_root_bonn: str
    out_root: str
    protocol: str
    static_sequences: str
    dynamic_sequences: str
    bonn_sequences: str
    methods: str
    frames_main: int
    stride_main: int
    frames_ablation: int
    stride_ablation: int
    frames_bonn: int
    stride_bonn: int
    max_points_per_frame: int
    voxel_size: float
    eval_thresh: float
    ghost_thresh: float
    bg_thresh: float
    seed_main: int
    seeds_significance: str
    run_main: bool
    run_multiseed: bool
    run_ablation: bool
    run_temporal: bool
    run_bonn: bool
    run_significance: bool
    force: bool
    download: bool
    dry_run: bool
    # EGF defaults for local-mapping-focused runs.
    egf_sigma_n0: float
    egf_rho_decay: float
    egf_phi_w_decay: float
    egf_forget_mode: str
    egf_dyn_forget_gain: float
    egf_raycast_clear_gain: float
    egf_surface_max_age_frames: int
    egf_surface_max_dscore: float
    egf_surface_max_free_ratio: float
    egf_surface_prune_free_min: float
    egf_surface_prune_residual_min: float
    egf_surface_max_clear_hits: float
    egf_static_sigma_n0: float
    egf_static_surface_phi_thresh: float
    egf_static_surface_rho_thresh: float
    egf_static_surface_min_weight: float
    egf_static_surface_max_dscore: float
    temporal_frames_list: str


def _benchmark_base_cmd(cfg: SuiteConfig) -> List[str]:
    return [
        cfg.python_bin,
        "scripts/run_benchmark.py",
        "--dataset_kind",
        "tum",
        "--dataset_root",
        cfg.dataset_root_tum,
        "--protocol",
        cfg.protocol,
        "--static_sequences",
        cfg.static_sequences,
        "--dynamic_sequences",
        cfg.dynamic_sequences,
        "--methods",
        cfg.methods,
        "--max_points_per_frame",
        str(cfg.max_points_per_frame),
        "--voxel_size",
        str(cfg.voxel_size),
        "--eval_thresh",
        str(cfg.eval_thresh),
        "--ghost_thresh",
        str(cfg.ghost_thresh),
        "--bg_thresh",
        str(cfg.bg_thresh),
        "--egf_sigma_n0",
        str(cfg.egf_sigma_n0),
        "--egf_rho_decay",
        str(cfg.egf_rho_decay),
        "--egf_phi_w_decay",
        str(cfg.egf_phi_w_decay),
        "--egf_forget_mode",
        cfg.egf_forget_mode,
        "--egf_dyn_forget_gain",
        str(cfg.egf_dyn_forget_gain),
        "--egf_raycast_clear_gain",
        str(cfg.egf_raycast_clear_gain),
        "--egf_surface_max_age_frames",
        str(cfg.egf_surface_max_age_frames),
        "--egf_surface_max_dscore",
        str(cfg.egf_surface_max_dscore),
        "--egf_surface_max_free_ratio",
        str(cfg.egf_surface_max_free_ratio),
        "--egf_surface_prune_free_min",
        str(cfg.egf_surface_prune_free_min),
        "--egf_surface_prune_residual_min",
        str(cfg.egf_surface_prune_residual_min),
        "--egf_surface_max_clear_hits",
        str(cfg.egf_surface_max_clear_hits),
        "--egf_static_sigma_n0",
        str(cfg.egf_static_sigma_n0),
        "--egf_static_surface_phi_thresh",
        str(cfg.egf_static_surface_phi_thresh),
        "--egf_static_surface_rho_thresh",
        str(cfg.egf_static_surface_rho_thresh),
        "--egf_static_surface_min_weight",
        str(cfg.egf_static_surface_min_weight),
        "--egf_static_surface_max_dscore",
        str(cfg.egf_static_surface_max_dscore),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-stop local mapping experiment suite for journal/top-tier prep."
    )
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--dataset_root_tum", type=str, default="data/tum")
    parser.add_argument("--dataset_root_bonn", type=str, default="data/bonn")
    parser.add_argument("--out_root", type=str, default="output/post_cleanup/local_top_tier")
    parser.add_argument("--protocol", type=str, default="oracle", choices=["oracle", "slam"])
    parser.add_argument("--static_sequences", type=str, default="rgbd_dataset_freiburg1_xyz")
    parser.add_argument(
        "--dynamic_sequences",
        type=str,
        default="rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_static,rgbd_dataset_freiburg3_walking_halfsphere",
    )
    parser.add_argument("--bonn_sequences", type=str, default="rgbd_bonn_balloon2")
    parser.add_argument("--methods", type=str, default="egf,tsdf,simple_removal")

    parser.add_argument("--frames_main", type=int, default=80)
    parser.add_argument("--stride_main", type=int, default=3)
    parser.add_argument("--frames_ablation", type=int, default=40)
    parser.add_argument("--stride_ablation", type=int, default=3)
    parser.add_argument("--frames_bonn", type=int, default=80)
    parser.add_argument("--stride_bonn", type=int, default=3)
    parser.add_argument("--max_points_per_frame", type=int, default=3000)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--eval_thresh", type=float, default=0.05)
    parser.add_argument("--ghost_thresh", type=float, default=0.08)
    parser.add_argument("--bg_thresh", type=float, default=0.05)

    parser.add_argument("--seed_main", type=int, default=7)
    parser.add_argument("--seeds_significance", type=str, default="0,1,2")

    parser.add_argument("--run_main", action="store_true", default=True)
    parser.add_argument("--run_multiseed", action="store_true", default=True)
    parser.add_argument("--run_ablation", action="store_true", default=True)
    parser.add_argument("--run_temporal", action="store_true", default=True)
    parser.add_argument("--run_bonn", action="store_true", default=True)
    parser.add_argument("--run_significance", action="store_true", default=True)
    parser.add_argument("--no_run_main", dest="run_main", action="store_false")
    parser.add_argument("--no_run_multiseed", dest="run_multiseed", action="store_false")
    parser.add_argument("--no_run_ablation", dest="run_ablation", action="store_false")
    parser.add_argument("--no_run_temporal", dest="run_temporal", action="store_false")
    parser.add_argument("--no_run_bonn", dest="run_bonn", action="store_false")
    parser.add_argument("--no_run_significance", dest="run_significance", action="store_false")

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--egf_sigma_n0", type=float, default=0.26)
    parser.add_argument("--egf_rho_decay", type=float, default=0.97)
    parser.add_argument("--egf_phi_w_decay", type=float, default=0.97)
    parser.add_argument("--egf_forget_mode", type=str, default="off", choices=["local", "global", "off"])
    parser.add_argument("--egf_dyn_forget_gain", type=float, default=0.0)
    parser.add_argument("--egf_raycast_clear_gain", type=float, default=0.0)
    parser.add_argument("--egf_surface_max_age_frames", type=int, default=1_000_000_000)
    parser.add_argument("--egf_surface_max_dscore", type=float, default=1.0)
    parser.add_argument("--egf_surface_max_free_ratio", type=float, default=1e9)
    parser.add_argument("--egf_surface_prune_free_min", type=float, default=1e9)
    parser.add_argument("--egf_surface_prune_residual_min", type=float, default=1e9)
    parser.add_argument("--egf_surface_max_clear_hits", type=float, default=1e9)
    parser.add_argument("--egf_static_sigma_n0", type=float, default=0.22)
    parser.add_argument("--egf_static_surface_phi_thresh", type=float, default=0.80)
    parser.add_argument("--egf_static_surface_rho_thresh", type=float, default=0.30)
    parser.add_argument("--egf_static_surface_min_weight", type=float, default=2.0)
    parser.add_argument("--egf_static_surface_max_dscore", type=float, default=0.80)

    parser.add_argument("--temporal_frames_list", type=str, default="15,30,45,60,90,120")
    args = parser.parse_args()

    cfg = SuiteConfig(**vars(args))
    out_root = Path(cfg.out_root)
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    (out_root / "suite_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    if cfg.run_main:
        cmd = _benchmark_base_cmd(cfg)
        cmd += [
            "--frames",
            str(cfg.frames_main),
            "--stride",
            str(cfg.stride_main),
            "--seed",
            str(cfg.seed_main),
            "--out_root",
            str(out_root / "main"),
        ]
        if cfg.download:
            cmd.append("--download")
        if cfg.force:
            cmd.append("--force")
        _run(cmd, dry_run=cfg.dry_run)

    if cfg.run_multiseed:
        cmd = _benchmark_base_cmd(cfg)
        cmd += [
            "--frames",
            str(cfg.frames_main),
            "--stride",
            str(cfg.stride_main),
            "--seeds",
            cfg.seeds_significance,
            "--out_root",
            str(out_root / "multiseed"),
        ]
        if cfg.download:
            cmd.append("--download")
        if cfg.force:
            cmd.append("--force")
        _run(cmd, dry_run=cfg.dry_run)

    if cfg.run_ablation:
        # Use single dynamic sequence for focused component analysis.
        base_ab = out_root / "ablation"
        variants = [
            ("r40_full", []),
            ("r40_no_evidence", ["--egf_ablation_no_evidence"]),
            ("r40_no_gradient", ["--egf_ablation_no_gradient"]),
            ("r40_classic_sdf", ["--egf_ablation_no_evidence", "--egf_ablation_no_gradient"]),
        ]
        for name, extra in variants:
            cmd = _benchmark_base_cmd(cfg)
            cmd += [
                "--static_sequences",
                "",
                "--dynamic_sequences",
                "rgbd_dataset_freiburg3_walking_xyz",
                "--methods",
                "egf",
                "--frames",
                str(cfg.frames_ablation),
                "--stride",
                str(cfg.stride_ablation),
                "--seed",
                str(cfg.seed_main),
                "--out_root",
                str(base_ab / name),
            ]
            cmd += extra
            if cfg.download:
                cmd.append("--download")
            if cfg.force:
                cmd.append("--force")
            _run(cmd, dry_run=cfg.dry_run)

        cmd = [
            cfg.python_bin,
            "scripts/build_ablation_summary.py",
            "--out_csv",
            str(tables_dir / "ablation_summary.csv"),
            "--variant",
            f"EGF-Full-v6={base_ab / 'r40_full'}",
            "--variant",
            f"EGF-No-Evidence={base_ab / 'r40_no_evidence'}",
            "--variant",
            f"EGF-No-Gradient={base_ab / 'r40_no_gradient'}",
            "--variant",
            f"EGF-Classic-SDF={base_ab / 'r40_classic_sdf'}",
        ]
        _run(cmd, dry_run=cfg.dry_run)

    if cfg.run_temporal:
        cmd = [
            cfg.python_bin,
            "scripts/run_temporal_ablation.py",
            "--dataset_root",
            cfg.dataset_root_tum,
            "--sequence",
            "rgbd_dataset_freiburg3_walking_xyz",
            "--frames_list",
            cfg.temporal_frames_list,
            "--stride",
            str(cfg.stride_main),
            "--max_points_per_frame",
            str(cfg.max_points_per_frame),
            "--voxel_size",
            str(cfg.voxel_size),
            "--eval_thresh",
            str(cfg.eval_thresh),
            "--ghost_thresh",
            str(cfg.ghost_thresh),
            "--bg_thresh",
            str(cfg.bg_thresh),
            "--out_root",
            str(out_root / "temporal_ablation"),
            "--curve_png",
            str(figs_dir / "temporal_convergence_curve.png"),
            "--rho_png",
            str(figs_dir / "temporal_rho_evolution.png"),
        ]
        if cfg.download:
            cmd.append("--download")
        if cfg.force:
            cmd.append("--force")
        _run(cmd, dry_run=cfg.dry_run)

    if cfg.run_bonn:
        cmd = [
            cfg.python_bin,
            "scripts/run_benchmark_bonn.py",
            "--dataset_root",
            cfg.dataset_root_bonn,
            "--sequences",
            cfg.bonn_sequences,
            "--frames",
            str(cfg.frames_bonn),
            "--stride",
            str(cfg.stride_bonn),
            "--max_points_per_frame",
            str(cfg.max_points_per_frame),
            "--voxel_size",
            str(cfg.voxel_size),
            "--eval_thresh",
            str(cfg.eval_thresh),
            "--ghost_thresh",
            str(cfg.ghost_thresh),
            "--bg_thresh",
            str(cfg.bg_thresh),
            "--out_root",
            str(out_root / "bonn"),
            "--compare_png",
            str(figs_dir / "bonn_comparison.png"),
        ]
        if cfg.download:
            cmd.append("--download")
        if cfg.force:
            cmd.append("--force")
        _run(cmd, dry_run=cfg.dry_run)

    if cfg.run_significance:
        cmd = [
            cfg.python_bin,
            "scripts/stats_significance.py",
            "--root",
            str(out_root / "multiseed"),
            "--out",
            str(tables_dir / "significance.csv"),
        ]
        _run(cmd, dry_run=cfg.dry_run)

    # Consolidate key tables into out_root/tables for paper/report usage.
    copied = []
    main_tables = out_root / "main" / "tables"
    copied_map = [
        (main_tables / "reconstruction_metrics.csv", tables_dir / "reconstruction_metrics.csv"),
        (main_tables / "dynamic_metrics.csv", tables_dir / "dynamic_metrics.csv"),
        (main_tables / "benchmark_summary.json", tables_dir / "benchmark_summary.json"),
        (out_root / "temporal_ablation" / "summary.csv", tables_dir / "temporal_ablation_summary.csv"),
        (out_root / "bonn" / "summary.csv", tables_dir / "bonn_summary.csv"),
        (out_root / "ablation" / "summary.csv", tables_dir / "ablation_summary_legacy.csv"),
    ]
    for src, dst in copied_map:
        if _copy_if_exists(src, dst):
            copied.append(str(dst))

    summary = {
        "out_root": str(out_root),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figs_dir),
        "copied_tables": copied,
        "run_main": cfg.run_main,
        "run_multiseed": cfg.run_multiseed,
        "run_ablation": cfg.run_ablation,
        "run_temporal": cfg.run_temporal,
        "run_bonn": cfg.run_bonn,
        "run_significance": cfg.run_significance,
        "dry_run": cfg.dry_run,
    }
    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Local Top-Tier Suite Run Summary",
        "",
        f"- out_root: `{out_root}`",
        f"- protocol: `{cfg.protocol}`",
        f"- methods: `{cfg.methods}`",
        f"- static_sequences: `{cfg.static_sequences}`",
        f"- dynamic_sequences: `{cfg.dynamic_sequences}`",
        f"- bonn_sequences: `{cfg.bonn_sequences}`",
        "",
        "## Key Outputs",
        f"- tables: `{tables_dir}`",
        f"- figures: `{figs_dir}`",
        f"- full config: `{out_root / 'suite_config.json'}`",
        f"- run summary: `{out_root / 'run_summary.json'}`",
    ]
    (out_root / "RUN_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] local top-tier suite finished, outputs at: {out_root}")


if __name__ == "__main__":
    main()

