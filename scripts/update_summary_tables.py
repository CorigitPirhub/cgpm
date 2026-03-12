#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import math


@dataclass(frozen=True)
class CsvData:
    headers: List[str]
    rows: List[Dict[str, str]]


def read_csv(path: Path) -> CsvData:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return CsvData(headers=headers, rows=rows)


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for row in rows:
            w.writerow({h: row.get(h, "") for h in headers})


def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def row_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        str(row.get("sequence", "")),
        str(row.get("scene_type", "")),
        str(row.get("method", "")),
    )


def sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    scene_order = {"static": 0, "dynamic": 1}
    method_order = {"egf": 0, "tsdf": 1, "simple_removal": 2}

    def k(r: Dict[str, str]) -> Tuple[int, str, int]:
        return (
            scene_order.get(str(r.get("scene_type", "")), 99),
            str(r.get("sequence", "")),
            method_order.get(str(r.get("method", "")), 99),
        )

    return sorted(rows, key=k)


def merge_with_override(
    base_rows: List[Dict[str, str]],
    override_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    merged: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in base_rows:
        merged[row_key(row)] = dict(row)
    for row in override_rows:
        merged[row_key(row)] = dict(row)
    return sort_rows(list(merged.values()))


def pick_first_existing(paths: Sequence[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def pick_first_existing_pair(candidates: Sequence[Tuple[Path, Path]]) -> Tuple[Path | None, Path | None]:
    for recon_path, dyn_path in candidates:
        if recon_path.exists() and dyn_path.exists():
            return recon_path, dyn_path
    return None, None


def build_bonn_summary_from_recon(recon_csv: Path) -> CsvData:
    data = read_csv(recon_csv)
    out_headers = [
        "sequence",
        "method",
        "points",
        "accuracy",
        "completeness",
        "chamfer",
        "hausdorff",
        "precision",
        "recall",
        "fscore",
        "ghost_count",
        "ghost_ratio",
        "ghost_tail_count",
        "ghost_tail_ratio",
        "background_recovery",
    ]
    out_rows: List[Dict[str, str]] = []
    for r in data.rows:
        seq = str(r.get("sequence", ""))
        if "bonn" not in seq:
            continue
        out_rows.append({k: r.get(k, "") for k in out_headers})
    return CsvData(headers=out_headers, rows=out_rows)


def copy_if_exists(src: Path, dst: Path, outputs: List[str], missing: List[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    data = read_csv(src)
    write_csv(dst, data.headers, data.rows)
    outputs.append(str(dst))


def run_python_script(
    python_exe: str,
    script_rel: str,
    script_args: Sequence[str],
    project_root: Path,
    verbose: bool = False,
) -> Tuple[bool, str]:
    cmd = [python_exe, script_rel, *script_args]
    if verbose:
        print("[run]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        if verbose and proc.stdout.strip():
            print(proc.stdout.strip())
        return True, ""
    msg = (
        f"script_failed: {script_rel} rc={proc.returncode} "
        f"stdout={proc.stdout[-500:].strip()} stderr={proc.stderr[-500:].strip()}"
    )
    if verbose:
        print("[warn]", msg)
    return False, msg


def build_static_fix_delta(
    base_recon_rows: List[Dict[str, str]],
    static_fix_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    base_map = {row_key(r): r for r in base_recon_rows}
    out: List[Dict[str, object]] = []
    for r in sort_rows(static_fix_rows):
        key = row_key(r)
        b = base_map.get(key)
        if b is None:
            continue
        out.append(
            {
                "sequence": r.get("sequence", ""),
                "scene_type": r.get("scene_type", ""),
                "method": r.get("method", ""),
                "points": to_float(r, "points"),
                "accuracy": to_float(r, "accuracy"),
                "completeness": to_float(r, "completeness"),
                "chamfer_new": to_float(r, "chamfer"),
                "hausdorff": to_float(r, "hausdorff"),
                "precision_new": to_float(r, "precision"),
                "recall": to_float(r, "recall"),
                "fscore_new": to_float(r, "fscore"),
                "ghost_count": to_float(r, "ghost_count"),
                "ghost_ratio": to_float(r, "ghost_ratio"),
                "ghost_tail_count": to_float(r, "ghost_tail_count"),
                "ghost_tail_ratio": to_float(r, "ghost_tail_ratio"),
                "background_recovery": to_float(r, "background_recovery"),
                "fscore_base": to_float(b, "fscore"),
                "precision_base": to_float(b, "precision"),
                "chamfer_base": to_float(b, "chamfer"),
                "fscore_delta": to_float(r, "fscore") - to_float(b, "fscore"),
                "precision_delta": to_float(r, "precision") - to_float(b, "precision"),
                "chamfer_delta": to_float(r, "chamfer") - to_float(b, "chamfer"),
            }
        )
    return out


def build_static_target_checks(
    base_recon_rows: List[Dict[str, str]],
    base_dyn_rows: List[Dict[str, str]],
    target_recon_rows: List[Dict[str, str]],
    target_dyn_rows: List[Dict[str, str]],
    static_sequence: str = "rgbd_dataset_freiburg1_xyz",
    ghost_delta_allow: float = 0.03,
) -> List[Dict[str, object]]:
    base_recon_map = {row_key(r): r for r in base_recon_rows}
    target_recon_map = {row_key(r): r for r in target_recon_rows}
    base_dyn_map = {row_key(r): r for r in base_dyn_rows}
    target_dyn_map = {row_key(r): r for r in target_dyn_rows}

    out: List[Dict[str, object]] = []

    # Static target check for EGF.
    base_static = base_recon_map.get((static_sequence, "static", "egf"))
    target_static = target_recon_map.get((static_sequence, "static", "egf"))
    if base_static is not None and target_static is not None:
        target_fscore = to_float(target_static, "fscore")
        target_chamfer = to_float(target_static, "chamfer")
        out.append(
            {
                "check_type": "static_target",
                "sequence": static_sequence,
                "method": "egf",
                "base_fscore": to_float(base_static, "fscore"),
                "new_fscore": target_fscore,
                "fscore_delta": target_fscore - to_float(base_static, "fscore"),
                "fscore_target": 0.93,
                "fscore_pass": 1 if target_fscore >= 0.93 else 0,
                "base_chamfer": to_float(base_static, "chamfer"),
                "new_chamfer": target_chamfer,
                "chamfer_delta": target_chamfer - to_float(base_static, "chamfer"),
                "chamfer_target": 0.04,
                "chamfer_pass": 1 if target_chamfer <= 0.04 else 0,
                "base_ghost_ratio": "",
                "new_ghost_ratio": "",
                "ghost_delta": "",
                "ghost_delta_allow": ghost_delta_allow,
                "ghost_pass": "",
                "overall_pass": 1 if (target_fscore >= 0.93 and target_chamfer <= 0.04) else 0,
            }
        )

    # Dynamic non-regression checks for EGF.
    for seq in [
        "rgbd_dataset_freiburg3_walking_xyz",
        "rgbd_dataset_freiburg3_walking_static",
        "rgbd_dataset_freiburg3_walking_halfsphere",
    ]:
        b = base_dyn_map.get((seq, "dynamic", "egf"))
        n = target_dyn_map.get((seq, "dynamic", "egf"))
        if b is None or n is None:
            continue
        base_ghost = to_float(b, "ghost_ratio")
        new_ghost = to_float(n, "ghost_ratio")
        d = new_ghost - base_ghost
        out.append(
            {
                "check_type": "dynamic_non_regression",
                "sequence": seq,
                "method": "egf",
                "base_fscore": "",
                "new_fscore": "",
                "fscore_delta": "",
                "fscore_target": "",
                "fscore_pass": "",
                "base_chamfer": "",
                "new_chamfer": "",
                "chamfer_delta": "",
                "chamfer_target": "",
                "chamfer_pass": "",
                "base_ghost_ratio": base_ghost,
                "new_ghost_ratio": new_ghost,
                "ghost_delta": d,
                "ghost_delta_allow": ghost_delta_allow,
                "ghost_pass": 1 if d <= ghost_delta_allow else 0,
                "overall_pass": 1 if d <= ghost_delta_allow else 0,
            }
        )
    return out


def build_multiseed_significance_tum_bonn(
    tum_sig_rows: List[Dict[str, str]],
    bonn_sig_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    keep_metrics = {"fscore", "ghost_ratio", "chamfer", "ghost_tail_ratio"}
    out: List[Dict[str, object]] = []

    def _append(dataset: str, rows: List[Dict[str, str]]) -> None:
        for r in rows:
            scene_type = str(r.get("scene_type", "")).strip().lower()
            metric = str(r.get("metric", "")).strip().lower()
            if scene_type != "dynamic" or metric not in keep_metrics:
                continue
            row = dict(r)
            row["dataset"] = dataset
            out.append(row)

    _append("tum", tum_sig_rows)
    _append("bonn", bonn_sig_rows)
    out.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("protocol", "")), str(r.get("metric", ""))))
    return out


def build_multiseed_mean_std_tum_bonn(
    tum_recon_rows: List[Dict[str, str]],
    bonn_recon_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    keep_metrics = ["fscore", "ghost_ratio", "chamfer", "ghost_tail_ratio"]
    groups: Dict[Tuple[str, str, str, str, str], List[float]] = {}

    def _ingest(dataset: str, rows: List[Dict[str, str]]) -> None:
        for r in rows:
            if str(r.get("scene_type", "")).strip().lower() != "dynamic":
                continue
            protocol = str(r.get("protocol", "")).strip().lower()
            if protocol not in {"oracle", "slam"}:
                continue
            method = str(r.get("method", "")).strip().lower()
            if method not in {"egf", "tsdf"}:
                continue
            for metric in keep_metrics:
                v = to_float(r, metric, default=float("nan"))
                if not math.isfinite(v):
                    continue
                key = (dataset, protocol, "dynamic", method, metric)
                groups.setdefault(key, []).append(v)

    _ingest("tum", tum_recon_rows)
    _ingest("bonn", bonn_recon_rows)

    out: List[Dict[str, object]] = []
    for key in sorted(groups.keys()):
        dataset, protocol, scene_type, method, metric = key
        vals = groups[key]
        if not vals:
            continue
        n = len(vals)
        mean_v = sum(vals) / n
        if n > 1:
            var = sum((x - mean_v) ** 2 for x in vals) / (n - 1)
            std_v = math.sqrt(var)
        else:
            std_v = 0.0
        out.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "scene_type": scene_type,
                "method": method,
                "metric": metric,
                "mean": mean_v,
                "std": std_v,
                "n": n,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Update output/summary_tables from post_cleanup results.")
    parser.add_argument("--post_cleanup_root", type=str, default="output/tmp")
    parser.add_argument("--summary_root", type=str, default="output/summary_tables")
    parser.add_argument("--legacy_ablation_csv", type=str, default="output/ablation_study/summary.csv")
    parser.add_argument(
        "--prefer_p4_final",
        action="store_true",
        help="Prefer output/tmp/p4_final_merged/oracle/tables as canonical main TUM table source.",
    )
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--refresh_p6_p9", dest="refresh_p6_p9", action="store_true")
    parser.add_argument("--no_refresh_p6_p9", dest="refresh_p6_p9", action="store_false")
    parser.set_defaults(refresh_p6_p9=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    post_cleanup_root = Path(args.post_cleanup_root)
    summary_root = Path(args.summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]

    outputs: List[str] = []
    missing: List[str] = []

    # Source CSVs.
    base_recon = post_cleanup_root / "benchmark_tum" / "tables" / "reconstruction_metrics.csv"
    base_dyn = post_cleanup_root / "benchmark_tum" / "tables" / "dynamic_metrics.csv"
    p1_consistency_v2_recon = post_cleanup_root / "p1_consistency_v2" / "oracle" / "tables" / "reconstruction_metrics.csv"
    p1_consistency_v2_dyn = post_cleanup_root / "p1_consistency_v2" / "oracle" / "tables" / "dynamic_metrics.csv"
    static_recon = post_cleanup_root / "static_fix_fullverify" / "tables" / "reconstruction_metrics.csv"
    static_dyn = post_cleanup_root / "static_fix_fullverify" / "tables" / "dynamic_metrics.csv"
    static_target_recon = post_cleanup_root / "static_target_v1" / "oracle" / "tables" / "reconstruction_metrics.csv"
    static_target_dyn = post_cleanup_root / "static_target_v1" / "oracle" / "tables" / "dynamic_metrics.csv"
    p1_consistency_recon = post_cleanup_root / "p1_consistency_v2" / "oracle" / "tables" / "reconstruction_metrics.csv"
    p1_consistency_dyn = post_cleanup_root / "p1_consistency_v2" / "oracle" / "tables" / "dynamic_metrics.csv"
    temporal = post_cleanup_root / "temporal_ablation" / "summary.csv"
    bonn = post_cleanup_root / "benchmark_bonn" / "summary.csv"
    bonn_recon_v2 = post_cleanup_root / "p3_bonn_expanded_v2" / "slam" / "tables" / "reconstruction_metrics.csv"
    bonn_recon_v1 = post_cleanup_root / "p3_bonn_expanded" / "slam" / "tables" / "reconstruction_metrics.csv"
    ablation_post = post_cleanup_root / "ablation_summary.csv"
    p3_real_external_recon = post_cleanup_root / "p3_real_external_niceslam" / "slam" / "tables" / "reconstruction_metrics.csv"
    p3_real_external_dyn = post_cleanup_root / "p3_real_external_niceslam" / "slam" / "tables" / "dynamic_metrics.csv"
    p3_real_external_dual_recon = post_cleanup_root / "p3_real_external_dual" / "slam" / "tables" / "reconstruction_metrics.csv"
    p3_real_external_dual_dyn = post_cleanup_root / "p3_real_external_dual" / "slam" / "tables" / "dynamic_metrics.csv"
    p3_real_external_dynaslam_real_recon = (
        post_cleanup_root / "p3_real_external_dynaslam_real" / "slam" / "tables" / "reconstruction_metrics.csv"
    )
    p3_real_external_dynaslam_real_dyn = (
        post_cleanup_root / "p3_real_external_dynaslam_real" / "slam" / "tables" / "dynamic_metrics.csv"
    )
    if p3_real_external_dynaslam_real_recon.exists() and p3_real_external_dynaslam_real_dyn.exists():
        p3_real_external_recon = p3_real_external_dynaslam_real_recon
        p3_real_external_dyn = p3_real_external_dynaslam_real_dyn
    elif p3_real_external_dual_recon.exists() and p3_real_external_dual_dyn.exists():
        p3_real_external_recon = p3_real_external_dual_recon
        p3_real_external_dyn = p3_real_external_dual_dyn
    ablation_legacy = Path(args.legacy_ablation_csv)
    stage_abc = summary_root / "stage_abc_progress.csv"
    stage_abc_check = summary_root / "stage_abc_target_check.csv"
    literature_gap = summary_root / "literature_vs_ours_tum_walking_gap.csv"
    literature_metrics = summary_root / "literature_tum_walking_metrics.csv"
    temporal_trend = summary_root / "temporal_trend_stats.csv"
    local_eff = summary_root / "local_mapping_efficiency.csv"
    ext_ind = summary_root / "external_baselines_independent.csv"
    repro_cmds = summary_root / "reproduce_commands.md"
    p4_goal_check = summary_root / "round20260302_goal_check.csv"
    dual_recon = summary_root / "dual_protocol_multiseed_reconstruction.csv"
    dual_recon_agg = summary_root / "dual_protocol_multiseed_reconstruction_agg.csv"
    dual_dyn = summary_root / "dual_protocol_multiseed_dynamic.csv"
    dual_dyn_agg = summary_root / "dual_protocol_multiseed_dynamic_agg.csv"
    dual_sig = summary_root / "dual_protocol_multiseed_significance.csv"
    paper_main_csv = summary_root / "paper_main_table_local_mapping.csv"
    paper_main_md = summary_root / "paper_main_table_local_mapping.md"
    p6_main_csv = summary_root / "local_mapping_main_metrics_toptier.csv"
    p6_main_md = summary_root / "local_mapping_main_metrics_toptier.md"
    p7_eff_v2_csv = summary_root / "local_mapping_efficiency_v2.csv"
    p7_eff_v2_md = summary_root / "local_mapping_efficiency_v2.md"
    p7_eff_v2_json = summary_root / "local_mapping_efficiency_v2.json"
    p8_native_recon = summary_root / "external_baselines_native_reconstruction.csv"
    p8_native_dyn = summary_root / "external_baselines_native_dynamic.csv"
    p8_native_runtime = summary_root / "external_baselines_native_runtime.csv"
    p9_stress_csv = summary_root / "stress_test_summary.csv"
    p9_failure_md = post_cleanup_root / "stress_test" / "FAILURE_CASES.md"
    p14_stress_csv = summary_root / "stress_test_summary_final.csv"
    p14_failure_md = post_cleanup_root / "stress_test" / "FAILURE_CASES_FINAL.md"
    p14_curve_png = project_root / "assets" / "stress_degradation_curves_final.png"
    p14_meta_json = post_cleanup_root / "stress_test" / "stress_meta_final.json"
    p10_precision_csv = summary_root / "local_mapping_precision_profile.csv"
    p11_eff_final_csv = summary_root / "local_mapping_efficiency_final.csv"
    p11_eff_final_json = summary_root / "local_mapping_efficiency_final.json"
    p10_probe_scan_csv = summary_root / "p10_probe_scan.csv"
    p10_tradeoff_png = project_root / "assets" / "acc_comp_tradeoff.png"
    p11_tradeoff_png = project_root / "assets" / "quality_speed_tradeoff_final.png"
    p15_eff_rt_csv = summary_root / "local_mapping_efficiency_realtime.csv"
    p15_eff_rt_json = summary_root / "local_mapping_efficiency_realtime.json"
    p15_profile_md = post_cleanup_root / "p15_realtime" / "PROFILE.md"
    p15_tradeoff_png = project_root / "assets" / "quality_speed_tradeoff_realtime.png"
    p4_tum_recon = post_cleanup_root / "p4_multiseed_tum_final_v2" / "oracle" / "tables" / "reconstruction_metrics.csv"
    p4_tum_dyn = post_cleanup_root / "p4_multiseed_tum_final_v2" / "oracle" / "tables" / "dynamic_metrics.csv"
    p4_tum_recon_agg = post_cleanup_root / "p4_multiseed_tum_final_v2" / "oracle" / "tables" / "reconstruction_metrics_agg.csv"
    p4_tum_dyn_agg = post_cleanup_root / "p4_multiseed_tum_final_v2" / "oracle" / "tables" / "dynamic_metrics_agg.csv"
    p4_tum_sig = post_cleanup_root / "p4_multiseed_tum_final_v2" / "tables" / "significance.csv"
    p5_bonn_recon = post_cleanup_root / "p5_multiseed_bonn_all3" / "slam" / "tables" / "reconstruction_metrics.csv"
    p5_bonn_dyn = post_cleanup_root / "p5_multiseed_bonn_all3" / "slam" / "tables" / "dynamic_metrics.csv"
    p5_bonn_recon_agg = post_cleanup_root / "p5_multiseed_bonn_all3" / "slam" / "tables" / "reconstruction_metrics_agg.csv"
    p5_bonn_dyn_agg = post_cleanup_root / "p5_multiseed_bonn_all3" / "slam" / "tables" / "dynamic_metrics_agg.csv"
    p5_bonn_sig = post_cleanup_root / "p5_multiseed_bonn_all3" / "tables" / "significance.csv"
    p4_final_merged_recon = post_cleanup_root / "p4_final_merged" / "oracle" / "tables" / "reconstruction_metrics.csv"
    p4_final_merged_dyn = post_cleanup_root / "p4_final_merged" / "oracle" / "tables" / "dynamic_metrics.csv"
    p4_bonn_recon = post_cleanup_root / "p4_multiseed_bonn_final" / "slam" / "tables" / "reconstruction_metrics.csv"
    p4_bonn_dyn = post_cleanup_root / "p4_multiseed_bonn_final" / "slam" / "tables" / "dynamic_metrics.csv"
    p4_bonn_recon_agg = post_cleanup_root / "p4_multiseed_bonn_final" / "slam" / "tables" / "reconstruction_metrics_agg.csv"
    p4_bonn_dyn_agg = post_cleanup_root / "p4_multiseed_bonn_final" / "slam" / "tables" / "dynamic_metrics_agg.csv"
    p4_bonn_sig = post_cleanup_root / "p4_multiseed_bonn_final" / "tables" / "significance.csv"

    # Protocol-isolated TUM sources.
    tum_slam_recon_primary = post_cleanup_root / "p3_tum_expanded" / "slam" / "tables" / "reconstruction_metrics.csv"
    tum_slam_dyn_primary = post_cleanup_root / "p3_tum_expanded" / "slam" / "tables" / "dynamic_metrics.csv"
    tum_slam_recon_secondary = post_cleanup_root / "p0p1_verify_bench" / "slam" / "tables" / "reconstruction_metrics.csv"
    tum_slam_dyn_secondary = post_cleanup_root / "p0p1_verify_bench" / "slam" / "tables" / "dynamic_metrics.csv"
    tum_slam_recon_cached = summary_root / "tum_reconstruction_metrics_slam.csv"
    tum_slam_dyn_cached = summary_root / "tum_dynamic_metrics_slam.csv"

    oracle_candidates: List[Tuple[Path, Path]]
    if bool(args.prefer_p4_final):
        oracle_candidates = [
            (p4_final_merged_recon, p4_final_merged_dyn),
            (p1_consistency_v2_recon, p1_consistency_v2_dyn),
            (base_recon, base_dyn),
        ]
    else:
        oracle_candidates = [
            (p1_consistency_v2_recon, p1_consistency_v2_dyn),
            (p4_final_merged_recon, p4_final_merged_dyn),
            (base_recon, base_dyn),
        ]
    slam_candidates: List[Tuple[Path, Path]] = [
        (tum_slam_recon_primary, tum_slam_dyn_primary),
        (tum_slam_recon_secondary, tum_slam_dyn_secondary),
        (tum_slam_recon_cached, tum_slam_dyn_cached),
    ]

    oracle_recon_src, oracle_dyn_src = pick_first_existing_pair(oracle_candidates)
    slam_recon_src, slam_dyn_src = pick_first_existing_pair(slam_candidates)

    if oracle_recon_src is not None and oracle_dyn_src is not None:
        oracle_recon_data = read_csv(oracle_recon_src)
        oracle_dyn_data = read_csv(oracle_dyn_src)
        write_csv(summary_root / "tum_reconstruction_metrics_oracle.csv", oracle_recon_data.headers, sort_rows(oracle_recon_data.rows))
        outputs.append(str(summary_root / "tum_reconstruction_metrics_oracle.csv"))
        write_csv(summary_root / "tum_dynamic_metrics_oracle.csv", oracle_dyn_data.headers, sort_rows(oracle_dyn_data.rows))
        outputs.append(str(summary_root / "tum_dynamic_metrics_oracle.csv"))
    else:
        missing.append("tum_oracle_source_pair")

    if slam_recon_src is not None and slam_dyn_src is not None:
        slam_recon_data = read_csv(slam_recon_src)
        slam_dyn_data = read_csv(slam_dyn_src)
        write_csv(summary_root / "tum_reconstruction_metrics_slam.csv", slam_recon_data.headers, sort_rows(slam_recon_data.rows))
        outputs.append(str(summary_root / "tum_reconstruction_metrics_slam.csv"))
        write_csv(summary_root / "tum_dynamic_metrics_slam.csv", slam_dyn_data.headers, sort_rows(slam_dyn_data.rows))
        outputs.append(str(summary_root / "tum_dynamic_metrics_slam.csv"))
    else:
        missing.append("tum_slam_source_pair")

    # Canonical unsuffixed table remains protocol-safe and oracle-first for
    # backward compatibility; protocol-specific tables are exported explicitly.
    main_source_recon: Path | None = None
    main_source_dyn: Path | None = None
    if oracle_recon_src is not None and oracle_dyn_src is not None:
        main_source_recon = oracle_recon_src
        main_source_dyn = oracle_dyn_src
    elif slam_recon_src is not None and slam_dyn_src is not None:
        main_source_recon = slam_recon_src
        main_source_dyn = slam_dyn_src

    if main_source_recon is None or main_source_dyn is None:
        raise FileNotFoundError(
            "missing both SLAM and oracle TUM source pairs; run benchmark first."
        )

    base_recon_data = read_csv(main_source_recon)
    base_dyn_data = read_csv(main_source_dyn)
    write_csv(summary_root / "tum_reconstruction_metrics.csv", base_recon_data.headers, sort_rows(base_recon_data.rows))
    outputs.append(str(summary_root / "tum_reconstruction_metrics.csv"))
    write_csv(summary_root / "tum_dynamic_metrics.csv", base_dyn_data.headers, sort_rows(base_dyn_data.rows))
    outputs.append(str(summary_root / "tum_dynamic_metrics.csv"))

    # Always export static_fix tables separately if available.
    if static_recon.exists():
        static_recon_data = read_csv(static_recon)
        write_csv(summary_root / "tum_reconstruction_metrics_static_fix.csv", static_recon_data.headers, sort_rows(static_recon_data.rows))
        outputs.append(str(summary_root / "tum_reconstruction_metrics_static_fix.csv"))
    else:
        missing.append(str(static_recon))

    if static_dyn.exists():
        static_dyn_data = read_csv(static_dyn)
        write_csv(summary_root / "tum_dynamic_metrics_static_fix.csv", static_dyn_data.headers, sort_rows(static_dyn_data.rows))
        outputs.append(str(summary_root / "tum_dynamic_metrics_static_fix.csv"))
    else:
        missing.append(str(static_dyn))

    # Strong fairness constraint: do not patch/override single-method rows in canonical tables.
    tuned_static_summary = post_cleanup_root / "static_target_v2_mw19" / "summary.json"
    if tuned_static_summary.exists():
        outputs.append(str(tuned_static_summary))

    # Static-fix delta table (requires static_fix + base).
    if static_recon.exists():
        delta_rows = build_static_fix_delta(base_recon_data.rows, read_csv(static_recon).rows)
        delta_headers = [
            "sequence",
            "scene_type",
            "method",
            "points",
            "accuracy",
            "completeness",
            "chamfer_new",
            "hausdorff",
            "precision_new",
            "recall",
            "fscore_new",
            "ghost_count",
            "ghost_ratio",
            "ghost_tail_count",
            "ghost_tail_ratio",
            "background_recovery",
            "fscore_base",
            "precision_base",
            "chamfer_base",
            "fscore_delta",
            "precision_delta",
            "chamfer_delta",
        ]
        write_csv(summary_root / "static_fix_delta_summary.csv", delta_headers, delta_rows)
        outputs.append(str(summary_root / "static_fix_delta_summary.csv"))

    # Static-vs-TSDF target run (freiburg1_xyz >=0.93 and chamfer <=0.04) + dynamic +0.03 check.
    if static_target_recon.exists():
        target_recon_data = read_csv(static_target_recon)
        write_csv(
            summary_root / "tum_reconstruction_metrics_static_target_v1.csv",
            target_recon_data.headers,
            sort_rows(target_recon_data.rows),
        )
        outputs.append(str(summary_root / "tum_reconstruction_metrics_static_target_v1.csv"))
    else:
        missing.append(str(static_target_recon))

    if static_target_dyn.exists():
        target_dyn_data = read_csv(static_target_dyn)
        write_csv(
            summary_root / "tum_dynamic_metrics_static_target_v1.csv",
            target_dyn_data.headers,
            sort_rows(target_dyn_data.rows),
        )
        outputs.append(str(summary_root / "tum_dynamic_metrics_static_target_v1.csv"))
    else:
        missing.append(str(static_target_dyn))

    if p1_consistency_recon.exists():
        p1_recon_data = read_csv(p1_consistency_recon)
        write_csv(
            summary_root / "tum_reconstruction_metrics_p1_consistency.csv",
            p1_recon_data.headers,
            sort_rows(p1_recon_data.rows),
        )
        outputs.append(str(summary_root / "tum_reconstruction_metrics_p1_consistency.csv"))
    else:
        missing.append(str(p1_consistency_recon))

    if p1_consistency_dyn.exists():
        p1_dyn_data = read_csv(p1_consistency_dyn)
        write_csv(
            summary_root / "tum_dynamic_metrics_p1_consistency.csv",
            p1_dyn_data.headers,
            sort_rows(p1_dyn_data.rows),
        )
        outputs.append(str(summary_root / "tum_dynamic_metrics_p1_consistency.csv"))
    else:
        missing.append(str(p1_consistency_dyn))

    if static_target_recon.exists() and static_target_dyn.exists():
        target_checks = build_static_target_checks(
            base_recon_data.rows,
            base_dyn_data.rows,
            read_csv(static_target_recon).rows,
            read_csv(static_target_dyn).rows,
            static_sequence="rgbd_dataset_freiburg1_xyz",
            ghost_delta_allow=0.03,
        )
        target_headers = [
            "check_type",
            "sequence",
            "method",
            "base_fscore",
            "new_fscore",
            "fscore_delta",
            "fscore_target",
            "fscore_pass",
            "base_chamfer",
            "new_chamfer",
            "chamfer_delta",
            "chamfer_target",
            "chamfer_pass",
            "base_ghost_ratio",
            "new_ghost_ratio",
            "ghost_delta",
            "ghost_delta_allow",
            "ghost_pass",
            "overall_pass",
        ]
        write_csv(summary_root / "static_target_constraint_check.csv", target_headers, target_checks)
        outputs.append(str(summary_root / "static_target_constraint_check.csv"))

    # Temporal / Bonn / Ablation passthrough.
    copy_if_exists(temporal, summary_root / "temporal_ablation_summary.csv", outputs, missing)
    bonn_recon_src = pick_first_existing([bonn_recon_v2, bonn_recon_v1])
    if bonn_recon_src is not None:
        bonn_data = build_bonn_summary_from_recon(bonn_recon_src)
        write_csv(summary_root / "bonn_summary.csv", bonn_data.headers, bonn_data.rows)
        outputs.append(str(summary_root / "bonn_summary.csv"))
    else:
        copy_if_exists(bonn, summary_root / "bonn_summary.csv", outputs, missing)
    copy_if_exists(
        p3_real_external_recon,
        summary_root / "p3_real_external_reconstruction.csv",
        outputs,
        missing,
    )
    copy_if_exists(
        p3_real_external_dyn,
        summary_root / "p3_real_external_dynamic.csv",
        outputs,
        missing,
    )
    if ablation_post.exists():
        copy_if_exists(ablation_post, summary_root / "ablation_summary.csv", outputs, missing)
    else:
        copy_if_exists(ablation_legacy, summary_root / "ablation_summary.csv", outputs, missing)

    # Optional pre-generated analysis tables in summary_root (kept in manifest for release packaging).
    if stage_abc.exists():
        outputs.append(str(stage_abc))
    if stage_abc_check.exists():
        outputs.append(str(stage_abc_check))
    if literature_gap.exists():
        outputs.append(str(literature_gap))
    if literature_metrics.exists():
        outputs.append(str(literature_metrics))
    if temporal_trend.exists():
        outputs.append(str(temporal_trend))
    if local_eff.exists():
        outputs.append(str(local_eff))
    if ext_ind.exists():
        outputs.append(str(ext_ind))
    if repro_cmds.exists():
        outputs.append(str(repro_cmds))
    if p4_goal_check.exists():
        outputs.append(str(p4_goal_check))
    if dual_recon.exists():
        outputs.append(str(dual_recon))
    if dual_recon_agg.exists():
        outputs.append(str(dual_recon_agg))
    if dual_dyn.exists():
        outputs.append(str(dual_dyn))
    if dual_dyn_agg.exists():
        outputs.append(str(dual_dyn_agg))
    if dual_sig.exists():
        outputs.append(str(dual_sig))
    if paper_main_csv.exists():
        outputs.append(str(paper_main_csv))
    if paper_main_md.exists():
        outputs.append(str(paper_main_md))
    if p10_precision_csv.exists():
        outputs.append(str(p10_precision_csv))
    if p11_eff_final_csv.exists():
        outputs.append(str(p11_eff_final_csv))
    if p11_eff_final_json.exists():
        outputs.append(str(p11_eff_final_json))
    if p10_probe_scan_csv.exists():
        outputs.append(str(p10_probe_scan_csv))
    if p10_tradeoff_png.exists():
        outputs.append(str(p10_tradeoff_png))
    if p11_tradeoff_png.exists():
        outputs.append(str(p11_tradeoff_png))
    if p15_eff_rt_csv.exists():
        outputs.append(str(p15_eff_rt_csv))
    if p15_eff_rt_json.exists():
        outputs.append(str(p15_eff_rt_json))
    if p15_profile_md.exists():
        outputs.append(str(p15_profile_md))
    if p15_tradeoff_png.exists():
        outputs.append(str(p15_tradeoff_png))

    # Optional: multi-seed round tables (TUM + Bonn).
    copy_if_exists(
        p4_tum_recon,
        summary_root / "tum_reconstruction_metrics_multiseed.csv",
        outputs,
        missing,
    )
    copy_if_exists(
        p4_tum_dyn,
        summary_root / "tum_dynamic_metrics_multiseed.csv",
        outputs,
        missing,
    )
    copy_if_exists(
        p4_tum_recon_agg,
        summary_root / "tum_reconstruction_metrics_multiseed_agg.csv",
        outputs,
        missing,
    )
    copy_if_exists(
        p4_tum_dyn_agg,
        summary_root / "tum_dynamic_metrics_multiseed_agg.csv",
        outputs,
        missing,
    )
    copy_if_exists(
        p4_tum_sig,
        summary_root / "tum_significance_multiseed.csv",
        outputs,
        missing,
    )
    bonn_multiseed_recon_src = pick_first_existing([p5_bonn_recon, p4_bonn_recon])
    bonn_multiseed_dyn_src = pick_first_existing([p5_bonn_dyn, p4_bonn_dyn])
    bonn_multiseed_recon_agg_src = pick_first_existing([p5_bonn_recon_agg, p4_bonn_recon_agg])
    bonn_multiseed_dyn_agg_src = pick_first_existing([p5_bonn_dyn_agg, p4_bonn_dyn_agg])
    bonn_multiseed_sig_src = pick_first_existing([p5_bonn_sig, p4_bonn_sig])

    if bonn_multiseed_recon_src is not None:
        copy_if_exists(
            bonn_multiseed_recon_src,
            summary_root / "bonn_reconstruction_metrics_multiseed.csv",
            outputs,
            missing,
        )
    else:
        missing.append(str(p5_bonn_recon))
        missing.append(str(p4_bonn_recon))

    if bonn_multiseed_dyn_src is not None:
        copy_if_exists(
            bonn_multiseed_dyn_src,
            summary_root / "bonn_dynamic_metrics_multiseed.csv",
            outputs,
            missing,
        )
    else:
        missing.append(str(p5_bonn_dyn))
        missing.append(str(p4_bonn_dyn))

    if bonn_multiseed_recon_agg_src is not None:
        copy_if_exists(
            bonn_multiseed_recon_agg_src,
            summary_root / "bonn_reconstruction_metrics_multiseed_agg.csv",
            outputs,
            missing,
        )
    else:
        missing.append(str(p5_bonn_recon_agg))
        missing.append(str(p4_bonn_recon_agg))

    if bonn_multiseed_dyn_agg_src is not None:
        copy_if_exists(
            bonn_multiseed_dyn_agg_src,
            summary_root / "bonn_dynamic_metrics_multiseed_agg.csv",
            outputs,
            missing,
        )
    else:
        missing.append(str(p5_bonn_dyn_agg))
        missing.append(str(p4_bonn_dyn_agg))

    if bonn_multiseed_sig_src is not None:
        copy_if_exists(
            bonn_multiseed_sig_src,
            summary_root / "bonn_significance_multiseed.csv",
            outputs,
            missing,
        )
    else:
        missing.append(str(p5_bonn_sig))
        missing.append(str(p4_bonn_sig))

    oracle_tables_root = p4_tum_recon.parent if p4_tum_recon.exists() and p4_tum_dyn.exists() else None
    slam_tables_root = None
    if p5_bonn_recon.exists() and p5_bonn_dyn.exists():
        slam_tables_root = p5_bonn_recon.parent
    elif p4_bonn_recon.exists() and p4_bonn_dyn.exists():
        slam_tables_root = p4_bonn_recon.parent

    if oracle_tables_root is not None and slam_tables_root is not None:
        ok, err = run_python_script(
            python_exe=str(args.python_exe),
            script_rel="scripts/build_dual_protocol_multiseed_summary.py",
            script_args=[
                "--summary_root",
                str(summary_root),
                "--oracle_tables_root",
                str(oracle_tables_root),
                "--slam_tables_root",
                str(slam_tables_root),
            ],
            project_root=project_root,
            verbose=bool(args.verbose),
        )
        if not ok:
            missing.append(err)
    else:
        if oracle_tables_root is None:
            missing.append(str(p4_tum_recon.parent))
        if slam_tables_root is None:
            missing.append(str(p5_bonn_recon.parent))
            missing.append(str(p4_bonn_recon.parent))

    if (
        (summary_root / "tum_reconstruction_metrics_multiseed_agg.csv").exists()
        and (summary_root / "bonn_reconstruction_metrics_multiseed_agg.csv").exists()
        and dual_sig.exists()
    ):
        ok, err = run_python_script(
            python_exe=str(args.python_exe),
            script_rel="scripts/build_paper_main_table.py",
            script_args=[
                "--tum_agg",
                str(summary_root / "tum_reconstruction_metrics_multiseed_agg.csv"),
                "--bonn_agg",
                str(summary_root / "bonn_reconstruction_metrics_multiseed_agg.csv"),
                "--dual_sig",
                str(dual_sig),
                "--out_csv",
                str(paper_main_csv),
                "--out_md",
                str(paper_main_md),
            ],
            project_root=project_root,
            verbose=bool(args.verbose),
        )
        if not ok:
            missing.append(err)
    else:
        if not (summary_root / "tum_reconstruction_metrics_multiseed_agg.csv").exists():
            missing.append(str(summary_root / "tum_reconstruction_metrics_multiseed_agg.csv"))
        if not (summary_root / "bonn_reconstruction_metrics_multiseed_agg.csv").exists():
            missing.append(str(summary_root / "bonn_reconstruction_metrics_multiseed_agg.csv"))
        if not dual_sig.exists():
            missing.append(str(dual_sig))

    # Combined TUM+Bonn multi-seed tables for paper-facing summaries.
    tum_sig_path = summary_root / "tum_significance_multiseed.csv"
    bonn_sig_path = summary_root / "bonn_significance_multiseed.csv"
    tum_recon_ms_path = summary_root / "tum_reconstruction_metrics_multiseed.csv"
    bonn_recon_ms_path = summary_root / "bonn_reconstruction_metrics_multiseed.csv"

    if tum_sig_path.exists() and bonn_sig_path.exists():
        merged_sig = build_multiseed_significance_tum_bonn(
            read_csv(tum_sig_path).rows,
            read_csv(bonn_sig_path).rows,
        )
        sig_headers = [
            "dataset",
            "protocol",
            "scene_type",
            "metric",
            "direction",
            "method_a",
            "method_b",
            "n_pairs",
            "mean_delta_egf_minus_baseline",
            "std_delta_egf_minus_baseline",
            "mean_improvement",
            "std_improvement",
            "cohens_d",
            "t_stat",
            "t_pvalue",
            "wilcoxon_stat",
            "wilcoxon_pvalue",
        ]
        write_csv(summary_root / "multiseed_significance_tum_bonn.csv", sig_headers, merged_sig)
        outputs.append(str(summary_root / "multiseed_significance_tum_bonn.csv"))

    if tum_recon_ms_path.exists() and bonn_recon_ms_path.exists():
        mean_std_rows = build_multiseed_mean_std_tum_bonn(
            read_csv(tum_recon_ms_path).rows,
            read_csv(bonn_recon_ms_path).rows,
        )
        mean_std_headers = ["dataset", "protocol", "scene_type", "method", "metric", "mean", "std", "n"]
        write_csv(summary_root / "multiseed_mean_std_tum_bonn.csv", mean_std_headers, mean_std_rows)
        outputs.append(str(summary_root / "multiseed_mean_std_tum_bonn.csv"))

    # Auto-refresh P6-P9 deliverables (default on). Use --no_refresh_p6_p9 to skip.
    project_root = Path(__file__).resolve().parents[1]
    if bool(args.refresh_p6_p9):
        # P7: efficiency v2 + quality-speed plot
        ok, err = run_python_script(
            python_exe=str(args.python_exe),
            script_rel="experiments/p7/run_p7_efficiency_v2.py",
            script_args=[
                "--out_root",
                str(post_cleanup_root / "p7_speed_probe"),
                "--out_csv",
                str(p7_eff_v2_csv),
                "--out_md",
                str(p7_eff_v2_md),
                "--out_json",
                str(p7_eff_v2_json),
            ],
            project_root=project_root,
            verbose=bool(args.verbose),
        )
        if not ok:
            missing.append(err)

        # P8: native external baseline matrix
        ok, err = run_python_script(
            python_exe=str(args.python_exe),
            script_rel="experiments/p8/run_p8_native_external.py",
            script_args=[],
            project_root=project_root,
            verbose=bool(args.verbose),
        )
        if not ok:
            missing.append(err)

        # P6: top-tier main metrics (Acc/Comp/Comp-R + significance)
        ok, err = run_python_script(
            python_exe=str(args.python_exe),
            script_rel="scripts/build_local_mapping_main_toptier.py",
            script_args=[
                "--tum_multiseed",
                str(tum_recon_ms_path),
                "--bonn_multiseed",
                str(bonn_recon_ms_path),
                "--out_csv",
                str(p6_main_csv),
                "--out_md",
                str(p6_main_md),
            ],
            project_root=project_root,
            verbose=bool(args.verbose),
        )
        if not ok:
            missing.append(err)

        # P9: stress summary + failure boundaries
        ok, err = run_python_script(
            python_exe=str(args.python_exe),
            script_rel="experiments/p9/build_p9_stress_report.py",
            script_args=[
                "--out_csv",
                str(p9_stress_csv),
                "--out_failure_md",
                str(p9_failure_md),
            ],
            project_root=project_root,
            verbose=bool(args.verbose),
        )
        if not ok:
            missing.append(err)

        # P14: final 4D stress summary + final failure boundaries
        ok, err = run_python_script(
            python_exe=str(args.python_exe),
            script_rel="experiments/p14/build_p14_stress_report.py",
            script_args=[
                "--out_csv",
                str(p14_stress_csv),
                "--out_png",
                str(p14_curve_png),
                "--out_failure_md",
                str(p14_failure_md),
            ],
            project_root=project_root,
            verbose=bool(args.verbose),
        )
        if not ok:
            missing.append(err)

    # Track P6-P14 artifacts in output list (and mark missing when absent).
    for p in [
        p6_main_csv,
        p6_main_md,
        p7_eff_v2_csv,
        p7_eff_v2_md,
        p7_eff_v2_json,
        p8_native_recon,
        p8_native_dyn,
        p8_native_runtime,
        p9_stress_csv,
        p9_failure_md,
        p14_stress_csv,
        p14_failure_md,
        p14_curve_png,
        p14_meta_json,
        p10_precision_csv,
        p11_eff_final_csv,
        p11_eff_final_json,
        p10_probe_scan_csv,
        p10_tradeoff_png,
        p11_tradeoff_png,
        p15_eff_rt_csv,
        p15_eff_rt_json,
        p15_profile_md,
        p15_tradeoff_png,
    ]:
        if p.exists():
            outputs.append(str(p))
        else:
            missing.append(str(p))

    # De-duplicate for a cleaner manifest.
    outputs = list(dict.fromkeys(outputs))
    missing = list(dict.fromkeys(missing))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_cleanup_root": str(post_cleanup_root),
        "summary_root": str(summary_root),
        "prefer_p4_final": bool(args.prefer_p4_final),
        "refresh_p6_p9": bool(args.refresh_p6_p9),
        "outputs": outputs,
        "missing_sources": missing,
    }
    with (summary_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    outputs.append(str(summary_root / "manifest.json"))

    print("[done] updated summary tables")
    if args.verbose:
        for p in outputs:
            print(f"  + {p}")
    if missing:
        print("[warn] some optional sources are missing:")
        for p in missing:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
