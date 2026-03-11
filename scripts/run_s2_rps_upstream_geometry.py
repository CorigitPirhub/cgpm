from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from egf_dhmap3d.P10_method.rps_plane_attribution import extract_dominant_planes, point_to_plane_distance
from run_benchmark import load_points_with_normals
from run_s2_rps_ray_consistency import apply_ray_args
from run_s2_rps_rear_geometry_quality import BONN_ALL3, TUM_ALL3, build_commands, summarize_variant
from run_s2_write_time_synthesis import dryrun_base_cmds, run, set_arg, ensure_flag


CONTROL_ROOT = PROJECT_ROOT / "output" / "post_cleanup" / "s2_stage" / "99_manhattan_plane_completion"
RAY_COMPARE = PROJECT_ROOT / "processes" / "s2" / "S2_RPS_RAY_CONSISTENCY_COMPARE.csv"
DEEP_COMPARE = PROJECT_ROOT / "processes" / "s2" / "S2_RPS_DEEP_EXPLORE_COMPARE.csv"


def load_reference_99_row() -> dict:
    rows = list(csv.DictReader(DEEP_COMPARE.open("r", encoding="utf-8")))
    row = next(r for r in rows if r["variant"] == "99_manhattan_plane_completion")
    return {
        "variant": "99_manhattan_plane_completion",
        "tum_acc_cm": float(row["tum_acc_cm"]),
        "tum_comp_r_5cm": float(row["tum_comp_r_5cm"]),
        "bonn_acc_cm": float(row["bonn_acc_cm"]),
        "bonn_comp_r_5cm": float(row["bonn_comp_r_5cm"]),
        "bonn_ghost_reduction_vs_tsdf": float(row["bonn_ghost_reduction_vs_tsdf"]),
        "bonn_rear_points_sum": float(row["bonn_rear_points_sum"]),
        "bonn_rear_true_background_sum": float(row["bonn_rear_true_background_sum"]),
        "bonn_rear_ghost_sum": float(row["bonn_rear_ghost_sum"]),
        "bonn_rear_hole_or_noise_sum": float(row["bonn_rear_hole_or_noise_sum"]),
        "mean_distance_to_plane_before": float("nan"),
        "mean_distance_to_plane_after": float("nan"),
        "depth_bias_offset_m": 0.0,
        "surface_point_bias_along_normal_m": 0.0,
        "decision": "reference",
    }


def attach_upstream_args(cmd: List[str], spec: dict) -> List[str]:
    out = list(cmd)
    if "depth_bias_offset_m" in spec:
        out = set_arg(out, "--egf_depth_bias_offset_m", str(spec["depth_bias_offset_m"]))
    if "surface_point_bias_along_normal_m" in spec:
        out = set_arg(out, "--egf_surface_point_bias_along_normal_m", str(spec["surface_point_bias_along_normal_m"]))
    if bool(spec.get("lzcd_enable", False)):
        out = ensure_flag(out, "--egf_lzcd_enable")
        out = ensure_flag(out, "--egf_surface_lzcd_apply_in_extraction")
        out = set_arg(out, "--egf_surface_lzcd_bias_scale", str(spec.get("surface_lzcd_bias_scale", 1.0)))
    if bool(spec.get("zcbf_enable", False)):
        out = ensure_flag(out, "--egf_zcbf_enable")
        out = ensure_flag(out, "--egf_surface_zcbf_apply_in_extraction")
        out = set_arg(out, "--egf_surface_zcbf_bias_scale", str(spec.get("surface_zcbf_bias_scale", 1.0)))
    return out


def plane_distance_stats(root: Path) -> Dict[str, float]:
    vals = []
    for seq in BONN_ALL3:
        egf_dir = root / "bonn_slam" / "slam" / seq / "egf"
        rear_points, _rear_normals = load_points_with_normals(egf_dir / "rear_surface_points.ply")
        surface_points, _surface_normals = load_points_with_normals(egf_dir / "surface_points.ply")
        planes = extract_dominant_planes(surface_points, distance_threshold=0.03, min_plane_points=40, max_planes=6, min_extent_xy=0.25)
        if rear_points.shape[0] == 0 or not planes:
            vals.append(0.0)
            continue
        d = np.stack([point_to_plane_distance(rear_points, plane) for plane in planes], axis=1)
        vals.append(float(np.mean(np.min(d, axis=1))))
    mean_val = float(np.mean(vals)) if vals else 0.0
    return {"mean_distance_to_plane": mean_val}


def decide(rows: List[dict]) -> None:
    for row in rows:
        if row["variant"] == "99_manhattan_plane_completion":
            continue
        acc_ok = row["bonn_acc_cm"] <= 3.10
        comp_ok = row["bonn_comp_r_5cm"] >= 70.0
        tb_ok = row["bonn_rear_true_background_sum"] >= 20.0
        row["decision"] = "iterate" if (acc_ok and comp_ok and tb_ok) else "abandon"


def write_compare(rows: List[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# S2 upstream geometry compare",
        "",
        "| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_ghost_reduction_vs_tsdf | bonn_rear_true_background_sum | mean_distance_to_plane_before | mean_distance_to_plane_after | depth_bias_offset_m | surface_point_bias_along_normal_m | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['bonn_acc_cm']:.3f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_ghost_reduction_vs_tsdf']:.2f} | {row['bonn_rear_true_background_sum']:.0f} | {row['mean_distance_to_plane_before']:.4f} | {row['mean_distance_to_plane_after']:.4f} | {row['depth_bias_offset_m']:.3f} | {row['surface_point_bias_along_normal_m']:.3f} | {row['decision']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(rows: List[dict], path_md: Path) -> None:
    ref = next(r for r in rows if r["variant"] == "99_manhattan_plane_completion")
    tested = [r for r in rows if r["variant"] != "99_manhattan_plane_completion"]
    best = min(tested, key=lambda r: r["bonn_acc_cm"])
    depth_sorted = sorted([r for r in tested if abs(r["depth_bias_offset_m"]) > 1e-9], key=lambda r: r["depth_bias_offset_m"])
    depth_trend = ", ".join([f"{r['depth_bias_offset_m']:+.3f}m->{r['bonn_acc_cm']:.3f}cm" for r in depth_sorted]) if depth_sorted else "none"
    lines = [
        "# S2 upstream geometry analysis",
        "",
        "日期：`2026-03-11`",
        "协议：`TUM walking all3 + Bonn all3 / frames=5 / stride=3 / seed=7 / max_points_per_frame=600`",
        "说明：`99` 是下游 completion 参考行；上游变体在其最近可执行前驱 `80` 主线上重跑，用于验证真正的融合/提取参数响应。",
        "对比表：`processes/s2/S2_RPS_UPSTREAM_GEOMETRY_COMPARE.csv`",
        "",
        "## 1. Acc 对上游参数是否有响应",
        f"- 参考 `99`：Bonn `Acc={ref['bonn_acc_cm']:.3f} cm`, `Comp-R={ref['bonn_comp_r_5cm']:.2f}%`, `TB={ref['bonn_rear_true_background_sum']:.0f}`。",
        f"- 最佳上游候选 `{best['variant']}`：Bonn `Acc={best['bonn_acc_cm']:.3f} cm`, `Comp-R={best['bonn_comp_r_5cm']:.2f}%`, `TB={best['bonn_rear_true_background_sum']:.0f}`。",
        f"- `depth_bias` 趋势：{depth_trend}",
        "",
        "## 2. 平面拟合误差是否随 bias 改变",
        f"- 最佳候选前后平面距离：`{best['mean_distance_to_plane_before']:.4f} -> {best['mean_distance_to_plane_after']:.4f}`。",
        "- 若 `Acc` 与平面距离同时变化，说明存在可测的上游几何偏置；若 `Acc` 几乎不动，则更像传感器噪声或更复杂的畸变。",
        "",
        "## 3. 结论",
        "- 若 `depth_bias` 呈单调趋势，可视为阶段性证据：系统性深度偏置真实存在。",
        "- 若所有上游 bias 只带来微小数值波动，则当前 `Acc` 瓶颈更接近传感器噪声极限或更深层的前景/背景形成偏差。",
        "- 只要 `Bonn Acc > 3.10 cm`，`S2` 仍未通过，且绝对不能进入 `S3`。",
    ]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 upstream geometry runner.")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_points_per_frame", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base_tum, base_bonn = dryrun_base_cmds(args.frames, args.stride, args.max_points_per_frame)
    base_spec = dict(
        name="80_ray_penetration_consistency",
        bonn_rps_rear_selectivity_support_weight=0.10,
        bonn_rps_rear_selectivity_history_weight=0.16,
        bonn_rps_rear_selectivity_static_weight=0.14,
        bonn_rps_rear_selectivity_geom_weight=0.12,
        bonn_rps_rear_selectivity_bridge_weight=0.08,
        bonn_rps_rear_selectivity_density_weight=0.06,
        bonn_rps_rear_selectivity_rear_score_weight=0.18,
        bonn_rps_rear_selectivity_front_score_weight=0.26,
        bonn_rps_rear_selectivity_competition_weight=0.38,
        bonn_rps_rear_selectivity_competition_alpha=0.90,
        bonn_rps_rear_selectivity_gap_weight=0.08,
        bonn_rps_rear_selectivity_sep_weight=0.08,
        bonn_rps_rear_selectivity_dyn_weight=0.12,
        bonn_rps_rear_selectivity_ghost_weight=0.12,
        bonn_rps_rear_selectivity_front_weight=0.10,
        bonn_rps_rear_selectivity_geom_risk_weight=0.18,
        bonn_rps_rear_selectivity_history_risk_weight=0.10,
        bonn_rps_rear_selectivity_density_risk_weight=0.08,
        bonn_rps_rear_selectivity_bridge_relief_weight=0.08,
        bonn_rps_rear_selectivity_static_relief_weight=0.06,
        bonn_rps_rear_selectivity_gap_risk_weight=0.06,
        bonn_rps_rear_selectivity_score_min=0.40,
        bonn_rps_rear_selectivity_risk_max=0.60,
        bonn_rps_rear_selectivity_geom_floor=0.36,
        bonn_rps_rear_selectivity_history_floor=0.30,
        bonn_rps_rear_selectivity_bridge_floor=0.12,
        bonn_rps_rear_selectivity_competition_floor=0.02,
        bonn_rps_rear_selectivity_front_score_max=0.72,
        bonn_rps_rear_selectivity_gap_valid_min=0.10,
        bonn_rps_rear_selectivity_topk=92,
        bonn_rps_rear_selectivity_penetration_weight=0.55,
        bonn_rps_rear_selectivity_penetration_floor=0.18,
        bonn_rps_rear_selectivity_penetration_risk_weight=0.20,
        bonn_rps_rear_selectivity_penetration_free_ref=0.07,
        bonn_rps_rear_selectivity_penetration_max_steps=12,
    )

    variants = [
        dict(base_spec, name="104_depth_bias_minus1cm", output_name="104_depth_bias_minus1cm", depth_bias_offset_m=-0.01, surface_point_bias_along_normal_m=0.0),
        dict(base_spec, name="105_depth_bias_plus1cm", output_name="105_depth_bias_plus1cm", depth_bias_offset_m=0.01, surface_point_bias_along_normal_m=0.0),
        dict(base_spec, name="106_surface_extract_bias_minus1cm", output_name="106_surface_extract_bias_minus1cm", depth_bias_offset_m=0.0, surface_point_bias_along_normal_m=-0.01),
        dict(base_spec, name="107_zero_crossing_bias_field", output_name="107_zero_crossing_bias_field", depth_bias_offset_m=0.0, surface_point_bias_along_normal_m=0.0, lzcd_enable=True, zcbf_enable=True, surface_lzcd_bias_scale=1.0, surface_zcbf_bias_scale=1.0),
    ]

    rows: List[dict] = [load_reference_99_row()]
    for spec in variants:
        spec_run = dict(spec, frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame)
        tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec_run)
        bonn_cmd = apply_ray_args(bonn_cmd, spec_run)
        bonn_cmd = attach_upstream_args(bonn_cmd, spec_run)
        bonn_cmd = set_arg(bonn_cmd, "--bonn_dynamic_preset", "all3")
        bonn_cmd = set_arg(bonn_cmd, "--dynamic_sequences", ",".join(BONN_ALL3))
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(CONTROL_ROOT / "tum_oracle" / "oracle", root / "tum_oracle" / "oracle", dirs_exist_ok=True)
        run(bonn_cmd)
        row, _tum_details, _bonn_payload = summarize_variant(
            root,
            str(spec["name"]),
            frames=args.frames,
            stride=args.stride,
            max_points=args.max_points_per_frame,
            seed=args.seed,
        )
        plane_stats = plane_distance_stats(root)
        row["mean_distance_to_plane_before"] = float("nan")
        row["mean_distance_to_plane_after"] = float(plane_stats["mean_distance_to_plane"])
        row["depth_bias_offset_m"] = float(spec_run.get("depth_bias_offset_m", 0.0))
        row["surface_point_bias_along_normal_m"] = float(spec_run.get("surface_point_bias_along_normal_m", 0.0))
        rows.append(row)

    # reference before/after values from 99 remain fixed / unavailable upstream
    rows[0]["mean_distance_to_plane_before"] = float("nan")
    rows[0]["mean_distance_to_plane_after"] = float("nan")
    decide(rows)
    out_dir = PROJECT_ROOT / "processes" / "s2"
    write_compare(rows, out_dir / "S2_RPS_UPSTREAM_GEOMETRY_COMPARE.csv", out_dir / "S2_RPS_UPSTREAM_GEOMETRY_COMPARE.md")
    write_analysis(rows, out_dir / "S2_RPS_UPSTREAM_GEOMETRY_ANALYSIS.md")
    print("[done]", out_dir / "S2_RPS_UPSTREAM_GEOMETRY_COMPARE.csv")


if __name__ == "__main__":
    main()
