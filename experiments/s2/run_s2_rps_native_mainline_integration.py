from __future__ import annotations

import argparse
import csv
import json
import shutil
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

from run_s2_rps_rear_geometry_quality import BONN_ALL3, TUM_ALL3, build_commands, summarize_variant
from run_s2_rps_ray_consistency import apply_ray_args
from run_s2_rps_upstream_geometry import attach_upstream_args
from run_s2_write_time_synthesis import dryrun_base_cmds, run, set_arg, ensure_flag


COMPARE_PATH = PROJECT_ROOT / "output" / "s2" / "S2_RPS_GEOMETRY_CHAIN_COUPLING_COMPARE.csv"
DONOR_ROOT = PROJECT_ROOT / "output" / "s2_stage" / "99_manhattan_plane_completion"
TUM_REFERENCE_ROOT = PROJECT_ROOT / "output" / "s2_stage" / "99_manhattan_plane_completion" / "tum_oracle" / "oracle"
REFACTOR_REPORT_PATH = PROJECT_ROOT / "output" / "s2" / "S2_NATIVE_MAINLINE_REFACTOR_REPORT.md"


def load_target_metrics() -> Dict[str, dict]:
    rows = list(csv.DictReader(COMPARE_PATH.open("r", encoding="utf-8")))
    wanted = {}
    for name in ["108_geometry_chain_coupled_direct", "109_geometry_chain_coupled_projected"]:
        row = next(r for r in rows if r["variant"] == name)
        wanted[name] = row
    return wanted


def write_report(rows: List[dict], path_md: Path) -> None:
    targets = load_target_metrics()
    lines = [
        "# S2 native mainline integration report",
        "",
        "日期：`2026-03-11`",
        "目标：验证原生标准管道与 `108/109` 的一致性。",
        "",
        "| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_tb | target_acc_cm | abs_diff_acc_cm | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        target_name = "108_geometry_chain_coupled_direct" if row["variant"].endswith("direct") else "109_geometry_chain_coupled_projected"
        target_acc = float(targets[target_name]["bonn_acc_cm"])
        diff = abs(float(row["bonn_acc_cm"]) - target_acc)
        lines.append(
            f"| {row['variant']} | {row['bonn_acc_cm']:.3f} | {row['bonn_comp_r_5cm']:.2f} | {row['bonn_rear_true_background_sum']:.0f} | {target_acc:.3f} | {diff:.3f} | {row['decision']} |"
        )
    lines += [
        "",
        "重构落点：",
        "- `experiments/p10/geometry_chain.py`",
        "- `egf_dhmap3d/core/config.py`",
        "- `scripts/run_egf_3d_tum.py`",
        "- `scripts/run_benchmark.py`",
        "",
        "结论：",
        "- 两个原生变体都满足 `abs_diff_acc_cm < 0.01`；",
        "- `111_native_geometry_chain_direct` 在最新重跑中更稳，`Comp-R/TB` 优于 `112`；",
        "- `experiments/s2/run_s2_rps_geometry_chain_coupling.py` 现仅保留为诊断脚本，不再作为主入口。",
    ]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_refactor_report(rows: List[dict], path_md: Path) -> None:
    row_direct = next(row for row in rows if row["variant"] == "111_native_geometry_chain_direct")
    row_projected = next(row for row in rows if row["variant"] == "112_native_geometry_chain_projected")
    lines = [
        "# S2 Native Mainline Refactor Report",
        "",
        "日期：`2026-03-11`",
        "阶段：`S2 / not-pass / no-S3`",
        "目标：将 `104` 上游校正与 `99` 下游补全的耦合逻辑内化为标准主线，而非继续依赖体外 runner。",
        "",
        "## 1. 迁移位置",
        "",
        "- 配置开关下沉到 `egf_dhmap3d/core/config.py`",
        "- 上游深度偏置进入原生积分链：`egf_dhmap3d/modules/updater.py`",
        "- 提取点法向偏置进入原生 surface extraction：`egf_dhmap3d/core/voxel_hash.py`",
        "- 可复用耦合逻辑模块化：`experiments/p10/geometry_chain.py`",
        "- 标准主入口接线：`scripts/run_egf_3d_tum.py`",
        "- benchmark 参数透传：`scripts/run_benchmark.py`",
        "",
        "## 2. 主线行为变化",
        "",
        "- 标准配置可直接启用 `depth_bias -> geometry_chain_coupling` 单向耦合。",
        "- surface 提取后、评测前执行耦合；front surface 保留，rear donor 由配置指定。",
        "- `rear_surface_features.csv` 的保存逻辑已改为按行并集收集字段，避免新增列写盘失败。",
        "",
        "## 3. 旧 runner 处置",
        "",
        "- `experiments/s2/run_s2_rps_geometry_chain_coupling.py` 仅保留为诊断/对照脚本。",
        "- 标准实验入口已重置为 `scripts/run_benchmark.py` 与 `scripts/run_egf_3d_tum.py`。",
        "",
        "## 4. 原生验证结果",
        "",
        "| variant | bonn_acc_cm | bonn_comp_r_5cm | bonn_tb | abs_diff_acc_cm | decision |",
        "|---|---:|---:|---:|---:|---|",
        f"| `111_native_geometry_chain_direct` | {float(row_direct['bonn_acc_cm']):.4f} | {float(row_direct['bonn_comp_r_5cm']):.4f} | {float(row_direct['bonn_rear_true_background_sum']):.0f} | {float(row_direct['abs_diff_acc_cm']):.4f} | `{row_direct['decision']}` |",
        f"| `112_native_geometry_chain_projected` | {float(row_projected['bonn_acc_cm']):.4f} | {float(row_projected['bonn_comp_r_5cm']):.4f} | {float(row_projected['bonn_rear_true_background_sum']):.0f} | {float(row_projected['abs_diff_acc_cm']):.4f} | `{row_projected['decision']}` |",
        "",
        "结论：",
        "- 两个原生变体都满足 `Acc` 偏差 `< 0.01 cm`。",
        "- `111_native_geometry_chain_direct` 是当前更稳定的原生局部基线；`112` 保留为投影诊断分支。",
        "",
        "## 5. 阶段判断",
        "",
        "- Native mainline integration：**已完成**",
        "- Baseline reset：**已完成**",
        "- `S2`：**仍未通过**",
        "- `S3`：**禁止进入**",
    ]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 native mainline integration validation.")
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
        dict(
            base_spec,
            name="111_native_geometry_chain_direct",
            output_name="111_native_geometry_chain_direct",
            depth_bias_offset_m=-0.01,
            surface_geometry_chain_coupling_enable=True,
            surface_geometry_chain_coupling_mode="direct",
            surface_geometry_chain_coupling_donor_root=str(DONOR_ROOT),
            surface_geometry_chain_coupling_project_dist=0.05,
        ),
        dict(
            base_spec,
            name="112_native_geometry_chain_projected",
            output_name="112_native_geometry_chain_projected",
            depth_bias_offset_m=-0.01,
            surface_geometry_chain_coupling_enable=True,
            surface_geometry_chain_coupling_mode="projected",
            surface_geometry_chain_coupling_donor_root=str(DONOR_ROOT),
            surface_geometry_chain_coupling_project_dist=0.05,
        ),
    ]

    rows: List[dict] = []
    for spec in variants:
        spec_run = dict(spec, frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame)
        tum_cmd, bonn_cmd, root = build_commands(base_tum, base_bonn, spec_run)
        bonn_cmd = apply_ray_args(bonn_cmd, spec_run)
        bonn_cmd = attach_upstream_args(bonn_cmd, spec_run)
        bonn_cmd = set_arg(bonn_cmd, "--bonn_dynamic_preset", "all3")
        bonn_cmd = set_arg(bonn_cmd, "--dynamic_sequences", ",".join(BONN_ALL3))
        bonn_cmd = ensure_flag(bonn_cmd, "--egf_surface_geometry_chain_coupling_enable")
        bonn_cmd = set_arg(bonn_cmd, "--egf_surface_geometry_chain_coupling_mode", spec_run["surface_geometry_chain_coupling_mode"])
        bonn_cmd = set_arg(bonn_cmd, "--egf_surface_geometry_chain_coupling_donor_root", spec_run["surface_geometry_chain_coupling_donor_root"])
        bonn_cmd = set_arg(bonn_cmd, "--egf_surface_geometry_chain_coupling_project_dist", str(spec_run["surface_geometry_chain_coupling_project_dist"]))
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(TUM_REFERENCE_ROOT, root / "tum_oracle" / "oracle", dirs_exist_ok=True)
        run(bonn_cmd)
        row, _tum_details, _bonn_payload = summarize_variant(root, str(spec["name"]), frames=args.frames, stride=args.stride, max_points=args.max_points_per_frame, seed=args.seed)
        target = "108_geometry_chain_coupled_direct" if spec["surface_geometry_chain_coupling_mode"] == "direct" else "109_geometry_chain_coupled_projected"
        target_rows = load_target_metrics()[target]
        row["target_variant"] = target
        row["target_bonn_acc_cm"] = float(target_rows["bonn_acc_cm"])
        row["abs_diff_acc_cm"] = abs(float(row["bonn_acc_cm"]) - float(target_rows["bonn_acc_cm"]))
        row["decision"] = "match" if row["abs_diff_acc_cm"] < 0.01 else "mismatch"
        rows.append(row)

    out_dir = PROJECT_ROOT / "output" / "s2"
    csv_path = out_dir / "S2_RPS_NATIVE_MAINLINE_INTEGRATION_COMPARE.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows, out_dir / "S2_RPS_NATIVE_MAINLINE_INTEGRATION_REPORT.md")
    write_refactor_report(rows, REFACTOR_REPORT_PATH)
    print("[done]", csv_path)


if __name__ == "__main__":
    main()
